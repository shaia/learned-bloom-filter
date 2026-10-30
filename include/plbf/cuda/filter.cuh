#pragma once
//
// The filter on an NVIDIA GPU. Histogramming the scores, inserting the keys and
// answering queries run as kernels; the threshold search stays on the CPU,
// where it takes well under a millisecond and would only wait on launches.
//
// Every kernel follows the CPU code step for step: the same segment index, the
// same region tables (built by region_tables::build), the same block index and
// bit pattern. A filter built here is therefore bit-identical to one built by
// plbf::filter from the same keys, and answers every query the same way;
// test_cuda holds it to both.
//
// Include from a .cu file compiled by nvcc (CUDA 12 or later, C++20).

#include "../filter.hpp"
#include "../optimize.hpp"
#include "../region_tables.hpp"
#include "../segments.hpp"
#include "../split_block.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace plbf::cuda {

// Throws on any CUDA error, naming the call that produced it.
inline void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("plbf::cuda: ") + what + ": " + cudaGetErrorString(e));
}

// An owning device allocation. Move-only; copies go through upload/download.
template <class T>
class device_buffer {
public:
    device_buffer() = default;
    explicit device_buffer(std::size_t n) : n_(n) {
        if (n != 0) check(cudaMalloc(reinterpret_cast<void**>(&p_), n * sizeof(T)), "cudaMalloc");
    }
    static device_buffer from_host(std::span<const T> h) {
        device_buffer b(h.size());
        b.upload(h);
        return b;
    }
    ~device_buffer() { release(); }
    device_buffer(device_buffer&& o) noexcept : p_(std::exchange(o.p_, nullptr)), n_(std::exchange(o.n_, 0)) {}
    device_buffer& operator=(device_buffer&& o) noexcept {
        if (this != &o) {
            release();
            p_ = std::exchange(o.p_, nullptr);
            n_ = std::exchange(o.n_, 0);
        }
        return *this;
    }
    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;

    T* data() noexcept { return p_; }
    const T* data() const noexcept { return p_; }
    std::size_t size() const noexcept { return n_; }

    void upload(std::span<const T> h) {
        if (h.size() != n_) throw std::invalid_argument("plbf::cuda: upload size mismatch");
        if (n_ != 0) check(cudaMemcpy(p_, h.data(), n_ * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    }
    std::vector<T> download() const {
        std::vector<T> h(n_);
        if (n_ != 0) check(cudaMemcpy(h.data(), p_, n_ * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
        return h;
    }

private:
    void release() noexcept {
        if (p_ != nullptr) cudaFree(p_);  // nothing useful to do with an error here
        p_ = nullptr;
        n_ = 0;
    }

    T* p_ = nullptr;
    std::size_t n_ = 0;
};

namespace detail {

// Device code cannot read split_block::salt (a host constexpr array), so the
// probe carries its own copy. The static_assert pins the two together.
#define PLBF_CUDA_SALTS \
    0x47b6137bU, 0x44974d91U, 0x8824ad5bU, 0xa2b7289dU, 0x705495c7U, 0x2df1424bU, 0x9efc4947U, 0x5c6bfb31U

constexpr bool salts_match() {
    constexpr std::uint32_t s[8] = {PLBF_CUDA_SALTS};
    for (int i = 0; i < 8; ++i) {
        if (s[i] != split_block::salt[i]) return false;
    }
    return true;
}
static_assert(salts_match(), "the device salts must be split_block::salt");
static_assert(sizeof(split_block::block) == 2 * sizeof(uint4), "a block is two 16-byte loads");

// What the routing kernels read, passed by value.
struct routing {
    const double* thresholds;                // N + 1
    const std::uint16_t* region_of_segment;  // N + 2
    const unsigned long long* slices;        // k, packed as region_tables::make_slice
    std::uint32_t segments;                  // N
};

// segment_grid::index(), on the device. ceil(score * N) is within one of the
// answer (the argument is at segment_grid::index4_avx2), so one step each way
// replaces index()'s loops. __dmul_rn keeps the multiply correctly rounded
// whatever fast-math flags the including TU was built with.
__device__ __forceinline__ std::uint32_t segment_index(const double* __restrict__ t, std::uint32_t n,
                                                      double score) {
    if (!(score > 0.0)) return 1;  // also catches NaN
    if (score >= 1.0) return n;
    std::uint32_t i = static_cast<std::uint32_t>(ceil(__dmul_rn(score, static_cast<double>(n))));
    if (i < 1) i = 1;
    if (i > n) i = n;
    if (i > 1 && __ldg(t + i - 1) >= score) {
        --i;
    } else if (i < n && __ldg(t + i) < score) {
        ++i;
    }
    return i;
}

// filter::block_for(), as a block index. __umulhi is the multiply-shift of
// split_block::block_index in one instruction.
__device__ __forceinline__ unsigned long long block_of(const routing& r, std::uint64_t hash, double score) {
    const std::uint32_t seg = segment_index(r.thresholds, r.segments, score);
    const unsigned long long sl = __ldg(r.slices + __ldg(r.region_of_segment + seg));
    return (sl >> 32) + __umulhi(static_cast<std::uint32_t>(hash >> 32), static_cast<std::uint32_t>(sl));
}

__device__ __forceinline__ std::uint32_t bit(std::uint32_t x, std::uint32_t salt) {
    return 1U << ((x * salt) >> 27);
}

// split_block::probe_scalar on a block read as two 16-byte loads.
__device__ __forceinline__ bool probe(const uint4* __restrict__ blocks, unsigned long long idx, std::uint32_t x) {
    constexpr std::uint32_t salt[8] = {PLBF_CUDA_SALTS};
    const uint4 a = __ldg(blocks + 2 * idx);
    const uint4 b = __ldg(blocks + 2 * idx + 1);
    const std::uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    std::uint32_t missing = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) missing |= bit(x, salt[i]) & ~w[i];
    return missing == 0;
}

// The kernels are templates only so that a header can define them: every TU
// that includes this gets the same instantiation, with no ODR clash.

// Per-segment counts. Shared: each block counts in shared memory and adds its
// totals once, which keeps the global atomics off the hot path; it needs
// N * 4 bytes of shared memory. Otherwise every score is a global atomic.
template <bool Shared>
__global__ void histogram_kernel(const double* __restrict__ t, std::uint32_t n_seg,
                                 const double* __restrict__ scores, std::size_t n,
                                 unsigned long long* __restrict__ counts) {
    extern __shared__ unsigned int local[];
    if constexpr (Shared) {
        for (std::uint32_t j = threadIdx.x; j < n_seg; j += blockDim.x) local[j] = 0;
        __syncthreads();
    }
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        const std::uint32_t s = segment_index(t, n_seg, scores[i]) - 1;
        if constexpr (Shared) {
            atomicAdd(&local[s], 1U);
        } else {
            atomicAdd(&counts[s], 1ULL);
        }
    }
    if constexpr (Shared) {
        __syncthreads();
        for (std::uint32_t j = threadIdx.x; j < n_seg; j += blockDim.x) {
            if (local[j] != 0) atomicAdd(&counts[j], static_cast<unsigned long long>(local[j]));
        }
    }
}

// split_block::insert through atomics: the 8 words of a block as 4 64-bit
// ORs. OR commutes, so the order threads land in cannot change the bits.
template <int Unused = 0>
__global__ void insert_kernel(routing r, unsigned long long* __restrict__ blocks,
                              const std::uint64_t* __restrict__ hashes, const double* __restrict__ scores,
                              std::size_t n) {
    constexpr std::uint32_t salt[8] = {PLBF_CUDA_SALTS};
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        const std::uint64_t h = hashes[i];
        const std::uint32_t seg = segment_index(r.thresholds, r.segments, scores[i]);
        const unsigned long long sl = __ldg(r.slices + __ldg(r.region_of_segment + seg));
        if ((sl >> 32) < 2) continue;  // a sentinel: nothing to insert
        const unsigned long long idx =
            (sl >> 32) + __umulhi(static_cast<std::uint32_t>(h >> 32), static_cast<std::uint32_t>(sl));
        const auto x = static_cast<std::uint32_t>(h);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned long long m = static_cast<unsigned long long>(bit(x, salt[2 * j])) |
                                         static_cast<unsigned long long>(bit(x, salt[2 * j + 1])) << 32;
            atomicOr(&blocks[4 * idx + j], m);
        }
    }
}

template <int Unused = 0>
__global__ void contains_kernel(routing r, const uint4* __restrict__ blocks, const std::uint64_t* __restrict__ hashes,
                                const double* __restrict__ scores, std::uint8_t* __restrict__ out, std::size_t n) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        const std::uint64_t h = hashes[i];
        out[i] = probe(blocks, block_of(r, h, scores[i]), static_cast<std::uint32_t>(h)) ? 1 : 0;
    }
}

constexpr int threads_per_block = 256;

// Enough blocks to fill every SM several times over; the grid-stride loops do
// the rest, so a launch never needs more than this whatever n is.
inline unsigned blocks_for(std::size_t n, int sm_count) {
    const std::size_t wanted = (n + threads_per_block - 1) / threads_per_block;
    const std::size_t cap = static_cast<std::size_t>(sm_count) * 16;
    const std::size_t b = wanted < cap ? wanted : cap;
    return static_cast<unsigned>(b == 0 ? 1 : b);
}

inline int current_sm_count() {
    int dev = 0;
    check(cudaGetDevice(&dev), "cudaGetDevice");
    int sms = 0;
    check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "cudaDeviceGetAttribute");
    return sms;
}

inline std::vector<double> thresholds_of(const segment_grid& grid) {
    std::vector<double> t(grid.segments() + 1);
    for (std::size_t i = 0; i < t.size(); ++i) t[i] = grid.threshold(i);
    return t;
}

// Raw per-segment counts of n device-resident scores; synchronous, since the
// caller needs them on the host.
inline std::vector<std::uint64_t> count_segments(const double* thresholds, std::uint32_t n_seg, const double* scores,
                                                 std::size_t n, int sm_count, cudaStream_t stream) {
    device_buffer<unsigned long long> counts(n_seg);
    check(cudaMemsetAsync(counts.data(), 0, n_seg * sizeof(unsigned long long), stream), "cudaMemsetAsync");
    if (n != 0) {
        const std::size_t shared = static_cast<std::size_t>(n_seg) * sizeof(unsigned int);
        if (shared <= 48 * 1024) {
            histogram_kernel<true><<<blocks_for(n, sm_count), threads_per_block, shared, stream>>>(
                thresholds, n_seg, scores, n, counts.data());
        } else {
            histogram_kernel<false><<<blocks_for(n, sm_count), threads_per_block, 0, stream>>>(
                thresholds, n_seg, scores, n, counts.data());
        }
        check(cudaGetLastError(), "histogram_kernel launch");
    }
    std::vector<std::uint64_t> h(n_seg);
    check(cudaMemcpyAsync(h.data(), counts.data(), n_seg * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream),
          "cudaMemcpyAsync D2H");
    check(cudaStreamSynchronize(stream), "histogram");
    return h;
}

}  // namespace detail

// Raw per-segment counts of n scores already on the device: plbf::segment_counts
// on the GPU. out[s-1] is the count of segment s.
inline std::vector<std::uint64_t> segment_counts(const segment_grid& grid, const double* scores, std::size_t n,
                                                 cudaStream_t stream = nullptr) {
    const auto t = device_buffer<double>::from_host(detail::thresholds_of(grid));
    return detail::count_segments(t.data(), static_cast<std::uint32_t>(grid.segments()), scores, n,
                                  detail::current_sm_count(), stream);
}

class device_filter {
public:
    // Upload a filter built on the CPU.
    explicit device_filter(const plbf::filter& f)
        : segments_(f.grid().segments()),
          sm_count_(detail::current_sm_count()),
          layout_(f.regions()),
          keys_per_region_(f.keys_per_region()),
          tables_(f.tables()) {
        upload_tables(f.grid());
        blocks_ = device_buffer<split_block::block>::from_host(f.blocks());
    }

    // Build from keys and scores already on the device, bit-identical to
    // plbf::filter::build_for_fpr on the same data: the histograms and the
    // insertion run on the GPU, the threshold search on the CPU.
    static device_filter build_for_fpr(const std::uint64_t* key_hashes, const double* key_scores,
                                       std::size_t n_keys, const double* non_key_scores, std::size_t n_non_keys,
                                       double target_fpr, const options& opt = {}, cudaStream_t stream = nullptr) {
        return build(key_hashes, key_scores, n_keys, non_key_scores, n_non_keys, opt, stream,
                     [&](auto& g, auto& h, std::size_t n) {
                         return optimize_for_fpr(g, h, n, opt.regions, target_fpr, opt.search);
                     });
    }

    // As plbf::filter::build_for_memory: memory_bits of classic Bloom filter.
    static device_filter build_for_memory(const std::uint64_t* key_hashes, const double* key_scores,
                                          std::size_t n_keys, const double* non_key_scores, std::size_t n_non_keys,
                                          double memory_bits, const options& opt = {}, cudaStream_t stream = nullptr) {
        return build(key_hashes, key_scores, n_keys, non_key_scores, n_non_keys, opt, stream,
                     [&](auto& g, auto& h, std::size_t n) {
                         return optimize_for_memory(g, h, n, opt.regions, memory_bits, opt.search);
                     });
    }

    // out[i] = contains(hashes[i], scores[i]) for n device-resident queries,
    // as 0 or 1. Asynchronous on `stream`, like any kernel launch.
    void contains(const std::uint64_t* hashes, const double* scores, std::uint8_t* out, std::size_t n,
                  cudaStream_t stream = nullptr) const {
        if (n == 0) return;
        detail::contains_kernel<<<detail::blocks_for(n, sm_count_), detail::threads_per_block, 0, stream>>>(
            routing_view(), reinterpret_cast<const uint4*>(blocks_.data()), hashes, scores, out, n);
        check(cudaGetLastError(), "contains_kernel launch");
    }

    // The same filter on the CPU: same bits, same answers.
    plbf::filter to_host() const {
        return plbf::filter::from_parts(segments_, layout_, keys_per_region_, tables_, blocks_.download());
    }

    const plbf::layout& regions() const noexcept { return layout_; }
    const std::vector<std::size_t>& keys_per_region() const noexcept { return keys_per_region_; }
    const region_tables& tables() const noexcept { return tables_; }

    // Device bytes: blocks, routing tables and thresholds.
    std::size_t memory_bytes() const noexcept {
        return blocks_.size() * sizeof(split_block::block) + region_of_segment_.size() * sizeof(std::uint16_t) +
               slices_.size() * sizeof(unsigned long long) + thresholds_.size() * sizeof(double);
    }

private:
    explicit device_filter(std::size_t segments) : segments_(segments), sm_count_(detail::current_sm_count()) {}

    detail::routing routing_view() const noexcept {
        return {thresholds_.data(), region_of_segment_.data(), slices_.data(), static_cast<std::uint32_t>(segments_)};
    }

    void upload_thresholds(const segment_grid& grid) {
        thresholds_ = device_buffer<double>::from_host(detail::thresholds_of(grid));
    }

    void upload_tables(const segment_grid& grid) {
        upload_thresholds(grid);
        region_of_segment_ = device_buffer<std::uint16_t>::from_host(tables_.region_of_segment);
        // std::uint64_t is unsigned long on some platforms; the device side
        // wants the type __ldg and atomicOr are declared for.
        static_assert(sizeof(unsigned long long) == sizeof(std::uint64_t));
        slices_ = device_buffer<unsigned long long>(tables_.slices.size());
        check(cudaMemcpy(slices_.data(), tables_.slices.data(), tables_.slices.size() * sizeof(std::uint64_t),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy H2D");
    }

    template <class Optimize>
    static device_filter build(const std::uint64_t* key_hashes, const double* key_scores, std::size_t n_keys,
                               const double* non_key_scores, std::size_t n_non_keys, const options& opt,
                               cudaStream_t stream, Optimize optimize) {
        const segment_grid grid(opt.segments);
        device_filter f(opt.segments);
        f.upload_thresholds(grid);
        const auto n_seg = static_cast<std::uint32_t>(opt.segments);

        // The same steps as plbf::filter::build, with the two passes over the
        // scores on the GPU and only N counts each coming back.
        const std::vector<std::uint64_t> key_counts =
            detail::count_segments(f.thresholds_.data(), n_seg, key_scores, n_keys, f.sm_count_, stream);
        const std::vector<std::uint64_t> non_key_counts =
            detail::count_segments(f.thresholds_.data(), n_seg, non_key_scores, n_non_keys, f.sm_count_, stream);
        const auto g = segment_distribution::from_counts(key_counts);
        const auto h = segment_distribution::from_counts(non_key_counts);
        f.layout_ = optimize(g, h, n_keys);
        f.keys_per_region_ = plbf::keys_per_region(f.layout_, key_counts);
        f.tables_ = region_tables::build(f.layout_, opt.segments, f.keys_per_region_);
        f.upload_tables(grid);

        f.blocks_ = device_buffer<split_block::block>(f.tables_.block_count);
        const std::size_t bytes = f.blocks_.size() * sizeof(split_block::block);
        check(cudaMemsetAsync(f.blocks_.data(), 0, bytes, stream), "cudaMemsetAsync");
        check(cudaMemsetAsync(f.blocks_.data() + region_tables::accept_block, 0xff, sizeof(split_block::block), stream),
              "cudaMemsetAsync");
        if (n_keys != 0) {
            detail::insert_kernel<<<detail::blocks_for(n_keys, f.sm_count_), detail::threads_per_block, 0, stream>>>(
                f.routing_view(), reinterpret_cast<unsigned long long*>(f.blocks_.data()), key_hashes, key_scores, n_keys);
            check(cudaGetLastError(), "insert_kernel launch");
        }
        check(cudaStreamSynchronize(stream), "insert");
        return f;
    }

    std::size_t segments_;
    int sm_count_;  // asked once: cudaDeviceGetAttribute per launch would cost more than a small batch
    plbf::layout layout_;
    std::vector<std::size_t> keys_per_region_;
    region_tables tables_;  // host copy, for to_host() and inspection
    device_buffer<double> thresholds_;
    device_buffer<std::uint16_t> region_of_segment_;
    device_buffer<unsigned long long> slices_;
    device_buffer<split_block::block> blocks_;
};

}  // namespace plbf::cuda

#undef PLBF_CUDA_SALTS  // expanded where it was used; keep it out of including code
