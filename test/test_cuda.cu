// The CUDA filter against the CPU one, piece by piece and whole.
//
// The claim in cuda/filter.cuh is strong -- bit-identical filters, identical
// answers -- so every check here is an equality, not a tolerance: the device
// segment index against segment_grid::index(), the GPU histogram against
// segment_counts(), a GPU-built filter's tables and blocks against a
// CPU-built one's, and device answers against contains().

#include "harness.hpp"

#include <plbf/cuda/filter.cuh>
#include <plbf/filter.hpp>
#include <plbf/hash.hpp>
#include <plbf/segments.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace plbf;
using namespace plbf::test;
using plbf::cuda::device_buffer;

namespace {

__global__ void index_kernel(const double* t, std::uint32_t n_seg, const double* s, std::size_t n,
                             std::uint32_t* out) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = plbf::cuda::detail::segment_index(t, n_seg, s[i]);
}

// Scores on, and one ulp either side of, every threshold, plus everything
// index() clamps: where a one-step correction would go wrong first.
std::vector<double> edge_scores(const segment_grid& grid, std::mt19937_64& rng, std::size_t random) {
    constexpr double inf = std::numeric_limits<double>::infinity();
    std::vector<double> s = {0.0, -0.0, 1.0, -1.0, 2.0, inf, -inf, std::nan(""),
                             std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::min(),
                             std::nextafter(1.0, 0.0), std::nextafter(1.0, 2.0), 1e300, -1e300};
    for (std::size_t i = 0; i <= grid.segments(); ++i) {
        const double t = grid.threshold(i);
        s.push_back(t);
        s.push_back(std::nextafter(t, -1.0));
        s.push_back(std::nextafter(t, 2.0));
    }
    for (std::size_t i = 0; i < random; ++i) s.push_back(std::uniform_real_distribution<double>(0, 1)(rng));
    return s;
}

void test_segment_index() {
    section("cuda: device segment index equals segment_grid::index()");
    std::mt19937_64 rng(31);
    std::size_t probes = 0;
    int mismatches = 0;
    for (const std::size_t n : {1u, 2u, 3u, 7u, 50u, 1000u, 4096u, 65537u, 1000003u, 16777217u}) {
        const segment_grid grid(n);
        const std::vector<double> s = edge_scores(grid, rng, 100000);
        const auto t = device_buffer<double>::from_host(plbf::cuda::detail::thresholds_of(grid));
        const auto ds = device_buffer<double>::from_host(s);
        device_buffer<std::uint32_t> out(s.size());
        index_kernel<<<static_cast<unsigned>((s.size() + 255) / 256), 256>>>(t.data(), static_cast<std::uint32_t>(n),
                                                                             ds.data(), s.size(), out.data());
        plbf::cuda::check(cudaDeviceSynchronize(), "index_kernel");
        const std::vector<std::uint32_t> got = out.download();
        for (std::size_t i = 0; i < s.size(); ++i) mismatches += got[i] != grid.index(s[i]);
        probes += s.size();
    }
    check_eq(mismatches, 0, std::to_string(probes) + " scores over 10 grids, N up to 2^24 + 1");
}

void test_histogram() {
    section("cuda: GPU histogram equals segment_counts()");
    std::mt19937_64 rng(37);
    // 1000 segments count in shared memory; 20011 overflow it and take the
    // global-atomic path.
    for (const std::size_t n : {std::size_t{1000}, std::size_t{20011}}) {
        const segment_grid grid(n);
        std::vector<double> s = edge_scores(grid, rng, 2000000);
        // Skew half of them towards 1, as key scores are, to load a few bins hard.
        for (std::size_t i = 0; i < s.size(); i += 2) s[i] = std::pow(std::uniform_real_distribution<double>(0, 1)(rng), 0.1);
        const auto ds = device_buffer<double>::from_host(s);
        const std::vector<std::uint64_t> gpu = plbf::cuda::segment_counts(grid, ds.data(), s.size());
        check(gpu == segment_counts(grid, s), "N = " + std::to_string(n) + ": " + std::to_string(s.size()) +
                                                  " scores, every count equal");
    }
    const segment_grid grid(10);
    check(plbf::cuda::segment_counts(grid, nullptr, 0) == std::vector<std::uint64_t>(10, 0), "no scores: all zero");
}

struct workload {
    std::vector<std::uint64_t> keys, non_keys;
    std::vector<double> key_scores, non_key_scores;
};

workload make_workload(std::size_t n_keys, std::size_t n_non_keys, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0, 1);
    workload w;
    std::uint64_t id = seed << 40;
    for (std::size_t i = 0; i < n_keys; ++i) {
        w.keys.push_back(hash_u64(id++));
        w.key_scores.push_back(std::pow(u(rng), 0.25));
    }
    for (std::size_t i = 0; i < n_non_keys; ++i) {
        w.non_keys.push_back(hash_u64(id++));
        w.non_key_scores.push_back(1 - std::sqrt(u(rng)));
    }
    return w;
}

// Device answers for (hashes, scores), downloaded.
std::vector<std::uint8_t> device_answers(const plbf::cuda::device_filter& f, const std::vector<std::uint64_t>& h,
                                         const std::vector<double>& s) {
    const auto dh = device_buffer<std::uint64_t>::from_host(h);
    const auto ds = device_buffer<double>::from_host(s);
    device_buffer<std::uint8_t> out(h.size());
    f.contains(dh.data(), ds.data(), out.data(), h.size());
    plbf::cuda::check(cudaDeviceSynchronize(), "contains");
    return out.download();
}

std::vector<std::uint8_t> host_answers(const filter& f, const std::vector<std::uint64_t>& h,
                                       const std::vector<double>& s) {
    std::vector<std::uint8_t> out(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) out[i] = f.contains(h[i], s[i]) ? 1 : 0;
    return out;
}

// Keys at their own scores, unseen hashes at edge scores, unseen non-keys.
void make_queries(const filter& f, const workload& w, const workload& unseen, std::vector<std::uint64_t>& h,
                  std::vector<double>& s) {
    std::mt19937_64 rng(41);
    const std::vector<double> edges = edge_scores(f.grid(), rng, 0);
    h = w.keys;
    s = w.key_scores;
    for (std::size_t i = 0; i < edges.size(); ++i) {
        h.push_back(hash_u64(0xedce0000ULL + i));
        s.push_back(edges[i]);
    }
    h.insert(h.end(), unseen.non_keys.begin(), unseen.non_keys.end());
    s.insert(s.end(), unseen.non_key_scores.begin(), unseen.non_key_scores.end());
}

bool same_layout(const layout& a, const layout& b) {
    return a.boundary == b.boundary && a.fpr == b.fpr && a.backup_bits == b.backup_bits &&
           a.expected_fpr == b.expected_fpr;
}

bool same_blocks(std::span<const split_block::block> a, const std::vector<split_block::block>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(split_block::block)) == 0;
}

void test_uploaded_filter() {
    section("cuda: a CPU-built filter, uploaded, answers exactly as contains()");
    const workload w = make_workload(200000, 400000, 1);
    const workload unseen = make_workload(0, 1000000, 2);
    for (const std::size_t k : {std::size_t{5}, std::size_t{9}}) {
        options opt;
        opt.segments = 997;
        opt.regions = k;
        const filter f = filter::build_for_fpr(w.keys, w.key_scores, w.non_key_scores, 0.01, opt);
        const plbf::cuda::device_filter d(f);
        std::vector<std::uint64_t> h;
        std::vector<double> s;
        make_queries(f, w, unseen, h, s);
        const auto got = device_answers(d, h, s);
        check(got == host_answers(f, h, s), "k = " + std::to_string(k) + ": " + std::to_string(h.size()) +
                                                " queries, every answer equal");
        check_eq(d.memory_bytes() - (opt.segments + 1) * sizeof(double), f.memory_bytes(),
                 "k = " + std::to_string(k) + ": device holds the same tables and blocks");
    }
}

// The compact query formats on the device: float scores answer as the CPU's
// contains() on the widened score, and 16-bit segment numbers as contains() on
// the score they came from. Scores are float values widened, as a model's
// float32 output would be.
void test_compact_queries() {
    section("cuda: float scores and segment numbers answer as the CPU's contains()");
    workload w = make_workload(200000, 400000, 5);
    workload unseen = make_workload(0, 1000000, 6);
    for (auto* v : {&w.key_scores, &w.non_key_scores, &unseen.non_key_scores}) {
        for (double& s : *v) s = static_cast<float>(s);
    }
    options opt;
    opt.segments = 997;
    opt.regions = 9;
    const filter f = filter::build_for_fpr(w.keys, w.key_scores, w.non_key_scores, 0.01, opt);
    const plbf::cuda::device_filter d(f);
    std::vector<std::uint64_t> h;
    std::vector<double> s;
    make_queries(f, w, unseen, h, s);
    const std::vector<std::uint8_t> want = host_answers(f, h, s);

    // Float scores: the edge scores from make_queries round to nearby floats,
    // so compare against contains() on what the floats widen to.
    std::vector<float> sf(s.size());
    std::vector<double> widened(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) widened[i] = sf[i] = static_cast<float>(s[i]);
    const auto dh = device_buffer<std::uint64_t>::from_host(h);
    const auto dsf = device_buffer<float>::from_host(sf);
    device_buffer<std::uint8_t> out(h.size());
    d.contains(dh.data(), dsf.data(), out.data(), h.size());
    plbf::cuda::check(cudaDeviceSynchronize(), "contains float");
    check(out.download() == host_answers(f, h, widened),
          std::to_string(h.size()) + " float-score queries equal contains() on the widened scores");

    // Segment numbers, converted on the host from the double scores.
    std::vector<std::uint16_t> seg(s.size());
    d.to_segments(s, seg);
    int conv = 0;
    for (std::size_t i = 0; i < s.size(); ++i) conv += seg[i] != f.grid().index(s[i]);
    check_eq(conv, 0, "to_segments() equals segment_grid::index()");
    const auto dseg = device_buffer<std::uint16_t>::from_host(seg);
    d.contains_segments(dh.data(), dseg.data(), out.data(), h.size());
    plbf::cuda::check(cudaDeviceSynchronize(), "contains_segments");
    check(out.download() == want, std::to_string(h.size()) + " segment-number queries equal contains() on the scores");

    // Out-of-range segment numbers clamp as the CPU's contains_segment does.
    std::vector<std::uint16_t> bad(h.size());
    std::vector<std::uint8_t> bad_want(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) {
        bad[i] = i % 2 ? 0 : static_cast<std::uint16_t>(opt.segments + 1 + i % 1000);
        bad_want[i] = f.contains_segment(h[i], bad[i]) ? 1 : 0;
    }
    const auto dbad = device_buffer<std::uint16_t>::from_host(bad);
    d.contains_segments(dh.data(), dbad.data(), out.data(), h.size());
    plbf::cuda::check(cudaDeviceSynchronize(), "contains_segments clamp");
    check(out.download() == bad_want, "segments 0 and past N clamp as on the CPU");

    // More segments than 16 bits can number: refused, not truncated.
    options wide;
    wide.segments = 70000;
    const filter fw = filter::build_for_fpr(w.keys, w.key_scores, w.non_key_scores, 0.01, wide);
    const plbf::cuda::device_filter dw(fw);
    bool threw = false;
    try {
        dw.contains_segments(dh.data(), dseg.data(), out.data(), h.size());
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "70,000 segments: contains_segments() refuses rather than truncating");
}

void test_gpu_build() {
    section("cuda: a GPU-built filter is bit-identical to a CPU-built one");
    const workload w = make_workload(300000, 500000, 3);
    const workload unseen = make_workload(0, 1000000, 4);
    const auto dk = device_buffer<std::uint64_t>::from_host(w.keys);
    const auto dks = device_buffer<double>::from_host(w.key_scores);
    const auto dns = device_buffer<double>::from_host(w.non_key_scores);

    struct config {
        std::size_t segments, regions;
        method search;
        bool memory_mode;
        double target;
    };
    const config configs[] = {
        {1000, 5, method::fast_sharp, false, 0.01},  {1000, 5, method::fast, false, 0.001},
        {997, 9, method::fast_pp, false, 0.02},      {1000, 5, method::fast_sharp, true, 6.0 * 300000},
        {20011, 7, method::fast_sharp, false, 0.01},  // histogram through the global-atomic path
    };
    for (const config& c : configs) {
        options opt;
        opt.segments = c.segments;
        opt.regions = c.regions;
        opt.search = c.search;
        const std::string name = "N=" + std::to_string(c.segments) + " k=" + std::to_string(c.regions) +
                                 (c.memory_mode ? " memory" : " fpr") + " " + std::to_string(c.target);

        const filter cpu =
            c.memory_mode ? filter::build_for_memory(w.keys, w.key_scores, w.non_key_scores, c.target, opt)
                          : filter::build_for_fpr(w.keys, w.key_scores, w.non_key_scores, c.target, opt);
        const plbf::cuda::device_filter gpu =
            c.memory_mode
                ? plbf::cuda::device_filter::build_for_memory(dk.data(), dks.data(), w.keys.size(), dns.data(),
                                                              w.non_keys.size(), c.target, opt)
                : plbf::cuda::device_filter::build_for_fpr(dk.data(), dks.data(), w.keys.size(), dns.data(),
                                                           w.non_keys.size(), c.target, opt);

        check(same_layout(gpu.regions(), cpu.regions()), name + ": same layout");
        check(gpu.keys_per_region() == cpu.keys_per_region(), name + ": same keys per region");
        check(gpu.tables().region_of_segment == cpu.tables().region_of_segment &&
                  gpu.tables().slices == cpu.tables().slices && gpu.tables().block_count == cpu.tables().block_count,
              name + ": same routing tables");

        const filter back = gpu.to_host();
        check(same_blocks(cpu.blocks(), std::vector<split_block::block>(back.blocks().begin(), back.blocks().end())),
              name + ": same " + std::to_string(cpu.blocks().size()) + " blocks, bit for bit");

        std::vector<std::uint64_t> h;
        std::vector<double> s;
        make_queries(cpu, w, unseen, h, s);
        const auto want = host_answers(cpu, h, s);
        check(device_answers(gpu, h, s) == want, name + ": GPU answers equal the CPU filter's");
        check(host_answers(back, h, s) == want, name + ": to_host() answers equal too");
    }
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("test_cuda: no CUDA device; nothing to test\n");
        return 0;
    }
    cudaDeviceProp prop{};
    plbf::cuda::check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::printf("device: %s, sm_%d%d, %d SMs\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    test_segment_index();
    test_histogram();
    test_uploaded_filter();
    test_compact_queries();
    test_gpu_build();
    return summary("test_cuda");
}
