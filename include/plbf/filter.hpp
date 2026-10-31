#pragma once
//
// The partitioned learned Bloom filter itself: a score selects a region, and
// the region's split-block filter answers for the key.
//
// The classifier stays outside. Callers pass each key's 64-bit hash and its
// score; the filter never sees a model, so the same build serves any scorer.
//
// Every region is a slice of one block array, including the two regions that
// need no filter, which point at sentinel blocks (see region_tables). A probe
// is therefore the same load-and-test whatever the region, with no branch on
// the region kind in the batch loop.

#include "optimize.hpp"
#include "region_tables.hpp"
#include "segments.hpp"
#include "simd.hpp"
#include "split_block.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace plbf {

enum class kernel : std::uint8_t {
    automatic,   // avx2 where the CPU has it, else scalar
    scalar,      // scalar lookup, scalar probe
    avx2,        // score -> block address four lanes at a time, AVX2 probe
    avx2_probe,  // scalar lookup, AVX2 probe: kept so the lookup's share can be measured
};

struct options {
    std::size_t segments = 1000;  // N
    std::size_t regions = 5;      // k
    method search = method::fast_sharp;
};

class filter {
public:
    // Least memory at an expected overall FPR of target_fpr. The key scores and
    // the non-key scores train the layout; the keys are then inserted.
    static filter build_for_fpr(std::span<const std::uint64_t> key_hashes,
                                std::span<const double> key_scores,
                                std::span<const double> non_key_scores, double target_fpr,
                                const options& opt = {}) {
        return build(key_hashes, key_scores, non_key_scores, opt, [&](auto& g, auto& h, std::size_t n) {
            return optimize_for_fpr(g, h, n, opt.regions, target_fpr, opt.search);
        });
    }

    // Lowest expected FPR for memory_bits of backup filter -- as the paper
    // counts bits, for classic Bloom filters. The split-block filters that
    // actually get built hit the same per-region FPRs with somewhat more
    // memory; memory_bytes() reports what was really allocated.
    static filter build_for_memory(std::span<const std::uint64_t> key_hashes,
                                   std::span<const double> key_scores,
                                   std::span<const double> non_key_scores, double memory_bits,
                                   const options& opt = {}) {
        return build(key_hashes, key_scores, non_key_scores, opt, [&](auto& g, auto& h, std::size_t n) {
            return optimize_for_memory(g, h, n, opt.regions, memory_bits, opt.search);
        });
    }

    // A filter from the parts of one built earlier -- downloaded from a GPU, or
    // read back from storage: the inverse of regions(), keys_per_region(),
    // tables() and blocks(). Sizes and slice bounds are checked, so a
    // malformed part fails here rather than reading out of bounds later; the
    // bits themselves are trusted.
    static filter from_parts(std::size_t segments, plbf::layout l, std::vector<std::size_t> keys_per_region,
                             region_tables t, std::vector<split_block::block> blocks) {
        const std::size_t k = l.regions();
        const bool ok = k >= 1 && l.boundary.size() == k + 1 && l.boundary.front() == 0 &&
                        l.boundary.back() == segments && keys_per_region.size() == k &&
                        t.region_of_segment.size() == segments + 2 && t.slices.size() == k &&
                        t.block_count == blocks.size() && blocks.size() >= 2;
        if (!ok) throw std::invalid_argument("plbf::filter: parts do not fit together");
        for (const std::uint64_t s : t.slices) {
            if (region_tables::slice_count(s) == 0 ||
                region_tables::slice_first(s) + region_tables::slice_count(s) > blocks.size()) {
                throw std::invalid_argument("plbf::filter: a slice runs past the block array");
            }
        }
        for (std::size_t s = 0; s < t.region_of_segment.size(); ++s) {
            if (t.region_of_segment[s] >= k) throw std::invalid_argument("plbf::filter: segment maps to no region");
        }
        filter f(segments);
        f.layout_ = std::move(l);
        f.keys_per_region_ = std::move(keys_per_region);
        f.tables_ = std::move(t);
        f.blocks_ = std::move(blocks);
        return f;
    }

    bool contains(std::uint64_t hash, double score) const noexcept {
        return split_block::probe_scalar(*block_for(hash, score), static_cast<std::uint32_t>(hash));
    }

    // contains() for a query that carries its segment number (1..N, as
    // segment_grid::index gives it) in place of its score. Out-of-range
    // segments are clamped, as out-of-range scores are.
    bool contains_segment(std::uint64_t hash, std::uint32_t segment) const noexcept {
        return split_block::probe_scalar(*block_for_segment(hash, segment), static_cast<std::uint32_t>(hash));
    }

    // out[i] = contains(hashes[i], scores[i]), as 0 or 1.
    void contains_batch(std::span<const std::uint64_t> hashes, std::span<const double> scores,
                        std::span<std::uint8_t> out, kernel kern = kernel::automatic) const {
        batch(hashes, scores, out, kern);
    }

    // The same for 32-bit scores, as most models emit them. Each score is
    // widened to double, exactly, so the answer is contains(hash, double(score)):
    // a filter built from the same values widened gives every key the segment
    // it was inserted from.
    void contains_batch(std::span<const std::uint64_t> hashes, std::span<const float> scores,
                        std::span<std::uint8_t> out, kernel kern = kernel::automatic) const {
        batch(hashes, scores, out, kern);
    }

    // The optimiser's decision: boundaries, per-region FPRs, predicted cost.
    const plbf::layout& regions() const noexcept { return layout_; }
    const segment_grid& grid() const noexcept { return grid_; }

    // Score threshold above which region r starts: region r holds scores in
    // (threshold(r), threshold(r+1)], region 0 also holding a score of 0.
    double threshold(std::size_t r) const noexcept { return grid_.threshold(layout_.boundary[r]); }

    // Blocks per region, sentinels excluded: 0 for a region with no filter.
    std::uint32_t region_blocks(std::size_t r) const noexcept {
        const std::uint64_t s = tables_.slices[r];
        return region_tables::slice_first(s) < 2 ? 0 : region_tables::slice_count(s);
    }
    std::size_t region_keys(std::size_t r) const noexcept { return keys_per_region_[r]; }
    const std::vector<std::size_t>& keys_per_region() const noexcept { return keys_per_region_; }

    // The raw parts, for moving the filter to another device or saving it.
    const region_tables& tables() const noexcept { return tables_; }
    std::span<const split_block::block> blocks() const noexcept { return blocks_; }

    // Bytes of backup filter, sentinels and region lookup tables included.
    std::size_t memory_bytes() const noexcept {
        return blocks_.size() * sizeof(split_block::block) +
               tables_.region_of_segment.size() * sizeof(std::uint16_t) +
               tables_.slices.size() * sizeof(std::uint64_t);
    }

private:
    // Large enough to cover a DRAM miss with the probes in flight, small enough
    // that the plan stays in L1. A multiple of 4 for the vector lookup.
    static constexpr std::size_t batch_chunk = 64;
    static_assert(batch_chunk % 4 == 0);

    // CPUID is serialising and costs more than a short batch; ask once per filter.
    explicit filter(std::size_t segments) : grid_(segments), has_avx2_(cpu_has_avx2()) {}

    template <class Optimize>
    static filter build(std::span<const std::uint64_t> key_hashes, std::span<const double> key_scores,
                        std::span<const double> non_key_scores, const options& opt, Optimize optimize) {
        if (key_hashes.size() != key_scores.size()) {
            throw std::invalid_argument("plbf::filter: one score per key");
        }
        filter f(opt.segments);
        const std::vector<std::uint64_t> key_counts = segment_counts(f.grid_, key_scores);
        const auto g = segment_distribution::from_counts(key_counts);
        const segment_distribution h(f.grid_, non_key_scores);
        f.layout_ = optimize(g, h, key_scores.size());
        f.keys_per_region_ = plbf::keys_per_region(f.layout_, key_counts);
        f.tables_ = region_tables::build(f.layout_, opt.segments, f.keys_per_region_);

        f.blocks_.assign(f.tables_.block_count, split_block::block{});
        for (auto& w : f.blocks_[region_tables::accept_block].word) w = ~0U;
        for (std::size_t i = 0; i < key_hashes.size(); ++i) {
            const std::uint64_t sl = f.tables_.slices[f.tables_.region_of_segment[f.grid_.index(key_scores[i])]];
            if (region_tables::slice_first(sl) < 2) continue;  // accept region: nothing to insert
            const std::uint64_t b =
                region_tables::slice_first(sl) + split_block::block_index(key_hashes[i], region_tables::slice_count(sl));
            split_block::insert(f.blocks_[b], static_cast<std::uint32_t>(key_hashes[i]));
        }
        return f;
    }

    const split_block::block* block_for(std::uint64_t hash, double score) const noexcept {
        const std::uint64_t sl = tables_.slices[tables_.region_of_segment[grid_.index(score)]];
        return &blocks_[region_tables::slice_first(sl) + split_block::block_index(hash, region_tables::slice_count(sl))];
    }

    // block_for() from a segment number the caller computed; the clamp keeps a
    // bad one in the table, where index() needs none.
    const split_block::block* block_for_segment(std::uint64_t hash, std::uint32_t segment) const noexcept {
        const auto n = static_cast<std::uint32_t>(grid_.segments());
        if (segment < 1) segment = 1;
        if (segment > n) segment = n;
        const std::uint64_t sl = tables_.slices[tables_.region_of_segment[segment]];
        return &blocks_[region_tables::slice_first(sl) + split_block::block_index(hash, region_tables::slice_count(sl))];
    }

    // The batch kernels, for double or float scores. A float is widened to
    // double exactly, so both answer as contains() on the widened value.
    template <class Score>
    void batch(std::span<const std::uint64_t> hashes, std::span<const Score> scores, std::span<std::uint8_t> out,
               kernel kern) const {
        if (hashes.size() != scores.size() || out.size() < hashes.size()) {
            throw std::invalid_argument("plbf::filter: batch spans differ in size");
        }
        if (kern == kernel::automatic) kern = has_avx2_ ? kernel::avx2 : kernel::scalar;
        if (kern == kernel::scalar) {
            batch_scalar(hashes.data(), scores.data(), out.data(), hashes.size());
            return;
        }
#if PLBF_X86_64
        if (!has_avx2_) throw std::runtime_error("plbf::filter: no AVX2 on this CPU");
        if (kern == kernel::avx2) {
            batch_avx2<true>(hashes.data(), scores.data(), out.data(), hashes.size());
        } else {
            batch_avx2<false>(hashes.data(), scores.data(), out.data(), hashes.size());
        }
#else
        throw std::runtime_error("plbf::filter: AVX2 kernels need x86-64");
#endif
    }

    static void prefetch(const void* p) noexcept {
#if PLBF_X86_64
        _mm_prefetch(static_cast<const char*>(p), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(p);
#else
        (void)p;
#endif
    }

    // Each batch kernel works a chunk at a time: first plan every probe in the
    // chunk and prefetch its block, then test them all, so the misses overlap.
    template <class Score>
    void batch_scalar(const std::uint64_t* hashes, const Score* scores, std::uint8_t* out,
                      std::size_t n) const noexcept {
        struct probe {
            const split_block::block* block;
            std::uint32_t x;
        };
        probe plan[batch_chunk];
        for (std::size_t base = 0; base < n; base += batch_chunk) {
            const std::size_t len = n - base < batch_chunk ? n - base : batch_chunk;
            for (std::size_t i = 0; i < len; ++i) {
                plan[i] = {block_for(hashes[base + i], static_cast<double>(scores[base + i])),
                           static_cast<std::uint32_t>(hashes[base + i])};
                prefetch(plan[i].block);
            }
            for (std::size_t i = 0; i < len; ++i) {
                out[base + i] = split_block::probe_scalar(*plan[i].block, plan[i].x) ? 1 : 0;
            }
        }
    }

#if PLBF_X86_64
    // Score to block address, four lanes at a time: segment index (see
    // segment_grid::index4_avx2), then two gathers -- the region, and the
    // region's packed slice -- and the multiply-shift block index on the high
    // halves of the hashes. Lane for lane it is block_for(). Float scores are
    // widened four at a time; the conversion is exact.
    template <class Score>
    PLBF_TARGET_AVX2 void plan4_avx2(const std::uint64_t* hashes, const Score* scores,
                                     std::uint64_t* addr) const noexcept {
        __m256d s;
        if constexpr (std::is_same_v<Score, float>) {
            s = _mm256_cvtps_pd(_mm_loadu_ps(scores));
        } else {
            s = _mm256_loadu_pd(scores);
        }
        const __m128i seg = grid_.index4_avx2(s);
        const __m128i region = _mm_and_si128(
            _mm_i32gather_epi32(reinterpret_cast<const int*>(tables_.region_of_segment.data()), seg, 2),
            _mm_set1_epi32(0xffff));
        const __m256i slice =
            _mm256_i32gather_epi64(reinterpret_cast<const long long*>(tables_.slices.data()), region, 8);
        const __m256i hash = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(hashes));
        const __m256i offset = _mm256_srli_epi64(_mm256_mul_epu32(_mm256_srli_epi64(hash, 32), slice), 32);
        const __m256i index = _mm256_add_epi64(offset, _mm256_srli_epi64(slice, 32));
        const __m256i base = _mm256_set1_epi64x(static_cast<long long>(reinterpret_cast<std::uintptr_t>(blocks_.data())));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(addr),
                            _mm256_add_epi64(base, _mm256_slli_epi64(index, 5)));  // 32-byte blocks
    }

    template <bool VectorLookup, class Score>
    PLBF_TARGET_AVX2 void batch_avx2(const std::uint64_t* hashes, const Score* scores,
                                     std::uint8_t* out, std::size_t n) const noexcept {
        static_assert(sizeof(split_block::block) == 32);
        std::uint64_t plan[batch_chunk];
        for (std::size_t base = 0; base < n; base += batch_chunk) {
            const std::size_t len = n - base < batch_chunk ? n - base : batch_chunk;
            std::size_t i = 0;
            if constexpr (VectorLookup) {
                for (; i + 4 <= len; i += 4) {
                    plan4_avx2(hashes + base + i, scores + base + i, plan + i);
                    for (std::size_t j = i; j < i + 4; ++j) prefetch(reinterpret_cast<const void*>(plan[j]));
                }
            }
            for (; i < len; ++i) {
                plan[i] = reinterpret_cast<std::uintptr_t>(
                    block_for(hashes[base + i], static_cast<double>(scores[base + i])));
                prefetch(reinterpret_cast<const void*>(plan[i]));
            }
            for (i = 0; i < len; ++i) {
                const auto* b = reinterpret_cast<const split_block::block*>(plan[i]);
                out[base + i] = split_block::probe_avx2(*b, static_cast<std::uint32_t>(hashes[base + i])) ? 1 : 0;
            }
        }
    }
#endif

    segment_grid grid_;
    bool has_avx2_;
    plbf::layout layout_;
    std::vector<std::size_t> keys_per_region_;
    region_tables tables_;
    std::vector<split_block::block> blocks_;
};

}  // namespace plbf
