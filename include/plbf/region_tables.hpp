#pragma once
//
// The tables that route a score to its region's slice of the block array.
//
// They follow from the layout and the number of keys in each region, nothing
// else. The CPU filter and the CUDA filter both build them through
// region_tables::build, which is what lets a GPU-built filter come out
// bit-identical to a CPU-built one.

#include "optimize.hpp"
#include "split_block.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace plbf {

struct region_tables {
    // Region for each segment 1..N, plus one pad entry past N: the AVX2 lookup
    // gathers 32 bits at the 16-bit entry for segment s, so the read for s = N
    // spills into it. Entry 0 copies entry 1; index() never returns 0.
    std::vector<std::uint16_t> region_of_segment;
    // Per region, its slice of the block array as one 64-bit word: first block
    // in the high half, block count (>= 1) in the low half. One gather fetches
    // both for the vector lookup, and _mm256_mul_epu32 takes the count straight
    // from the low half. Unpacking it on the scalar paths measured no slower
    // than a two-field struct.
    std::vector<std::uint64_t> slices;
    // Blocks in the whole array, the two sentinels included.
    std::uint64_t block_count = 0;

    // Regions that need no filter point at a sentinel: all ones for FPR 1,
    // all zeros for FPR 0. Every probe is then the same load-and-test.
    static constexpr std::uint64_t accept_block = 0;
    static constexpr std::uint64_t reject_block = 1;

    static constexpr std::uint64_t make_slice(std::uint64_t first, std::uint32_t count) noexcept {
        return first << 32 | count;
    }
    static constexpr std::uint64_t slice_first(std::uint64_t s) noexcept { return s >> 32; }
    static constexpr std::uint32_t slice_count(std::uint64_t s) noexcept { return static_cast<std::uint32_t>(s); }

    // Size every region's split-block filter for its keys at its FPR, and lay
    // the slices out one after another behind the sentinels.
    static region_tables build(const layout& l, std::size_t segments,
                               std::span<const std::size_t> keys_per_region) {
        const std::size_t k = l.regions();
        if (k > 0xffff) throw std::invalid_argument("plbf: at most 65535 regions");
        if (keys_per_region.size() != k) throw std::invalid_argument("plbf: one key count per region");

        region_tables t;
        t.region_of_segment.assign(segments + 2, 0);
        for (std::size_t r = 0; r < k; ++r) {
            for (std::size_t s = l.boundary[r] + 1; s <= l.boundary[r + 1]; ++s) {
                t.region_of_segment[s] = static_cast<std::uint16_t>(r);
            }
        }
        t.region_of_segment[0] = t.region_of_segment[1];

        std::uint64_t next = 2;  // after the two sentinels
        t.slices.resize(k);
        for (std::size_t r = 0; r < k; ++r) {
            const double fpr = l.fpr[r];
            if (fpr >= 1) {
                t.slices[r] = make_slice(accept_block, 1);
            } else if (fpr <= 0) {
                // The smoothing in segment_distribution makes this unreachable
                // from scores; refuse rather than build false negatives.
                if (keys_per_region[r] != 0) throw std::logic_error("plbf: keys in a reject region");
                t.slices[r] = make_slice(reject_block, 1);
            } else {
                const std::uint32_t count = split_block::blocks_for(keys_per_region[r], fpr);
                t.slices[r] = make_slice(next, count);
                next += count;
            }
            // The packed slice holds the first block in 32 bits.
            if (next > std::numeric_limits<std::uint32_t>::max()) {
                throw std::length_error("plbf: more than 2^32 blocks (128 GiB)");
            }
        }
        t.block_count = next;
        return t;
    }
};

// Keys in each region of `l`, from the unsmoothed per-segment key counts
// (counts[s-1] is segment s). The same numbers as indexing every key again,
// for the cost of N additions.
inline std::vector<std::size_t> keys_per_region(const layout& l, std::span<const std::uint64_t> counts) {
    std::vector<std::size_t> out(l.regions(), 0);
    for (std::size_t r = 0; r < out.size(); ++r) {
        for (std::size_t s = l.boundary[r] + 1; s <= l.boundary[r + 1]; ++s) {
            out[r] += static_cast<std::size_t>(counts[s - 1]);
        }
    }
    return out;
}

}  // namespace plbf
