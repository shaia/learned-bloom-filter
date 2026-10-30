#pragma once
//
// Split-block Bloom filter primitives: the backing store for every region.
//
// A block is 256 bits as eight 32-bit words, and a key sets exactly one bit in
// each word -- the layout Apache Parquet standardised, salts included. A probe
// is then one 32-byte load and an AND, which is the whole reason to prefer it
// over a classic Bloom filter here: every region, whatever its FPR, runs the
// same branch-free probe, so a batch spanning regions never diverges. The cost
// is a worse FPR per bit than a classic filter, which blocks_for accounts for.

#include "simd.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace plbf::split_block {

struct alignas(32) block {
    std::uint32_t word[8];
};
static_assert(sizeof(block) == 32);

inline constexpr std::uint32_t salt[8] = {0x47b6137bU, 0x44974d91U, 0x8824ad5bU, 0xa2b7289dU,
                                          0x705495c7U, 0x2df1424bU, 0x9efc4947U, 0x5c6bfb31U};

// Which block of `count` a hash selects: the high half, scaled by
// multiply-shift so no modulo is needed and any block count works.
constexpr std::uint64_t block_index(std::uint64_t hash, std::uint32_t count) noexcept {
    return ((hash >> 32) * count) >> 32;
}

inline void insert(block& b, std::uint32_t x) noexcept {
    for (int i = 0; i < 8; ++i) b.word[i] |= 1U << ((x * salt[i]) >> 27);
}

inline bool probe_scalar(const block& b, std::uint32_t x) noexcept {
    // Accumulate rather than return early: eight independent tests the
    // compiler can schedule together beat eight dependent branches.
    std::uint32_t missing = 0;
    for (int i = 0; i < 8; ++i) {
        const std::uint32_t bit = 1U << ((x * salt[i]) >> 27);
        missing |= bit & ~b.word[i];
    }
    return missing == 0;
}

#if PLBF_X86_64
PLBF_TARGET_AVX2 inline bool probe_avx2(const block& b, std::uint32_t x) noexcept {
    const __m256i salts = _mm256_setr_epi32(
        static_cast<int>(salt[0]), static_cast<int>(salt[1]), static_cast<int>(salt[2]),
        static_cast<int>(salt[3]), static_cast<int>(salt[4]), static_cast<int>(salt[5]),
        static_cast<int>(salt[6]), static_cast<int>(salt[7]));
    const __m256i h = _mm256_srli_epi32(_mm256_mullo_epi32(_mm256_set1_epi32(static_cast<int>(x)), salts), 27);
    const __m256i mask = _mm256_sllv_epi32(_mm256_set1_epi32(1), h);
    const __m256i bits = _mm256_load_si256(reinterpret_cast<const __m256i*>(&b));
    return _mm256_testc_si256(bits, mask) != 0;  // (~bits & mask) == 0
}
#endif

// Expected FPR of a split-block filter holding `load` keys per block on
// average. The number of keys in a block is Poisson(load); a block holding x
// keys reports a non-key present with probability (1 - (31/32)^x)^8.
//
// The Poisson weights are carried in log space so that a heavy load, where
// e^-load underflows, still sums correctly.
inline double false_positive_rate(double load) noexcept {
    if (!(load > 0)) return 0.0;
    // Past ~1000 keys per block, (31/32)^load < 1e-13 and the FPR is 1 to
    // within 1e-12; summing ~load Poisson terms to say so is wasted work.
    if (load > 1100) return 1.0;
    const double log_load = std::log(load);
    const double log_miss = std::log(31.0 / 32.0);
    const double limit = load + 12.0 * std::sqrt(load) + 30.0;
    double log_p = -load;  // log P(x = 0)
    double sum = 0.0;
    for (double x = 0; x <= limit; x += 1.0) {
        if (x > 0) log_p += log_load - std::log(x);
        const double one_word = 1.0 - std::exp(x * log_miss);
        const double term = std::exp(log_p) * std::pow(one_word, 8);
        sum += term;
        if (x > load && term < sum * 1e-17) break;
    }
    return sum < 1.0 ? sum : 1.0;
}

// The fewest blocks that hold `keys` keys at an expected FPR of at most `target`.
inline std::uint32_t blocks_for(std::size_t keys, double target) {
    if (!(target > 0 && target < 1)) throw std::invalid_argument("split_block: target FPR must be in (0, 1)");
    if (keys == 0) return 1;
    const auto n = static_cast<double>(keys);
    auto fits = [&](std::uint64_t blocks) { return false_positive_rate(n / static_cast<double>(blocks)) <= target; };

    // Start from the load at which a block holding exactly `load` keys hits the
    // target. Poisson spread only makes the real FPR worse, so the true answer
    // is at or above this guess -- and starting here, rather than at one block
    // holding every key, keeps each model evaluation short.
    const double guess_load = std::log(1.0 - std::pow(target, 1.0 / 8.0)) / std::log(31.0 / 32.0);
    constexpr std::uint64_t max_blocks = std::numeric_limits<std::uint32_t>::max();
    auto clamp_blocks = [&](double b) -> std::uint64_t {
        if (!(b >= 1)) return 1;
        return b >= static_cast<double>(max_blocks) ? max_blocks : static_cast<std::uint64_t>(b);
    };

    // Invariant from here on: fits(hi), and lo == 0 or !fits(lo).
    std::uint64_t lo = 0;
    std::uint64_t hi = clamp_blocks(n / guess_load);
    if (fits(hi)) {
        while (hi > 1 && fits(hi / 2)) hi /= 2;
        lo = hi / 2;
    } else {
        while (!fits(hi)) {
            if (hi >= max_blocks) throw std::length_error("split_block: target FPR needs more than 2^32 blocks");
            lo = hi;
            hi = hi * 2 > max_blocks ? max_blocks : hi * 2;
        }
    }
    while (hi - lo > 1) {
        const std::uint64_t mid = lo + (hi - lo) / 2;
        (fits(mid) ? hi : lo) = mid;
    }
    return static_cast<std::uint32_t>(hi);
}

}  // namespace plbf::split_block
