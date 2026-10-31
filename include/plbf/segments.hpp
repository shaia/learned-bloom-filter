#pragma once
//
// Score segments, and the per-segment key and non-key distributions.
//
// Everything downstream works on segment indices, never on raw scores: the
// optimiser places region boundaries on the N+1 segment thresholds, and the
// filter maps a query's score to its segment exactly as the build did. Both
// sides going through segment_grid::index is what makes "no false negatives"
// hold however a score rounds near a threshold.

#include "simd.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace plbf {

// The N equal-width segments of [0, 1]. Segment s (1-based) is ((s-1)/N, s/N];
// a score of exactly 0 belongs to segment 1.
//
// The thresholds are stored rather than recomputed so that comparisons are
// against the same doubles the reference uses (i / N, correctly rounded). A
// score sitting exactly on a threshold then lands in the same segment here as
// in the Python, which is what lets the golden tests demand exact boundaries.
class segment_grid {
public:
    explicit segment_grid(std::size_t segments) : thresholds_(segments + 1) {
        if (segments == 0) throw std::invalid_argument("segment_grid: need at least one segment");
        if (segments >= (std::size_t{1} << 31)) throw std::invalid_argument("segment_grid: too many segments");
        const auto n = static_cast<double>(segments);
        for (std::size_t i = 0; i <= segments; ++i) thresholds_[i] = static_cast<double>(i) / n;
    }

    std::size_t segments() const noexcept { return thresholds_.size() - 1; }
    double threshold(std::size_t i) const noexcept { return thresholds_[i]; }

    // The most segments a 16-bit segment number can name: the limit for the
    // compact query format, a 2-byte segment in place of an 8-byte score.
    static constexpr std::size_t max_segments16 = 65535;

    // Smallest i with threshold(i) >= score, but at least 1: the reference's
    // bisect_left plus its "0 maps to segment 1" rule.
    //
    // Out-of-range scores and NaN are clamped rather than rejected. Build and
    // query clamp identically, so a key is always looked up in the segment it
    // was inserted from.
    std::size_t index(double score) const noexcept {
        const std::size_t n = segments();
        if (!(score > 0.0)) return 1;  // also catches NaN
        if (score >= 1.0) return n;
        // ceil(score * N) is within one of the answer; the two loops make it exact.
        auto i = static_cast<std::size_t>(std::ceil(score * static_cast<double>(n)));
        if (i < 1) i = 1;
        if (i > n) i = n;
        while (i > 1 && thresholds_[i - 1] >= score) --i;
        while (i < n && thresholds_[i] < score) ++i;
        return i;
    }

#if PLBF_X86_64
    // index() for four scores at once, lane for lane the same answer.
    //
    // The loops in index() never take more than one step. With c = ceil(fl(s N))
    // and u = 2^-53, threshold(c+1) >= s needs (1 - u) >= s N (2u - u^2), and
    // threshold(c-2) < s needs s N (2u + u^2) < 1 + u; both hold for any
    // N < 2^31, which segment_grid enforces. So the answer is c-1, c or c+1,
    // and one compare each way decides it.
    //
    // Clamping covers what index() returns early for: a score <= 0 or NaN goes
    // to 1, a score >= 1 to N. max_pd returns its second operand when the first
    // is NaN, which is what sends NaN to 1; the ordered compares are then false.
    PLBF_TARGET_AVX2 __m128i index4_avx2(__m256d score) const noexcept {
        const __m256d n = _mm256_set1_pd(static_cast<double>(segments()));
        const __m256d one = _mm256_set1_pd(1.0);
        __m256d c = _mm256_ceil_pd(_mm256_mul_pd(score, n));
        c = _mm256_min_pd(_mm256_max_pd(c, one), n);
        const __m128i ci = _mm256_cvttpd_epi32(c);
        const double* t = thresholds_.data();
        const __m256d below = _mm256_i32gather_pd(t, _mm_sub_epi32(ci, _mm_set1_epi32(1)), 8);
        const __m256d at = _mm256_i32gather_pd(t, ci, 8);
        const __m256d down = _mm256_and_pd(_mm256_cmp_pd(below, score, _CMP_GE_OQ), one);
        const __m256d up = _mm256_and_pd(_mm256_cmp_pd(at, score, _CMP_LT_OQ), one);
        c = _mm256_add_pd(_mm256_sub_pd(c, down), up);
        // Only reachable outside (0, 1): down at c = 1 for s <= 0, up at c = N for s > 1.
        c = _mm256_min_pd(_mm256_max_pd(c, one), n);
        return _mm256_cvttpd_epi32(c);
    }

    // out[i] = index(scores[i]), four at a time. The caller checks for AVX2.
    PLBF_TARGET_AVX2 void index_batch_avx2(const double* scores, std::size_t count,
                                           std::uint32_t* out) const noexcept {
        std::size_t i = 0;
        for (; i + 4 <= count; i += 4) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), index4_avx2(_mm256_loadu_pd(scores + i)));
        }
        for (; i < count; ++i) out[i] = static_cast<std::uint32_t>(index(scores[i]));
    }

    // The same as 16-bit segment numbers; needs segments() <= max_segments16,
    // which the caller checks.
    PLBF_TARGET_AVX2 void index_batch_avx2(const double* scores, std::size_t count,
                                           std::uint16_t* out) const noexcept {
        std::size_t i = 0;
        for (; i + 4 <= count; i += 4) {
            const __m128i s = index4_avx2(_mm256_loadu_pd(scores + i));
            _mm_storel_epi64(reinterpret_cast<__m128i*>(out + i), _mm_packus_epi32(s, s));
        }
        for (; i < count; ++i) out[i] = static_cast<std::uint16_t>(index(scores[i]));
    }
#endif

    // out[i] = index(scores[i]) as a 16-bit segment number, one at a time; for
    // CPUs without AVX2. Needs segments() <= max_segments16.
    void index_batch(const double* scores, std::size_t count, std::uint16_t* out) const noexcept {
        for (std::size_t i = 0; i < count; ++i) out[i] = static_cast<std::uint16_t>(index(scores[i]));
    }

private:
    std::vector<double> thresholds_;
};

// Raw per-segment counts: out[s-1] is how many scores land in segment s.
inline std::vector<std::uint64_t> segment_counts(const segment_grid& grid, std::span<const double> scores) {
    std::vector<std::uint64_t> counts(grid.segments(), 0);
    for (const double s : scores) ++counts[grid.index(s) - 1];
    return counts;
}

// The probability mass of one population (keys, or non-keys) in each segment,
// held as prefix sums so the optimiser reads any range in O(1).
class segment_distribution {
public:
    // Histogram the scores, with every segment's count starting at 1.
    //
    // That smoothing is the reference's, and it is load-bearing: every segment
    // then has non-zero mass in both populations, so the divergence term
    // pos * ln(pos / neg) is always finite and no region ever ends up with zero
    // key mass and a false-positive rate of 0.
    segment_distribution(const segment_grid& grid, std::span<const double> scores)
        : segment_distribution(from_counts(segment_counts(grid, scores))) {}

    // counts[s-1] is the raw count of segment s; the +1 smoothing is applied here.
    // For callers that histogram elsewhere -- on a GPU, or across threads.
    static segment_distribution from_counts(std::span<const std::uint64_t> counts) {
        if (counts.empty()) throw std::invalid_argument("segment_distribution: no segments");
        std::uint64_t total = 0;
        for (const auto c : counts) total += c + 1;
        const auto t = static_cast<double>(total);
        segment_distribution d;
        d.acc_.assign(counts.size() + 1, 0.0);
        for (std::size_t i = 1; i <= counts.size(); ++i) {
            d.acc_[i] = d.acc_[i - 1] + static_cast<double>(counts[i - 1] + 1) / t;
        }
        d.check_total();
        return d;
    }

    // pr[s-1] is the mass of segment s, used as given: no smoothing. A segment
    // with zero mass is allowed, which is exactly what the property tests need
    // to reach the reference's zero-mass branches.
    static segment_distribution from_probabilities(std::span<const double> pr) {
        if (pr.empty()) throw std::invalid_argument("segment_distribution: no segments");
        segment_distribution d;
        d.acc_.assign(pr.size() + 1, 0.0);
        for (std::size_t i = 1; i <= pr.size(); ++i) {
            if (!(pr[i - 1] >= 0.0)) throw std::invalid_argument("segment_distribution: negative mass");
            d.acc_[i] = d.acc_[i - 1] + pr[i - 1];
        }
        d.check_total();
        return d;
    }

    std::size_t segments() const noexcept { return acc_.size() - 1; }

    // Mass of segments 1..i (0 for i = 0).
    double cumulative(std::size_t i) const noexcept { return acc_[i]; }

    // Mass of segments lo..hi inclusive, 1 <= lo <= hi <= N.
    double range(std::size_t lo, std::size_t hi) const noexcept { return acc_[hi] - acc_[lo - 1]; }

private:
    segment_distribution() = default;

    // The reference asserts the same tolerance.
    void check_total() const {
        if (std::abs(acc_.back() - 1.0) >= 1e-8) {
            throw std::invalid_argument("segment_distribution: masses do not sum to 1");
        }
    }

    std::vector<double> acc_;  // acc_[i] = mass of segments 1..i
};

}  // namespace plbf
