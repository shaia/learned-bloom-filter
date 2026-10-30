#pragma once
//
// Choosing the region boundaries and the per-region false-positive rates.
//
// The search space is fixed by the paper: k contiguous regions of segments,
// the last of which is segments j..N for some j. For each j, a DP picks the
// first k-1 regions over segments 1..j-1 to maximise the key/non-key
// divergence; the FPRs then follow in closed form, and the j whose layout costs
// the least wins. The four methods differ only in how the DP's row maxima are
// found, which is why they share everything else here.
//
// Numerical parity with the reference (FastPLBF, Sato and Matsui) is a design
// goal, not an accident: every expression is evaluated in the reference's
// order, so a build without FMA contraction reproduces its boundaries exactly
// and its FPRs to the last bit. The golden tests hold it to that.

#include "row_maxima.hpp"
#include "segments.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

namespace plbf {

enum class method : std::uint8_t {
    plbf,        // Vaidya et al.: a fresh DP for every last-region start. O(N^3 k).
    fast,        // One DP, every split point scanned. Same answer as plbf. O(N^2 k).
    fast_pp,     // One DP, divide-and-conquer row maxima. O(N k log N).
    fast_sharp,  // One DP, SMAWK row maxima. O(N k log k), the j loop now dominating.
};

// What the optimiser decided.
struct layout {
    // Region r (0-based) is segments boundary[r]+1 .. boundary[r+1].
    // boundary[0] = 0 and boundary[k] = N.
    std::vector<std::uint32_t> boundary;
    // Per region. 1 means no backup filter (accept everything that scores into
    // the region); 0 means reject everything.
    std::vector<double> fpr;
    // The cost the optimiser minimises in fpr mode: total bits of classic
    // Bloom filters at these FPRs. Real split-block filters need somewhat more.
    double backup_bits = 0;
    // Sum over regions of non-key mass times FPR: the overall FPR the layout
    // predicts for non-keys distributed like the training sample.
    double expected_fpr = 0;

    std::size_t regions() const noexcept { return fpr.size(); }
};

namespace detail {

inline constexpr double neg_inf = -std::numeric_limits<double>::infinity();

// DP over (segment prefix length n, number of regions q). Only q <= k-1 is
// ever read back, so the k-th layer the reference also computes is skipped.
struct dp_table {
    std::size_t rows = 0;    // N
    std::size_t layers = 0;  // k - 1
    std::vector<double> value;
    std::vector<std::int32_t> pre;  // split point; -1 is the reference's None

    dp_table(std::size_t n, std::size_t l)
        : rows(n), layers(l), value((n + 1) * (l + 1), neg_inf), pre((n + 1) * (l + 1), -1) {
        value[0] = 0.0;
    }

    double& v(std::size_t n, std::size_t q) noexcept { return value[n * (layers + 1) + q]; }
    double v(std::size_t n, std::size_t q) const noexcept { return value[n * (layers + 1) + q]; }
    std::int32_t& p(std::size_t n, std::size_t q) noexcept { return pre[n * (layers + 1) + q]; }
    std::int32_t p(std::size_t n, std::size_t q) const noexcept { return pre[n * (layers + 1) + q]; }
};

// The reference's A matrix: DP[i-1][q-1] plus the divergence of segments i..p.
inline double transition(const segment_distribution& g, const segment_distribution& h, double prev,
                         std::size_t i, std::size_t p) noexcept {
    const double pos = g.range(i, p);
    const double neg = h.range(i, p);
    if (neg == 0) return neg_inf;
    if (pos == 0) return prev + 0.0;
    return prev + pos * std::log(pos / neg);
}

// Rows 1..rows, every split point tried: the reference's calc_DPKL.
//
// The loop order is (n, i, q) where the reference's is (n, q, i), so each
// divergence is computed once for all layers instead of once per layer. Every
// cell still sees its candidates in increasing i under a strict comparison, so
// every value and every predecessor comes out bit-identical.
inline void full_scan_dp(const segment_distribution& g, const segment_distribution& h,
                         std::size_t rows, dp_table& dp) {
    for (std::size_t n = 1; n <= rows; ++n) {
        for (std::size_t i = 1; i <= n; ++i) {
            const double pos = g.range(i, n);
            const double neg = h.range(i, n);
            if (neg == 0) continue;
            const double d = pos == 0 ? 0.0 : pos * std::log(pos / neg);
            for (std::size_t q = 1; q <= dp.layers; ++q) {
                const double cand = dp.v(i - 1, q - 1) + d;
                if (dp.v(n, q) < cand) {
                    dp.v(n, q) = cand;
                    dp.p(n, q) = static_cast<std::int32_t>(i - 1);
                }
            }
        }
    }
}

// One layer at a time, each layer's row maxima found by `solve`: the
// reference's fast_calc_DPKL and fast_calc_DPKL_sharp.
template <class Solver>
void layered_dp(const segment_distribution& g, const segment_distribution& h, dp_table& dp,
                Solver solve) {
    const std::size_t n_rows = dp.rows;
    std::vector<std::uint32_t> argmax;
    for (std::size_t q = 1; q <= dp.layers; ++q) {
        auto a = [&](std::size_t p, std::size_t i) -> double {
            if (i >= p + 1) return neg_inf;
            return transition(g, h, dp.v(i - 1, q - 1), i, p);
        };
        solve(a, n_rows, n_rows, argmax);
        for (std::size_t n = 1; n <= n_rows; ++n) {
            const std::uint32_t best = argmax[n];
            dp.v(n, q) = a(n, best);
            dp.p(n, q) = static_cast<std::int32_t>(best) - 1;
        }
    }
}

inline dp_table build_dp(const segment_distribution& g, const segment_distribution& h,
                         std::size_t layers, method m) {
    dp_table dp(g.segments(), layers);
    switch (m) {
        case method::plbf:
        case method::fast: full_scan_dp(g, h, dp.rows, dp); break;
        case method::fast_pp:
            layered_dp(g, h, dp, [](auto& a, std::size_t n, std::size_t c, auto& out) {
                monotone_maxima(a, n, c, out);
            });
            break;
        case method::fast_sharp:
            layered_dp(g, h, dp, [](auto& a, std::size_t n, std::size_t c, auto& out) {
                smawk(a, n, c, out);
            });
            break;
    }
    return dp;
}

// Boundaries for "segments j..N form the last region", read back from the DP.
//
// Returns false where the reference returns None, and also where the
// predecessor chain is not a strictly decreasing walk to 0. That second case
// only arises when an approximate solver ran on a matrix that is not totally
// monotone and handed back a column with no feasible value. The reference would
// carry on and take the log of a negative mass; skipping the candidate is the
// conservative reading.
inline bool trace(const dp_table& dp, std::size_t j, std::size_t k, std::size_t n_segments,
                  std::vector<std::uint32_t>& boundary) {
    boundary.assign(k + 1, 0);
    boundary[k] = static_cast<std::uint32_t>(n_segments);
    std::size_t now = j - 1;
    if (dp.p(now, k - 1) < 0) return false;
    boundary[k - 1] = static_cast<std::uint32_t>(now);
    for (std::size_t q = k - 1; q >= 1; --q) {
        const std::int32_t prev = dp.p(now, q);
        if (prev < 0 || static_cast<std::size_t>(prev) >= now) return false;
        now = static_cast<std::size_t>(prev);
        boundary[q - 1] = static_cast<std::uint32_t>(now);
    }
    return boundary[0] == 0;
}

inline void region_masses(const segment_distribution& d, std::span<const std::uint32_t> boundary,
                          std::vector<double>& out) {
    out.resize(boundary.size() - 1);
    for (std::size_t r = 0; r + 1 < boundary.size(); ++r) {
        out[r] = d.cumulative(boundary[r + 1]) - d.cumulative(boundary[r]);
    }
}

// Scratch for the per-j work, so the j loop allocates nothing.
struct fpr_scratch {
    std::vector<std::uint32_t> order;
    std::vector<double> ratio, suf_a, suf_b;
};

// Regions with non-zero non-key mass, by key/non-key ratio, highest first.
// Stable, as Python's sort is, so equal ratios keep region order.
inline void order_by_ratio(std::span<const double> pos, std::span<const double> neg, fpr_scratch& s) {
    s.order.clear();
    s.ratio.assign(pos.size(), 0.0);
    for (std::size_t i = 0; i < pos.size(); ++i) {
        if (neg[i] != 0) {
            s.order.push_back(static_cast<std::uint32_t>(i));
            s.ratio[i] = pos[i] / neg[i];
        }
    }
    std::stable_sort(s.order.begin(), s.order.end(),
                     [&](std::uint32_t a, std::uint32_t b) { return s.ratio[a] > s.ratio[b]; });
}

// The FPRs minimising classic-Bloom-filter bits subject to an overall FPR of
// `target`: the reference's OptimalFPR.
//
// Each region's FPR is proportional to its key/non-key ratio. A region whose
// FPR would exceed 1 is clamped to 1 -- it gets no filter -- and its non-key
// mass comes out of the budget. Because regions are visited by descending
// ratio, once the top remaining region fits, every region after it does too.
inline void optimal_fpr(std::span<const double> pos, std::span<const double> neg, double target,
                        std::vector<double>& fpr, fpr_scratch& s) {
    const std::size_t k = pos.size();
    order_by_ratio(pos, neg, s);
    const std::size_t len = s.order.size();
    s.suf_a.assign(len + 1, 0.0);
    s.suf_b.assign(len + 1, 0.0);
    for (std::size_t j = len; j-- > 0;) {
        s.suf_a[j] = s.suf_a[j + 1] + pos[s.order[j]];
        s.suf_b[j] = s.suf_b[j + 1] + neg[s.order[j]];
    }

    std::size_t m = 0;
    double invalid_neg = 0.0;
    double valid_pos = 0.0;
    double valid_neg = 0.0;
    double normed = 0.0;
    while (true) {
        valid_pos = s.suf_a[m];
        valid_neg = s.suf_b[m];
        if (valid_pos == 0) {
            fpr.assign(k, 1.0);
            for (std::size_t j = m; j < len; ++j) fpr[s.order[j]] = 0.0;
            return;
        }
        normed = (target - invalid_neg) / (1 - invalid_neg);
        const std::uint32_t top = s.order[m];
        const double top_fpr = normed * (pos[top] / valid_pos) / (neg[top] / valid_neg);
        if (top_fpr > 1) {
            invalid_neg += neg[top];
            ++m;
            continue;
        }
        break;
    }
    fpr.assign(k, 1.0);
    for (std::size_t j = m; j < len; ++j) {
        const std::uint32_t i = s.order[j];
        fpr[i] = normed * (pos[i] / valid_pos) / (neg[i] / valid_neg);
    }
}

// The FPRs minimising the overall FPR subject to `memory_bits` of classic
// Bloom filter: the reference's OptimalFPR_M, with the same clamping argument.
inline void optimal_fpr_for_memory(std::span<const double> pos, std::span<const double> neg,
                                   double memory_bits, std::size_t key_count,
                                   std::vector<double>& fpr, fpr_scratch& s) {
    const std::size_t k = pos.size();
    const double c = std::log2(std::numbers::e);
    const auto n = static_cast<double>(key_count);
    order_by_ratio(pos, neg, s);
    const std::size_t len = s.order.size();

    double g_sum = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
        if (neg[i] == 0) g_sum += pos[i];
    }
    s.suf_a.assign(len + 1, 0.0);
    for (std::size_t j = len; j-- > 0;) {
        const std::uint32_t i = s.order[j];
        const double term = pos[i] != 0 ? pos[i] * std::log2(pos[i] / neg[i]) : 0.0;
        s.suf_a[j] = s.suf_a[j + 1] + term;
    }

    std::size_t m = 0;
    double beta = 0.0;
    while (m < len) {
        beta = (memory_bits + c * n * s.suf_a[m]) / (c * n * (1 - g_sum));
        const std::uint32_t top = s.order[m];
        if (std::pow(2.0, -beta) * pos[top] / neg[top] > 1) {
            g_sum += pos[top];
            ++m;
            continue;
        }
        break;
    }
    fpr.assign(k, 1.0);
    for (std::size_t j = m; j < len; ++j) {
        const std::uint32_t i = s.order[j];
        fpr[i] = std::pow(2.0, -beta) * pos[i] / neg[i];
    }
}

// The reference's SpaceUsed: bits of a classic Bloom filter per region, n keys
// in total. A region at FPR 1 costs nothing.
inline double space_used(std::span<const double> pos, std::span<const double> fpr,
                         std::size_t key_count) {
    double bits = 0;
    for (std::size_t r = 0; r < pos.size(); ++r) {
        const double pos_num = pos[r] * static_cast<double>(key_count);
        if (pos_num == 0) continue;
        const double hash_num = std::log(fpr[r]) / std::log(0.5);
        bits += hash_num * pos_num / std::log(2.0);
    }
    return bits;
}

// The reference's ExpectedFPR.
inline double expected_fpr(std::span<const double> neg, std::span<const double> fpr) {
    double e = 0;
    for (std::size_t r = 0; r < neg.size(); ++r) e += neg[r] * fpr[r];
    return e;
}

enum class objective : std::uint8_t { min_space_at_fpr, min_fpr_at_space };

inline layout optimize(const segment_distribution& g, const segment_distribution& h,
                       std::size_t key_count, std::size_t k, double target, method m,
                       objective obj) {
    const std::size_t n_segments = g.segments();
    if (h.segments() != n_segments) throw std::invalid_argument("plbf: key and non-key segment counts differ");
    // The reference cannot trace k = 1 (it would read DP layer 0, which has no
    // predecessors), so neither do we: one region is just a Bloom filter.
    if (k < 2 || k > n_segments) throw std::invalid_argument("plbf: need 2 <= regions <= segments");
    if (key_count == 0) throw std::invalid_argument("plbf: no keys");
    if (obj == objective::min_space_at_fpr && !(target > 0 && target < 1)) {
        throw std::invalid_argument("plbf: target FPR must be in (0, 1)");
    }
    if (obj == objective::min_fpr_at_space && !(target > 0)) {
        throw std::invalid_argument("plbf: memory budget must be positive");
    }

    const std::size_t layers = k - 1;
    dp_table dp(0, 0);
    if (m != method::plbf) dp = build_dp(g, h, layers, m);

    layout best;
    double best_cost = std::numeric_limits<double>::infinity();
    std::vector<std::uint32_t> boundary;
    std::vector<double> pos, neg, fpr;
    fpr_scratch scratch;

    for (std::size_t j = k; j <= n_segments; ++j) {
        if (m == method::plbf) {
            // What makes the original cubic: the DP for segments 1..j-1 is
            // rebuilt from scratch for every j, though it is a prefix of the
            // previous one. Kept so the benchmarks measure what the paper did.
            dp = dp_table(j - 1, layers);
            full_scan_dp(g, h, j - 1, dp);
        }
        if (!trace(dp, j, k, n_segments, boundary)) continue;
        region_masses(g, boundary, pos);
        region_masses(h, boundary, neg);

        double cost = 0;
        if (obj == objective::min_space_at_fpr) {
            optimal_fpr(pos, neg, target, fpr, scratch);
            cost = space_used(pos, fpr, key_count);
        } else {
            optimal_fpr_for_memory(pos, neg, target, key_count, fpr, scratch);
            cost = expected_fpr(neg, fpr);
        }
        // Strict, as in the reference: the first j to reach the minimum wins.
        if (best_cost > cost) {
            best_cost = cost;
            best.boundary = boundary;
            best.fpr = fpr;
            best.backup_bits = space_used(pos, fpr, key_count);
            best.expected_fpr = expected_fpr(neg, fpr);
        }
    }
    if (best.boundary.empty()) throw std::runtime_error("plbf: no feasible layout");
    return best;
}

}  // namespace detail

// Least backup-filter memory such that the expected overall FPR is target_fpr.
// key_count is the number of keys the distributions were built from.
inline layout optimize_for_fpr(const segment_distribution& keys, const segment_distribution& non_keys,
                               std::size_t key_count, std::size_t regions, double target_fpr,
                               method m = method::fast_sharp) {
    return detail::optimize(keys, non_keys, key_count, regions, target_fpr, m,
                            detail::objective::min_space_at_fpr);
}

// Lowest expected overall FPR with memory_bits of (classic) backup filter.
//
// The layout's backup_bits is the space of the chosen layout. The reference's
// _M classes record the space of the last candidate j they tried instead, which
// is harmless in practice -- every candidate is sized to the same budget, so the
// two agree to within rounding -- but it is the chosen layout's that is reported
// here.
inline layout optimize_for_memory(const segment_distribution& keys,
                                  const segment_distribution& non_keys, std::size_t key_count,
                                  std::size_t regions, double memory_bits,
                                  method m = method::fast_sharp) {
    return detail::optimize(keys, non_keys, key_count, regions, memory_bits, m,
                            detail::objective::min_fpr_at_space);
}

}  // namespace plbf
