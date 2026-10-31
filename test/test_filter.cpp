// The filter: segment mapping, the split-block kernels, the sizing model, and
// end-to-end behaviour -- no false negatives, and an FPR that meets its target.

#include "harness.hpp"

#include <plbf/filter.hpp>
#include <plbf/hash.hpp>
#include <plbf/region_tables.hpp>
#include <plbf/segments.hpp>
#include <plbf/simd.hpp>
#include <plbf/split_block.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace plbf;
using namespace plbf::test;

namespace {

// --- segment mapping ---------------------------------------------------------

// The reference's mapping, verbatim: bisect_left over the thresholds, 0 -> 1.
std::size_t reference_index(const segment_grid& grid, double score) {
    std::vector<double> t(grid.segments() + 1);
    for (std::size_t i = 0; i < t.size(); ++i) t[i] = grid.threshold(i);
    const auto i = static_cast<std::size_t>(std::lower_bound(t.begin(), t.end(), score) - t.begin());
    return i == 0 ? 1 : i;
}

void test_segment_index() {
    section("segments: index() is the reference's bisect, including on the thresholds");
    std::mt19937_64 rng(3);
    int mismatches = 0;
    for (const std::size_t n : {1u, 3u, 7u, 50u, 100u, 999u, 1000u, 4096u}) {
        const segment_grid grid(n);
        std::vector<double> probes = {0.0, 1.0, std::numeric_limits<double>::denorm_min(),
                                      std::nextafter(1.0, 0.0)};
        for (std::size_t i = 0; i <= n; ++i) {
            const double t = grid.threshold(i);
            probes.push_back(t);
            probes.push_back(std::nextafter(t, 0.0));
            probes.push_back(std::nextafter(t, 2.0));
        }
        for (int i = 0; i < 20000; ++i) probes.push_back(std::uniform_real_distribution<double>(0, 1)(rng));
        for (const double s : probes) {
            if (s < 0 || s > 1) continue;
            if (grid.index(s) != reference_index(grid, s)) ++mismatches;
        }
    }
    check_eq(mismatches, 0, "index() == bisect_left for random, threshold and next-to-threshold scores");

    const segment_grid grid(10);
    check_eq(grid.index(-0.5), 1, "negative score clamps to segment 1");
    check_eq(grid.index(std::nan("")), 1, "NaN clamps to segment 1");
    check_eq(grid.index(7.0), 10, "score above 1 clamps to segment N");
}

// Scores that sit on, or one ulp either side of, every threshold -- where a
// one-step correction would go wrong first -- plus everything index() clamps.
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

void test_segment_index_avx2() {
    section("segments: the four-lane index equals index() everywhere");
    if (!cpu_has_avx2()) {
        std::printf("  (no AVX2 on this CPU; skipped)\n");
        return;
    }
#if PLBF_X86_64
    std::mt19937_64 rng(17);
    std::size_t probes = 0;
    int mismatches = 0;
    // Large N is where score * N carries the most rounding; 2^24 + 1 and the
    // primes keep the thresholds from being exact binary fractions.
    for (const std::size_t n : {1u, 2u, 3u, 7u, 50u, 1000u, 4096u, 65537u, 1000003u, 16777217u}) {
        const segment_grid grid(n);
        const std::vector<double> s = edge_scores(grid, rng, 100000);
        std::vector<std::uint32_t> got(s.size());
        grid.index_batch_avx2(s.data(), s.size(), got.data());
        for (std::size_t i = 0; i < s.size(); ++i) {
            if (got[i] != grid.index(s[i])) ++mismatches;
        }
        probes += s.size();
    }
    check_eq(mismatches, 0, std::to_string(probes) + " scores over 10 grids, N up to 2^24 + 1");
#endif
}

// Every batch kernel against contains(), on scores that stress the lookup and
// on lengths that leave a ragged tail in both the 4-lane loop and the chunk.
void test_batch_kernels_agree() {
    section("filter: every batch kernel equals contains(), edge scores and ragged lengths");
    std::mt19937_64 rng(23);
    std::uniform_real_distribution<double> u(0, 1);
    std::vector<std::uint64_t> keys(30000);
    std::vector<double> key_scores(keys.size()), non_key_scores(60000);
    for (std::size_t i = 0; i < keys.size(); ++i) {
        keys[i] = hash_u64(i);
        key_scores[i] = std::pow(u(rng), 0.25);
    }
    for (auto& s : non_key_scores) s = 1 - std::sqrt(u(rng));
    options opt;
    opt.segments = 997;
    opt.regions = 9;
    const filter f = filter::build_for_fpr(keys, key_scores, non_key_scores, 0.01, opt);

    // Queries: half inserted keys at their own scores, half arbitrary hashes
    // at edge scores, interleaved.
    const std::vector<double> edges = edge_scores(f.grid(), rng, 20000);
    std::vector<std::uint64_t> qh;
    std::vector<double> qs;
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const std::size_t k = rng() % keys.size();
        qh.push_back(keys[k]);
        qs.push_back(key_scores[k]);
        qh.push_back(hash_u64(1'000'000 + i));
        qs.push_back(edges[i]);
    }

    int mismatches = 0;
    std::size_t runs = 0;
    for (const std::size_t len : {std::size_t{1}, std::size_t{3}, std::size_t{4}, std::size_t{63},
                                  std::size_t{64}, std::size_t{65}, std::size_t{1001}, qh.size()}) {
        const std::span<const std::uint64_t> h(qh.data(), len);
        const std::span<const double> s(qs.data(), len);
        std::vector<std::uint8_t> want(len), got(len);
        for (std::size_t i = 0; i < len; ++i) want[i] = f.contains(qh[i], qs[i]) ? 1 : 0;
        for (const kernel kern : {kernel::scalar, kernel::avx2, kernel::avx2_probe}) {
            if (kern != kernel::scalar && !cpu_has_avx2()) continue;
            std::fill(got.begin(), got.end(), std::uint8_t{2});
            f.contains_batch(h, s, got, kern);
            if (got != want) ++mismatches;
            ++runs;
        }
    }
    check_eq(mismatches, 0, std::to_string(runs) + " kernel x length runs identical to contains()");
}

// The compact query formats: 32-bit scores answer as contains() on the score
// widened to double, and 16-bit segment numbers as contains() on the score the
// segment came from. The filter is built from float scores widened, as a
// model's float32 output would be.
void test_compact_queries() {
    section("filter: float scores and segment numbers answer as contains()");
    std::mt19937_64 rng(29);
    std::uniform_real_distribution<double> u(0, 1);
    std::vector<std::uint64_t> keys(30000);
    std::vector<float> key_f(keys.size());
    std::vector<double> key_scores(keys.size()), non_key_scores(60000);
    for (std::size_t i = 0; i < keys.size(); ++i) {
        keys[i] = hash_u64(i);
        key_f[i] = static_cast<float>(std::pow(u(rng), 0.25));
        key_scores[i] = key_f[i];
    }
    for (auto& s : non_key_scores) s = static_cast<float>(1 - std::sqrt(u(rng)));
    options opt;
    opt.segments = 997;
    opt.regions = 9;
    const filter f = filter::build_for_fpr(keys, key_scores, non_key_scores, 0.01, opt);
    const segment_grid& grid = f.grid();

    // Keys at their float scores, then unseen hashes at every threshold rounded
    // to float and the floats either side of it, then the values index() clamps.
    std::vector<std::uint64_t> qh(keys);
    std::vector<float> qf(key_f);
    constexpr float finf = std::numeric_limits<float>::infinity();
    std::vector<float> edges = {0.0f, -0.0f, 1.0f, -1.0f, 2.0f, finf, -finf, std::nanf("")};
    for (std::size_t i = 0; i <= grid.segments(); ++i) {
        const auto t = static_cast<float>(grid.threshold(i));
        edges.insert(edges.end(), {std::nextafter(t, -1.0f), t, std::nextafter(t, 2.0f)});
    }
    for (std::size_t i = 0; i < edges.size(); ++i) {
        qh.push_back(hash_u64(2'000'000 + i));
        qf.push_back(edges[i]);
    }
    std::vector<std::uint8_t> want(qh.size());
    for (std::size_t i = 0; i < qh.size(); ++i) want[i] = f.contains(qh[i], static_cast<double>(qf[i])) ? 1 : 0;

    std::size_t missed = 0;
    for (std::size_t i = 0; i < keys.size(); ++i) missed += want[i] == 0;
    check_eq(missed, 0, "every key found from its float score");

    int mismatches = 0, runs = 0;
    for (const kernel kern : {kernel::scalar, kernel::avx2, kernel::avx2_probe}) {
        if (kern != kernel::scalar && !cpu_has_avx2()) continue;
        std::vector<std::uint8_t> got(qh.size(), 2);
        f.contains_batch(qh, qf, got, kern);
        if (got != want) ++mismatches;
        ++runs;
    }
    check_eq(mismatches, 0, std::to_string(runs) + " float-score kernels identical to contains() on the widened scores");

    // Segment numbers: on double edge scores, every threshold's neighbours.
    const std::vector<double> ds = edge_scores(grid, rng, 20000);
    int seg_mismatches = 0;
    for (std::size_t i = 0; i < ds.size(); ++i) {
        const std::uint64_t h = hash_u64(3'000'000 + i);
        const auto seg = static_cast<std::uint32_t>(grid.index(ds[i]));
        if (f.contains_segment(h, seg) != f.contains(h, ds[i])) ++seg_mismatches;
    }
    check_eq(seg_mismatches, 0, std::to_string(ds.size()) + " segment-number queries identical to contains()");

    int clamp_mismatches = 0;
    const auto n = static_cast<std::uint32_t>(grid.segments());
    for (std::uint64_t i = 0; i < 10000; ++i) {
        const std::uint64_t h = hash_u64(4'000'000 + i);
        if (f.contains_segment(h, 0) != f.contains_segment(h, 1)) ++clamp_mismatches;
        if (f.contains_segment(h, n + 7) != f.contains_segment(h, n)) ++clamp_mismatches;
    }
    check_eq(clamp_mismatches, 0, "segment 0 answers as 1, and segments past N as N");

    // The 16-bit batch conversions equal index(), with ragged tails.
    int conv_mismatches = 0;
    for (const std::size_t len : {ds.size(), ds.size() - 1, ds.size() - 2, ds.size() - 3}) {
        std::vector<std::uint16_t> got(len, 0xffff);
        grid.index_batch(ds.data(), len, got.data());
        for (std::size_t i = 0; i < len; ++i) conv_mismatches += got[i] != grid.index(ds[i]);
#if PLBF_X86_64
        if (cpu_has_avx2()) {
            std::fill(got.begin(), got.end(), std::uint16_t{0xffff});
            grid.index_batch_avx2(ds.data(), len, got.data());
            for (std::size_t i = 0; i < len; ++i) conv_mismatches += got[i] != grid.index(ds[i]);
        }
#endif
    }
    check_eq(conv_mismatches, 0, "16-bit segment numbers, scalar and AVX2, equal index()");
}

// --- split-block kernels and sizing -----------------------------------------

void test_kernels_agree() {
    section("split block: scalar and AVX2 probes agree bit for bit");
    if (!cpu_has_avx2()) {
        std::printf("  (no AVX2 on this CPU; skipped)\n");
        return;
    }
#if PLBF_X86_64
    std::mt19937_64 rng(5);
    int mismatches = 0;
    std::vector<split_block::block> blocks(64);
    for (int round = 0; round < 50; ++round) {
        for (auto& b : blocks) {
            b = {};
            const int fill = static_cast<int>(rng() % 40);
            for (int i = 0; i < fill; ++i) split_block::insert(b, static_cast<std::uint32_t>(rng()));
        }
        for (int i = 0; i < 20000; ++i) {
            const auto& b = blocks[rng() % blocks.size()];
            const auto x = static_cast<std::uint32_t>(rng());
            if (split_block::probe_scalar(b, x) != split_block::probe_avx2(b, x)) ++mismatches;
        }
    }
    check_eq(mismatches, 0, "1M random probes over blocks of varying fill");
#endif
}

void test_fpr_model() {
    section("split block: the FPR model predicts what a filter measures");
    std::mt19937_64 rng(9);
    for (const double load : {2.0, 6.0, 15.0, 40.0}) {
        const std::size_t blocks = 4096;
        const auto keys = static_cast<std::size_t>(load * blocks);
        std::vector<split_block::block> f(blocks);
        for (std::size_t i = 0; i < keys; ++i) {
            const std::uint64_t h = hash_u64(rng());
            split_block::insert(f[split_block::block_index(h, blocks)], static_cast<std::uint32_t>(h));
        }
        const std::size_t trials = 2000000;
        std::size_t hits = 0;
        for (std::size_t i = 0; i < trials; ++i) {
            const std::uint64_t h = hash_u64(rng());
            hits += split_block::probe_scalar(f[split_block::block_index(h, blocks)], static_cast<std::uint32_t>(h));
        }
        const double measured = static_cast<double>(hits) / static_cast<double>(trials);
        const double predicted = split_block::false_positive_rate(load);
        // Binomial noise at 2M trials is under 1% relative for every load here
        // except the smallest, whose FPR is ~1e-5; allow 3 sigma either way.
        const double sigma = std::sqrt(predicted * (1 - predicted) / static_cast<double>(trials));
        std::printf("  load %4.0f keys/block: predicted %.3e, measured %.3e\n", load, predicted, measured);
        check(std::abs(measured - predicted) <= 3 * sigma + 1e-7,
              "load " + std::to_string(load) + ": measured within 3 sigma of the model");
    }

    for (const double target : {0.3, 0.05, 0.01, 1e-3, 1e-4}) {
        for (const std::size_t n : {std::size_t{1}, std::size_t{37}, std::size_t{100000}}) {
            const std::uint32_t b = split_block::blocks_for(n, target);
            const auto fpr_at = [&](std::uint32_t blocks) {
                return split_block::false_positive_rate(static_cast<double>(n) / blocks);
            };
            check(fpr_at(b) <= target, "blocks_for meets target " + std::to_string(target));
            check(b == 1 || fpr_at(b - 1) > target, "blocks_for is minimal at target " + std::to_string(target));
        }
    }
}

// --- end to end ----------------------------------------------------------------

// A classifier that is good but not perfect: key scores lean towards 1, non-key
// scores towards 0, with real overlap. The density ratio is increasing, so
// this is the paper's ideal case.
struct workload {
    std::vector<std::uint64_t> keys, train_non_keys, test_non_keys;
    std::vector<double> key_scores, train_scores, test_scores;
};

workload make_workload(std::size_t n_keys, std::size_t n_train, std::size_t n_test, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0, 1);
    workload w;
    std::uint64_t id = 0;
    auto key_score = [&] { return std::pow(u(rng), 0.25); };        // density 4 s^3
    auto non_key_score = [&] { return 1 - std::sqrt(u(rng)); };     // density 2 (1 - s)
    for (std::size_t i = 0; i < n_keys; ++i) {
        w.keys.push_back(hash_u64(id++));
        w.key_scores.push_back(key_score());
    }
    for (std::size_t i = 0; i < n_train; ++i) {
        w.train_non_keys.push_back(hash_u64(id++));
        w.train_scores.push_back(non_key_score());
    }
    for (std::size_t i = 0; i < n_test; ++i) {
        w.test_non_keys.push_back(hash_u64(id++));
        w.test_scores.push_back(non_key_score());
    }
    return w;
}

const char* name_of(method m) {
    switch (m) {
        case method::plbf: return "plbf";
        case method::fast: return "fast";
        case method::fast_pp: return "fast_pp";
        case method::fast_sharp: return "fast_sharp";
    }
    return "?";
}

void check_filter(const filter& f, const workload& w, double target, const std::string& name) {
    // No false negatives, through every query path.
    std::size_t missed = 0;
    for (std::size_t i = 0; i < w.keys.size(); ++i) missed += !f.contains(w.keys[i], w.key_scores[i]);
    check_eq(missed, 0, name + ": every key found (contains)");

    std::vector<std::uint8_t> out(w.keys.size());
    for (const kernel kern : {kernel::scalar, kernel::avx2, kernel::avx2_probe}) {
        if (kern != kernel::scalar && !cpu_has_avx2()) continue;
        f.contains_batch(w.keys, w.key_scores, out, kern);
        check(std::all_of(out.begin(), out.end(), [](std::uint8_t v) { return v == 1; }),
              name + ": every key found (batch, kernel " + std::to_string(static_cast<int>(kern)) + ")");
    }

    // The FPR the layout promised, measured on non-keys it never saw.
    std::vector<std::uint8_t> single(w.test_non_keys.size()), batch(w.test_non_keys.size());
    for (std::size_t i = 0; i < w.test_non_keys.size(); ++i) {
        single[i] = f.contains(w.test_non_keys[i], w.test_scores[i]) ? 1 : 0;
    }
    for (const kernel kern : {kernel::scalar, kernel::avx2, kernel::avx2_probe}) {
        if (kern != kernel::scalar && !cpu_has_avx2()) continue;
        f.contains_batch(w.test_non_keys, w.test_scores, batch, kern);
        check(single == batch, name + ": batch answers equal single answers on non-keys (kernel " +
                                   std::to_string(static_cast<int>(kern)) + ")");
    }

    std::size_t fp = 0;
    for (const auto v : single) fp += v;
    const double measured = static_cast<double>(fp) / static_cast<double>(single.size());
    std::printf("  %-10s target %.4f  expected %.4f  measured %.4f  %8.1f KiB  (classic-BF model %8.1f KiB)\n",
                name.c_str(), target, f.regions().expected_fpr, measured,
                static_cast<double>(f.memory_bytes()) / 1024, f.regions().backup_bits / 8 / 1024);
    // Split-block sizing rounds up per region, and the smoothed training
    // histogram is not the test distribution: a little slack, no more.
    check(measured <= target * 1.1, name + ": measured FPR within 10% of target");
}

void test_end_to_end() {
    section("filter: no false negatives, FPR on held-out non-keys meets the target");
    const workload w = make_workload(100000, 200000, 1000000, 21);
    for (const double target : {0.01, 0.001}) {
        for (const method m : {method::fast, method::fast_pp, method::fast_sharp}) {
            options opt;
            opt.search = m;
            const filter f = filter::build_for_fpr(w.keys, w.key_scores, w.train_scores, target, opt);
            check_filter(f, w, target, name_of(m));
        }
    }

    // The cubic original at a size where it finishes quickly, to show it
    // builds the same filter as fast.
    options small;
    small.segments = 100;
    small.search = method::plbf;
    const filter orig = filter::build_for_fpr(w.keys, w.key_scores, w.train_scores, 0.01, small);
    small.search = method::fast;
    const filter fast = filter::build_for_fpr(w.keys, w.key_scores, w.train_scores, 0.01, small);
    check(orig.regions().boundary == fast.regions().boundary && orig.regions().fpr == fast.regions().fpr,
          "plbf and fast build the same layout at N = 100");
    check_eq(orig.memory_bytes(), fast.memory_bytes(), "plbf and fast allocate the same filter");

    section("filter: memory-budget build");
    for (const double bits_per_key : {4.0, 8.0}) {
        const double budget = bits_per_key * static_cast<double>(w.keys.size());
        const filter f = filter::build_for_memory(w.keys, w.key_scores, w.train_scores, budget);
        check(std::abs(f.regions().backup_bits - budget) <= 1e-6 * budget,
              "layout spends the budget (" + std::to_string(bits_per_key) + " bits/key)");
        check_filter(f, w, f.regions().expected_fpr, "budget " + std::to_string(static_cast<int>(bits_per_key)));
    }
}

void test_region_mapping() {
    section("filter: region lookup is the reference's get_region_idx");
    const workload w = make_workload(20000, 20000, 0, 4);
    options opt;
    opt.segments = 200;
    opt.regions = 7;
    const filter f = filter::build_for_fpr(w.keys, w.key_scores, w.train_scores, 0.01, opt);

    // bisect_left over the region thresholds, 0 -> region 1 (1-based there).
    std::vector<double> t;
    for (std::size_t r = 0; r <= f.regions().regions(); ++r) t.push_back(f.threshold(r));
    std::mt19937_64 rng(8);
    int mismatches = 0;
    std::vector<double> probes(t.begin(), t.end());
    for (const double x : t) {
        probes.push_back(std::nextafter(x, 0.0));
        probes.push_back(std::nextafter(x, 2.0));
    }
    for (int i = 0; i < 100000; ++i) probes.push_back(std::uniform_real_distribution<double>(0, 1)(rng));
    for (const double s : probes) {
        if (s < 0 || s > 1) continue;
        auto ref = static_cast<std::size_t>(std::lower_bound(t.begin(), t.end(), s) - t.begin());
        if (ref == 0) ref = 1;
        // A region at FPR 1 answers yes for any key; elsewhere the answer
        // depends on the key, so compare regions through the boundaries.
        const std::size_t seg = f.grid().index(s);
        const std::size_t region = static_cast<std::size_t>(
            std::lower_bound(f.regions().boundary.begin(), f.regions().boundary.end(), seg) -
            f.regions().boundary.begin());
        if (region != ref) ++mismatches;
    }
    check_eq(mismatches, 0, "segment -> region equals bisect on thresholds");
}

void test_from_parts() {
    section("filter: from_parts rebuilds a filter, and rejects parts that do not fit");
    const workload w = make_workload(20000, 20000, 200000, 6);
    options opt;
    opt.segments = 300;
    opt.regions = 6;
    const filter f = filter::build_for_fpr(w.keys, w.key_scores, w.train_scores, 0.01, opt);
    const auto blocks = [&] { return std::vector<split_block::block>(f.blocks().begin(), f.blocks().end()); };

    const filter g = filter::from_parts(opt.segments, f.regions(), f.keys_per_region(), f.tables(), blocks());
    int differ = 0;
    for (std::size_t i = 0; i < w.test_non_keys.size(); ++i) {
        differ += f.contains(w.test_non_keys[i], w.test_scores[i]) != g.contains(w.test_non_keys[i], w.test_scores[i]);
    }
    for (std::size_t i = 0; i < w.keys.size(); ++i) differ += !g.contains(w.keys[i], w.key_scores[i]);
    check_eq(differ, 0, "rebuilt filter answers as the original on 220k queries");
    check_eq(g.memory_bytes(), f.memory_bytes(), "rebuilt filter is the same size");

    auto throws = [](auto fn) {
        try {
            fn();
        } catch (const std::exception&) {
            return true;
        }
        return false;
    };
    check(throws([&] { (void)filter::from_parts(opt.segments + 1, f.regions(), f.keys_per_region(), f.tables(), blocks()); }),
          "wrong segment count is rejected");
    check(throws([&] {
              auto b = blocks();
              b.pop_back();
              (void)filter::from_parts(opt.segments, f.regions(), f.keys_per_region(), f.tables(), std::move(b));
          }),
          "a short block array is rejected");
    check(throws([&] {
              region_tables t = f.tables();
              t.slices.back() = region_tables::make_slice(t.block_count, 1);
              (void)filter::from_parts(opt.segments, f.regions(), f.keys_per_region(), std::move(t), blocks());
          }),
          "a slice past the end is rejected");
    check(throws([&] {
              region_tables t = f.tables();
              t.region_of_segment[5] = static_cast<std::uint16_t>(opt.regions);
              (void)filter::from_parts(opt.segments, f.regions(), f.keys_per_region(), std::move(t), blocks());
          }),
          "a segment mapped to no region is rejected");
}

}  // namespace

int main() {
    test_segment_index();
    test_segment_index_avx2();
    test_kernels_agree();
    test_batch_kernels_agree();
    test_compact_queries();
    test_fpr_model();
    test_region_mapping();
    test_from_parts();
    test_end_to_end();
    return summary("test_filter");
}
