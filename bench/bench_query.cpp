// Query throughput: one contains() at a time, the scalar batch, the AVX2
// batch with and without the vector lookup, and a plain split-block Bloom
// filter at the same FPR for scale.
//
// Each query is a (hash, score) pair; the classifier that would produce the
// score is not part of this. Queries are half keys, half unseen non-keys, in
// random order, so the probes land all over the block array.
//
// The variants run interleaved and the best of several rounds is reported:
// on a shared machine the minimum is the least noisy estimate of what the
// code itself costs. Compare variants within one run, never across runs.
//
// Usage: bench_query [keys] [queries]

#include "workload.hpp"

#include <plbf/filter.hpp>
#include <plbf/split_block.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <vector>

using namespace plbf;
using namespace plbf::bench;

namespace {

struct queries {
    std::vector<std::uint64_t> hashes;
    std::vector<double> scores;
    std::vector<std::uint8_t> is_key;
};

queries make_queries(const workload& w, const workload& unseen, std::size_t count) {
    std::mt19937_64 rng(99);
    queries q;
    for (std::size_t i = 0; i < count; ++i) {
        if (i % 2 == 0) {
            const std::size_t j = rng() % w.keys.size();
            q.hashes.push_back(w.keys[j]);
            q.scores.push_back(w.key_scores[j]);
            q.is_key.push_back(1);
        } else {
            const std::size_t j = rng() % unseen.non_keys.size();
            q.hashes.push_back(unseen.non_keys[j]);
            q.scores.push_back(unseen.non_key_scores[j]);
            q.is_key.push_back(0);
        }
    }
    return q;
}

// The non-learned baseline: every key in one split-block filter at the target.
struct plain_filter {
    std::vector<split_block::block> blocks;
    std::uint32_t count = 0;

    plain_filter(const std::vector<std::uint64_t>& keys, double fpr)
        : count(split_block::blocks_for(keys.size(), fpr)) {
        blocks.assign(count, split_block::block{});
        for (const auto h : keys) {
            split_block::insert(blocks[split_block::block_index(h, count)], static_cast<std::uint32_t>(h));
        }
    }
    bool contains(std::uint64_t h) const noexcept {
        return split_block::probe_scalar(blocks[split_block::block_index(h, count)], static_cast<std::uint32_t>(h));
    }

#if PLBF_X86_64
    // The same chunked prefetch-then-probe loop as filter::contains_batch, so
    // the learned/plain comparison is batch against batch.
    PLBF_TARGET_AVX2 void contains_batch_avx2(const std::uint64_t* h, std::uint8_t* out, std::size_t n) const {
        constexpr std::size_t chunk = 64;
        const split_block::block* plan[chunk];
        for (std::size_t base = 0; base < n; base += chunk) {
            const std::size_t len = n - base < chunk ? n - base : chunk;
            for (std::size_t i = 0; i < len; ++i) {
                plan[i] = &blocks[split_block::block_index(h[base + i], count)];
                _mm_prefetch(reinterpret_cast<const char*>(plan[i]), _MM_HINT_T0);
            }
            for (std::size_t i = 0; i < len; ++i) {
                out[base + i] = split_block::probe_avx2(*plan[i], static_cast<std::uint32_t>(h[base + i])) ? 1 : 0;
            }
        }
    }
#endif
};

double fpr_of(const std::vector<std::uint8_t>& out, const queries& q) {
    std::size_t fp = 0, nk = 0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (!q.is_key[i]) {
            ++nk;
            fp += out[i];
        }
    }
    return static_cast<double>(fp) / static_cast<double>(nk);
}

void run(std::size_t n_keys, std::size_t n_queries) {
    // The training sample only feeds a histogram; past a few million it buys
    // nothing but memory.
    const std::size_t n_train = std::min<std::size_t>(n_keys * 2, 4000000);
    const workload w = make_workload(n_keys, n_train, shape::ideal, 1);
    const workload unseen = make_workload(0, n_queries, shape::ideal, 2, 1ULL << 40);
    const queries q = make_queries(w, unseen, n_queries);
    const double target = 0.01;

    const filter f = filter::build_for_fpr(w.keys, w.key_scores, w.non_key_scores, target);
    const plain_filter plain(w.keys, target);
    const auto bits_per_key = [&](std::size_t bytes) { return 8.0 * static_cast<double>(bytes) / static_cast<double>(n_keys); };

    std::printf("\n%zu keys, %zu queries, target FPR %.2f\n", n_keys, n_queries, target);
    std::printf("  learned filter %10zu bytes (%5.2f bits/key)   plain split-block %10zu bytes (%5.2f bits/key)\n",
                f.memory_bytes(), bits_per_key(f.memory_bytes()), plain.blocks.size() * sizeof(split_block::block),
                bits_per_key(plain.blocks.size() * sizeof(split_block::block)));

    std::vector<std::uint8_t> out(n_queries);
    struct variant {
        const char* label;
        bool needs_avx2;
        std::function<void()> run;
        double best = std::numeric_limits<double>::infinity();
        double fpr = 0;
    };
    // The loops copy what they read into locals first. A byte store through
    // out[i] may alias anything, including the closure's captured references,
    // so reading through those would reload the pointers and the count on every
    // iteration and bill it to the filter: 0.25-0.31 ns/query with the filter in
    // L2 (bench/results/2026-10-30/captured_vs_locals.txt).
    std::vector<variant> variants = {
        {"learned, contains() loop", false,
         [&] {
             const std::uint64_t* qh = q.hashes.data();
             const double* qs = q.scores.data();
             std::uint8_t* o = out.data();
             const std::size_t n = n_queries;
             for (std::size_t i = 0; i < n; ++i) o[i] = f.contains(qh[i], qs[i]) ? 1 : 0;
         }},
        {"learned, batch scalar", false, [&] { f.contains_batch(q.hashes, q.scores, out, kernel::scalar); }},
        {"learned, scalar lookup + AVX2 probe", true,
         [&] { f.contains_batch(q.hashes, q.scores, out, kernel::avx2_probe); }},
        {"learned, AVX2 lookup + AVX2 probe", true, [&] { f.contains_batch(q.hashes, q.scores, out, kernel::avx2); }},
        {"plain split-block, loop", false,
         [&] {
             const std::uint64_t* qh = q.hashes.data();
             std::uint8_t* o = out.data();
             const std::size_t n = n_queries;
             for (std::size_t i = 0; i < n; ++i) o[i] = plain.contains(qh[i]) ? 1 : 0;
         }},
#if PLBF_X86_64
        {"plain split-block, AVX2 batch", true,
         [&] { plain.contains_batch_avx2(q.hashes.data(), out.data(), n_queries); }},
#endif
    };

    const bool avx2 = cpu_has_avx2();
    for (int round = 0; round < 7; ++round) {
        for (auto& v : variants) {
            if (v.needs_avx2 && !avx2) continue;
            v.best = std::min(v.best, median_ms(1, [&] {
                v.run();
                keep(out[n_queries - 1]);
            }));
            v.fpr = fpr_of(out, q);
        }
    }
    for (const auto& v : variants) {
        if (v.needs_avx2 && !avx2) {
            std::printf("  %-36s (no AVX2)\n", v.label);
            continue;
        }
        std::printf("  %-36s %7.2f ns/query   measured FPR %.4f\n", v.label,
                    v.best * 1e6 / static_cast<double>(n_queries), v.fpr);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t queries_n = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 10000000;
    std::printf("bench_query: best of 7 interleaved rounds per variant\n");
    if (argc > 1) {
        run(std::strtoull(argv[1], nullptr, 10), queries_n);
        return 0;
    }
    // At ~7 bits/key: filter in L2 (~180 KB), in L3 (~3.6 MB), and well past
    // a 36 MB L3 (~58 MB).
    for (const std::size_t keys : {std::size_t{200000}, std::size_t{4000000}, std::size_t{64000000}}) {
        run(keys, queries_n);
    }
    return 0;
}
