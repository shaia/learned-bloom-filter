// Construction time, step by step, at the scale of the paper's URL dataset.
//
// The paper times the threshold search in Python. This measures the same
// search in C++, alongside the steps around it -- histogramming the scores,
// sizing and filling the backup filters -- because once the search takes
// microseconds, those are what a rebuild actually waits on.
//
// Usage: bench_construct [keys] [non_keys]

#include "workload.hpp"

#include <plbf/filter.hpp>
#include <plbf/optimize.hpp>
#include <plbf/segments.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace plbf;
using namespace plbf::bench;

namespace {

const char* name_of(method m) {
    switch (m) {
        case method::plbf: return "plbf";
        case method::fast: return "fast";
        case method::fast_pp: return "fast_pp";
        case method::fast_sharp: return "fast_sharp";
    }
    return "?";
}

void run(const workload& w, shape s, std::size_t n_seg, std::size_t k, bool include_plbf) {
    std::printf("\n%s scores, N = %zu, k = %zu, target FPR 0.01\n", s == shape::ideal ? "ideal" : "bumpy",
                n_seg, k);
    const segment_grid grid(n_seg);
    const double t_hist = median_ms(5, [&] {
        const segment_distribution g(grid, w.key_scores);
        const segment_distribution h(grid, w.non_key_scores);
        keep(g.cumulative(1));
        keep(h.cumulative(1));
    });
    std::printf("  %-28s %10.3f ms\n", "histogram keys + non-keys", t_hist);

    const segment_distribution g(grid, w.key_scores);
    const segment_distribution h(grid, w.non_key_scores);
    double exact_bits = 0;
    for (const method m : {method::plbf, method::fast, method::fast_pp, method::fast_sharp}) {
        if (m == method::plbf && !include_plbf) continue;
        layout l;
        const double t = median_ms(m == method::plbf ? 1 : 7, [&] {
            l = optimize_for_fpr(g, h, w.keys.size(), k, 0.01, m);
        });
        if (m == method::fast) exact_bits = l.backup_bits;
        const double ratio = exact_bits > 0 ? l.backup_bits / exact_bits : 1.0;
        std::printf("  search %-21s %10.3f ms   backup %9.0f bits  (x%.5f of exact)\n", name_of(m), t,
                    l.backup_bits, ratio);
    }

    options opt;
    opt.segments = n_seg;
    opt.regions = k;
    std::size_t bytes = 0;
    const double t_build = median_ms(5, [&] {
        const filter f = filter::build_for_fpr(w.keys, w.key_scores, w.non_key_scores, 0.01, opt);
        bytes = f.memory_bytes();
    });
    std::printf("  %-28s %10.3f ms   %zu bytes (%.2f bits/key)\n", "whole build (fast_sharp)", t_build, bytes,
                8.0 * static_cast<double>(bytes) / static_cast<double>(w.keys.size()));
}

}  // namespace

int main(int argc, char** argv) {
    // Defaults: the paper's URL dataset has 223,088 keys and 428,118 non-keys.
    const std::size_t keys = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 223088;
    const std::size_t non_keys = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 428118;
    std::printf("bench_construct: %zu keys, %zu non-keys (medians; plbf is one run)\n", keys, non_keys);

    for (const shape s : {shape::ideal, shape::bumpy}) {
        const workload w = make_workload(keys, non_keys, s, 42);
        run(w, s, 1000, 5, true);
        run(w, s, 1000, 50, false);
    }
    return 0;
}
