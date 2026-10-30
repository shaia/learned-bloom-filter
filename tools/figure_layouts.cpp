// The numbers behind the blog series' explorable figures, from the library
// itself rather than from a JavaScript replica of it.
//
// Build and run from the repository root:
//   g++ -std=c++20 -O2 -ffp-contract=off -Iinclude tools/figure_layouts.cpp -o figure_layouts
//   ./figure_layouts > plbf_layouts.json
//
// Two figures read the output (as data/plbf_layouts.json in the blog repo):
//
// - plbf-regions: the layout optimize_for_fpr picks for every slider setting,
//   on N = 100 segments, for the benchmarks' two score shapes. The segment
//   masses are the shapes' exact probabilities (CDF differences), not a sample:
//   what an infinitely large training set would histogram to. Bits are per key
//   (key_count = 1; the cost is linear in it).
// - plbf-staircase: one layer of the DP on N = 16 segments, the matrix its row
//   maxima are taken over, and every cell each solver evaluates, in order.

#include <plbf/optimize.hpp>
#include <plbf/row_maxima.hpp>
#include <plbf/segments.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace plbf;

namespace {

// The shapes of bench/workload.hpp, as CDFs. Keys: density 4 s^3. Non-keys:
// density 2 (1 - s); "bumpy" moves `bump` of that mass into a uniform band on
// [lo, lo + width], so the key/non-key ratio dips there.
double key_cdf(double s) { return s * s * s * s; }
double non_key_cdf(double s, double bump, double lo, double width) {
    const double base = 1 - (1 - s) * (1 - s);
    double band = (s - lo) / width;
    band = band < 0 ? 0 : band > 1 ? 1 : band;
    return (1 - bump) * base + bump * band;
}

std::vector<double> masses(std::size_t n, auto cdf) {
    const segment_grid grid(n);
    std::vector<double> m(n);
    for (std::size_t i = 1; i <= n; ++i) m[i - 1] = cdf(grid.threshold(i)) - cdf(grid.threshold(i - 1));
    return m;
}

void print_doubles(const std::vector<double>& v) {
    std::printf("[");
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (std::isinf(v[i])) {
            std::printf("%snull", i ? "," : "");
        } else {
            std::printf("%s%.17g", i ? "," : "", v[i]);
        }
    }
    std::printf("]");
}

template <class T>
void print_ints(const std::vector<T>& v, std::size_t from = 0) {
    std::printf("[");
    for (std::size_t i = from; i < v.size(); ++i) std::printf("%s%llu", i > from ? "," : "", static_cast<unsigned long long>(v[i]));
    std::printf("]");
}

const char* method_name(method m) { return m == method::fast ? "fast" : "fast_sharp"; }

// --- plbf-regions --------------------------------------------------------------

void regions_section() {
    constexpr std::size_t n_seg = 100;
    const double fprs[] = {0.1, 0.05, 0.02, 0.01, 0.005, 0.001};
    struct shape {
        const char* name;
        double bump;
    };
    const shape shapes[] = {{"ideal", 0.0}, {"bumpy", 0.2}};

    std::printf("\"regions\":{\"segments\":%zu,\"band\":[0.69,0.02],\"shapes\":{", n_seg);
    for (std::size_t si = 0; si < 2; ++si) {
        const auto g = masses(n_seg, key_cdf);
        const auto h = masses(n_seg, [&](double s) { return non_key_cdf(s, shapes[si].bump, 0.69, 0.02); });
        std::printf("%s\"%s\":{\"keys\":", si ? "," : "", shapes[si].name);
        print_doubles(g);
        std::printf(",\"non_keys\":");
        print_doubles(h);
        std::printf("}");
    }
    std::printf("},\"plain_bits\":{");
    for (std::size_t fi = 0; fi < std::size(fprs); ++fi) {
        // A classic Bloom filter at FPR f: log2(1/f) / ln 2 bits per key, the
        // same model the optimiser's backup_bits uses.
        std::printf("%s\"%g\":%.17g", fi ? "," : "", fprs[fi], -std::log(fprs[fi]) / (std::log(2.0) * std::log(2.0)));
    }
    std::printf("},\"cases\":[");
    bool first = true;
    for (const shape& sh : shapes) {
        const auto g = segment_distribution::from_probabilities(masses(n_seg, key_cdf));
        const auto h = segment_distribution::from_probabilities(
            masses(n_seg, [&](double s) { return non_key_cdf(s, sh.bump, 0.69, 0.02); }));
        for (std::size_t k = 2; k <= 10; ++k) {
            for (const double f : fprs) {
                for (const method m : {method::fast, method::fast_sharp}) {
                    const layout l = optimize_for_fpr(g, h, 1, k, f, m);
                    std::printf("%s\n{\"shape\":\"%s\",\"k\":%zu,\"fpr_target\":%g,\"method\":\"%s\",\"boundary\":",
                                first ? "" : ",", sh.name, k, f, method_name(m));
                    print_ints(l.boundary);
                    std::printf(",\"fpr\":");
                    print_doubles(l.fpr);
                    std::printf(",\"bits_per_key\":%.17g,\"expected_fpr\":%.17g}", l.backup_bits, l.expected_fpr);
                    first = false;
                }
            }
        }
    }
    std::printf("]}");
}

// --- plbf-staircase ------------------------------------------------------------

struct trace {
    std::vector<std::uint32_t> argmax;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> evaluated;  // (row, col), in order
};

// One layer's matrix, A[p][i] = DP[i-1][q-1] + divergence(i..p), exactly as
// detail::layered_dp builds it, and each solver's walk over it.
struct layer {
    std::vector<std::vector<double>> a;  // 1-based rows and columns; -inf where infeasible
    trace full, monotone, sharp;
};

layer solve_layer(const segment_distribution& g, const segment_distribution& h, std::size_t q) {
    const std::size_t n = g.segments();
    const detail::dp_table dp = detail::build_dp(g, h, q - 1, method::fast);
    auto value = [&](std::size_t p, std::size_t i) {
        if (i >= p + 1) return -std::numeric_limits<double>::infinity();
        return detail::transition(g, h, dp.v(i - 1, q - 1), i, p);
    };
    layer out;
    out.a.assign(n + 1, std::vector<double>(n + 1, -std::numeric_limits<double>::infinity()));
    for (std::size_t p = 1; p <= n; ++p) {
        for (std::size_t i = 1; i <= n; ++i) out.a[p][i] = value(p, i);
    }
    auto recorder = [&](trace& t) {
        return [&value, &t](std::size_t p, std::size_t i) {
            t.evaluated.emplace_back(static_cast<std::uint32_t>(p), static_cast<std::uint32_t>(i));
            return value(p, i);
        };
    };
    full_scan_maxima(recorder(out.full), n, n, out.full.argmax);
    monotone_maxima(recorder(out.monotone), n, n, out.monotone.argmax);
    smawk(recorder(out.sharp), n, n, out.sharp.argmax);
    return out;
}

void print_trace(const char* name, const trace& t) {
    std::printf("\"%s\":{\"argmax\":", name);
    print_ints(t.argmax, 1);
    std::printf(",\"evaluations\":%zu,\"order\":[", t.evaluated.size());
    for (std::size_t i = 0; i < t.evaluated.size(); ++i) {
        std::printf("%s[%u,%u]", i ? "," : "", t.evaluated[i].first, t.evaluated[i].second);
    }
    std::printf("]}");
}

void staircase_section() {
    constexpr std::size_t n_seg = 16;
    constexpr std::size_t q = 2;  // layer 1 is trivial: only i = 1 is feasible
    // The bumpy variant needs a band that breaks total monotonicity badly enough
    // for the approximate solvers to miss a maximum at this size; search for the
    // mildest one that does, so the figure shows a real failure, not a staged one.
    struct candidate {
        double bump, lo, width;
    };
    std::vector<candidate> tried;
    candidate chosen{0, 0, 0};
    layer bumpy;
    bool found = false;
    for (double bump = 0.2; bump <= 0.95 && !found; bump += 0.05) {
        for (double lo = 0.30; lo <= 0.80 && !found; lo += 0.0625) {
            const candidate c{bump, lo, 0.0625};
            const auto g = segment_distribution::from_probabilities(masses(n_seg, key_cdf));
            const auto h = segment_distribution::from_probabilities(
                masses(n_seg, [&](double s) { return non_key_cdf(s, c.bump, c.lo, c.width); }));
            layer l = solve_layer(g, h, q);
            if (l.monotone.argmax != l.full.argmax || l.sharp.argmax != l.full.argmax) {
                chosen = c;
                bumpy = std::move(l);
                found = true;
            }
        }
    }
    const auto g = segment_distribution::from_probabilities(masses(n_seg, key_cdf));
    const auto h_ideal = segment_distribution::from_probabilities(
        masses(n_seg, [&](double s) { return non_key_cdf(s, 0.0, 0.69, 0.02); }));
    const layer ideal = solve_layer(g, h_ideal, q);

    std::printf("\"staircase\":{\"segments\":%zu,\"layer\":%zu,", n_seg, q);
    const std::pair<const char*, const layer*> layers[] = {{"ideal", &ideal}, {"bumpy", found ? &bumpy : nullptr}};
    for (std::size_t li = 0; li < 2; ++li) {
        std::printf("%s\"%s\":", li ? "," : "", layers[li].first);
        if (layers[li].second == nullptr) {
            std::printf("null");
            continue;
        }
        const layer& l = *layers[li].second;
        std::printf("{");
        if (li == 1) std::printf("\"band\":{\"bump\":%.17g,\"lo\":%.17g,\"width\":%.17g},", chosen.bump, chosen.lo, chosen.width);
        std::printf("\"a\":[");
        for (std::size_t p = 1; p <= n_seg; ++p) {
            std::printf("%s", p > 1 ? "," : "");
            print_doubles(std::vector<double>(l.a[p].begin() + 1, l.a[p].end()));
        }
        std::printf("],");
        print_trace("full_scan", l.full);
        std::printf(",");
        print_trace("monotone", l.monotone);
        std::printf(",");
        print_trace("smawk", l.sharp);
        std::printf("}");
    }
    std::printf("}");
}

// How many matrix cells each solver evaluates for one layer as N grows, on the
// ideal shape. At N = 16 divide and conquer beats SMAWK; the asymptotics only
// show further out, and this is where.
void scaling_section() {
    std::printf("\"scaling\":[");
    bool first = true;
    for (const std::size_t n : {16u, 32u, 64u, 128u, 256u, 512u, 1000u, 2048u, 4096u}) {
        const auto g = segment_distribution::from_probabilities(masses(n, key_cdf));
        const auto h = segment_distribution::from_probabilities(
            masses(n, [&](double s) { return non_key_cdf(s, 0.0, 0.69, 0.02); }));
        const detail::dp_table dp = detail::build_dp(g, h, 1, method::fast);
        std::size_t counts[3] = {0, 0, 0};
        auto counted = [&](std::size_t& c) {
            return [&](std::size_t p, std::size_t i) {
                ++c;
                if (i >= p + 1) return -std::numeric_limits<double>::infinity();
                return detail::transition(g, h, dp.v(i - 1, 1), i, p);
            };
        };
        std::vector<std::uint32_t> a0, a1, a2;
        full_scan_maxima(counted(counts[0]), n, n, a0);
        monotone_maxima(counted(counts[1]), n, n, a1);
        smawk(counted(counts[2]), n, n, a2);
        std::printf("%s{\"segments\":%zu,\"full_scan\":%zu,\"monotone\":%zu,\"smawk\":%zu,\"agree\":%s}",
                    first ? "" : ",", n, counts[0], counts[1], counts[2], a0 == a1 && a0 == a2 ? "true" : "false");
        first = false;
    }
    std::printf("]");
}

}  // namespace

int main() {
    std::printf("{\"generated_by\":\"tools/figure_layouts.cpp\",");
    regions_section();
    std::printf(",\n");
    staircase_section();
    std::printf(",\n");
    scaling_section();
    std::printf("}\n");
    return 0;
}
