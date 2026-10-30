// The optimiser against the Python reference, and the four methods against
// each other.
//
// The golden section is the one that matters: test/golden/reference.txt holds
// what FastPLBF itself decided for 116 configurations (tools/gen_golden.py),
// and every boundary has to match exactly. FPRs and costs may differ only by
// the libm: C++ and CPython call different log implementations here, so a last
// bit can move. That is why the golden check counts bit-identical results
// separately from the ulp tolerance -- a drift in the former is worth noticing
// even while the latter passes.
//
// The property sections then hold the methods to the paper's claims on inputs
// the golden file does not cover.

#include "harness.hpp"

#include <plbf/optimize.hpp>
#include <plbf/row_maxima.hpp>
#include <plbf/segments.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef PLBF_TEST_DIR
#error "PLBF_TEST_DIR must point at the test directory"
#endif

using namespace plbf;
using namespace plbf::test;

namespace {

// --- golden: the Python reference ------------------------------------------

struct dataset {
    std::vector<double> key_scores;
    std::vector<double> non_key_scores;
};

double parse_double(const std::string& s) {
    double v = 0;
    const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || ptr != s.data() + s.size()) {
        std::printf("bad number '%s'\n", s.c_str());
        std::exit(2);
    }
    return v;
}

// Tolerate CRLF files: .gitattributes keeps a clone LF, but a copy that went
// through an editor or a zip download may not have.
void chomp(std::string& line) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
}

dataset load_dataset(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::printf("cannot open %s\n", path.c_str());
        std::exit(2);
    }
    dataset d;
    std::string line;
    std::getline(in, line);  // header: key,label,score
    while (std::getline(in, line)) {
        chomp(line);
        const auto c1 = line.find(',');
        const auto c2 = line.find(',', c1 + 1);
        const bool is_key = line.substr(c1 + 1, c2 - c1 - 1) == "1";
        (is_key ? d.key_scores : d.non_key_scores).push_back(parse_double(line.substr(c2 + 1)));
    }
    return d;
}

struct golden_case {
    std::map<std::string, std::string> field;
    const std::string& operator[](const std::string& k) const { return field.at(k); }
};

std::vector<golden_case> load_golden(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::printf("cannot open %s\n", path.c_str());
        std::exit(2);
    }
    std::vector<golden_case> cases;
    std::string line;
    while (std::getline(in, line)) {
        chomp(line);
        if (line.empty() || line[0] == '#') continue;
        if (line == "case") {
            cases.emplace_back();
        } else if (line != "end") {
            const auto sp = line.find(' ');
            cases.back().field[line.substr(0, sp)] = line.substr(sp + 1);
        }
    }
    return cases;
}

template <class T>
std::vector<T> split(const std::string& s) {
    std::vector<T> out;
    std::istringstream in(s);
    std::string tok;
    while (in >> tok) {
        if constexpr (std::is_same_v<T, double>) {
            out.push_back(parse_double(tok));
        } else {
            out.push_back(static_cast<T>(std::stoull(tok)));
        }
    }
    return out;
}

method parse_method(const std::string& s) {
    if (s == "plbf") return method::plbf;
    if (s == "fast") return method::fast;
    if (s == "fast_pp") return method::fast_pp;
    return method::fast_sharp;
}

void test_golden() {
    section("golden: decisions of the Python reference");
    const std::string dir = PLBF_TEST_DIR;
    const auto cases = load_golden(dir + "/golden/reference.txt");
    check(cases.size() == 116, "golden file has all 116 cases");

    std::map<std::string, dataset> data;
    std::size_t values = 0, identical = 0;
    std::uint64_t worst = 0;
    for (const auto& c : cases) {
        const std::string name = c["data"] + " " + c["objective"] + " " + c["method"] + " target=" +
                                 c["target"] + " N=" + c["segments"] + " k=" + c["regions"];
        if (!data.count(c["data"])) data[c["data"]] = load_dataset(dir + "/data/" + c["data"]);
        const dataset& d = data[c["data"]];

        const std::size_t n_seg = std::stoul(c["segments"]);
        const std::size_t k = std::stoul(c["regions"]);
        const segment_grid grid(n_seg);
        const segment_distribution g(grid, d.key_scores);
        const segment_distribution h(grid, d.non_key_scores);
        check_eq(d.key_scores.size(), std::stoul(c["keys"]), name + ": key count");

        const double target = parse_double(c["target"]);
        const method m = parse_method(c["method"]);
        const layout got = c["objective"] == "fpr"
                               ? optimize_for_fpr(g, h, d.key_scores.size(), k, target, m)
                               : optimize_for_memory(g, h, d.key_scores.size(), k, target, m);

        const auto boundary = split<std::uint32_t>(c["boundary"]);
        check(got.boundary == boundary, name + ": boundaries match exactly");

        const auto fpr = split<double>(c["fpr"]);
        check_eq(got.fpr.size(), fpr.size(), name + ": region count");
        std::vector<std::pair<double, double>> pairs;
        for (std::size_t r = 0; r < fpr.size() && r < got.fpr.size(); ++r) pairs.emplace_back(got.fpr[r], fpr[r]);
        pairs.emplace_back(got.backup_bits, parse_double(c["backup_bits"]));
        pairs.emplace_back(got.expected_fpr, parse_double(c["expected_fpr"]));
        for (const auto& [a, b] : pairs) {
            const std::uint64_t u = ulps(a, b);
            ++values;
            if (u == 0) ++identical;
            if (u > worst) worst = u;
            check_ulps(a, b, 64, name + ": value within 64 ulps");
        }
    }
    std::printf("  %zu cases, %zu values: %zu bit-identical to the reference, worst %llu ulps\n",
                cases.size(), values, identical, static_cast<unsigned long long>(worst));
}

// --- the row-maxima solvers ------------------------------------------------

// -(x_r - y_c)^2 with both coordinates sorted is totally monotone for maxima:
// expanding it, the cross term 2 x_r y_c is supermodular, which is exactly the
// Monge condition. Rounding to integers makes ties common, which exercises the
// leftmost-maximum rule.
std::vector<double> sorted_coords(std::mt19937_64& rng, std::size_t n, bool integral) {
    std::uniform_real_distribution<double> u(0.0, 100.0);
    std::vector<double> v(n);
    for (auto& x : v) x = integral ? std::floor(u(rng)) : u(rng);
    std::sort(v.begin(), v.end());
    return v;
}

void test_row_maxima() {
    section("row maxima: the three solvers agree on totally monotone matrices");
    std::mt19937_64 rng(7);
    int disagreements = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const std::size_t n = 1 + rng() % 90;
        const std::size_t m = 1 + rng() % 90;
        const bool integral = trial % 2 == 0;
        const auto x = sorted_coords(rng, n, integral);
        const auto y = sorted_coords(rng, m, integral);
        auto f = [&](std::size_t r, std::size_t c) {
            const double d = x[r - 1] - y[c - 1];
            return -d * d;
        };
        std::vector<std::uint32_t> exact, mono, sm;
        full_scan_maxima(f, n, m, exact);
        monotone_maxima(f, n, m, mono);
        smawk(f, n, m, sm);
        if (exact != mono || exact != sm) ++disagreements;
    }
    check_eq(disagreements, 0, "400 random Monge matrices, ties included: identical argmax");

    // Not totally monotone: the approximate solvers may miss the maximum, but
    // must still name a real column for every row.
    int out_of_range = 0;
    for (int trial = 0; trial < 200; ++trial) {
        const std::size_t n = 1 + rng() % 60;
        const std::size_t m = 1 + rng() % 60;
        std::vector<double> a(n * m);
        for (auto& v : a) v = std::uniform_real_distribution<double>(-1, 1)(rng);
        auto f = [&](std::size_t r, std::size_t c) { return a[(r - 1) * m + (c - 1)]; };
        std::vector<std::uint32_t> mono, sm;
        monotone_maxima(f, n, m, mono);
        smawk(f, n, m, sm);
        for (std::size_t r = 1; r <= n; ++r) {
            if (mono[r] < 1 || mono[r] > m || sm[r] < 1 || sm[r] > m) ++out_of_range;
        }
    }
    check_eq(out_of_range, 0, "arbitrary matrices: every row still gets a column in range");
}

// --- the methods against each other -----------------------------------------

// Masses for N segments. With ideal = true the key/non-key ratio is
// non-decreasing across segments -- the paper's assumption, under which
// fast_pp and fast_sharp must reproduce the exact DP. The ratio is built from
// sorted increments, as the reference's own test generator does.
void random_masses(std::mt19937_64& rng, std::size_t n, bool ideal, std::vector<double>& g,
                   std::vector<double>& h) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    h.resize(n);
    g.resize(n);
    double hs = 0, gs = 0, ratio = 0;
    for (std::size_t i = 0; i < n; ++i) {
        h[i] = 0.05 + u(rng);
        hs += h[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
        h[i] /= hs;
        ratio += u(rng);
        g[i] = ideal ? h[i] * ratio : 0.05 + u(rng);
        gs += g[i];
    }
    for (auto& x : g) x /= gs;
}

bool same_layout(const layout& a, const layout& b) {
    return a.boundary == b.boundary && a.fpr == b.fpr && a.backup_bits == b.backup_bits &&
           a.expected_fpr == b.expected_fpr;
}

void test_methods_agree() {
    section("methods: plbf == fast always; fast_pp == fast_sharp == fast on ideal inputs");
    std::mt19937_64 rng(11);
    std::vector<double> gm, hm;
    int plbf_vs_fast = 0, ideal_mismatch = 0, cases = 0;
    for (int trial = 0; trial < 60; ++trial) {
        const std::size_t n = 20 + rng() % 60;
        const std::size_t k = 2 + rng() % 8;
        const bool ideal = trial % 2 == 0;
        random_masses(rng, n, ideal, gm, hm);
        const auto g = segment_distribution::from_probabilities(gm);
        const auto h = segment_distribution::from_probabilities(hm);
        const std::size_t keys = 1000 + rng() % 100000;
        for (const double target : {0.05, 0.001}) {
            ++cases;
            const layout fast = optimize_for_fpr(g, h, keys, k, target, method::fast);
            if (!same_layout(optimize_for_fpr(g, h, keys, k, target, method::plbf), fast)) ++plbf_vs_fast;
            if (ideal) {
                if (!same_layout(optimize_for_fpr(g, h, keys, k, target, method::fast_pp), fast)) ++ideal_mismatch;
                if (!same_layout(optimize_for_fpr(g, h, keys, k, target, method::fast_sharp), fast)) ++ideal_mismatch;
            }
        }
        const double budget = 4.0 * static_cast<double>(keys);
        const layout fast_m = optimize_for_memory(g, h, keys, k, budget, method::fast);
        if (!same_layout(optimize_for_memory(g, h, keys, k, budget, method::plbf), fast_m)) ++plbf_vs_fast;
        if (ideal) {
            if (!same_layout(optimize_for_memory(g, h, keys, k, budget, method::fast_pp), fast_m)) ++ideal_mismatch;
            if (!same_layout(optimize_for_memory(g, h, keys, k, budget, method::fast_sharp), fast_m)) ++ideal_mismatch;
        }
    }
    check_eq(plbf_vs_fast, 0, "plbf and fast: bit-identical layouts (" + std::to_string(cases) + " fpr cases + memory)");
    check_eq(ideal_mismatch, 0, "ideal inputs: fast_pp and fast_sharp bit-identical to fast");
}

void test_approximation_bounds() {
    section("methods: on non-ideal inputs the exact DP dominates, cell by cell");
    std::mt19937_64 rng(13);
    std::vector<double> gm, hm;
    int above_exact = 0, beat_exact = 0, differed = 0;
    for (int trial = 0; trial < 40; ++trial) {
        const std::size_t n = 30 + rng() % 70;
        const std::size_t k = 3 + rng() % 6;
        random_masses(rng, n, false, gm, hm);
        const auto g = segment_distribution::from_probabilities(gm);
        const auto h = segment_distribution::from_probabilities(hm);
        const auto exact = detail::build_dp(g, h, k - 1, method::fast);
        for (const method m : {method::fast_pp, method::fast_sharp}) {
            const auto approx = detail::build_dp(g, h, k - 1, m);
            for (std::size_t row = 1; row <= n; ++row) {
                for (std::size_t q = 1; q < k; ++q) {
                    if (approx.v(row, q) > exact.v(row, q)) ++above_exact;
                }
            }
            const layout a = optimize_for_fpr(g, h, 10000, k, 0.01, m);
            const layout e = optimize_for_fpr(g, h, 10000, k, 0.01, method::fast);
            if (a.backup_bits < e.backup_bits) ++beat_exact;
            if (!same_layout(a, e)) ++differed;
        }
    }
    check_eq(above_exact, 0, "no approximate DP cell exceeds the exact one");
    // Not a theorem -- the j loop and the FPR clamping sit between the DP and
    // the final cost -- but it is what the reference's own test asserts.
    check_eq(beat_exact, 0, "no approximate layout needs less memory than fast's");
    std::printf("  approximate layouts differing from fast on non-ideal inputs: %d of 80\n", differed);
}

void test_validation() {
    section("arguments: rejected rather than misread");
    const std::vector<double> u4(4, 0.25);
    const auto d = segment_distribution::from_probabilities(u4);
    auto throws = [&](auto fn) {
        try {
            fn();
        } catch (const std::exception&) {
            return true;
        }
        return false;
    };
    check(throws([&] { (void)optimize_for_fpr(d, d, 10, 1, 0.01); }), "k = 1 is rejected");
    check(throws([&] { (void)optimize_for_fpr(d, d, 10, 5, 0.01); }), "k > N is rejected");
    check(throws([&] { (void)optimize_for_fpr(d, d, 10, 2, 1.0); }), "target FPR 1 is rejected");
    check(throws([&] { (void)optimize_for_memory(d, d, 10, 2, 0.0); }), "zero memory is rejected");
    check(throws([&] { (void)segment_distribution::from_probabilities(std::vector<double>{0.5, 0.4}); }),
          "masses not summing to 1 are rejected");
}

}  // namespace

int main() {
    test_golden();
    test_row_maxima();
    test_methods_agree();
    test_approximation_bounds();
    test_validation();
    return summary("test_optimize");
}
