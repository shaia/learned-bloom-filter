#pragma once
//
// Minimal assertion harness. Deliberately no test framework: the library has
// no dependencies and neither do its tests, so a build is hermetic.
//
// The exit code is derived from the failure count, so a violated invariant
// fails ctest rather than merely the eye of whoever reads the output. Passing
// checks are counted, not printed: the golden suite runs thousands of them, and
// a wall of "ok" lines hides the one that matters.

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

namespace plbf::test {

inline int g_failures = 0;
inline int g_checks = 0;

inline void check(bool cond, const std::string& what) {
    ++g_checks;
    if (cond) return;
    ++g_failures;
    std::printf("  FAIL  %s\n", what.c_str());
}

template <class A, class B>
void check_eq(A a, B b, const std::string& what) {
    const auto ua = static_cast<std::uint64_t>(a);
    const auto ub = static_cast<std::uint64_t>(b);
    ++g_checks;
    if (ua == ub) return;
    ++g_failures;
    std::printf("  FAIL  %s  (%llu != %llu)\n", what.c_str(), static_cast<unsigned long long>(ua),
                static_cast<unsigned long long>(ub));
}

// Distance in units in the last place; 0 means bit-identical. Both values must
// be finite and of the same sign for the answer to mean anything, which is
// all the callers here need.
inline std::uint64_t ulps(double a, double b) {
    if (a == b) return 0;
    const auto ia = std::bit_cast<std::int64_t>(a);
    const auto ib = std::bit_cast<std::int64_t>(b);
    if ((ia < 0) != (ib < 0)) return UINT64_MAX;
    return ia > ib ? static_cast<std::uint64_t>(ia - ib) : static_cast<std::uint64_t>(ib - ia);
}

inline void check_ulps(double a, double b, std::uint64_t max_ulps, const std::string& what) {
    ++g_checks;
    const std::uint64_t d = ulps(a, b);
    if (d <= max_ulps) return;
    ++g_failures;
    std::printf("  FAIL  %s  (%.17g vs %.17g, %llu ulps)\n", what.c_str(), a, b,
                static_cast<unsigned long long>(d));
}

inline void section(const char* name) { std::printf("%s\n", name); }

inline int summary(const char* suite) {
    std::printf("\n%s: %d checks, %d failed\n", suite, g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

}  // namespace plbf::test
