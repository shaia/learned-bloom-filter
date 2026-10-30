#pragma once
//
// Synthetic workloads and timing for the benchmarks.
//
// Two score distributions. `ideal`: key scores with density 4 s^3, non-key
// scores with density 2 (1 - s) -- the key/non-key ratio rises monotonically,
// the paper's assumption. `bumpy`: the same, except a fifth of the non-keys sit
// in a narrow bump at 0.7, so the ratio dips there and the approximate methods
// have something to get wrong.

#include <plbf/hash.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace plbf::bench {

enum class shape { ideal, bumpy };

struct workload {
    std::vector<std::uint64_t> keys, non_keys;
    std::vector<double> key_scores, non_key_scores;
};

inline workload make_workload(std::size_t n_keys, std::size_t n_non_keys, shape s, std::uint64_t seed,
                              std::uint64_t first_id = 0) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0, 1);
    workload w;
    w.keys.reserve(n_keys);
    w.key_scores.reserve(n_keys);
    w.non_keys.reserve(n_non_keys);
    w.non_key_scores.reserve(n_non_keys);
    std::uint64_t id = first_id;
    for (std::size_t i = 0; i < n_keys; ++i) {
        w.keys.push_back(hash_u64(id++));
        w.key_scores.push_back(std::pow(u(rng), 0.25));
    }
    for (std::size_t i = 0; i < n_non_keys; ++i) {
        w.non_keys.push_back(hash_u64(id++));
        const bool bump = s == shape::bumpy && u(rng) < 0.2;
        w.non_key_scores.push_back(bump ? 0.69 + 0.02 * u(rng) : 1 - std::sqrt(u(rng)));
    }
    return w;
}

// Median wall time in milliseconds of `reps` runs of fn.
template <class F>
double median_ms(int reps, F&& fn) {
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
        const auto a = std::chrono::steady_clock::now();
        fn();
        const auto b = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// Keeps a result alive without the optimiser seeing through it.
template <class T>
void keep(const T& v) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r,m"(v) : "memory");
#else
    static_cast<void>(*static_cast<const volatile T*>(&v));
#endif
}

}  // namespace plbf::bench
