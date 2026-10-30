// Prints, once a second for `seconds`, a single-thread compute rate and a DRAM
// pointer-chase latency, to see when and how the machine slows down under load.
//   g++ -std=c++20 -O2 throttle_probe.cpp -o throttle_probe && ./throttle_probe 90
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const int seconds = argc > 1 ? std::atoi(argv[1]) : 90;
    // 64 MB of 8-byte indices forming one random cycle: every load misses cache.
    const std::size_t n = std::size_t{1} << 23;
    std::vector<std::uint64_t> next(n);
    std::vector<std::uint64_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin() + 1, perm.end(), std::mt19937_64(1));
    for (std::size_t i = 0; i < n; ++i) next[perm[i]] = perm[(i + 1) % n];

    using clk = std::chrono::steady_clock;
    std::uint64_t x = 1, p = 0;
    const auto t0 = clk::now();
    for (int s = 1; s <= seconds; ++s) {
        // Half a second of arithmetic: a dependent multiply-xorshift chain.
        std::uint64_t ops = 0;
        auto start = clk::now();
        while (clk::now() - start < std::chrono::milliseconds(500)) {
            for (int i = 0; i < 4096; ++i) x = (x * 0x9e3779b97f4a7c15ULL) ^ (x >> 29);
            ops += 4096;
        }
        const double mops = ops / std::chrono::duration<double>(clk::now() - start).count() / 1e6;
        // Then a fixed number of dependent DRAM loads.
        start = clk::now();
        for (int i = 0; i < 200000; ++i) p = next[p];
        const double ns = std::chrono::duration<double, std::nano>(clk::now() - start).count() / 200000;
        std::printf("t=%5.1fs  compute %7.1f Mops/s  dram %6.1f ns/load\n",
                    std::chrono::duration<double>(clk::now() - t0).count(), mops, ns);
        std::fflush(stdout);
    }
    return static_cast<int>((x ^ p) & 1);
}
