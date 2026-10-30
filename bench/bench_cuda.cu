// The GPU against the CPU: building the filter, and querying it.
//
// Every GPU figure comes with the CPU figure it should be judged against --
// one thread, and every thread the machine has -- and with what moving the
// data over PCIe costs, because a GPU only wins on queries that are already
// on it (scored there by a GPU model, say). Every GPU result is also checked
// against the CPU's, so a fast wrong answer cannot pass for a result.
//
// Usage: bench_cuda [query_keys...]   (filter sizes for the query section)

#include "workload.hpp"

#include <plbf/cuda/filter.cuh>
#include <plbf/filter.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

using namespace plbf;
using namespace plbf::bench;
using plbf::cuda::check;
using plbf::cuda::device_buffer;

namespace {

// Pinned host memory: what PCIe transfers actually run at. Pageable memory
// goes through a staging copy and would understate the link.
template <class T>
struct pinned {
    T* p = nullptr;
    std::size_t n = 0;
    explicit pinned(std::size_t count) : n(count) {
        check(cudaMallocHost(reinterpret_cast<void**>(&p), n * sizeof(T)), "cudaMallocHost");
    }
    ~pinned() { cudaFreeHost(p); }
    pinned(const pinned&) = delete;
    pinned& operator=(const pinned&) = delete;
    std::span<T> span() { return {p, n}; }
};

struct gpu_timer {
    cudaEvent_t a{}, b{};
    gpu_timer() {
        check(cudaEventCreate(&a), "cudaEventCreate");
        check(cudaEventCreate(&b), "cudaEventCreate");
    }
    ~gpu_timer() {
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
    template <class F>
    double ms(F&& fn) {
        check(cudaEventRecord(a), "cudaEventRecord");
        fn();
        check(cudaEventRecord(b), "cudaEventRecord");
        check(cudaEventSynchronize(b), "cudaEventSynchronize");
        float t = 0;
        check(cudaEventElapsedTime(&t, a, b), "cudaEventElapsedTime");
        return t;
    }
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

bool same_filter(const filter& a, const filter& b) {
    return a.regions().boundary == b.regions().boundary && a.regions().fpr == b.regions().fpr &&
           a.tables().slices == b.tables().slices && a.blocks().size() == b.blocks().size() &&
           std::memcmp(a.blocks().data(), b.blocks().data(), a.blocks().size_bytes()) == 0;
}

// --- building ---------------------------------------------------------------

void bench_build(std::size_t n_keys, std::size_t n_non_keys, const char* label) {
    std::printf("\nbuild: %s -- %zu keys, %zu non-keys, N = 1000, k = 5, target FPR 0.01\n", label, n_keys,
                n_non_keys);
    const workload w = make_workload(n_keys, n_non_keys, shape::ideal, 7);
    const options opt;
    const int reps = n_keys > 10000000 ? 3 : 7;

    filter cpu = filter::build_for_fpr(w.keys, w.key_scores, w.non_key_scores, 0.01, opt);
    const double t_cpu = median_ms(reps, [&] {
        cpu = filter::build_for_fpr(w.keys, w.key_scores, w.non_key_scores, 0.01, opt);
    });

    device_buffer<std::uint64_t> dk(n_keys);
    device_buffer<double> dks(n_keys), dns(n_non_keys);
    const double t_h2d = median_ms(3, [&] {
        dk.upload(w.keys);
        dks.upload(w.key_scores);
        dns.upload(w.non_key_scores);
    });

    // The parts of a GPU build, timed one at a time, then the whole thing.
    const segment_grid grid(opt.segments);
    std::vector<std::uint64_t> kc, nc;
    const double t_hist = median_ms(reps, [&] {
        kc = plbf::cuda::segment_counts(grid, dks.data(), n_keys);
        nc = plbf::cuda::segment_counts(grid, dns.data(), n_non_keys);
    });
    const auto g = segment_distribution::from_counts(kc);
    const auto h = segment_distribution::from_counts(nc);
    const double t_search = median_ms(reps, [&] { keep(optimize_for_fpr(g, h, n_keys, opt.regions, 0.01, opt.search)); });

    plbf::cuda::device_filter gpu =
        plbf::cuda::device_filter::build_for_fpr(dk.data(), dks.data(), n_keys, dns.data(), n_non_keys, 0.01, opt);
    const double t_gpu = median_ms(reps, [&] {
        gpu = plbf::cuda::device_filter::build_for_fpr(dk.data(), dks.data(), n_keys, dns.data(), n_non_keys, 0.01,
                                                       opt);
    });
    const bool identical = same_filter(gpu.to_host(), cpu);

    std::printf("  %-40s %10.2f ms\n", "CPU build (one thread)", t_cpu);
    std::printf("  %-40s %10.2f ms   (%.1fx the CPU)\n", "GPU build, data already on the GPU", t_gpu, t_cpu / t_gpu);
    std::printf("  %-40s %10.2f ms\n", "  of which: two GPU histograms", t_hist);
    std::printf("  %-40s %10.2f ms\n", "  of which: threshold search (CPU)", t_search);
    std::printf("  %-40s %10.2f ms\n", "  the rest: tables, allocation, insert", t_gpu - t_hist - t_search);
    std::printf("  %-40s %10.2f ms   (pageable memory)\n", "copying keys and scores to the GPU", t_h2d);
    std::printf("  %-40s %10s\n", "GPU filter bit-identical to CPU filter", identical ? "yes" : "NO");
    if (!identical) std::exit(1);
}

// --- querying -----------------------------------------------------------------

void bench_query(std::size_t n_keys, std::size_t n_queries) {
    const std::size_t n_train = std::min<std::size_t>(n_keys * 2, 4000000);
    const workload w = make_workload(n_keys, n_train, shape::ideal, 1);
    const workload unseen = make_workload(0, n_queries / 2 + 1, shape::ideal, 2, 1ULL << 40);

    // Build on the GPU, then bring a copy back for the CPU side of the comparison.
    plbf::cuda::device_filter gpu = [&] {
        const auto dk = device_buffer<std::uint64_t>::from_host(w.keys);
        const auto dks = device_buffer<double>::from_host(w.key_scores);
        const auto dns = device_buffer<double>::from_host(w.non_key_scores);
        return plbf::cuda::device_filter::build_for_fpr(dk.data(), dks.data(), n_keys, dns.data(), n_train, 0.01);
    }();
    const filter cpu = gpu.to_host();

    pinned<std::uint64_t> qh(n_queries);
    pinned<double> qs(n_queries);
    pinned<std::uint8_t> out(n_queries);
    std::mt19937_64 rng(99);
    for (std::size_t i = 0; i < n_queries; ++i) {
        if (i % 2 == 0) {
            const std::size_t j = rng() % n_keys;
            qh.p[i] = w.keys[j];
            qs.p[i] = w.key_scores[j];
        } else {
            qh.p[i] = unseen.non_keys[i / 2];
            qs.p[i] = unseen.non_key_scores[i / 2];
        }
    }

    std::printf("\nquery: %zu keys, filter %.1f MB (%.2f bits/key), %zu queries, half keys\n", n_keys,
                static_cast<double>(cpu.memory_bytes()) / 1e6,
                8.0 * static_cast<double>(cpu.memory_bytes()) / static_cast<double>(n_keys), n_queries);

    device_buffer<std::uint64_t> dh(n_queries);
    device_buffer<double> ds(n_queries);
    device_buffer<std::uint8_t> dout(n_queries);
    gpu_timer t;
    std::vector<double> h2d, kern, d2h;
    for (int rep = 0; rep < 7; ++rep) {
        h2d.push_back(t.ms([&] {
            check(cudaMemcpyAsync(dh.data(), qh.p, n_queries * sizeof(std::uint64_t), cudaMemcpyHostToDevice), "H2D");
            check(cudaMemcpyAsync(ds.data(), qs.p, n_queries * sizeof(double), cudaMemcpyHostToDevice), "H2D");
        }));
        kern.push_back(t.ms([&] { gpu.contains(dh.data(), ds.data(), dout.data(), n_queries); }));
        d2h.push_back(t.ms([&] {
            check(cudaMemcpyAsync(out.p, dout.data(), n_queries, cudaMemcpyDeviceToHost), "D2H");
        }));
    }
    const std::vector<std::uint8_t> gpu_answers(out.p, out.p + n_queries);

    // CPU: the AVX2 batch on one thread, then on every hardware thread, each
    // taking a contiguous share. contains_batch is const and touches no
    // shared mutable state, so the threads need no coordination.
    std::vector<std::uint8_t> cpu_answers(n_queries);
    const auto run_cpu = [&](unsigned threads) {
        std::vector<std::thread> pool;
        const std::size_t per = (n_queries + threads - 1) / threads;
        for (unsigned i = 0; i < threads; ++i) {
            const std::size_t lo = std::min(n_queries, i * per);
            const std::size_t hi = std::min(n_queries, lo + per);
            pool.emplace_back([&, lo, hi] {
                cpu.contains_batch(std::span<const std::uint64_t>(qh.p + lo, hi - lo),
                                   std::span<const double>(qs.p + lo, hi - lo),
                                   std::span<std::uint8_t>(cpu_answers.data() + lo, hi - lo));
            });
        }
        for (auto& th : pool) th.join();
    };
    const unsigned hw = std::max(1U, std::thread::hardware_concurrency());
    const double t_cpu1 = median_ms(3, [&] { run_cpu(1); });
    const double t_cpun = median_ms(5, [&] { run_cpu(hw); });
    const bool agree = cpu_answers == gpu_answers;

    const auto ns = [&](double ms) { return ms * 1e6 / static_cast<double>(n_queries); };
    const double k_ms = median(kern), up = median(h2d), down = median(d2h);
    std::printf("  %-42s %8.3f ns/query  (%6.2f G queries/s)\n", "GPU kernel, queries already on the GPU", ns(k_ms),
                static_cast<double>(n_queries) / (k_ms * 1e6));
    std::printf("  %-42s %8.3f ns/query  (%.1f GB/s)\n", "  + copying queries in (16 B each, pinned)", ns(up),
                static_cast<double>(n_queries) * 16 / (up * 1e6));
    std::printf("  %-42s %8.3f ns/query\n", "  + copying answers out (1 B each)", ns(down));
    std::printf("  %-42s %8.3f ns/query\n", "GPU, queries starting on the host", ns(up + k_ms + down));
    std::printf("  %-42s %8.3f ns/query\n", "CPU AVX2 batch, 1 thread", ns(t_cpu1));
    std::printf("  %-42s %8.3f ns/query  (%u threads)\n", "CPU AVX2 batch, all threads", ns(t_cpun), hw);
    std::printf("  %-42s %8s\n", "GPU answers equal CPU answers", agree ? "yes" : "NO");
    if (!agree) std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::printf("bench_cuda: %s, %d SMs, %.0f MB L2; CPU: %u hardware threads\n", prop.name,
                prop.multiProcessorCount, prop.l2CacheSize / 1e6, std::thread::hardware_concurrency());
    std::printf("medians; CPU and GPU runs interleave, so compare within a section, not across runs\n");

    bench_build(223088, 428118, "the paper's URL scale");
    bench_build(50000000, 50000000, "the paper's DNA scale");

    std::vector<std::size_t> sizes = {4000000, 50000000, 200000000};
    if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; ++i) sizes.push_back(std::strtoull(argv[i], nullptr, 10));
    }
    for (const std::size_t keys : sizes) bench_query(keys, 100000000);
    return 0;
}
