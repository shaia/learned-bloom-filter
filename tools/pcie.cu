// What a query that starts on the CPU costs a GPU filter, in three query
// formats: the 8-byte hash with an 8-byte double score (16 B), with a 4-byte
// float score (12 B), or with a 2-byte segment number (10 B); 1 byte of answer
// comes back. The filter is a plain split-block filter with the Parquet salts,
// as in Part 9 of the blog series, so the kernel reads only the hash: the score
// or segment crosses the link as it would for the learned filter, which is the
// cost this measures. bench/bench_cuda.cu times the library's real kernels for
// the same formats. The formats run interleaved in one process, so they share
// the link's state, and every answer is checked against the CPU's.
//
// Version e2a8ec4 of this file is the Part 10 reducer, which sent one
// 16-byte struct per query; bench/results/2026-10-30/pcie_ten_runs.txt is it.
//   nvcc -O3 -std=c++17 pcie.cu -o pcie && ./pcie
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#define CHECK(x)                                                                  \
    do {                                                                          \
        if (cudaError_t err = (x); err != cudaSuccess) {                          \
            std::printf("%s failed: %s\n", #x, cudaGetErrorString(err));          \
            return 1;                                                             \
        }                                                                         \
    } while (0)

__host__ __device__ inline std::uint64_t mix(std::uint64_t x) {  // splitmix64's finaliser
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__host__ __device__ inline bool probe(const std::uint32_t* blocks, std::uint32_t count, std::uint64_t h) {
    const std::uint32_t salt[8] = {0x47b6137bU, 0x44974d91U, 0x8824ad5bU, 0xa2b7289dU,
                                   0x705495c7U, 0x2df1424bU, 0x9efc4947U, 0x5c6bfb31U};
    const std::uint32_t* b = blocks + 8 * (((h >> 32) * count) >> 32);
    const auto x = static_cast<std::uint32_t>(h);
    std::uint32_t missing = 0;
    for (int i = 0; i < 8; ++i) missing |= (1U << ((x * salt[i]) >> 27)) & ~b[i];
    return missing == 0;
}

__global__ void answer(const std::uint32_t* blocks, std::uint32_t count, const std::uint64_t* hashes,
                       std::uint8_t* out, std::size_t n) {
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n;
         i += std::size_t(blockDim.x) * gridDim.x) {
        out[i] = probe(blocks, count, hashes[i]);
    }
}

int main() {
    // 4 million keys in 131,072 blocks: a 4 MB filter, about 8.4 bits a key.
    const std::uint32_t count = 131072;
    const std::size_t keys = 4000000, n = 32 << 20;  // 32 Mi queries
    std::vector<std::uint32_t> blocks(8 * std::size_t(count), 0);
    const std::uint32_t salt[8] = {0x47b6137bU, 0x44974d91U, 0x8824ad5bU, 0xa2b7289dU,
                                   0x705495c7U, 0x2df1424bU, 0x9efc4947U, 0x5c6bfb31U};
    for (std::size_t k = 0; k < keys; ++k) {
        const std::uint64_t h = mix(k);
        std::uint32_t* b = blocks.data() + 8 * (((h >> 32) * count) >> 32);
        for (int i = 0; i < 8; ++i) b[i] |= 1U << ((std::uint32_t(h) * salt[i]) >> 27);
    }

    // Pinned host memory, one array per field: a struct would pad 12 bytes to 16.
    std::uint64_t* hh;
    double* hd;
    float* hf;
    std::uint16_t* hs;
    std::uint8_t* ha;
    CHECK(cudaMallocHost(&hh, n * 8));
    CHECK(cudaMallocHost(&hd, n * 8));
    CHECK(cudaMallocHost(&hf, n * 4));
    CHECK(cudaMallocHost(&hs, n * 2));
    CHECK(cudaMallocHost(&ha, n));
    for (std::size_t i = 0; i < n; ++i) {
        hh[i] = i % 2 ? mix(i / 2 % keys) : mix((1ULL << 40) + i);  // half keys
        hd[i] = double(i % 1000) / 1000;
        hf[i] = float(hd[i]);
        hs[i] = std::uint16_t(i % 1000 + 1);
    }

    std::uint32_t* db;
    std::uint64_t* dh;
    void* dp;  // the payload: scores or segments, the largest is 8 bytes
    std::uint8_t* da;
    CHECK(cudaMalloc(&db, blocks.size() * 4));
    CHECK(cudaMalloc(&dh, n * 8));
    CHECK(cudaMalloc(&dp, n * 8));
    CHECK(cudaMalloc(&da, n));
    CHECK(cudaMemcpy(db, blocks.data(), blocks.size() * 4, cudaMemcpyHostToDevice));

    struct format {
        const char* name;
        const void* payload;
        std::size_t bytes;  // payload bytes per query
        std::vector<float> in, kernel, out;
    };
    std::vector<format> formats = {{"hash + double score", hd, 8, {}, {}, {}},
                                   {"hash + float score", hf, 4, {}, {}, {}},
                                   {"hash + 16-bit segment", hs, 2, {}, {}, {}}};
    cudaEvent_t t0, t1, t2, t3;
    for (cudaEvent_t* ev : {&t0, &t1, &t2, &t3}) CHECK(cudaEventCreate(ev));
    std::size_t wrong = 0, present = 0;
    for (int rep = 0; rep < 7; ++rep) {
        for (format& f : formats) {
            CHECK(cudaEventRecord(t0));
            CHECK(cudaMemcpy(dh, hh, n * 8, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(dp, f.payload, n * f.bytes, cudaMemcpyHostToDevice));
            CHECK(cudaEventRecord(t1));
            answer<<<1024, 256>>>(db, count, dh, da, n);
            CHECK(cudaEventRecord(t2));
            CHECK(cudaMemcpy(ha, da, n, cudaMemcpyDeviceToHost));
            CHECK(cudaEventRecord(t3));
            CHECK(cudaEventSynchronize(t3));
            float a, b, c;
            CHECK(cudaEventElapsedTime(&a, t0, t1));
            CHECK(cudaEventElapsedTime(&b, t1, t2));
            CHECK(cudaEventElapsedTime(&c, t2, t3));
            f.in.push_back(a), f.kernel.push_back(b), f.out.push_back(c);
            if (rep == 0) {
                for (std::size_t i = 0; i < n; ++i) {
                    wrong += ha[i] != std::uint8_t(probe(blocks.data(), count, hh[i]));
                    present += ha[i];
                }
            }
        }
    }
    auto median_ns = [&](std::vector<float> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2] * 1e6 / double(n);  // ms per batch -> ns per query
    };
    std::printf("%-22s %5s %22s %10s %10s %12s\n", "query format", "bytes", "copy in", "kernel", "copy out",
                "from the host");
    for (const format& f : formats) {
        const double in = median_ns(f.in), k = median_ns(f.kernel), out = median_ns(f.out);
        std::printf("%-22s %5zu %8.3f ns %6.1f GB/s %7.3f ns %7.3f ns %9.3f ns\n", f.name, 8 + f.bytes, in,
                    double(8 + f.bytes) / in, k, out, in + k + out);
    }
    std::printf("answers equal the CPU's: %s (%zu wrong; %.1f%% present)\n", wrong ? "no" : "yes", wrong,
                100.0 * double(present) / double(3 * n));
}
