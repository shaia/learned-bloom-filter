// What a query that starts on the CPU costs a GPU filter: 16 bytes in, the
// probe, 1 byte out. The filter is a plain split-block filter with the Parquet
// salts, as in Part 9; the answers are checked against the CPU's.
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

struct query {
    std::uint64_t hash;
    double score;  // unused by a plain filter, but it crosses the bus all the same
};

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

__global__ void answer(const std::uint32_t* blocks, std::uint32_t count, const query* q, std::uint8_t* out,
                       std::size_t n) {
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n;
         i += std::size_t(blockDim.x) * gridDim.x) {
        out[i] = probe(blocks, count, q[i].hash);
    }
}

int main() {
    // 4 million keys in 131,072 blocks: a 4 MB filter, about 8.4 bits a key.
    const std::uint32_t count = 131072;
    const std::size_t keys = 4000000, n = 32 << 20;  // 32 Mi queries: 512 MB in, 32 MB out
    std::vector<std::uint32_t> blocks(8 * std::size_t(count), 0);
    const std::uint32_t salt[8] = {0x47b6137bU, 0x44974d91U, 0x8824ad5bU, 0xa2b7289dU,
                                   0x705495c7U, 0x2df1424bU, 0x9efc4947U, 0x5c6bfb31U};
    for (std::size_t k = 0; k < keys; ++k) {
        const std::uint64_t h = mix(k);
        std::uint32_t* b = blocks.data() + 8 * (((h >> 32) * count) >> 32);
        for (int i = 0; i < 8; ++i) b[i] |= 1U << ((std::uint32_t(h) * salt[i]) >> 27);
    }

    query* hq;  // pinned host memory: what the link really runs at
    std::uint8_t* ha;
    CHECK(cudaMallocHost(&hq, n * sizeof(query)));
    CHECK(cudaMallocHost(&ha, n));
    for (std::size_t i = 0; i < n; ++i) hq[i] = {i % 2 ? mix(i / 2 % keys) : mix((1ULL << 40) + i), 0.5};  // half keys

    std::uint32_t* db;
    query* dq;
    std::uint8_t* da;
    CHECK(cudaMalloc(&db, blocks.size() * 4));
    CHECK(cudaMalloc(&dq, n * sizeof(query)));
    CHECK(cudaMalloc(&da, n));
    CHECK(cudaMemcpy(db, blocks.data(), blocks.size() * 4, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1, t2, t3;
    for (cudaEvent_t* ev : {&t0, &t1, &t2, &t3}) CHECK(cudaEventCreate(ev));
    std::vector<float> in, kernel, out;
    for (int rep = 0; rep < 7; ++rep) {
        CHECK(cudaEventRecord(t0));
        CHECK(cudaMemcpy(dq, hq, n * sizeof(query), cudaMemcpyHostToDevice));
        CHECK(cudaEventRecord(t1));
        answer<<<1024, 256>>>(db, count, dq, da, n);
        CHECK(cudaEventRecord(t2));
        CHECK(cudaMemcpy(ha, da, n, cudaMemcpyDeviceToHost));
        CHECK(cudaEventRecord(t3));
        CHECK(cudaEventSynchronize(t3));
        float a, b, c;
        CHECK(cudaEventElapsedTime(&a, t0, t1));
        CHECK(cudaEventElapsedTime(&b, t1, t2));
        CHECK(cudaEventElapsedTime(&c, t2, t3));
        in.push_back(a), kernel.push_back(b), out.push_back(c);
    }
    auto median_ns = [&](std::vector<float> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2] * 1e6 / double(n);  // ms per batch -> ns per query
    };
    const double ns_in = median_ns(in), ns_kernel = median_ns(kernel), ns_out = median_ns(out);

    std::size_t wrong = 0, present = 0;
    for (std::size_t i = 0; i < n; ++i) {
        wrong += ha[i] != std::uint8_t(probe(blocks.data(), count, hq[i].hash));
        present += ha[i];
    }
    std::printf("copy in:  %.3f ns per 16-byte query (%.1f GB/s)\n", ns_in, 16 / ns_in);
    std::printf("kernel:   %.3f ns per query\n", ns_kernel);
    std::printf("copy out: %.3f ns per 1-byte answer (%.1f GB/s)\n", ns_out, 1 / ns_out);
    std::printf("a query that starts on the host: %.3f ns, %.0f%% of it the kernel\n", ns_in + ns_kernel + ns_out,
                100 * ns_kernel / (ns_in + ns_kernel + ns_out));
    std::printf("answers equal the CPU's: %s (%zu wrong; %.1f%% present)\n", wrong ? "no" : "yes", wrong,
                100.0 * double(present) / double(n));
}
