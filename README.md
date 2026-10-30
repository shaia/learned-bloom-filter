# plbf

A header-only C++20 partitioned learned Bloom filter: the construction methods
of Sato and Matsui, *Fast Construction of Partitioned Learned Bloom Filter with
Theoretical Guarantees* (The VLDB Journal, 2026), ported from their Python
reference, with split-block backup filters, an AVX2 batch query, and a CUDA
backend. Source: <https://github.com/shaia/learned-bloom-filter> (MIT; see
[LICENSE](LICENSE)). The library, its namespace and its include path keep the
short name `plbf`.

- **Same decisions as the reference.** On all 116 golden configurations the
  boundaries match exactly, and every FPR and cost is bit-identical to the
  Python's (g++ 16, clang 19 and MSVC 19.44, x86-64).
- **The threshold search stops mattering.** At the scale of the paper's URL
  dataset, Fast PLBF# picks its layout in about 0.4 ms. The whole build takes
  about 4.5 ms, most of it histogramming and filling the filters.
- **A third less memory than a plain filter.** About 7 bits per key against
  10.5 for a split-block Bloom filter at the same 1% FPR, on the synthetic
  classifier below. The classifier's own size is not counted.
- **On a GPU, bit-identical and much faster to build.** The CUDA backend builds
  the same filter, block for block, 135× faster than one CPU thread at the
  paper's DNA scale: 13 ms against 1.8 s for 50 M keys. GPU queries only pay off
  when the queries are already on the GPU, though. Copying them over PCIe costs
  more than the CPU spends answering them.

> **Status:** CPU and CUDA. Builds and tests pass on Windows with g++ 16
> (MinGW), clang 19 (MSVC target), MSVC 19.44, and CUDA 13.0 on an RTX 4090
> Laptop GPU. Not yet built on Linux or macOS; no serialisation yet.

## What it is for

A learned Bloom filter answers "is this key in the set?" with help from a
classifier that scores how key-like each item is. The partitioned variant
(Vaidya et al., ICLR 2021) splits the score range into k regions, each with its
own backup filter at its own false-positive rate. High-scoring items get a
loose filter, or none. Low-scoring items get a strict one. The paper this
library implements makes choosing those regions fast enough to rebuild often.

It fits when approximate-membership filters are rebuilt frequently and a
classifier already separates keys from non-keys well. Malicious URLs and
malware hashes are the paper's examples. If no such classifier exists, or the
model costs more per query than the memory is worth, a plain Bloom filter is
the better choice.

## Quickstart

```cpp
#include <plbf/plbf.hpp>

// One hash and one classifier score per key; scores for a sample of non-keys.
std::vector<std::uint64_t> key_hashes;   // e.g. plbf::hash_bytes(url)
std::vector<double> key_scores;          // model(url), in [0, 1]
std::vector<double> non_key_scores;      // model(x) for known non-members

plbf::options opt;                       // N = 1000 segments, k = 5 regions,
opt.search = plbf::method::fast_sharp;   // Fast PLBF# (the default)
const auto f = plbf::filter::build_for_fpr(key_hashes, key_scores, non_key_scores, 0.01, opt);

bool maybe = f.contains(plbf::hash_bytes(query), model(query));

// Or many at once: prefetched, AVX2 where the CPU has it.
f.contains_batch(hashes, scores, out);   // out[i] is 0 or 1
```

`build_for_memory` takes a bit budget instead and minimises the expected FPR.
`f.regions()` reports what the optimiser chose: boundaries, per-region FPRs, and
the predicted cost.

The optimiser is usable on its own, for example with histograms computed
elsewhere:

```cpp
auto g = plbf::segment_distribution::from_counts(key_counts);      // +1 smoothing applied
auto h = plbf::segment_distribution::from_counts(non_key_counts);
plbf::layout l = plbf::optimize_for_fpr(g, h, n_keys, /*k=*/5, /*fpr=*/0.01);
```

On a GPU, from a `.cu` file, with every array already in device memory:

```cpp
#include <plbf/cuda/filter.cuh>

auto d = plbf::cuda::device_filter::build_for_fpr(d_key_hashes, d_key_scores, n_keys,
                                                  d_non_key_scores, n_non_keys, 0.01, opt);
d.contains(d_hashes, d_scores, d_out, n);   // asynchronous, like any kernel launch
plbf::filter f = d.to_host();               // the same filter, on the CPU
```

`plbf::cuda::device_filter(f)` uploads a filter built on the CPU instead.

## Build and test

```bash
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=g++
```

```bash
cmake --build build
```

```bash
ctest --test-dir build --output-on-failure
```

Binaries land in `bin/`. Release is the default for every toolchain, MSVC
included. CMake's MSVC platform file otherwise defaults to Debug, which runs
3–24× slower here (`bench/results/2026-10-30/msvc_debug_vs_release.txt`) and
reorders the kernels: in Debug the AVX2 lookup is the slowest batch. On
Windows, the clang++ bundled with Visual Studio
targets 32-bit x86 by default. Pass `--target=x86_64-pc-windows-msvc` for the
AVX2 kernels; the 32-bit build works, with scalar probes only.

**CUDA.** On Windows, nvcc needs MSVC as its host compiler, so configure a
separate tree from a VS 2022 x64 developer prompt (or after running
`vcvars64.bat`):

```bat
cmake -S . -B build-cuda -G Ninja -DCMAKE_CXX_COMPILER=cl -DPLBF_BUILD_CUDA=ON
cmake --build build-cuda
ctest --test-dir build-cuda --output-on-failure
```

This adds `test_cuda` and `bench_cuda` to the CPU suites. MSVC-built binaries
get an `_msvc` suffix (`PLBF_BIN_SUFFIX`), so they don't overwrite the g++
ones in `bin/`. `CMAKE_CUDA_ARCHITECTURES` defaults to `native`, the GPU in
the machine. CUDA 13.0 accepts VS 2022's MSVC 19.44; the newer Visual Studio
18 toolset was not tried.

## The four construction methods

All four search the same space: k contiguous regions of the N score segments,
the last of which is segments j..N for some j. A DP over the first j-1 segments
maximises the key/non-key divergence. Each region's FPR then follows in closed
form. They differ only in how the DP's row maxima are found.

| `method` | Paper name | Row maxima | Complexity | Result |
|---|---|---|---|---|
| `plbf` | PLBF | full scan, DP rebuilt for every j | O(N³k) | optimal |
| `fast` | Fast PLBF | full scan, one DP | O(N²k) | identical to `plbf` |
| `fast_pp` | Fast PLBF++ | divide and conquer | O(Nk log N) | identical when the key/non-key ratio rises with the score |
| `fast_sharp` | Fast PLBF# | SMAWK | O(Nk log k) | the same as `fast_pp` |

When the ratio does not rise monotonically, `fast_pp` and `fast_sharp` can pick
slightly worse layouts. The paper bounds the gap, and the tests check that
neither ever beats `fast`.

## Results

These are from an i9-13980HX laptop (2 MB L2 per P-core, 36 MB L3) with g++ 16
at `-O3` (CMake's Release default). Other work was running on the machine, so
compare rows within a table, not across runs. `bin/bench_construct` and
`bin/bench_query` reproduce them.

**Construction**, at the paper's URL scale (223,088 keys and 428,118 non-keys,
N = 1000, target FPR 1%):

| | k = 5 | k = 50 |
|---|---|---|
| histogram, keys + non-keys | 1.7 ms | 1.8 ms |
| search, `plbf` | 1,282 ms | not run |
| search, `fast` | 3.6 ms | 16.9 ms |
| search, `fast_pp` | 0.40 ms | 4.3 ms |
| search, `fast_sharp` | 0.43 ms | 4.4 ms |
| whole build, `fast_sharp` | 4.5 ms | 10.4 ms |

For comparison, the Python reference spends about 1.2 s on `fast` and 45 ms on
`fast_sharp` at N = 1000, k = 5, though on fewer keys (5,000; see
`reference_seconds` in the golden file). The search cost depends on N and k, not
on the number of keys.

**Queries** (target FPR 1%; half keys, half unseen non-keys, random order; best
of 7 interleaved rounds; the recorded run, `bench/results/2026-10-30/bench_query_gcc.txt`):

| keys | filter size | AVX2 lookup + AVX2 probe | scalar lookup + AVX2 probe | scalar batch | plain split-block, AVX2 batch |
|---|---|---|---|---|---|
| 200 K | 178 KB (7.1 bits/key) | 3.0 ns | 4.5 ns | 8.4 ns | 1.2 ns (10.5 bits/key) |
| 4 M | 3.4 MB (6.8 bits/key) | 3.1 ns | 5.7 ns | 9.4 ns | 2.1 ns |
| 64 M | 56 MB (7.0 bits/key) | 6.4 ns | 7.9 ns | 10.9 ns | 7.2 ns |

The first column is `kernel::avx2`, the default wherever the CPU has AVX2: it
maps four scores to block addresses at once (see the design notes). The second
is `kernel::avx2_probe`, the same probe behind the scalar lookup, kept so the
lookup's share can be measured.

The vector lookup was checked with an A/B: the same benchmark source compiled
against the old headers and the new ones, six alternating runs of each, each
run alone. With the filter in L2, the AVX2 batch went from 4.31 to 2.76 ns per
query (1.56×). The scalar paths did not move: 9.96 against 9.97 ns one query at
a time, 8.16 against 8.03 ns for the scalar batch.

With the filter in cache, the learned filter now costs about 2 ns per query
more than a plain split-block filter (3.0 against 1.2 ns). That is the price of
mapping a score to its region, paid for a third less memory.

**Past L3, memory latency dominates.** At 64M keys the vector lookup takes
6.4 ns against 7.9 ns for the scalar lookup (1.23×), and the learned filter's
batch is faster than the plain filter's (7.2 ns), which is a third larger.
DRAM-sized runs during development came after the machine's memory latency
rose 7–8× for every variant, the plain filter included, and are not quoted.
Batching is 3.9× faster than querying one at a time at this size (42.5 against
10.9 ns), because the chunk prefetches 64 blocks before testing any of them.

**GPU** (RTX 4090 Laptop GPU: 76 SMs, 67 MB L2; CUDA 13.0, MSVC 19.44 host
code). These are from one `bin/bench_cuda_msvc` run, in the "Silent" power
profile, with other applications holding about 23% of the GPU. Every GPU
filter and every GPU answer in the run was checked equal to the CPU's.

*Building* (N = 1000, k = 5, target FPR 1%; the CPU is one thread):

| | URL scale (223 K keys, 428 K non-keys) | DNA scale (50 M keys, 50 M non-keys) |
|---|---|---|
| CPU build | 6.1 ms | 1,798 ms |
| GPU build, data already on the GPU | 0.88 ms | 13.4 ms |
| of which: two histograms on the GPU | 0.11 ms | 3.1 ms |
| of which: threshold search on the CPU | 0.52 ms | 0.58 ms |
| of which: tables, allocation, insert | 0.25 ms | 9.7 ms |
| copying keys and scores in (pageable) | 0.90 ms | 105 ms |

At DNA scale, the GPU build is 135× faster than one CPU thread, and still about
15× faster when the keys and scores have to be copied in first. The CPU build is
single-threaded, so a multithreaded one would narrow that gap. The threshold
search is the same 0.5 ms on both, and on the GPU path it is now the part that
does not shrink.

*Querying* (100 M queries, half keys, half unseen non-keys, target FPR 1%):

| keys | filter | GPU kernel | GPU, kernel + copies | CPU, 1 thread | CPU, 32 threads |
|---|---|---|---|---|---|
| 4 M | 3.4 MB | 0.034 ns (29 G/s) | 1.43 ns | 3.44 ns | 0.47 ns |
| 50 M | 44 MB, in GPU L2 | 0.046 ns (22 G/s) | 1.41 ns | 6.00 ns | 0.76 ns |
| 200 M | 170 MB | 0.173 ns (5.8 G/s) | 1.57 ns | 8.19 ns | 1.10 ns |

The kernel answers 6–17× faster than all 32 CPU threads, but only for queries
that are already on the GPU. Copying 16 bytes of query in and 1 byte of answer
out costs about 1.4 ns per query over this machine's PCIe link (12 GB/s
measured, pinned memory). That is slower than every thread of the CPU at every
size here. GPU queries therefore make sense when the scores come from a model
that runs on the GPU. When queries arrive on the CPU, the multithreaded AVX2
batch is the faster path.

## Design notes

- **Bit-exact parity is deliberate.** Every expression is evaluated in the
  reference's order, ties go to the leftmost split point as in the Python, and
  the dev build sets `-ffp-contract=off`. The golden suite therefore demands
  identical boundaries. It counts bit-identical values separately from the
  64-ulp tolerance, so a drift shows up even while the suite passes. The AVX2
  kernels are compiled with a per-function `target("avx2")` attribute rather
  than `-mavx2`, which keeps FMA contraction out of the optimiser.
- **Every segment's count starts at 1, as in the reference.** This smoothing is
  load-bearing: every segment then has non-zero mass in both populations, so the
  divergence is always finite and no region ends up at FPR 0 with keys in it.
- **The backup filters are split-block, not classic.** A key sets one bit in
  each of the eight 32-bit words of a 256-bit block, so a probe is one load and
  one AND. Every region uses the same probe whatever its FPR, so a batch never
  branches on region type. The two regions that need no filter (FPR 1 and FPR 0)
  point at sentinel blocks of all ones and all zeros.
- **The vector lookup gives the scalar answer, not an approximation.** For four
  scores at once it takes c = ceil(score × N), then one compare against each
  neighbouring threshold. Rounding in score × N and in i / N is relative 2⁻⁵³,
  so for any N < 2³¹ the true segment is always c−1, c or c+1. The derivation
  is in `segment_grid::index4_avx2`. Gathers then fetch the region, and the
  region's slice packed into one 64-bit word, and the block index comes out of
  `_mm256_mul_epu32`. The tests compare it lane for lane with the scalar path on
  54 million scores across ten grid sizes up to N = 2²⁴ + 1, including every
  threshold and its neighbouring doubles. Deleting the downward correction makes
  that check fail on 11,541 of them.
- **Filter sizing uses the split-block FPR model, not the classic one.** The
  optimiser still minimises classic-Bloom-filter bits, which is the paper's cost
  model and what keeps parity. Each region's split-block filter is then sized
  from its own Poisson FPR model, so the measured FPR meets the target. This
  costs about 20–25% more memory than the paper's model predicts. In the tests
  the model matches measured FPRs within sampling noise.
- **`build_for_memory` budgets classic-filter bits, as the paper does.** The
  split-block filters that get built use somewhat more; `memory_bytes()` reports
  what was actually allocated.
- **The CUDA backend mirrors the CPU code instead of reimplementing it.** Its
  kernels use the same segment index (the one-step correction above, with
  `__dmul_rn` so fast-math flags cannot change the multiply), the same block
  index (`__umulhi` is the multiply-shift), and the same bit pattern, with the
  salts pinned to the CPU's by a `static_assert`. The routing tables come from
  the same `region_tables::build`. Insertion uses 64-bit `atomicOr`, which
  commutes, so the order threads land in cannot change a bit. That is why a
  GPU-built filter can be compared with a CPU-built one byte for byte, and
  `test_cuda` does exactly that. Breaking the device index or dropping one of
  the four ORs makes it fail.
- **The threshold search stays on the CPU.** It is a k-layer DP with
  recursive, branchy row maxima, and takes about 0.5 ms. Only the per-segment
  counts come back for it, 8 bytes per segment per population (16 KB at
  N = 1000). The routing tables it produces are about the same size going up.

## Differences from the reference

- **`k = 1` is rejected, as in the reference,** which cannot trace back through
  one region; a single region is just a Bloom filter.
- **Broken traces are skipped.** An approximate method on a very non-monotone
  input can return a predecessor chain that does not step down to 0. The
  reference would go on to take the log of a negative mass and raise an error.
  This port skips that candidate j instead.
- **Memory mode reports the chosen layout's memory.** The reference's `_M`
  classes record the memory of the last candidate j instead. Every candidate is
  sized to the same budget, so the two figures agree to within rounding.

## Regenerating the golden file

`tools/gen_golden.py` runs the reference's own classes and writes
`test/data/*.csv` and `test/golden/reference.txt`. It stubs pandas,
scikit-learn and bloom_filter, which the reference imports but this path never
calls, so nothing needs installing.

```bash
python -I tools/gen_golden.py path/to/FastPLBF --out-dir test
```

## Other tools

- **`tools/time_reference.py`** times the Python reference's four searches, and nothing else, at a
  given scale.
- **`tools/figure_layouts.cpp`** writes the numbers behind the blog series' explorable figures from
  the library itself: the layout for every slider setting, the DP traces of the three row-maxima
  solvers, and their evaluation counts as N grows.
- **`tools/throttle_probe.cpp`** prints a compute rate and a DRAM latency about twice a second.
  Run it before trusting a benchmark on a laptop: the first attempt at recording
  `bench/results/2026-10-30/` ran up to 10× slow once the CPU throttled, within 5 seconds of load,
  and only this showed it.
- **`tools/pcie.cu`** measures what a query that starts on the CPU costs a GPU filter: the copy
  in, the probe and the copy out. It is the reducer from Part 10 of the blog series.

Raw benchmark outputs, with the machine and its power state, live in `bench/results/<date>/`.

## Not done yet

- A multithreaded CPU build. The CPU figures above are one thread; the
  histogram and the insert both parallelise.
- Serialisation. `filter::from_parts` and the `tables()`/`blocks()` accessors
  are the seam it would use.
- Builds on Linux and macOS, including CUDA there.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the reference's MIT license.
