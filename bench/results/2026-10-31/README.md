# Benchmark results, 2026-10-31: smaller query formats for the GPU

The raw outputs behind the 32-bit-score and segment-number query formats. Nothing here is edited by
hand. Machine as in `../2026-10-30/README.md`; Windows reported its "Performance" power scheme.

| File | Program | Build |
|---|---|---|
| `bench_cuda_formats.txt` | `bench/bench_cuda.cu`, with `tools/throttle_probe.cpp 20` before and after | CUDA 13.0.88, host MSVC 19.44 `/O2 /Ob2 /DNDEBUG` |
| `pcie_formats_five_runs.txt` | `tools/pcie.cu`, five runs in a row | CUDA 13.0.88, `nvcc -O3 -std=c++17` |

## What changed in the code

- `plbf::filter::contains_batch` takes 32-bit scores, widened to double exactly, so the answer is
  `contains()` on the widened value. `filter::contains_segment` takes a segment number.
- `plbf::cuda::device_filter::contains` takes 32-bit scores (12 bytes a query across PCIe), and
  `contains_segments` takes 16-bit segment numbers (10 bytes), which `to_segments` computes on the
  host. Segment numbers need at most 65,535 segments.
- `bench_cuda` now rounds every score to float before building, as a model's float32 output would be,
  so all three formats give identical answers; it checks that they do. Its query rows for the double
  format are otherwise as before, with two rows added per size.

## What it shows

Per query, from the host, with the link at about 12 GB/s in every run here:

| Filter | 16 B (double) | 12 B (float) | 10 B (segment) | CPU, 32 threads |
|---|---|---|---|---|
| 3.4 MB | 1.459 ns | 1.124 ns | 0.932 ns | 0.421 ns |
| 43.7 MB | 1.476 ns | 1.126 ns | 0.939 ns | 0.825 ns |
| 169.7 MB | 1.589 ns | 1.250 ns | 1.085 ns | 1.058 ns |

- The copy scales with the bytes: `pcie.cu` measures 1.29, 0.96 and 0.81 ns in for 16, 12 and 10
  bytes, as the arithmetic predicted.
- The CPU's 32 threads still answer faster at every size. The 169.7 MB filter is within 3%.
- The segment format adds 1.0–1.2 ns a query of score-to-segment work on one CPU thread, the work
  it moves off the GPU. It suits a pipeline that already has the segment, or a CPU that is idle.
- Compare within this run only: the CPU's 32-thread batch measured 0.421 ns at 4M keys here, against
  0.613 ns in `../2026-10-30/bench_cuda_msvc.txt`, on the same machine a day apart.
