# Benchmark results, 2026-10-30

The raw outputs the blog series quotes. Nothing here is edited by hand.

| File | Program | Build |
|---|---|---|
| `time_reference.txt` | `tools/time_reference.py`, the FastPLBF Python reference's four searches | Python 3.13.9 |
| `bench_construct_gcc.txt` | `bench/bench_construct.cpp` | g++ 16.1.0 (MinGW-w64 UCRT), `-O3 -DNDEBUG -ffp-contract=off` |
| `bench_query_gcc.txt` | `bench/bench_query.cpp` | the same |
| `bench_cuda_msvc.txt` | `bench/bench_cuda.cu` | CUDA 13.0.88, host MSVC 19.44.35229 `/O2 /Ob2 /DNDEBUG` |
| `probe_before.txt`, `probe_after.txt` | `tools/throttle_probe.cpp`, 20 s each side of the three runs | g++, `-O2` |

## The machine

- **CPU:** Intel Core i9-13980HX (8 P-cores and 16 E-cores, 32 threads; 2 MB L2 per P-core; 36 MB
  L3), with 64 GB of RAM, in a Razer laptop on AC power.
- **GPU:** NVIDIA GeForce RTX 4090 Laptop GPU (76 SMs, 16 GB, driver 617.42).
- **OS:** Windows 11 Pro 10.0.26300.

## Power state, which matters here

Windows reported its "Silent" power scheme throughout. The user switched Razer Synapse to its
performance mode, which Windows does not reflect. The probe files show what the CPU actually did:
a single-thread compute rate and a DRAM pointer-chase latency, once a second.

The first recording attempt, in `throttled/`, ran before that switch held. Within 5 seconds of
sustained load the probe's compute rate fell from about 1,190 to about 160 Mops/s, and its DRAM
latency rose from 94 to 155–200 ns. Every CPU figure in `throttled/` from the second minute on is 4
to 12 times slower than in a clean run, while the GPU kernel figures are unaffected. Those files are
kept as evidence of that failure, and are **not** quoted as results.
