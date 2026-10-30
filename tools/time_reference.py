"""Time the FastPLBF Python reference's threshold search, method by method.

Usage:
    python -I tools/time_reference.py <path to FastPLBF checkout> [--methods plbf,fast,fast_pp,fast_sharp]

The C++ benchmark (bench_construct) times the same search at the same N and k.
Only find_best_t_and_f is timed: the histogram (divide_into_segments) is a
separate pass in both, and its cost grows with the number of scores while the
search's depends on N and k alone. Scores follow bench/workload.hpp's ideal
shape (key density 4 s^3, non-key density 2 (1 - s)), drawn with Python's own
RNG -- the layouts may differ from the C++ run's, the amount of work does not.
"""

import argparse
import os
import platform
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_golden  # noqa: E402  (stubs pandas, scikit-learn and bloom_filter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_root")
    ap.add_argument("--methods", default="fast_sharp,fast_pp,fast,plbf")
    ap.add_argument("--keys", type=int, default=223088)
    ap.add_argument("--non-keys", type=int, default=428118)
    ap.add_argument("--segments", type=int, default=1000)
    ap.add_argument("--regions", type=int, default=5)
    ap.add_argument("--fpr", type=float, default=0.01)
    args = ap.parse_args()

    gen_golden.stub_third_party()
    classes, _, _, _ = gen_golden.load_reference(args.ref_root)

    rng = random.Random(42)
    pos = [rng.random() ** 0.25 for _ in range(args.keys)]
    neg = [1 - rng.random() ** 0.5 for _ in range(args.non_keys)]

    print(f"time_reference: Python {platform.python_version()}, {platform.processor() or platform.machine()}")
    print(f"{args.keys} keys, {args.non_keys} non-keys, N = {args.segments}, k = {args.regions}, "
          f"target FPR {args.fpr}; search only, one run each", flush=True)
    for method in args.methods.split(","):
        cls = classes[("fpr", method)]
        obj = cls.__new__(cls)
        obj.F, obj.N, obj.k, obj.n = args.fpr, args.segments, args.regions, len(pos)
        thre, g, h = obj.divide_into_segments(pos, neg)
        start = time.perf_counter()
        obj.find_best_t_and_f(thre, g, h)
        seconds = time.perf_counter() - start
        bits = obj.memory_usage_of_backup_bf
        boundary = [round(t * args.segments) for t in obj.t]
        print(f"  search {method:<12} {seconds * 1000:12.3f} ms   backup {bits:10.0f} bits   boundary {boundary}",
              flush=True)


if __name__ == "__main__":
    main()
