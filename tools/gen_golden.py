"""Generate golden layouts from the FastPLBF Python reference.

Usage:
    python -I tools/gen_golden.py <path to FastPLBF checkout> [--out-dir test]

Writes the score files to <out-dir>/data/ and the reference's decisions to
<out-dir>/golden/reference.txt. test_optimize reads both and demands the same
boundaries, and the same FPRs and costs to within a few ulps.

Only the reference's optimisation step runs. Its module-level imports of
pandas, scikit-learn and bloom_filter are satisfied with empty stubs: the code
paths that use them (CSV loading, the train/test split, inserting keys) are
never called, so nothing needs installing.

The reference's _M classes record the memory of the last candidate they tried
rather than of the layout they chose. Every candidate is sized to the same
budget, so the two agree to within rounding; the golden file still carries the
chosen layout's figure (SpaceUsed) as backup_bits, and the reference's own as
reference_reported_bits.
"""

import argparse
import importlib.util
import os
import random
import sys
import time
import types


def stub_third_party():
    for name in ("pandas", "sklearn", "sklearn.model_selection", "bloom_filter"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["sklearn"].model_selection = sys.modules["sklearn.model_selection"]
    sys.modules["sklearn.model_selection"].train_test_split = None
    sys.modules["bloom_filter"].BloomFilter = None


def load_reference(ref_root):
    src = os.path.join(ref_root, "src")
    sys.path.insert(0, src)
    from PLBFs.PLBF import PLBF
    from PLBFs.FastPLBF import FastPLBF
    from PLBFs.FastPLBFpp import FastPLBFpp
    from PLBFs.FastPLBFsharp import FastPLBFsharp
    from PLBFs.PLBF_M import PLBF_M
    from PLBFs.FastPLBF_M import FastPLBF_M
    from PLBFs.FastPLBFpp_M import FastPLBFpp_M
    from PLBFs.FastPLBFsharp_M import FastPLBFsharp_M
    from PLBFs.utils.SpaceUsed import SpaceUsed
    from PLBFs.utils.ExpectedFPR import ExpectedFPR

    # The synthetic generators live in src/test.py; load it under another name
    # so it cannot shadow the standard library's `test` package.
    spec = importlib.util.spec_from_file_location("fastplbf_test", os.path.join(src, "test.py"))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    classes = {
        ("fpr", "plbf"): PLBF,
        ("fpr", "fast"): FastPLBF,
        ("fpr", "fast_pp"): FastPLBFpp,
        ("fpr", "fast_sharp"): FastPLBFsharp,
        ("memory", "plbf"): PLBF_M,
        ("memory", "fast"): FastPLBF_M,
        ("memory", "fast_pp"): FastPLBFpp_M,
        ("memory", "fast_sharp"): FastPLBFsharp_M,
    }
    return classes, SpaceUsed, ExpectedFPR, gen


def read_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        assert header == ["key", "label", "score"], header
        for line in f:
            key, label, score = line.strip().split(",")
            rows.append((key, int(label), float(score)))
    return rows


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("key,label,score\n")
        for key, label, score in rows:
            f.write(f"{key},{label},{score!r}\n")


def synthetic(gen, seed, ideal, n_gen=1000, keys=5000, non_keys=5000):
    random.seed(seed)
    g, h = gen.create_ieal_g_h(n_gen) if ideal else gen.create_not_ieal_g_h(n_gen)
    pos = gen.make_scores_list(g, n_gen, keys)
    neg = gen.make_scores_list(h, n_gen, non_keys)
    return [(f"key_{i}", 1, s) for i, s in enumerate(pos)] + [
        (f"nonkey_{i}", 0, s) for i, s in enumerate(neg)
    ]


def run_case(classes, space_used, expected_fpr, rows, objective, method, target, n_seg, k):
    pos_scores = [s for _, label, s in rows if label == 1]
    neg_scores = [s for _, label, s in rows if label != 1]
    cls = classes[(objective, method)]
    obj = cls.__new__(cls)  # skip __init__: it would insert keys into a real Bloom filter
    if objective == "fpr":
        obj.F = float(target)
    else:
        obj.M = float(target)
    obj.N, obj.k, obj.n = n_seg, k, len(pos_scores)

    start = time.perf_counter()
    thre, g, h = obj.divide_into_segments(pos_scores, neg_scores)
    obj.find_best_t_and_f(thre, g, h)
    seconds = time.perf_counter() - start

    boundary = [round(t * n_seg) for t in obj.t]
    assert all(abs(b - t * n_seg) < 1e-9 for b, t in zip(boundary, obj.t))
    fpr = [float(x) for x in obj.f[1:]]
    return {
        "keys": len(pos_scores),
        "boundary": boundary,
        "fpr": fpr,
        "backup_bits": float(space_used(g, h, obj.t, obj.f, obj.n)),
        "expected_fpr": float(expected_fpr(g, h, obj.t, obj.f, obj.n)),
        "reference_reported_bits": float(obj.memory_usage_of_backup_bf),
        "reference_seconds": seconds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_root", help="path to a FastPLBF checkout (the directory holding src/ and data/)")
    ap.add_argument("--out-dir", default="test")
    args = ap.parse_args()

    stub_third_party()
    classes, space_used, expected_fpr, gen = load_reference(args.ref_root)

    data_dir = os.path.join(args.out_dir, "data")
    golden_dir = os.path.join(args.out_dir, "golden")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(golden_dir, exist_ok=True)

    datasets = {
        "example.csv": read_csv(os.path.join(args.ref_root, "data", "example.csv")),
        "synth_ideal.csv": synthetic(gen, seed=0, ideal=True),
        "synth_nonideal.csv": synthetic(gen, seed=1, ideal=False),
    }
    for name, rows in datasets.items():
        write_csv(os.path.join(data_dir, name), rows)

    all_methods = ["plbf", "fast", "fast_pp", "fast_sharp"]
    fast_methods = ["fast", "fast_pp", "fast_sharp"]  # plbf is O(N^3 k): minutes at N = 1000
    cases = []
    # The configurations of the reference's own sh/runF.sh and sh/runM.sh.
    for f in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]:
        cases += [("example.csv", "fpr", m, f, 50, 5) for m in all_methods]
    for mem in [100, 200, 500, 1000, 1500, 2000, 2500, 3000]:
        cases += [("example.csv", "memory", m, float(mem), 50, 5) for m in all_methods]
    for ds in ["synth_ideal.csv", "synth_nonideal.csv"]:
        for k in [5, 10]:
            for f in [0.01, 0.001]:
                cases += [(ds, "fpr", m, f, 100, k) for m in all_methods]
        cases += [(ds, "memory", m, 40000.0, 100, 5) for m in all_methods]
        cases += [(ds, "fpr", m, 0.001, 1000, 5) for m in fast_methods]
        cases += [(ds, "memory", m, 40000.0, 1000, 5) for m in fast_methods]

    out_path = os.path.join(golden_dir, "reference.txt")
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        out.write("# FastPLBF reference decisions. Generated by tools/gen_golden.py; do not edit.\n")
        for i, (ds, objective, method, target, n_seg, k) in enumerate(cases):
            r = run_case(classes, space_used, expected_fpr, datasets[ds], objective, method, target, n_seg, k)
            print(f"[{i + 1}/{len(cases)}] {ds} {objective} {method} target={target} N={n_seg} k={k}"
                  f"  {r['reference_seconds']:.3f}s", flush=True)
            out.write("case\n")
            out.write(f"data {ds}\n")
            out.write(f"objective {objective}\n")
            out.write(f"method {method}\n")
            out.write(f"target {float(target)!r}\n")
            out.write(f"segments {n_seg}\n")
            out.write(f"regions {k}\n")
            out.write(f"keys {r['keys']}\n")
            out.write("boundary " + " ".join(str(b) for b in r["boundary"]) + "\n")
            out.write("fpr " + " ".join(repr(x) for x in r["fpr"]) + "\n")
            out.write(f"backup_bits {r['backup_bits']!r}\n")
            out.write(f"expected_fpr {r['expected_fpr']!r}\n")
            out.write(f"reference_reported_bits {r['reference_reported_bits']!r}\n")
            out.write(f"reference_seconds {r['reference_seconds']:.6f}\n")
            out.write("end\n")
    print(f"wrote {len(cases)} cases to {out_path}")


if __name__ == "__main__":
    main()
