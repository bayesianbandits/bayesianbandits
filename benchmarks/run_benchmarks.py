"""Benchmark all Takahashi Cython variants.

Generates a realistic sparse lower-triangular Cholesky factor (banded + random
fill-in to simulate supernodal structure), then times each variant.
"""
import importlib
import sys
import time

import numpy as np
from scipy.sparse import csc_array, random as sparse_random
from scipy.linalg import cholesky


def make_test_cholesky(p: int, bandwidth: int = 5, density: float = 0.01, seed: int = 42):
    """Create a sparse lower-triangular CSC matrix resembling a Cholesky factor.

    Builds a banded SPD matrix with some random fill-in, then takes its
    Cholesky decomposition.
    """
    rng = np.random.default_rng(seed)

    # Banded structure + random fill-in
    A = np.eye(p) * (p + 10.0)  # strong diagonal dominance
    for i in range(p):
        for j in range(max(0, i - bandwidth), i):
            v = rng.standard_normal()
            A[i, j] = v
            A[j, i] = v

    # Small random fill-in
    S = sparse_random(p, p, density=density, random_state=seed, format="coo")
    fill = S.toarray()
    fill = (fill + fill.T) * 0.1
    A += fill
    # Re-enforce diagonal dominance
    A += np.eye(p) * np.abs(A).sum(axis=1).max()

    L = cholesky(A, lower=True)
    # Sparsify: zero out near-zero entries
    L[np.abs(L) < 1e-12] = 0.0
    L_csc = csc_array(L)
    L_csc.eliminate_zeros()
    L_csc.sort_indices()
    return L_csc


def bench_variant(module_name: str, L_csc: csc_array, p: int,
                  n_warmup: int = 3, n_iters: int = 50):
    """Import a variant module and benchmark it."""
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        return None, str(e)

    func = mod.takahashi_diagonal
    data = L_csc.data.astype(np.float64)
    indices = L_csc.indices.astype(np.int32)
    indptr = L_csc.indptr.astype(np.int32)

    # Warmup
    for _ in range(n_warmup):
        result = func(data, indices, indptr, p)

    # Timed runs
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        result = func(data, indices, indptr, p)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return times, result


def main():
    sizes = [100, 500, 1000]

    VARIANTS = [
        ("takahashi_baseline",       "Baseline (ndarray buffers, no flags)"),
        ("takahashi_memview",        "Memoryviews (double[::1])"),
        ("takahashi_baseline_ofast", "Baseline + -O3 -ffast-math -march=native"),
        ("takahashi_memview_ofast",  "Memoryviews + -O3 -ffast-math -march=native"),
        ("takahashi_poisoned",       "POISONED: ** float exponents"),
        ("takahashi_no_cdiv",        "NO cdivision(True)"),
    ]

    print("=" * 90)
    print("Takahashi Diagonal — Cython Minefield Benchmark")
    print("=" * 90)

    for p in sizes:
        print(f"\n{'─' * 90}")
        print(f"  Matrix size p={p}")
        print(f"{'─' * 90}")

        L_csc = make_test_cholesky(p)
        nnz = L_csc.nnz
        print(f"  L has {nnz} nonzeros ({nnz / (p*p) * 100:.1f}% dense)")

        results = {}
        reference_result = None

        for mod_name, label in VARIANTS:
            times, result = bench_variant(mod_name, L_csc, p)
            if times is None:
                print(f"  {label:50s}  SKIPPED ({result})")
                continue

            if reference_result is not None:
                # Verify correctness
                maxdiff = np.max(np.abs(np.asarray(result) - np.asarray(reference_result)))
                if maxdiff > 1e-10:
                    print(f"  WARNING: {label} differs from baseline by {maxdiff:.2e}")
            else:
                reference_result = result

            median_us = np.median(times) * 1e6
            min_us = np.min(times) * 1e6
            std_us = np.std(times) * 1e6
            results[mod_name] = {
                "label": label,
                "median_us": median_us,
                "min_us": min_us,
                "std_us": std_us,
            }

        if not results:
            print("  No variants built successfully!")
            continue

        # Find fastest
        baseline_med = results.get("takahashi_baseline", {}).get("median_us", None)

        print(f"\n  {'Variant':50s} {'Median':>10s} {'Min':>10s} {'vs baseline':>12s}")
        print(f"  {'─' * 84}")
        for mod_name, label in VARIANTS:
            if mod_name not in results:
                continue
            r = results[mod_name]
            ratio_str = ""
            if baseline_med and mod_name != "takahashi_baseline":
                ratio = r["median_us"] / baseline_med
                if ratio > 1.02:
                    ratio_str = f"{ratio:.2f}x SLOWER"
                elif ratio < 0.98:
                    ratio_str = f"{1/ratio:.2f}x faster"
                else:
                    ratio_str = "~same"
            print(f"  {r['label']:50s} {r['median_us']:>8.1f}us {r['min_us']:>8.1f}us {ratio_str:>12s}")

    print(f"\n{'=' * 90}")
    print("Summary of Cython Minefields for this implementation:")
    print("=" * 90)
    print("""
  Minefield 1 (** float exponents):
    The current code AVOIDS this — uses manual multiplication instead of **.
    The 'poisoned' variant shows what happens if you use ** (-2.0) etc.

  Minefield 2 (pair index arrays vs nested loops):
    The current code AVOIDS this — uses nested for-loops with range().
    The inner scatter/gather loop is index-driven by necessity (sparse structure),
    but the pair iteration (a, b) uses proper nested loops.

  Minefield 3 (missing cdivision=True):
    The current code AVOIDS this — cdivision=True is set at module level.
    The 'no_cdiv' variant shows the penalty of the default (cdivision=False).

  Additional findings:
    - Old ndarray buffer syntax vs typed memoryviews: measured above.
    - Compiler optimization flags (-O3, -ffast-math): measured above.
    - Function is 'def' not 'cpdef': negligible for a function called once
      per matrix (the inner loops dominate).
""")


if __name__ == "__main__":
    main()
