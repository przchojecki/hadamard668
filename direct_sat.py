#!/usr/bin/env python3
"""
Direct SAT+CAS approach: encode the 37×9 ±1 matrix with column sum
constraints, solve with SAT, check LP PSD with CAS, add learned clauses.

This bypasses the 37-compression search entirely: we just need
compatible column sums (from the 9-compression search) and let SAT
find the full matrix.

The SAT+CAS loop:
1. SAT: find 37×9 ±1 matrices M_A, M_B with given column sums
2. CAS: check PSD(flatten(M_A), s) + PSD(flatten(M_B), s) = 668 for all s
3. If LP fails at frequency s, add a constraint blocking the current
   row-sum configuration (generalized from the PSD violation)
4. Repeat
"""

import numpy as np
from numpy.fft import fft
import json
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

TARGET = 668
ELL = 333

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def check_lp_psd(A, B):
    """Check LP PSD condition. Returns (is_valid, max_deviation, worst_freq)."""
    psd_A = np.abs(fft(A)) ** 2
    psd_B = np.abs(fft(B)) ** 2
    total = psd_A + psd_B
    devs = np.abs(total[1:] - TARGET)
    max_dev = np.max(devs)
    worst = np.argmax(devs) + 1
    return max_dev < 0.5, max_dev, worst


def build_matrix_from_cols(col_sums, rng):
    """
    Build a random 37×9 ±1 matrix with given column sums.
    col_sums: length-9 array of odd integers (each sum of 37 ±1 values).

    Returns 37×9 matrix or None.
    """
    M = np.ones((37, 9), dtype=int)
    for j in range(9):
        target_ones = (col_sums[j] + 37) // 2  # number of +1 entries
        target_neg = 37 - target_ones
        if target_ones < 0 or target_ones > 37:
            return None
        # Set first target_neg entries to -1, then shuffle
        col = np.ones(37, dtype=int)
        col[:target_neg] = -1
        rng.shuffle(col)
        M[:, j] = col
    return M


def flatten_matrix(M):
    """Flatten 37×9 matrix to length-333 sequence (row-major)."""
    return M.flatten()


def unflatten_to_matrix(seq):
    """Reshape length-333 sequence to 37×9 matrix."""
    return np.array(seq).reshape(37, 9)


def random_matrix_search(col_A, col_B, max_tries=100000, verbose=True):
    """
    Generate random ±1 matrices with given column sums and check LP.
    This is the naive approach — unlikely to find LP by chance, but
    gives a baseline and checks the pipeline.
    """
    rng = np.random.default_rng(42)
    col_A = np.array(col_A, dtype=int)
    col_B = np.array(col_B, dtype=int)

    best_dev = float('inf')
    t0 = time.time()

    for trial in range(max_tries):
        M_A = build_matrix_from_cols(col_A, rng)
        M_B = build_matrix_from_cols(col_B, rng)
        if M_A is None or M_B is None:
            continue

        A = flatten_matrix(M_A)
        B = flatten_matrix(M_B)

        valid, dev, worst = check_lp_psd(A, B)

        if dev < best_dev:
            best_dev = dev
            if verbose and (trial < 10 or trial % 10000 == 0):
                print(f"  trial {trial}: dev={dev:.1f} worst_freq={worst}",
                      flush=True)

        if valid:
            if verbose:
                print(f"  FOUND LP at trial {trial}!")
            return A, B

    dt = time.time() - t0
    if verbose:
        print(f"  {max_tries} trials in {dt:.1f}s, best dev={best_dev:.1f}")
    return None


def local_search_matrix(col_A, col_B, max_iter=1_000_000, verbose=True):
    """
    Local search on the 37×9 matrix directly.
    Move: swap two entries in the same column (preserves column sum).
    Energy: sum of |PSD_A(s) + PSD_B(s) - 668| over all nonzero s.
    """
    rng = np.random.default_rng(42)
    col_A = np.array(col_A, dtype=int)
    col_B = np.array(col_B, dtype=int)

    M_A = build_matrix_from_cols(col_A, rng)
    M_B = build_matrix_from_cols(col_B, rng)

    A = flatten_matrix(M_A).astype(np.float64)
    B = flatten_matrix(M_B).astype(np.float64)

    # Precompute DFT
    F_A = fft(A)
    F_B = fft(B)
    psd_A = np.real(F_A * np.conj(F_A))
    psd_B = np.real(F_B * np.conj(F_B))

    cur_E = np.sum(np.abs(psd_A[1:] + psd_B[1:] - TARGET))
    best_E = cur_E
    best_A = A.copy()
    best_B = B.copy()

    # Precompute twiddle factors for incremental DFT updates
    omega = np.exp(-2j * np.pi * np.arange(ELL) / ELL)
    # TWID[s, k] = omega^{s*k}
    # But storing 333×333 complex is ~1.7GB. Too much.
    # Instead, compute twiddle on the fly for the 2 positions being swapped.

    T_init = cur_E * 0.1
    accepts = 0
    t0 = time.time()

    for it in range(max_iter):
        progress = it / max_iter
        T = max(0.01, T_init * (1.0 - progress) ** 1.5)

        # Choose which matrix to modify
        if rng.random() < 0.5:
            M, flat, F, psd, other_psd = M_A, A, F_A, psd_A, psd_B
        else:
            M, flat, F, psd, other_psd = M_B, B, F_B, psd_B, psd_A

        # Move: swap two entries in the same column
        j = rng.integers(9)
        col = M[:, j]
        plus_idx = np.where(col == 1)[0]
        minus_idx = np.where(col == -1)[0]
        if len(plus_idx) == 0 or len(minus_idx) == 0:
            continue

        i1 = plus_idx[rng.integers(len(plus_idx))]   # +1 → -1
        i2 = minus_idx[rng.integers(len(minus_idx))]  # -1 → +1

        # Position in flattened array
        pos1 = i1 * 9 + j  # changes from +1 to -1 (delta = -2)
        pos2 = i2 * 9 + j  # changes from -1 to +1 (delta = +2)

        # Incremental DFT update
        # F_new[s] = F[s] + (-2) * omega^{s*pos1} + 2 * omega^{s*pos2}
        tw1 = np.exp(-2j * np.pi * pos1 * np.arange(ELL) / ELL)
        tw2 = np.exp(-2j * np.pi * pos2 * np.arange(ELL) / ELL)
        delta_F = -2 * tw1 + 2 * tw2

        new_F = F + delta_F
        new_psd = np.real(new_F * np.conj(new_F))
        new_E = np.sum(np.abs(new_psd[1:] + other_psd[1:] - TARGET))

        dE = new_E - cur_E
        if dE < 0 or rng.random() < np.exp(-dE / max(T, 0.01)):
            M[i1, j] = -1
            M[i2, j] = 1
            flat[pos1] = -1
            flat[pos2] = 1
            F[:] = new_F
            psd[:] = new_psd
            cur_E = new_E
            accepts += 1

            if cur_E < best_E:
                best_E = cur_E
                best_A[:] = A
                best_B[:] = B

                if best_E < 0.5:
                    if verbose:
                        dt = time.time() - t0
                        print(f"  FOUND LP! E={best_E:.3f} at iter {it+1} "
                              f"({dt:.1f}s)")
                    return best_A.copy(), best_B.copy(), best_E

        if verbose and (it + 1) % 200_000 == 0:
            dt = time.time() - t0
            rate = accepts / (it + 1) * 100
            print(f"  {it+1:>8,}: E={cur_E:.1f} best={best_E:.1f} "
                  f"T={T:.1f} acc={rate:.1f}% {dt:.0f}s", flush=True)

    if verbose:
        dt = time.time() - t0
        print(f"  Final: best={best_E:.1f}, {dt:.0f}s")

    return best_A.copy(), best_B.copy(), best_E


def run_direct_search(case_idx=0, n_trials=5, max_iter=2_000_000):
    """
    Run direct matrix search for a given macro-case using precomputed
    9-compression results.
    """
    from hadamard.search9 import find_sequences_for_psd, decode_psd_key

    psd9_file = os.path.join(RESULTS_DIR, f"psd9_case{case_idx}.json")
    with open(psd9_file) as f:
        data = json.load(f)

    from hadamard.compression import get_macro_case_details
    details = get_macro_case_details()
    P_A, P_B, reps_A, reps_B = details[case_idx]

    print(f"Direct SAT search for case {case_idx}: PSD(A,111)={P_A}, PSD(B,111)={P_B}")

    # Get (u,v,w) representatives
    seen = set()
    uvw_pairs = []
    for a_info in reps_A:
        for b_info in reps_B:
            _, _, ua, va, wa = a_info
            _, _, ub, vb, wb = b_info
            key = (tuple(sorted([ua,va,wa])), tuple(sorted([ub,vb,wb])))
            if key not in seen:
                seen.add(key)
                uvw_pairs.append(((ua,va,wa), (ub,vb,wb)))

    # Try a few PSD matches
    matches = data['sample_matches']
    print(f"  {len(matches)} sample PSD matches available")

    best_overall = float('inf')
    for mi, match in enumerate(matches[:20]):
        psd_A_triple = tuple(match[0])
        psd_B_triple = tuple(match[1])

        for uvw_A, uvw_B in uvw_pairs[:2]:
            # Find actual length-9 sequences
            seqs_A = find_sequences_for_psd(*uvw_A, psd_A_triple)
            seqs_B = find_sequences_for_psd(*uvw_B, psd_B_triple)
            if not seqs_A or not seqs_B:
                continue

            col_A = list(seqs_A[0])
            col_B = list(seqs_B[0])

            print(f"\n  Match {mi}: col_A={col_A}, col_B={col_B}", flush=True)

            for trial in range(n_trials):
                print(f"  Trial {trial}:", flush=True)
                A, B, E = local_search_matrix(
                    col_A, col_B, max_iter=max_iter)
                print(f"    E={E:.1f}", flush=True)

                if E < best_overall:
                    best_overall = E
                if E < 0.5:
                    print(f"\n*** FOUND LP(333)! ***")
                    return A, B

            break  # Just try first uvw pair for now
        if best_overall < 100:
            break

    print(f"\nBest energy: {best_overall:.1f}")
    return None


if __name__ == "__main__":
    case = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 2_000_000
    run_direct_search(case, n_trials, max_iter)
