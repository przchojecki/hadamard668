#!/usr/bin/env python3
"""
Full pipeline: SA for 37-compression → intersect with 9-compression → SAT → LP verify.

For each macro-case:
1. Run joint SA to find PSD-compatible 37-compressed pairs (row sums)
2. For each SA solution with energy 0, intersect with 9-compression results
3. For compatible (row_sums, col_sums) pairs, attempt SAT decompression
4. Verify LP and construct Hadamard matrix
"""

import numpy as np
import json
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

from hadamard.search37_fast import joint_sa, compute_dft_all, compute_psd_from_dft
from hadamard.search9 import find_sequences_for_psd, decode_psd_key
from hadamard.marginals import gale_ryser_check
from hadamard.core import check_lp, lp_to_hadamard, verify_hadamard

TARGET = 668
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def try_intersect_and_decompress(r_A, r_B, case_idx):
    """
    Given exact 37-compressed pairs (row sums), try to:
    1. Find compatible 9-compressed pairs (column sums) from precomputed results
    2. Check Gale-Ryser
    3. Attempt SAT decompression
    """
    r_A = np.array(r_A, dtype=int)
    r_B = np.array(r_B, dtype=int)

    # Verify row-sum PSD
    dft_A = compute_dft_all(r_A)
    dft_B = compute_dft_all(r_B)
    psd_A = compute_psd_from_dft(dft_A)
    psd_B = compute_psd_from_dft(dft_B)

    max_dev = max(abs(psd_A[k] + psd_B[k] - TARGET) for k in range(18))
    if max_dev > 0.5:
        print(f"  Row sums fail PSD check: max deviation {max_dev:.1f}")
        return None

    # Row sum 3-compressions
    uvw_A = (int(np.sum(r_A[0::3])), int(np.sum(r_A[1::3])),
             int(np.sum(r_A[2::3])))
    uvw_B = (int(np.sum(r_B[0::3])), int(np.sum(r_B[1::3])),
             int(np.sum(r_B[2::3])))
    print(f"  Row 3-comp A: {uvw_A}, B: {uvw_B}")

    # Load 9-compression PSD results
    psd9_file = os.path.join(RESULTS_DIR, f"psd9_case{case_idx}.json")
    if not os.path.exists(psd9_file):
        print(f"  No 9-compression results for case {case_idx}")
        return None

    with open(psd9_file) as f:
        psd9_data = json.load(f)

    print(f"  Loaded {psd9_data['total_matches']} PSD matches from 9-compression")

    # For each 9-compression PSD match, find actual sequences
    # and check Gale-Ryser compatibility
    matches = psd9_data['sample_matches']
    compatible_count = 0

    for match in matches:
        psd_A_9 = tuple(match[0])
        psd_B_9 = tuple(match[1])

        # Find length-9 sequences with these PSDs
        # Need to try with the correct (u,v,w)
        seqs_A_9 = find_sequences_for_psd(
            uvw_A[0], uvw_A[1], uvw_A[2], psd_A_9)
        if not seqs_A_9:
            continue

        seqs_B_9 = find_sequences_for_psd(
            uvw_B[0], uvw_B[1], uvw_B[2], psd_B_9)
        if not seqs_B_9:
            continue

        # Check Gale-Ryser for each pair
        for c_A in seqs_A_9[:10]:
            for c_B in seqs_B_9[:10]:
                if (gale_ryser_check(list(r_A), list(c_A)) and
                        gale_ryser_check(list(r_B), list(c_B))):
                    compatible_count += 1
                    print(f"  COMPATIBLE marginals found!")
                    print(f"    col_A: {c_A}")
                    print(f"    col_B: {c_B}")

                    # Attempt SAT decompression
                    result = attempt_sat(c_A, r_A, c_B, r_B)
                    if result is not None:
                        return result

    print(f"  Checked all PSD matches, {compatible_count} Gale-Ryser compatible")
    return None


def attempt_sat(col_A, row_A, col_B, row_B, timeout=300):
    """Attempt SAT decompression and verify LP."""
    from hadamard.sat_complete import attempt_decompression

    print(f"  Attempting SAT decompression...")
    result = attempt_decompression(
        list(col_A), list(row_A), list(col_B), list(row_B),
        timeout=timeout)

    if result is not None:
        A, B = result
        is_lp, dev = check_lp(A, B)
        if is_lp:
            print(f"\n{'!'*70}")
            print(f"FOUND LEGENDRE PAIR OF LENGTH 333!")
            print(f"{'!'*70}")
            H, err = lp_to_hadamard(A, B)
            is_h, herr = verify_hadamard(H)
            print(f"Hadamard matrix order {H.shape[0]}, error {herr:.6e}")
            save_solution(A, B, H)
            return A, B, H
        else:
            print(f"  SAT solution does not satisfy LP (dev={dev:.6f})")
            print(f"  Need to add PSD constraints and retry (SAT+CAS)")
    return None


def save_solution(A, B, H):
    """Save a found LP(333) solution."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(os.path.join(RESULTS_DIR, "LP333_SOLUTION.npz"),
             A=np.array(A), B=np.array(B), H=np.array(H))
    with open(os.path.join(RESULTS_DIR, "LP333_SOLUTION.json"), 'w') as f:
        json.dump({'A': list(int(x) for x in A),
                   'B': list(int(x) for x in B)}, f)
    print(f"  Solution saved to {RESULTS_DIR}/LP333_SOLUTION.*")


def run_pipeline(case_idx=None, n_sa_trials=100, sa_max_iter=10_000_000):
    """Run the full pipeline for one or all macro-cases."""
    from hadamard.compression import get_macro_case_details

    details = get_macro_case_details()

    if case_idx is not None:
        cases = [case_idx]
    else:
        cases = list(range(len(details)))

    for ci in cases:
        P_A, P_B, reps_A, reps_B = details[ci]
        print(f"\n{'='*70}")
        print(f"PIPELINE: Case {ci}, PSD(A,111)={P_A}, PSD(B,111)={P_B}")
        print(f"{'='*70}", flush=True)

        # Get unique (u,v,w)
        seen_A = set()
        unique_A = []
        for _, _, u, v, w in reps_A:
            canon = tuple(sorted([u, v, w]))
            if canon not in seen_A:
                seen_A.add(canon)
                unique_A.append((u, v, w))

        seen_B = set()
        unique_B = []
        for _, _, u, v, w in reps_B:
            canon = tuple(sorted([u, v, w]))
            if canon not in seen_B:
                seen_B.add(canon)
                unique_B.append((u, v, w))

        for uvw_A in unique_A:
            for uvw_B in unique_B:
                print(f"\n  37-comp SA: A={uvw_A}, B={uvw_B}", flush=True)

                for trial in range(n_sa_trials):
                    seed = ci * 100000 + trial * 137 + 7
                    r_A, r_B, energy = joint_sa(
                        uvw_A, uvw_B,
                        max_iter=sa_max_iter,
                        seed=seed,
                        verbose=(trial % 10 == 0))

                    if energy == 0:
                        print(f"\n  SA trial {trial}: EXACT MATCH (energy=0)!")
                        result = try_intersect_and_decompress(
                            r_A, r_B, ci)
                        if result is not None:
                            return result
                    elif trial % 10 == 0:
                        print(f"  SA trial {trial}: best_energy={energy:.1f}")

    print("\nPipeline complete. No LP(333) found.")
    return None


if __name__ == "__main__":
    case = int(sys.argv[1]) if len(sys.argv) > 1 else None
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    sa_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000_000
    run_pipeline(case_idx=case, n_sa_trials=n_trials, sa_max_iter=sa_iters)
