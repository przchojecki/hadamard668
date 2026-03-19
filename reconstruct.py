#!/usr/bin/env python3
"""
Given a matching PSD configuration from the length-9 compressed search,
reconstruct actual length-9 sequences, then attempt SAT decompression
to find the full LP(333).

Pipeline:
1. From PSD match → find actual length-9 compressed sequences (A, B)
2. Generate compatible length-37 compressed sequences via random search
3. Check Gale-Ryser compatibility
4. Attempt SAT decompression for compatible marginals
5. Verify LP and construct Hadamard matrix
"""

import numpy as np
from numpy.fft import fft
import json
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

from hadamard.search9 import find_sequences_for_psd, decode_psd_key
from hadamard.search37 import (
    compute_psd_full, stochastic_search_37, enumerate_profiles,
    generate_many_sequences, simulated_annealing_37
)
from hadamard.marginals import check_marginal_compatibility, gale_ryser_check
from hadamard.core import check_lp, lp_to_hadamard, verify_hadamard

TARGET = 668
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def reconstruct_from_psd_match(uvw_A, uvw_B, psd_A, psd_B, max_seqs=100):
    """
    Given a PSD match:
    - uvw_A: 3-compression of A
    - uvw_B: 3-compression of B
    - psd_A: (PSD(1), PSD(2), PSD(4)) for the 9-compressed A
    - psd_B: (PSD(1), PSD(2), PSD(4)) for the 9-compressed B

    Find actual length-9 compressed sequences and attempt completion.
    """
    print(f"\n  Finding A sequences with PSD={psd_A}, uvw={uvw_A}")
    seqs_A = find_sequences_for_psd(uvw_A[0], uvw_A[1], uvw_A[2], psd_A)
    print(f"    Found {len(seqs_A)} A sequences")

    print(f"  Finding B sequences with PSD={psd_B}, uvw={uvw_B}")
    seqs_B = find_sequences_for_psd(uvw_B[0], uvw_B[1], uvw_B[2], psd_B)
    print(f"    Found {len(seqs_B)} B sequences")

    if not seqs_A or not seqs_B:
        print("  No sequences found for this PSD match")
        return None

    # Limit to manageable number
    seqs_A = seqs_A[:max_seqs]
    seqs_B = seqs_B[:max_seqs]

    # For each pair (A_9comp, B_9comp), try to find compatible 37-compression
    for i, s9_A in enumerate(seqs_A[:5]):  # Try first 5
        for j, s9_B in enumerate(seqs_B[:5]):
            print(f"\n  Trying A seq {i}, B seq {j}:")
            print(f"    A_9: {s9_A}")
            print(f"    B_9: {s9_B}")

            # Verify PSD
            p_A = compute_psd_full(s9_A)
            p_B = compute_psd_full(s9_B)
            for k in range(1, 9):
                total = p_A[k] + p_B[k]
                if abs(total - TARGET) > 1.0:
                    print(f"    PSD mismatch at freq {k}: {total:.1f} ≠ {TARGET}")
                    break
            else:
                print(f"    ✓ PSD verified at all 8 nonzero frequencies")

                # Now search for compatible length-37 compressed pairs
                result = search_compatible_37(s9_A, s9_B, uvw_A, uvw_B)
                if result is not None:
                    return result

    return None


def search_compatible_37(s9_A, s9_B, uvw_A, uvw_B, n_trials=10000):
    """
    Given length-9 column sums and 3-compressions, search for compatible
    length-37 row sums.

    The row sums must:
    1. Be odd integers in [-9, 9]
    2. Sum to 1
    3. Have the correct 3-compression
    4. Satisfy Gale-Ryser with the column sums
    5. Have PSD_A(k) + PSD_B(k) = 668 at row frequencies
    """
    col_A = np.array(s9_A)
    col_B = np.array(s9_B)

    # Compute the sum-of-squares constraint for length-37
    # sum(c_A_j^2) + sum(c_B_j^2) = 650
    col_sq_A = np.sum(col_A ** 2)
    col_sq_B = np.sum(col_B ** 2)

    # For the 37-compression, Parseval gives:
    # 37 * sum(row_j^2) = sum_k PSD(k) = PSD(0) + sum_{k≠0} PSD(k)
    # PSD(0) = 1. sum_{k≠0} PSD(k) = 37*sum(row_j^2) - 1.
    # And for the 9-compression: 9 * sum(col_j^2) = sum_k PSD(k).
    # The overlap: at multiples of 37, PSD equals 9-compressed PSD.
    # At multiples of 9, PSD equals 37-compressed PSD.

    print(f"    Searching for compatible 37-compressed pairs...")
    print(f"    Column sums A: {list(col_A)}")
    print(f"    Column sums B: {list(col_B)}")

    # Use simulated annealing for a few trials
    for trial in range(3):
        print(f"\n    Trial {trial+1}: SA for A 37-compression")
        result_A, energy_A = simulated_annealing_37(
            uvw_A, max_iter=500000, verbose=False)
        print(f"      Best energy: {energy_A:.2f}")

        if result_A is not None and energy_A < 1000:
            # Check Gale-Ryser
            row_A = list(result_A)
            if gale_ryser_check(row_A, list(col_A)):
                print(f"      ✓ Gale-Ryser OK for A")

                # Now find compatible B
                p_A_37 = compute_psd_full(result_A)
                target_B_psd = [TARGET - p_A_37[k] for k in range(1, 19)]

                print(f"    SA for B 37-compression (target PSD)")
                result_B, energy_B = simulated_annealing_37(
                    uvw_B, target_psd_complement=list(p_A_37[1:19]),
                    max_iter=500000, verbose=False)
                print(f"      Best energy: {energy_B:.2f}")

                if result_B is not None and energy_B < 100:
                    if gale_ryser_check(list(result_B), list(col_B)):
                        print(f"      ✓ Gale-Ryser OK for B")
                        print(f"      Attempting SAT decompression...")

                        from hadamard.sat_complete import attempt_decompression
                        lp = attempt_decompression(
                            list(col_A), list(result_A),
                            list(col_B), list(result_B),
                            timeout=600)
                        if lp is not None:
                            return lp

    return None


def run_reconstruction(case_file, max_matches=100):
    """
    Load PSD matches from a search9 results file and attempt reconstruction.
    """
    with open(case_file) as f:
        data = json.load(f)

    case_idx = data['case']
    P_A = data['P_A']
    P_B = data['P_B']

    from hadamard.compression import get_macro_case_details
    details = get_macro_case_details()
    _, _, reps_A, reps_B = details[case_idx]

    print(f"Reconstructing from case {case_idx}: PSD(A,111)={P_A}, PSD(B,111)={P_B}")
    print(f"Total matches: {data['total_matches']}")

    # Get unique (u,v,w) triples
    seen_A = set()
    unique_A = {}
    for _, _, u, v, w in reps_A:
        canon = tuple(sorted([u, v, w]))
        if canon not in seen_A:
            seen_A.add(canon)
            unique_A[canon] = (u, v, w)

    seen_B = set()
    unique_B = {}
    for _, _, u, v, w in reps_B:
        canon = tuple(sorted([u, v, w]))
        if canon not in seen_B:
            seen_B.add(canon)
            unique_B[canon] = (u, v, w)

    matches = data['sample_matches']
    print(f"Processing {min(len(matches), max_matches)} sample matches...")

    for i, match in enumerate(matches[:max_matches]):
        psd_A = tuple(match[0])
        psd_B = tuple(match[1])

        print(f"\n{'='*50}")
        print(f"Match {i+1}: A PSD={psd_A}, B PSD={psd_B}")

        # Try each uvw combination
        for canon_A, uvw_A in unique_A.items():
            for canon_B, uvw_B in unique_B.items():
                result = reconstruct_from_psd_match(
                    uvw_A, uvw_B, psd_A, psd_B)
                if result is not None:
                    A, B = result
                    is_lp, dev = check_lp(A, B)
                    if is_lp:
                        print("\n" + "!" * 70)
                        print("FOUND LEGENDRE PAIR OF LENGTH 333!")
                        print("!" * 70)
                        H, err = lp_to_hadamard(A, B)
                        print(f"Hadamard matrix error: {err:.6e}")
                        save_solution(A, B, H)
                        return A, B, H

    print("\nNo solution found in this batch.")
    return None


def save_solution(A, B, H):
    """Save a found solution."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outfile = os.path.join(RESULTS_DIR, "LP333_SOLUTION.json")
    with open(outfile, 'w') as f:
        json.dump({
            'A': list(A) if hasattr(A, 'tolist') else A,
            'B': list(B) if hasattr(B, 'tolist') else B,
        }, f)
    print(f"Saved to {outfile}")

    np.savez(os.path.join(RESULTS_DIR, "LP333_SOLUTION.npz"),
             A=np.array(A), B=np.array(B), H=np.array(H))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_reconstruction(sys.argv[1])
    else:
        # Look for result files
        for f in sorted(os.listdir(RESULTS_DIR)):
            if f.startswith("psd9_case") and f.endswith(".json"):
                fpath = os.path.join(RESULTS_DIR, f)
                print(f"\nProcessing {f}...")
                result = run_reconstruction(fpath, max_matches=10)
                if result:
                    break
