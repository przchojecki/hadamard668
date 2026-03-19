#!/usr/bin/env python3
"""
Main driver for the LP(333) → Hadamard(668) search.

Usage:
  python -m hadamard.run verify          # Verify math claims + enumerate macro-cases
  python -m hadamard.run estimate        # Estimate search space sizes
  python -m hadamard.run search9 CASE    # Search 9-compressed pairs for case CASE (0-7)
  python -m hadamard.run search37 CASE   # Search 37-compressed pairs for case CASE
  python -m hadamard.run intersect CASE  # Intersect compressed pairs for case CASE
  python -m hadamard.run decompress      # SAT decompression of surviving candidates
  python -m hadamard.run full            # Full pipeline
"""

import sys
import json
import os
import time
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def cmd_verify():
    from hadamard.verify import verify_all
    verify_all()


def cmd_estimate():
    from hadamard.search9 import estimate_search_sizes
    estimate_search_sizes()


def cmd_search9(case_idx):
    from hadamard.compression import get_macro_case_details
    from hadamard.search9 import search_compressed_pairs_for_case
    ensure_results_dir()

    details = get_macro_case_details()
    if case_idx < 0 or case_idx >= len(details):
        print(f"Invalid case index {case_idx}. Must be 0-{len(details)-1}")
        return

    P_A, P_B, reps_A, reps_B = details[case_idx]
    print(f"Searching 9-compressed pairs for macro-case {case_idx}: "
          f"({P_A}, {P_B})")

    all_pairs = []
    for a_info in reps_A:
        _, _, u_A, v_A, w_A = a_info
        for b_info in reps_B:
            _, _, u_B, v_B, w_B = b_info
            t0 = time.time()
            pairs = search_compressed_pairs_for_case(
                (u_A, v_A, w_A), (u_B, v_B, w_B))
            dt = time.time() - t0
            print(f"  ({u_A},{v_A},{w_A}) × ({u_B},{v_B},{w_B}): "
                  f"{len(pairs)} pairs in {dt:.1f}s")
            all_pairs.extend(pairs)

    # Save results
    outfile = os.path.join(RESULTS_DIR, f"pairs9_case{case_idx}.json")
    with open(outfile, 'w') as f:
        json.dump({
            'case': case_idx,
            'P_A': P_A, 'P_B': P_B,
            'n_pairs': len(all_pairs),
            'pairs': [(list(a), list(b)) for a, b in all_pairs[:10000]],
        }, f)
    print(f"\nSaved {min(len(all_pairs), 10000)} pairs to {outfile}")
    return all_pairs


def cmd_search37(case_idx):
    from hadamard.compression import get_macro_case_details
    from hadamard.search37 import search_37_backtrack, search_37_pairs_by_psd
    ensure_results_dir()

    details = get_macro_case_details()
    if case_idx < 0 or case_idx >= len(details):
        print(f"Invalid case index {case_idx}. Must be 0-{len(details)-1}")
        return

    P_A, P_B, reps_A, reps_B = details[case_idx]
    print(f"Searching 37-compressed pairs for macro-case {case_idx}: "
          f"({P_A}, {P_B})")

    all_pairs = []
    for a_info in reps_A:
        _, _, u_A, v_A, w_A = a_info
        for b_info in reps_B:
            _, _, u_B, v_B, w_B = b_info
            print(f"\n  3-comp A=({u_A},{v_A},{w_A}), B=({u_B},{v_B},{w_B})")

            t0 = time.time()
            seqs_A = search_37_backtrack(
                target_3comp=(u_A, v_A, w_A), max_solutions=50000)
            seqs_B = search_37_backtrack(
                target_3comp=(u_B, v_B, w_B), max_solutions=50000)
            dt = time.time() - t0
            print(f"  Generated {len(seqs_A)} A-seqs, {len(seqs_B)} B-seqs "
                  f"in {dt:.1f}s")

            if seqs_A and seqs_B:
                pairs = search_37_pairs_by_psd(seqs_A, seqs_B)
                all_pairs.extend(pairs)

    outfile = os.path.join(RESULTS_DIR, f"pairs37_case{case_idx}.json")
    with open(outfile, 'w') as f:
        json.dump({
            'case': case_idx,
            'P_A': P_A, 'P_B': P_B,
            'n_pairs': len(all_pairs),
            'pairs': [(list(a), list(b)) for a, b in all_pairs[:10000]],
        }, f)
    print(f"\nSaved {min(len(all_pairs), 10000)} pairs to {outfile}")
    return all_pairs


def cmd_intersect(case_idx):
    from hadamard.marginals import intersect_compressed_pairs
    ensure_results_dir()

    # Load precomputed compressed pairs
    file9 = os.path.join(RESULTS_DIR, f"pairs9_case{case_idx}.json")
    file37 = os.path.join(RESULTS_DIR, f"pairs37_case{case_idx}.json")

    if not os.path.exists(file9) or not os.path.exists(file37):
        print(f"Need precomputed pairs. Run search9 and search37 first.")
        return

    with open(file9) as f:
        data9 = json.load(f)
    with open(file37) as f:
        data37 = json.load(f)

    pairs_9 = [(tuple(a), tuple(b)) for a, b in data9['pairs']]
    pairs_37 = [(tuple(a), tuple(b)) for a, b in data37['pairs']]

    print(f"Intersecting {len(pairs_9)} 9-comp pairs with "
          f"{len(pairs_37)} 37-comp pairs")

    compatible = intersect_compressed_pairs(pairs_9, pairs_37)
    print(f"Found {len(compatible)} compatible marginal configurations")

    outfile = os.path.join(RESULTS_DIR, f"marginals_case{case_idx}.json")
    with open(outfile, 'w') as f:
        json.dump({
            'case': case_idx,
            'n_compatible': len(compatible),
            'configs': [
                {'col_A': list(ca), 'row_A': list(ra),
                 'col_B': list(cb), 'row_B': list(rb)}
                for (ca, ra), (cb, rb) in compatible[:1000]
            ],
        }, f)
    print(f"Saved to {outfile}")
    return compatible


def cmd_decompress(case_idx=None):
    from hadamard.sat_complete import attempt_decompression
    from hadamard.core import check_lp, lp_to_hadamard, verify_hadamard

    # Load marginal configurations
    if case_idx is not None:
        files = [os.path.join(RESULTS_DIR, f"marginals_case{case_idx}.json")]
    else:
        files = sorted(
            f for f in os.listdir(RESULTS_DIR)
            if f.startswith("marginals_case"))
        files = [os.path.join(RESULTS_DIR, f) for f in files]

    for mfile in files:
        if not os.path.exists(mfile):
            continue
        with open(mfile) as f:
            data = json.load(f)

        print(f"\nDecompressing {data['n_compatible']} configurations "
              f"from {mfile}")

        for i, config in enumerate(data['configs']):
            print(f"\n  Config {i+1}/{len(data['configs'])}:")
            result = attempt_decompression(
                config['col_A'], config['row_A'],
                config['col_B'], config['row_B'],
                timeout=600
            )

            if result is not None:
                A, B = result
                is_lp, dev = check_lp(A, B)
                if is_lp:
                    print("\n" + "!" * 70)
                    print("FOUND LEGENDRE PAIR OF LENGTH 333!")
                    print("!" * 70)
                    H, err = lp_to_hadamard(A, B)
                    is_h, herr = verify_hadamard(H)
                    print(f"Hadamard verification: {is_h} (err={herr:.6e})")

                    # Save the result
                    outfile = os.path.join(RESULTS_DIR, "LP333_SOLUTION.json")
                    with open(outfile, 'w') as f:
                        json.dump({
                            'A': A.tolist(),
                            'B': B.tolist(),
                            'H': H.tolist(),
                        }, f)
                    print(f"SAVED TO {outfile}")

                    # Also save as numpy
                    np.savez(
                        os.path.join(RESULTS_DIR, "LP333_SOLUTION.npz"),
                        A=A, B=B, H=H)
                    return A, B, H

    print("\nNo solution found in current batch.")
    return None


def cmd_full():
    """Run the full pipeline for all macro-cases."""
    from hadamard.compression import get_macro_case_details

    cmd_verify()

    details = get_macro_case_details()
    for case_idx in range(len(details)):
        print(f"\n\n{'#'*70}")
        print(f"# MACRO-CASE {case_idx}")
        print(f"{'#'*70}")

        pairs9 = cmd_search9(case_idx)
        if not pairs9:
            print(f"  No 9-compressed pairs for case {case_idx}, skipping")
            continue

        pairs37 = cmd_search37(case_idx)
        if not pairs37:
            print(f"  No 37-compressed pairs for case {case_idx}, skipping")
            continue

        compatible = cmd_intersect(case_idx)
        if not compatible:
            print(f"  No compatible marginals for case {case_idx}, skipping")
            continue

        result = cmd_decompress(case_idx)
        if result:
            return result

    print("\n\nFull search complete. No LP(333) found.")
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "verify":
        cmd_verify()
    elif cmd == "estimate":
        cmd_estimate()
    elif cmd == "search9":
        case = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        cmd_search9(case)
    elif cmd == "search37":
        case = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        cmd_search37(case)
    elif cmd == "intersect":
        case = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        cmd_intersect(case)
    elif cmd == "decompress":
        case = int(sys.argv[2]) if len(sys.argv) > 2 else None
        cmd_decompress(case)
    elif cmd == "full":
        cmd_full()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
