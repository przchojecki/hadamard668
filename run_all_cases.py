#!/usr/bin/env python3
"""
Run the length-9 compressed search for all 8 macro-cases.
Outputs number of matching PSD triples per case.
"""
import sys
import time
import json
import os

sys.stdout.reconfigure(line_buffering=True)

from hadamard.compression import get_macro_case_details
from hadamard.search9 import fast_psd_catalog, find_matching_pairs

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

details = get_macro_case_details()
summary = []

for idx, (P_A, P_B, reps_A, reps_B) in enumerate(details):
    print(f"\n{'='*60}")
    print(f"MACRO-CASE {idx}: PSD(A,111)={P_A}, PSD(B,111)={P_B}")
    print(f"{'='*60}", flush=True)

    # Get unique (u,v,w) up to cyclic permutation
    # For A: since transposing v,w gives same PSD set, we only need
    # one representative per S_3 orbit (not just cyclic orbit)
    seen_A = set()
    unique_A = []
    for _, _, u, v, w in reps_A:
        canon = tuple(sorted([u, v, w]))  # canonical form
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

    print(f"  Unique A triples (up to S3): {unique_A}")
    print(f"  Unique B triples (up to S3): {unique_B}", flush=True)

    total_matches = 0
    case_matches = []

    for uvw_A in unique_A:
        print(f"\n  Cataloguing A: {uvw_A}", flush=True)
        t0 = time.time()
        psd_A = fast_psd_catalog(*uvw_A, verbose=True)
        print(f"  A: {len(psd_A)} PSD triples in {time.time()-t0:.1f}s", flush=True)

        for uvw_B in unique_B:
            print(f"  Cataloguing B: {uvw_B}", flush=True)
            t0 = time.time()
            psd_B = fast_psd_catalog(*uvw_B, verbose=True)
            print(f"  B: {len(psd_B)} PSD triples in {time.time()-t0:.1f}s", flush=True)

            matches = find_matching_pairs(psd_A, psd_B, verbose=True)
            total_matches += len(matches)
            if matches:
                case_matches.extend(matches[:100])  # Keep sample

    result = {
        'case': idx,
        'P_A': P_A,
        'P_B': P_B,
        'total_matches': total_matches,
        'sample_matches': case_matches[:100],
    }
    summary.append(result)

    outfile = os.path.join(RESULTS_DIR, f"psd9_case{idx}.json")
    with open(outfile, 'w') as f:
        json.dump(result, f)
    print(f"\n  CASE {idx} TOTAL: {total_matches} matching PSD configurations")
    print(f"  Saved to {outfile}", flush=True)

# Print summary
print(f"\n\n{'='*60}")
print("SUMMARY: Length-9 compressed PSD matching")
print(f"{'='*60}")
print(f"{'Case':>4} {'PSD_A':>6} {'PSD_B':>6} {'Matches':>12}")
print("-" * 35)
grand_total = 0
for r in summary:
    print(f"{r['case']:>4} {r['P_A']:>6} {r['P_B']:>6} {r['total_matches']:>12,}")
    grand_total += r['total_matches']
print("-" * 35)
print(f"{'':>4} {'':>6} {'TOTAL':>6} {grand_total:>12,}")
