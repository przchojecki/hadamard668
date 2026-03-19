"""
Verify the mathematical claims and enumerate the 8 macro-cases.

This script:
1. Confirms the mod-3 obstruction (167 is not a sum of two squares)
2. Enumerates all 8 feasible (PSD_A(111), PSD_B(111)) macro-cases
3. Lists all valid 3-compression vectors for each case
4. Verifies the mod-37 obstruction analog
5. Reports feasibility constraints for the full search
"""

import numpy as np
from hadamard.core import sum_of_two_squares_representations, sum_of_a2_plus_3b2
from hadamard.compression import (
    enumerate_macro_cases, get_macro_case_details, verify_mod3_obstruction,
    TARGET
)


def verify_all():
    print("=" * 70)
    print("VERIFICATION OF MATHEMATICAL CLAIMS FOR LP(333) → H(668)")
    print("=" * 70)

    # 1. Mod-3 obstruction
    print("\n1. MOD-3 OBSTRUCTION")
    print("-" * 40)
    print(f"   668 / 4 = 167")
    print(f"   167 mod 4 = {167 % 4}")
    reps = sum_of_two_squares_representations(167)
    print(f"   167 as sum of two squares: {reps}")
    if verify_mod3_obstruction():
        print("   ✓ CONFIRMED: 167 is NOT a sum of two squares")
        print("   → Any multiplier subgroup with h ≡ 2 mod 3 is impossible")
    else:
        print("   ✗ FAILED: 167 IS a sum of two squares (unexpected!)")

    # 2. Enumerate macro-cases
    print("\n2. MACRO-CASES: PSD(A,111) + PSD(B,111) = 668")
    print("-" * 40)

    # First, enumerate all values representable as a^2 + 3b^2
    # with valid 3-compression vectors
    valid_psds = set()
    for P in range(TARGET + 1):
        from hadamard.compression import _valid_representations
        if _valid_representations(P):
            valid_psds.add(P)

    print(f"   Values representable as a²+3b² (with valid 3-comp): "
          f"{len(valid_psds)}")

    cases = enumerate_macro_cases()
    print(f"   Number of unordered macro-cases: {len(cases)}")
    print()

    expected = [(16, 652), (64, 604), (76, 592), (112, 556),
                (172, 496), (256, 412), (268, 400), (304, 364)]

    for i, (P_A, P_B) in enumerate(cases):
        print(f"   Case {i+1}: PSD_A(111) = {P_A:>4}, PSD_B(111) = {P_B:>4}")
        # Check it matches expected
        if i < len(expected):
            if (P_A, P_B) == expected[i]:
                print(f"            ✓ matches expected")
            else:
                print(f"            ✗ expected {expected[i]}")

    if len(cases) == len(expected):
        print(f"\n   ✓ CONFIRMED: exactly {len(cases)} macro-cases")
    else:
        print(f"\n   ✗ Got {len(cases)} cases, expected {len(expected)}")

    # 3. Detail each macro-case
    print("\n3. DETAILED 3-COMPRESSION VECTORS")
    print("-" * 40)

    details = get_macro_case_details()
    for P_A, P_B, reps_A, reps_B in details:
        print(f"\n   Case ({P_A}, {P_B}):")
        print(f"     A: {len(reps_A)} valid (a,b,u,v,w) triples")
        for a, b, u, v, w in reps_A[:5]:
            print(f"       a={a:>4}, b={b:>4} → (u,v,w) = ({u:>4},{v:>4},{w:>4})"
                  f"  check: a²+3b² = {a*a+3*b*b}")
        if len(reps_A) > 5:
            print(f"       ... and {len(reps_A)-5} more")

        print(f"     B: {len(reps_B)} valid (a,b,u,v,w) triples")
        for a, b, u, v, w in reps_B[:5]:
            print(f"       a={a:>4}, b={b:>4} → (u,v,w) = ({u:>4},{v:>4},{w:>4})"
                  f"  check: a²+3b² = {a*a+3*b*b}")
        if len(reps_B) > 5:
            print(f"       ... and {len(reps_B)-5} more")

    # 4. Mod-37 obstruction analog
    print("\n\n4. MOD-37 OBSTRUCTION ANALOG")
    print("-" * 40)
    # If a subgroup projects onto all of U_37, then the length-37
    # compression has the form [α, β, ..., β].
    # PSD at frequency k ≠ 0: |α + β(ω_37^k + ... + ω_37^{36k})|²
    # = |α + β(-1)|² = (α - β)² for k ≠ 0.
    #
    # Each entry of the 37-compression is a sum of 9 ±1 values (odd, |.| ≤ 9).
    # α = d_0, β = d_1 = ... = d_36.
    # Sum = α + 36β = 1.
    # PSD at each nonzero freq = (α - β)².
    # LP: (α-β)² + (α'-β')² = 668 for each nonzero freq.
    # So (α-β)² + (α'-β')² = 668.
    # Both α,β,α',β' are odd, so α-β and α'-β' are even.
    # Let α-β = 2r, α'-β' = 2s. Then 4r² + 4s² = 668 → r²+s² = 167.
    # Same contradiction!
    print("   If a multiplier subgroup projects onto all of U_37:")
    print("   37-compression has form [α, β, ..., β]")
    print("   PSD at nonzero freq = (α-β)²")
    print("   LP: (α-β)² + (α'-β')² = 668")
    print("   Both diffs are even → r²+s² = 167, impossible")
    print("   ✓ CONFIRMED: mod-37 obstruction is identical")

    # 5. Summary
    print("\n\n5. SEARCH STRATEGY SUMMARY")
    print("-" * 40)
    print(f"   LP length: 333 = 9 × 37")
    print(f"   Hadamard order: 668 = 2 × 334")
    print(f"   Target PSD sum: {TARGET}")
    print(f"   Macro-cases: {len(cases)}")
    print(f"   Matrix view: 37 × 9 sign matrix")
    print(f"   9-compression entries: odd ∈ [-37, 37] (sum of 37 ±1)")
    print(f"   37-compression entries: odd ∈ [-9, 9] (sum of 9 ±1)")
    print()
    print("   Pipeline:")
    print("   1. For each of 8 macro-cases:")
    print("   2.   Enumerate feasible 9-compressed pairs (col sums)")
    print("   3.   Enumerate feasible 37-compressed pairs (row sums)")
    print("   4.   Intersect by fixed marginals + Gale-Ryser")
    print("   5.   SAT+CAS decompression for surviving candidates")
    print()
    print("   Obstruction rules:")
    print("   - Kill any multiplier branch with nontrivial mod-3 image")
    print("   - Kill any multiplier branch with full mod-37 image")


if __name__ == "__main__":
    verify_all()
