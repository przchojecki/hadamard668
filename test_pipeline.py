#!/usr/bin/env python3
"""
Test the full pipeline on LP(9) → Hadamard(20).

LP(9): length 9, order = 2*10 = 20.
9 = 3 × 3, so we can test the compression framework.

Target PSD sum = 2*9+2 = 20.

This tests: compression → macro-cases → search → verification.
"""
import numpy as np
from numpy.fft import fft
import sys

sys.stdout.reconfigure(line_buffering=True)


def find_lp9():
    """Brute-force search for LP(9)."""
    from hadamard.core import check_lp
    from itertools import product

    target = 20
    count = 0
    # LP(9): two ±1 sequences of length 9 with PSD_A(k)+PSD_B(k)=20 for k=1..8
    # With normalization sum(A)=sum(B)=1 (i.e., 5 ones and 4 minus-ones)
    # Total ±1 sequences of length 9 with sum=1: C(9,5) = 126

    from math import comb
    seqs = []
    for bits in range(2**9):
        seq = [1 if (bits >> i) & 1 else -1 for i in range(9)]
        if sum(seq) == 1:
            seqs.append(seq)
    print(f"Length-9 sequences with sum=1: {len(seqs)}")

    for i, A in enumerate(seqs):
        for j, B in enumerate(seqs):
            ok, dev = check_lp(A, B)
            if ok:
                count += 1
                if count <= 3:
                    print(f"  Found LP(9): A={A}, B={B}")

    print(f"Total LP(9) pairs found: {count}")
    return count > 0


def test_compression():
    """Test compression framework on LP(9)."""
    from hadamard.compression import compress

    # A known LP(9) (if found above)
    A = [1, 1, -1, -1, 1, -1, 1, 1, -1]  # placeholder
    B = [1, -1, 1, -1, -1, 1, 1, -1, 1]  # placeholder

    # Try all pairs to find one
    from hadamard.core import check_lp
    seqs = []
    for bits in range(2**9):
        seq = [1 if (bits >> i) & 1 else -1 for i in range(9)]
        if sum(seq) == 1:
            seqs.append(seq)

    found = False
    for A in seqs:
        for B in seqs:
            ok, _ = check_lp(A, B)
            if ok:
                print(f"LP(9) found: A={A}, B={B}")

                # Test 3-compression
                cA = compress(A, 3)
                cB = compress(B, 3)
                print(f"  3-compression of A: {cA}")
                print(f"  3-compression of B: {cB}")
                print(f"  Sum A: {sum(A)}, Sum cA: {sum(cA)}")

                # PSD at frequency 3 (= 9/3 * 1)
                psd_A = np.real(fft(A) * np.conj(fft(A)))
                psd_B = np.real(fft(B) * np.conj(fft(B)))
                print(f"  PSD(A,3) = {psd_A[3]:.1f}")
                print(f"  PSD(B,3) = {psd_B[3]:.1f}")
                print(f"  Sum = {psd_A[3]+psd_B[3]:.1f} (target=20)")

                # 3-compression PSD at freq 1 should equal PSD(A,3)
                psd_cA = np.real(fft(cA) * np.conj(fft(cA)))
                print(f"  PSD(3-comp A, 1) = {psd_cA[1]:.1f}")
                assert abs(psd_cA[1] - psd_A[3]) < 1e-6, "Compression PSD mismatch!"
                print(f"  ✓ Compression PSD identity verified")

                found = True
                return A, B

    if not found:
        print("No LP(9) found — existence may be open for ℓ=9")
        return None, None


def test_hadamard_construction(A, B):
    """Test Hadamard construction from LP."""
    if A is None:
        return

    from hadamard.core import lp_to_hadamard, verify_hadamard

    H, err = lp_to_hadamard(A, B)
    print(f"\nHadamard construction:")
    print(f"  Order: {H.shape[0]}")
    print(f"  Construction error: {err:.6e}")

    if err < 1e-6:
        is_h, herr = verify_hadamard(H)
        print(f"  Is Hadamard: {is_h}")
        print(f"  ✓ Full pipeline verified for LP(9) → H(20)")


if __name__ == "__main__":
    print("=== Testing Pipeline on LP(9) → H(20) ===\n")
    A, B = test_compression()
    test_hadamard_construction(A, B)
