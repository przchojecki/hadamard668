"""
Compression framework for LP(333).

333 = 9 × 37, so we can view a length-333 sequence as a 37×9 matrix.
- Column sums → 9-compression (length 9, entries are sums of 37 ±1 values)
- Row sums → 37-compression (length 37, entries are sums of 9 ±1 values)

The p-compression of a length-ℓ sequence A (where p | ℓ) is the length-p
sequence C where C[j] = sum_{k ≡ j mod p} A[k].

Key property: PSD(A, (ℓ/p)·t) = PSD(C, t) for t = 0, ..., p-1.

For the 3-compression (length 3):
  PSD at frequency 111 is determined, giving 8 macro-cases.

Mod-3 obstruction: if any multiplier h ≡ 2 mod 3, then v = w in the
3-compression, forcing (u-v)^2 + (u'-v')^2 = 668, i.e., r^2 + s^2 = 167.
Since 167 ≡ 3 mod 4, this is impossible.
"""

import numpy as np
from itertools import product
from hadamard.core import psd, sum_of_a2_plus_3b2


ELL = 333
TARGET = 2 * ELL + 2  # = 668


def compress(seq, p):
    """
    Compute the p-compression of seq.
    Returns a length-p sequence where entry j = sum of seq[k] for k ≡ j mod p.
    """
    seq = np.array(seq)
    ell = len(seq)
    assert ell % p == 0
    c = np.zeros(p)
    for j in range(p):
        c[j] = np.sum(seq[j::p])
    return c


def compression_psd_frequencies(ell, p):
    """
    The p-compression preserves PSD at frequencies that are multiples of ℓ/p.
    Returns the list of original frequencies preserved.
    """
    step = ell // p
    return [step * t for t in range(p)]


def enumerate_macro_cases():
    """
    Enumerate the 8 feasible (PSD_A(111), PSD_B(111)) macro-cases.

    The 3-compression gives (u, v, w) with u + v + w = 1, all odd.
    PSD at frequency 1 of the 3-compression = a^2 + 3b^2 where
      a = (3u - 1)/2, b = (v - w)/2.

    We need PSD_A + PSD_B = 668, both values of the form a^2 + 3b^2,
    with a ≡ 1 mod 3 and valid entry ranges.
    """
    cases = []

    # For each P_A value, check if P_A and 668 - P_A are both
    # representable as a^2 + 3b^2 with valid compression vectors
    for P_A in range(0, TARGET + 1):
        P_B = TARGET - P_A
        if P_A > P_B:
            break  # Only need unordered pairs

        reps_A = _valid_representations(P_A)
        if not reps_A:
            continue

        reps_B = _valid_representations(P_B)
        if not reps_B:
            continue

        cases.append((P_A, P_B))

    return cases


def _valid_representations(P):
    """
    Find (a, b) such that P = a^2 + 3b^2, with a ≡ 1 mod 3,
    and the resulting (u, v, w) are valid 3-compression values.

    Each of u, v, w is a sum of 111 values from {±1}, so odd and |.| ≤ 111.
    Returns deduplicated list of (a, b, u, v, w) with distinct (u, v, w).
    """
    valid = []
    seen_uvw = set()
    reps = sum_of_a2_plus_3b2(P)
    for a, b in reps:
        # Consider both signs of a
        for a_sign in set([a, -a]):
            if (a_sign % 3 + 3) % 3 != 1:
                continue

            if (2 * a_sign + 1) % 3 != 0:
                continue
            u = (2 * a_sign + 1) // 3
            if u % 2 == 0:
                continue
            if abs(u) > 111:
                continue

            # v - w = 2b_val, v + w = 1 - u
            vw_sum = 1 - u
            for b_val in set([b, -b]):
                vw_diff = 2 * b_val
                if (vw_sum + vw_diff) % 2 != 0:
                    continue
                v = (vw_sum + vw_diff) // 2
                w = (vw_sum - vw_diff) // 2
                if v % 2 == 0 or w % 2 == 0:
                    continue
                if abs(v) > 111 or abs(w) > 111:
                    continue
                uvw = (u, v, w)
                if uvw not in seen_uvw:
                    seen_uvw.add(uvw)
                    valid.append((a_sign, b_val, u, v, w))

    return valid


def get_macro_case_details():
    """
    Return detailed information about each macro-case:
    (P_A, P_B, list of valid (u,v,w) for A, list of valid (u',v',w') for B)
    """
    cases = enumerate_macro_cases()
    detailed = []
    for P_A, P_B in cases:
        reps_A = _valid_representations(P_A)
        reps_B = _valid_representations(P_B)
        detailed.append((P_A, P_B, reps_A, reps_B))
    return detailed


def verify_mod3_obstruction():
    """
    Verify that 167 = r^2 + s^2 has no solution, proving the mod-3 obstruction.
    167 ≡ 3 mod 4, so it cannot be a sum of two squares.
    """
    assert 167 % 4 == 3, "167 should be 3 mod 4"
    for r in range(13):  # sqrt(167) < 13
        for s in range(r + 1):
            if r * r + s * s == 167:
                return False  # Would disprove the obstruction
    return True  # Obstruction confirmed


def get_9compression_constraints(macro_case_uvw):
    """
    Given a 3-compression (u, v, w), derive constraints on the 9-compression.

    The 9-compression (c_0, ..., c_8) must satisfy:
    - c_0 + c_3 + c_6 = u  (entries in residue class 0 mod 3)
    - c_1 + c_4 + c_7 = v  (entries in residue class 1 mod 3)
    - c_2 + c_5 + c_8 = w  (entries in residue class 2 mod 3)
    - Each c_j is odd, |c_j| ≤ 37
    - sum(c_j) = 1
    """
    u, v, w = macro_case_uvw
    return {
        'group_sums': [(u, [0, 3, 6]), (v, [1, 4, 7]), (w, [2, 5, 8])],
        'entry_bound': 37,
        'entry_parity': 'odd',
        'total_sum': 1,
    }


def get_37compression_constraints(macro_case_uvw):
    """
    Given a 3-compression (u, v, w), derive constraints on the 37-compression.

    The 37-compression (d_0, ..., d_36) groups entries by residue mod 37.
    Since 37 ≡ 1 mod 3, the residue classes mod 3 partition {0,...,36} as:
    - Class 0 mod 3: {0, 3, 6, ..., 36} → 13 entries (0,3,...,36 but 37/3≈12.33)
    Actually, we need: for each j in {0,...,36}, j mod 3 determines which
    3-compression group it maps to.

    The 37-compression entry d_j is a sum of 9 values from {±1}.
    For the 3-compression compatibility:
      sum_{j ≡ 0 mod 3} d_j = u
      sum_{j ≡ 1 mod 3} d_j = v
      sum_{j ≡ 2 mod 3} d_j = w

    Residue classes of {0,...,36} mod 3:
      0 mod 3: {0,3,6,9,12,15,18,21,24,27,30,33,36} → 13 elements
      1 mod 3: {1,4,7,10,13,16,19,22,25,28,31,34} → 12 elements
      2 mod 3: {2,5,8,11,14,17,20,23,26,29,32,35} → 12 elements
    """
    u, v, w = macro_case_uvw
    class0 = [j for j in range(37) if j % 3 == 0]  # 13 elements
    class1 = [j for j in range(37) if j % 3 == 1]  # 12 elements
    class2 = [j for j in range(37) if j % 3 == 2]  # 12 elements

    return {
        'group_sums': [(u, class0), (v, class1), (w, class2)],
        'entry_bound': 9,
        'entry_parity': 'odd',
        'total_sum': 1,
    }


def matrix_view_constraints(uvw_A, uvw_B):
    """
    Constraints for the 37×9 matrix view of LP(333).

    M[i][j] = A[9*i + j], i=0..36, j=0..8.
    - Column sums (sum over i for fixed j): 9-compression of A
    - Row sums (sum over j for fixed i): 37-compression of A
    - All entries ±1
    - Row sums are odd, |row_sum| ≤ 9
    - Column sums are odd, |col_sum| ≤ 37
    - Total sum = 1
    """
    return {
        'rows': 37,
        'cols': 9,
        'entries': [-1, 1],
        'row_sum_bound': 9,
        'col_sum_bound': 37,
        'total_sum': 1,
        'uvw_A': uvw_A,
        'uvw_B': uvw_B,
    }
