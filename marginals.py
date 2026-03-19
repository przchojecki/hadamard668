"""
Fixed-marginals intersection for LP(333).

Given feasible 9-compressed pairs (column sums) and 37-compressed pairs (row sums),
find compatible (row_sums, col_sums) configurations for the 37×9 ±1 matrix.

A ±1 matrix of size 37×9 has:
- Row sums: length-37 vector, each entry odd in [-9, 9]
- Column sums: length-9 vector, each entry odd in [-37, 37]
- Grand total: sum of all entries = 1

Compatibility conditions:
- sum(row_sums) = sum(col_sums) = 1
- For each residue class mod 3:
    sum of row_sums at positions ≡ r mod 3 = sum of col_sums at positions ≡ r mod 3
    (both equal the r-th entry of the 3-compression)

After finding compatible marginal pairs, the question is whether a ±1 matrix
with those exact row and column sums exists. This is checked via a flow/matching
argument or SAT.
"""

import numpy as np
from itertools import product


def check_marginal_compatibility(row_sums, col_sums):
    """
    Check if row_sums (length 37) and col_sums (length 9) are compatible
    marginals for a 37×9 ±1 matrix.

    Basic necessary conditions:
    1. sum(row_sums) = sum(col_sums)
    2. Both sums equal 1
    3. Each row_sum is odd, |row_sum| ≤ 9
    4. Each col_sum is odd, |col_sum| ≤ 37
    5. The 3-compressions match
    """
    rs = np.array(row_sums)
    cs = np.array(col_sums)

    if np.sum(rs) != np.sum(cs):
        return False
    if np.sum(rs) != 1:
        return False
    if not np.all(np.abs(rs) <= 9):
        return False
    if not np.all(np.abs(cs) <= 37):
        return False
    if not np.all(rs % 2 != 0):
        return False
    if not np.all(cs % 2 != 0):
        return False

    # Check 3-compression compatibility
    # Row sums grouped by position mod 3:
    u_row = np.sum(rs[0::3])  # positions 0,3,6,...,36
    v_row = np.sum(rs[1::3])  # positions 1,4,7,...,34
    w_row = np.sum(rs[2::3])  # positions 2,5,8,...,35

    # Column sums grouped by position mod 3:
    u_col = np.sum(cs[0::3])  # positions 0,3,6
    v_col = np.sum(cs[1::3])  # positions 1,4,7
    w_col = np.sum(cs[2::3])  # positions 2,5,8

    if u_row != u_col or v_row != v_col or w_row != w_col:
        return False

    return True


def gale_ryser_check(row_sums, col_sums):
    """
    Check the Gale-Ryser condition for existence of a 0-1 matrix with given
    row and column sums. We convert from ±1 to 0-1 first.

    For a ±1 matrix M of size m×n with row sums r_i and column sums c_j:
    Let N = (M + J) / 2 where J is all-ones. Then N is 0-1 with
    row sums (r_i + n) / 2 and column sums (c_j + m) / 2.
    """
    m, n = len(row_sums), len(col_sums)

    # Convert to 0-1 marginals
    r01 = [(r + n) // 2 for r in row_sums]
    c01 = [(c + m) // 2 for c in col_sums]

    # Check: all values must be non-negative integers in valid range
    for r in r01:
        if r < 0 or r > n:
            return False
    for c in c01:
        if c < 0 or c > m:
            return False

    # Check sum
    if sum(r01) != sum(c01):
        return False

    # Gale-Ryser: sort row sums in decreasing order
    r_sorted = sorted(r01, reverse=True)
    c_sorted = sorted(c01, reverse=True)

    # Check: for each k = 1, ..., m:
    # sum_{i=1}^{k} r_sorted[i-1] <= sum_{j=1}^{n} min(c_sorted[j-1], k)
    for k in range(1, m + 1):
        lhs = sum(r_sorted[:k])
        rhs = sum(min(c, k) for c in c_sorted)
        if lhs > rhs:
            return False

    return True


def intersect_compressed_pairs(pairs_9, pairs_37):
    """
    Find compatible (9-compressed, 37-compressed) configurations.

    pairs_9: list of (C9_A, C9_B) tuples (each C9 is length 9)
    pairs_37: list of (C37_A, C37_B) tuples (each C37 is length 37)

    Returns list of ((C9_A, C37_A), (C9_B, C37_B)) compatible configurations.
    """
    compatible = []

    # Index pairs_9 by their 3-compression
    index_9 = {}
    for c9_A, c9_B in pairs_9:
        c9_A = np.array(c9_A)
        c9_B = np.array(c9_B)
        key_A = (int(np.sum(c9_A[0::3])), int(np.sum(c9_A[1::3])),
                 int(np.sum(c9_A[2::3])))
        key_B = (int(np.sum(c9_B[0::3])), int(np.sum(c9_B[1::3])),
                 int(np.sum(c9_B[2::3])))
        key = (key_A, key_B)
        if key not in index_9:
            index_9[key] = []
        index_9[key].append((tuple(c9_A), tuple(c9_B)))

    for c37_A, c37_B in pairs_37:
        c37_A = np.array(c37_A)
        c37_B = np.array(c37_B)
        key_A = (int(np.sum(c37_A[0::3])), int(np.sum(c37_A[1::3])),
                 int(np.sum(c37_A[2::3])))
        key_B = (int(np.sum(c37_B[0::3])), int(np.sum(c37_B[1::3])),
                 int(np.sum(c37_B[2::3])))
        key = (key_A, key_B)

        if key in index_9:
            for c9_A, c9_B in index_9[key]:
                # Check Gale-Ryser for both A and B matrices
                if (gale_ryser_check(list(c37_A), list(c9_A)) and
                        gale_ryser_check(list(c37_B), list(c9_B))):
                    compatible.append(((c9_A, tuple(c37_A)),
                                       (c9_B, tuple(c37_B))))

    return compatible
