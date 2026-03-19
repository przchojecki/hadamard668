"""
Search for feasible length-9 compressed pairs for LP(333).

The 9-compression of A has length 9, each entry is a sum of 37 ±1 values
(odd integer in [-37, 37]), total sum = 1.

Strategy: exploit the group structure. The 9 entries form 3 groups of 3
(by residue mod 3), with prescribed group sums from the macro-case.

The DFT at frequency k of the length-9 sequence decomposes as:
  Ĉ(k) = F_0(k) + ω^k F_1(k) + ω^{2k} F_2(k)
where ω = e^{2πi/9} and F_r(k) is the intra-group DFT contribution.

Key identity: F_r(1) = c_{3r} + c_{3r+3} ζ + c_{3r+6} ζ²  (ζ = e^{2πi/3})
  F_r(2) = conj(F_r(1)),  F_r(4) = F_r(1)

So PSD(1), PSD(2), PSD(4) are all determined by (F_0(1), F_1(1), F_2(1)).
PSD(3) is fixed by the macro-case.

PSD values are non-negative integers (elements of Z[ζ_9] have integer norms).

Algorithm:
1. For each group, enumerate ~1000 triples → compute F_r(1)
2. Vectorized loop: for each F_0, compute all PSD triples for F_1 × F_2
3. Collect distinct (PSD(1), PSD(2), PSD(4)) triples in a set
4. Match A and B sets by complement: P_A + P_B = 668
"""

import numpy as np
from collections import defaultdict
import sys
import time

ELL = 333
TARGET = 668
BOUND = 37  # max abs value of 9-compression entries


def enumerate_group_triples(group_sum, bound=BOUND):
    """
    All triples of odd integers in [-bound, bound] summing to group_sum.
    Returns list of (a, b, c) tuples.
    """
    results = []
    odds = list(range(-bound, bound + 1, 2))
    for a in odds:
        for b in odds:
            c = group_sum - a - b
            if abs(c) <= bound and c % 2 != 0:
                results.append((a, b, c))
    return results


def compute_group_F_values(triples, zeta):
    """
    For a list of group triples (a, b, c), compute F(1) = a + b*ζ + c*ζ².
    Returns complex numpy array.
    """
    triples = np.array(triples, dtype=np.float64)
    return triples[:, 0] + triples[:, 1] * zeta + triples[:, 2] * zeta ** 2


def fast_psd_catalog(u, v, w, bound=BOUND, verbose=True):
    """
    Enumerate all achievable (PSD(1), PSD(2), PSD(4)) triples for length-9
    compressed sequences with 3-compression (u, v, w).

    Returns a set of (int, int, int) PSD triples.
    """
    zeta = np.exp(2j * np.pi / 3)
    omega = np.exp(2j * np.pi / 9)
    omega_powers = [omega ** k for k in range(9)]

    # Enumerate group triples
    g0_triples = enumerate_group_triples(u, bound)
    g1_triples = enumerate_group_triples(v, bound)
    g2_triples = enumerate_group_triples(w, bound)

    if verbose:
        print(f"  Groups: {len(g0_triples)} × {len(g1_triples)} × "
              f"{len(g2_triples)} = {len(g0_triples)*len(g1_triples)*len(g2_triples)}")

    # Compute F_r(1) for each group
    F0 = compute_group_F_values(g0_triples, zeta)
    F1 = compute_group_F_values(g1_triples, zeta)
    F2 = compute_group_F_values(g2_triples, zeta)

    n0, n1, n2 = len(F0), len(F1), len(F2)

    # Precompute omega powers
    w1, w2, w4, w5, w7, w8 = (omega_powers[k] for k in [1, 2, 4, 5, 7, 8])

    # Precompute omega-scaled F values to avoid repeated multiplication
    wF1_1 = w1 * F1      # ω^1 * F1
    wF2_1 = w2 * F2      # ω^2 * F2
    cF1_2 = w2 * np.conj(F1)  # ω^2 * conj(F1)
    cF2_2 = w4 * np.conj(F2)  # ω^4 * conj(F2)
    wF1_4 = w4 * F1      # ω^4 * F1
    wF2_4 = w8 * F2      # ω^8 * F2

    # Collect all keys in chunks, use numpy for dedup
    all_keys_list = []
    t0 = time.time()
    chunk_size = 50  # Process in chunks for memory efficiency

    for i in range(n0):
        f0 = F0[i]
        f0c = np.conj(f0)

        C1 = f0 + wF1_1[:, None] + wF2_1[None, :]
        C2 = f0c + cF1_2[:, None] + cF2_2[None, :]
        C4 = f0 + wF1_4[:, None] + wF2_4[None, :]

        P1 = np.rint(np.real(C1 * np.conj(C1))).astype(np.int64).ravel()
        P2 = np.rint(np.real(C2 * np.conj(C2))).astype(np.int64).ravel()
        P4 = np.rint(np.real(C4 * np.conj(C4))).astype(np.int64).ravel()

        keys = P1 * 447561 + P2 * 669 + P4
        all_keys_list.append(keys)

        # Periodically deduplicate to control memory
        if (i + 1) % chunk_size == 0:
            merged = np.unique(np.concatenate(all_keys_list))
            all_keys_list = [merged]
            if verbose and (i + 1) % 200 == 0:
                dt = time.time() - t0
                print(f"    {i+1}/{n0} F0 values, ~{len(merged)} distinct, "
                      f"{dt:.1f}s")

    # Final dedup
    if all_keys_list:
        all_keys = np.unique(np.concatenate(all_keys_list))
    else:
        all_keys = np.array([], dtype=np.int64)

    if verbose:
        dt = time.time() - t0
        print(f"  Catalogued {len(all_keys)} distinct PSD triples in {dt:.1f}s")

    # Return as a set for O(1) lookup
    psd_set = set(all_keys.tolist())
    return psd_set


def decode_psd_key(key):
    """Decode a PSD key back to (P1, P2, P4)."""
    P4 = key % 669
    key //= 669
    P2 = key % 669
    P1 = key // 669
    return P1, P2, P4


def find_matching_pairs(psd_set_A, psd_set_B, verbose=True):
    """
    Find PSD triples that match: P_A(k) + P_B(k) = 668 for k = 1, 2, 4.
    """
    matches = []
    for key_A in psd_set_A:
        P1_A, P2_A, P4_A = decode_psd_key(key_A)
        P1_B = TARGET - P1_A
        P2_B = TARGET - P2_A
        P4_B = TARGET - P4_A
        if P1_B < 0 or P2_B < 0 or P4_B < 0:
            continue
        key_B = P1_B * 447561 + P2_B * 669 + P4_B
        if key_B in psd_set_B:
            matches.append(((P1_A, P2_A, P4_A), (P1_B, P2_B, P4_B)))

    if verbose:
        print(f"  Found {len(matches)} matching PSD pairs")
    return matches


def find_sequences_for_psd(u, v, w, target_psd, bound=BOUND):
    """
    Find actual length-9 sequences with 3-compression (u,v,w) and the
    given PSD triple (PSD(1), PSD(2), PSD(4)).

    Returns list of 9-tuples.
    """
    zeta = np.exp(2j * np.pi / 3)
    omega = np.exp(2j * np.pi / 9)

    g0_triples = enumerate_group_triples(u, bound)
    g1_triples = enumerate_group_triples(v, bound)
    g2_triples = enumerate_group_triples(w, bound)

    F0 = compute_group_F_values(g0_triples, zeta)
    F1 = compute_group_F_values(g1_triples, zeta)
    F2 = compute_group_F_values(g2_triples, zeta)

    target_P1, target_P2, target_P4 = target_psd
    results = []

    w1 = omega ** 1
    w2 = omega ** 2
    w4 = omega ** 4
    w8 = omega ** 8

    for i, f0 in enumerate(F0):
        f0c = np.conj(f0)

        C1 = f0 + w1 * F1[:, None] + w2 * F2[None, :]
        C2 = f0c + w2 * np.conj(F1[:, None]) + w4 * np.conj(F2[None, :])
        C4 = f0 + w4 * F1[:, None] + w8 * F2[None, :]

        P1 = np.rint(np.real(C1 * np.conj(C1))).astype(int)
        P2 = np.rint(np.real(C2 * np.conj(C2))).astype(int)
        P4 = np.rint(np.real(C4 * np.conj(C4))).astype(int)

        mask = (P1 == target_P1) & (P2 == target_P2) & (P4 == target_P4)
        indices = np.argwhere(mask)

        for j1, j2 in indices:
            g0 = g0_triples[i]
            g1 = g1_triples[j1]
            g2 = g2_triples[j2]
            # Interleave: positions 0,1,2,3,4,5,6,7,8
            # Group 0 → positions 0,3,6
            # Group 1 → positions 1,4,7
            # Group 2 → positions 2,5,8
            seq = [0] * 9
            seq[0], seq[3], seq[6] = g0
            seq[1], seq[4], seq[7] = g1
            seq[2], seq[5], seq[8] = g2
            results.append(tuple(seq))

    return results


def search_macro_case(case_idx, verbose=True):
    """
    Full search for 9-compressed pairs for a given macro-case.

    Returns list of matching PSD pairs and, optionally, actual sequences.
    """
    from hadamard.compression import get_macro_case_details

    details = get_macro_case_details()
    P_A, P_B, reps_A, reps_B = details[case_idx]

    if verbose:
        print(f"\nMacro-case {case_idx}: PSD(A,111)={P_A}, PSD(B,111)={P_B}")

    # Due to S_3 symmetry on (u,v,w), permutations of the 3-compression
    # give equivalent sequences (related by cyclic shifts of 3).
    # We can pick one representative per orbit.
    # The orbits are: all permutations of the sorted triple.
    # But PSD is NOT invariant under all permutations, only cyclic shifts
    # of the full length-9 sequence. A cyclic shift by 3 positions corresponds
    # to a cyclic permutation of the groups, which cyclically permutes (u,v,w).
    # So PSD is invariant under cyclic perms of (u,v,w) but NOT transpositions.

    # For the catalog, we need to be careful. Different (u,v,w) permutations
    # may give different PSD sets. But cyclic permutations of (u,v,w) give
    # the same PSD set (since cycling the groups = cycling the length-9 seq
    # by 3, which only rotates the DFT phases).

    # Extract unique (u,v,w) triples up to cyclic permutation
    seen_orbits_A = set()
    unique_A = []
    for _, _, u, v, w in reps_A:
        orbit = frozenset([(u, v, w), (v, w, u), (w, u, v)])
        if orbit not in seen_orbits_A:
            seen_orbits_A.add(orbit)
            unique_A.append((u, v, w))

    seen_orbits_B = set()
    unique_B = []
    for _, _, u, v, w in reps_B:
        orbit = frozenset([(u, v, w), (v, w, u), (w, u, v)])
        if orbit not in seen_orbits_B:
            seen_orbits_B.add(orbit)
            unique_B.append((u, v, w))

    if verbose:
        print(f"  Unique A triples (up to cyclic): {unique_A}")
        print(f"  Unique B triples (up to cyclic): {unique_B}")

    # Build PSD catalogs for each unique triple
    all_matches = []
    for uvw_A in unique_A:
        if verbose:
            print(f"\n  A triple: {uvw_A}")
        psd_A = fast_psd_catalog(*uvw_A, verbose=verbose)

        for uvw_B in unique_B:
            if verbose:
                print(f"  B triple: {uvw_B}")
            psd_B = fast_psd_catalog(*uvw_B, verbose=verbose)

            matches = find_matching_pairs(psd_A, psd_B, verbose=verbose)
            for psd_match_A, psd_match_B in matches:
                all_matches.append({
                    'uvw_A': uvw_A,
                    'uvw_B': uvw_B,
                    'psd_A': psd_match_A,
                    'psd_B': psd_match_B,
                })

    if verbose:
        print(f"\n  Total matching PSD configurations: {len(all_matches)}")

    return all_matches


def estimate_search_sizes():
    """Estimate the search space for each macro-case."""
    from hadamard.compression import get_macro_case_details

    details = get_macro_case_details()
    print("Macro-case search space estimates:")
    print(f"{'Case':>4} {'PSD_A':>6} {'PSD_B':>6} {'uvw_A':>20} "
          f"{'groups':>30}")
    print("-" * 90)

    for idx, (P_A, P_B, reps_A, reps_B) in enumerate(details):
        for a_info in reps_A[:1]:  # Just first representative
            _, _, u, v, w = a_info
            g0 = len(enumerate_group_triples(u))
            g1 = len(enumerate_group_triples(v))
            g2 = len(enumerate_group_triples(w))
            print(f"{idx:>4} {P_A:>6} {P_B:>6} ({u:>3},{v:>3},{w:>3})  "
                  f"{g0:>5} × {g1:>5} × {g2:>5} = {g0*g1*g2:>12,}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--estimate":
        estimate_search_sizes()
    elif len(sys.argv) > 1:
        case = int(sys.argv[1])
        search_macro_case(case)
    else:
        # Default: estimate then search case 0
        estimate_search_sizes()
