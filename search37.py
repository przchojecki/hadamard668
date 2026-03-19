"""
Search for feasible length-37 compressed pairs for LP(333).

The 37-compression of A has length 37, each entry is a sum of 9 ±1 values
(odd integer in [-9, 9]), total sum = 1.

The pair (C37_A, C37_B) must satisfy:
  PSD(C37_A, k) + PSD(C37_B, k) = 668  for k = 1, ..., 18
  (frequencies 19..36 are conjugate symmetric)

Key constraint: Parseval gives sum(c_A^2) + sum(c_B^2) = 650.

Entry values: {-9, -7, -5, -3, -1, 1, 3, 5, 7, 9} (10 possibilities each)

Strategy:
1. Enumerate valid "profiles" (count of entries with each abs value)
   constrained by sum and sum-of-squares
2. For each profile, use random generation + PSD filtering
3. Match A and B sequences by complement PSD
"""

import numpy as np
from numpy.fft import fft
from collections import defaultdict
import sys
import time

ELL = 333
TARGET = 668
LEN = 37
ODD_VALUES = np.array([-9, -7, -5, -3, -1, 1, 3, 5, 7, 9])
SQUARED_VALUES = np.array([1, 9, 25, 49, 81])  # |val|^2 for abs values 1,3,5,7,9
PARSEVAL_TARGET = 650  # sum(c_A^2) + sum(c_B^2) must equal this


def compute_psd_full(seq):
    """Compute PSD at all frequencies for a real sequence."""
    f = fft(np.array(seq, dtype=np.float64))
    return np.real(f * np.conj(f))


def psd_key_37(seq):
    """
    Compute PSD at independent frequencies 1..18 for a length-37 sequence.
    Returns tuple of rounded integers.
    """
    p = compute_psd_full(seq)
    return tuple(int(round(p[k])) for k in range(1, 19))


def enumerate_profiles(total_sum, sum_sq, n_entries=37):
    """
    Enumerate all valid "profiles": how many entries have each absolute value.

    n_1 + n_3 + n_5 + n_7 + n_9 = n_entries
    n_1*1 + n_3*9 + n_5*25 + n_7*49 + n_9*81 = sum_sq
    |signed_sum| and sum constraint handled separately.

    Returns list of (n_1, n_3, n_5, n_7, n_9) tuples.
    """
    profiles = []
    for n9 in range(min(n_entries, sum_sq // 81) + 1):
        rem9 = sum_sq - 81 * n9
        left9 = n_entries - n9
        for n7 in range(min(left9, rem9 // 49) + 1):
            rem7 = rem9 - 49 * n7
            left7 = left9 - n7
            for n5 in range(min(left7, rem7 // 25) + 1):
                rem5 = rem7 - 25 * n5
                left5 = left7 - n5
                for n3 in range(min(left5, rem5 // 9) + 1):
                    rem3 = rem5 - 9 * n3
                    n1 = left5 - n3
                    if n1 >= 0 and n1 * 1 == rem3:
                        profiles.append((n1, n3, n5, n7, n9))
    return profiles


def generate_random_sequence(profile, target_sum=1, max_attempts=1000):
    """
    Generate a random length-37 sequence with the given profile and target sum.

    profile: (n_1, n_3, n_5, n_7, n_9) counts
    target_sum: required sum of all entries
    """
    n1, n3, n5, n7, n9 = profile
    abs_vals = [1]*n1 + [3]*n3 + [5]*n5 + [7]*n7 + [9]*n9
    total = sum(abs_vals)

    for _ in range(max_attempts):
        signs = np.ones(37, dtype=int)
        np.random.shuffle(abs_vals_arr := list(abs_vals))

        # We need sum(signs * abs_vals) = target_sum
        # Start all positive: sum = total
        # Flipping sign of entry with value v changes sum by -2v
        # Need to reduce sum by (total - target_sum), must be even
        deficit = total - target_sum
        if deficit % 2 != 0:
            continue

        half_deficit = deficit // 2
        if half_deficit < 0 or half_deficit > total:
            continue

        # Greedily flip signs to reach target
        seq = list(abs_vals_arr)
        remaining = half_deficit
        indices = list(range(37))
        np.random.shuffle(indices)

        for idx in indices:
            v = seq[idx]
            if remaining >= v:
                seq[idx] = -v
                remaining -= v
            if remaining == 0:
                break

        if remaining != 0:
            continue

        # Shuffle positions
        np.random.shuffle(seq)
        return tuple(seq)

    return None


def generate_many_sequences(profile, target_sum=1, target_3comp=None,
                            n_seqs=10000, max_attempts_per=100):
    """Generate many random sequences with given constraints."""
    sequences = []
    for _ in range(n_seqs * max_attempts_per):
        if len(sequences) >= n_seqs:
            break
        seq = generate_random_sequence(profile, target_sum, max_attempts=1)
        if seq is None:
            continue

        # Check 3-compression constraint
        if target_3comp is not None:
            u, v, w = target_3comp
            s = np.array(seq)
            if (np.sum(s[0::3]) != u or np.sum(s[1::3]) != v or
                    np.sum(s[2::3]) != w):
                continue

        sequences.append(seq)

    return sequences


def stochastic_search_37(target_3comp, sum_sq_target, n_candidates=10000,
                         verbose=True):
    """
    Stochastic search for length-37 compressed sequences.

    Uses random generation with profile-based constraints and PSD computation.
    Returns dict mapping PSD key → list of sequences.
    """
    u, v, w = target_3comp
    profiles = enumerate_profiles(1, sum_sq_target, n_entries=37)

    if verbose:
        print(f"  sum_sq={sum_sq_target}: {len(profiles)} valid profiles")

    psd_catalog = defaultdict(list)
    total_found = 0

    for prof in profiles:
        seqs = generate_many_sequences(
            prof, target_sum=1, target_3comp=target_3comp,
            n_seqs=max(100, n_candidates // max(1, len(profiles))))

        for seq in seqs:
            key = psd_key_37(seq)
            psd_catalog[key].append(seq)
            total_found += 1

    if verbose:
        print(f"  Generated {total_found} sequences, "
              f"{len(psd_catalog)} distinct PSD classes")

    return psd_catalog


def search_37_pairs(target_3comp_A, target_3comp_B, n_candidates=50000,
                    verbose=True):
    """
    Search for compatible length-37 compressed pairs.

    Generates candidates for A and B, then matches by complement PSD.
    """
    if verbose:
        print(f"\nSearching 37-compressed pairs:")
        print(f"  A 3-comp: {target_3comp_A}")
        print(f"  B 3-comp: {target_3comp_B}")

    # Enumerate valid sum-of-squares splits
    pairs = []

    for sum_sq_A in range(37, PARSEVAL_TARGET - 37 + 1):
        sum_sq_B = PARSEVAL_TARGET - sum_sq_A

        # Check if valid profiles exist for both
        prof_A = enumerate_profiles(1, sum_sq_A)
        if not prof_A:
            continue
        prof_B = enumerate_profiles(1, sum_sq_B)
        if not prof_B:
            continue

        if verbose:
            print(f"\n  sum_sq split: A={sum_sq_A}, B={sum_sq_B}")

        cat_A = stochastic_search_37(
            target_3comp_A, sum_sq_A,
            n_candidates=n_candidates // 10, verbose=verbose)

        cat_B = stochastic_search_37(
            target_3comp_B, sum_sq_B,
            n_candidates=n_candidates // 10, verbose=verbose)

        # Match
        for key_A, seqs_A in cat_A.items():
            comp_key = tuple(TARGET - p for p in key_A)
            if comp_key in cat_B:
                for sa in seqs_A:
                    for sb in cat_B[comp_key]:
                        pairs.append((sa, sb))

    if verbose:
        print(f"\n  Total matching pairs: {len(pairs)}")

    return pairs


def simulated_annealing_37(target_3comp, target_psd_complement=None,
                           sum_sq=None, max_iter=1000000, verbose=True):
    """
    Use simulated annealing to find a length-37 sequence whose PSD
    best matches a target complement spectrum.

    target_psd_complement: if given, we want PSD_A(k) = 668 - target_psd_complement[k-1]
    """
    u, v, w = target_3comp

    # Initialize: random sequence with correct sum and 3-compression
    best_seq = None
    for _ in range(10000):
        seq = list(np.random.choice(ODD_VALUES, size=37))
        # Fix sum
        while sum(seq) != 1:
            idx = np.random.randint(37)
            old = seq[idx]
            new_val = int(np.random.choice(ODD_VALUES))
            seq[idx] = new_val
            if abs(sum(seq) - 1) > abs(sum(seq) - 1 + old - new_val):
                seq[idx] = old
        if sum(seq) == 1:
            s = np.array(seq)
            if (np.sum(s[0::3]) == u and np.sum(s[1::3]) == v and
                    np.sum(s[2::3]) == w):
                best_seq = seq
                break

    if best_seq is None:
        if verbose:
            print("  Could not initialize sequence with constraints")
        return None

    # Define energy: sum of squared deviations from target PSD
    def energy(seq):
        p = compute_psd_full(seq)
        if target_psd_complement is not None:
            targets = [TARGET - target_psd_complement[k-1] for k in range(1, 19)]
            return sum((p[k] - targets[k-1])**2 for k in range(1, 19))
        else:
            # Just minimize deviation from uniform PSD
            avg = (37 * sum(s**2 for s in seq) - 1) / 36
            return sum((p[k] - avg)**2 for k in range(1, 19))

    current = list(best_seq)
    current_energy = energy(current)
    best_energy = current_energy
    best = list(current)
    T = 100.0

    for it in range(max_iter):
        T = 100.0 * (1 - it / max_iter)
        if T < 0.01:
            T = 0.01

        # Propose: swap values at two positions (preserving constraints)
        i, j = np.random.choice(37, size=2, replace=False)
        # Swap must preserve 3-compression
        if i % 3 != j % 3:
            continue

        current[i], current[j] = current[j], current[i]
        new_energy = energy(current)

        if new_energy < current_energy or \
           np.random.random() < np.exp(-(new_energy - current_energy) / max(T, 0.01)):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best = list(current)
        else:
            current[i], current[j] = current[j], current[i]

        if verbose and (it + 1) % 100000 == 0:
            print(f"    iter {it+1}: energy={current_energy:.2f}, "
                  f"best={best_energy:.2f}, T={T:.2f}")

    if verbose:
        print(f"  Final best energy: {best_energy:.2f}")

    return tuple(best), best_energy


if __name__ == "__main__":
    from hadamard.compression import get_macro_case_details

    details = get_macro_case_details()
    P_A, P_B, reps_A, reps_B = details[0]
    _, _, u_A, v_A, w_A = reps_A[0]
    _, _, u_B, v_B, w_B = reps_B[0]

    print(f"Testing with macro-case 0: ({P_A}, {P_B})")
    print(f"A 3-comp: ({u_A}, {v_A}, {w_A})")
    print(f"B 3-comp: ({u_B}, {v_B}, {w_B})")

    # Quick profile enumeration
    for ss in [37, 100, 200, 300, 400, 500, 600]:
        profs = enumerate_profiles(1, ss)
        print(f"  sum_sq={ss}: {len(profs)} profiles")

    # Quick stochastic search
    print("\nStochastic search for A sequences:")
    cat = stochastic_search_37((u_A, v_A, w_A), 200, n_candidates=1000)
