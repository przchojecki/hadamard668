#!/usr/bin/env python3
"""
Fast joint simulated annealing for length-37 compressed LP pairs.

Search for (r_A, r_B) where each is a length-37 sequence of odd integers
in [-9, 9] with sum = 1 and prescribed 3-compression, such that
PSD(r_A, k) + PSD(r_B, k) = 668 for all k = 1, ..., 18.

Key optimization: incremental DFT updates. When changing one entry,
update all 18 DFT values in O(18) instead of O(37 log 37).
"""

import numpy as np
import sys
import time
import json
import os

TARGET = 668
LEN = 37
N_FREQ = 18  # independent nonzero frequencies

# Precompute twiddle factors
OMEGA = np.exp(2j * np.pi * np.arange(LEN) / LEN)  # ω^0, ω^1, ..., ω^36
# TWIDDLE[k][j] = ω^{jk} for freq k, position j
TWIDDLE = np.zeros((N_FREQ, LEN), dtype=complex)
for k in range(N_FREQ):
    for j in range(LEN):
        TWIDDLE[k][j] = np.exp(2j * np.pi * (k + 1) * j / LEN)

ODD_VALS = [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9]


def compute_dft_all(seq):
    """Compute DFT at frequencies 1..18 for a length-37 sequence."""
    seq = np.array(seq, dtype=np.float64)
    dft = np.zeros(N_FREQ, dtype=complex)
    for k in range(N_FREQ):
        dft[k] = np.sum(seq * TWIDDLE[k])
    return dft


def compute_psd_from_dft(dft):
    """PSD = |DFT|^2 at each frequency."""
    return np.real(dft * np.conj(dft))


def init_sequence(rng, target_sumsq=None):
    """
    Initialize a length-37 sequence with:
    - entries odd in [-9, 9]
    - sum = 1
    - sum of squares ≈ target_sumsq (if given)

    No 3-compression constraint (the 3-compression of the 37-compression
    is NOT the 3-compression of the original, since gcd(3,37)=1 but
    the position groupings differ).
    """
    if target_sumsq is not None:
        # Initialize with mostly ±1 and ±3 to approximate target sum of squares
        # avg d² = target_sumsq/37 ≈ 8.8 for target=325
        # Mix of ±1 (d²=1) and ±3 (d²=9): frac_3 = (avg-1)/8
        avg_sq = target_sumsq / LEN
        frac_3 = min(1.0, max(0.0, (avg_sq - 1) / 8))

        for _ in range(10000):
            vals = np.zeros(LEN, dtype=np.float64)
            for i in range(LEN):
                if rng.random() < frac_3:
                    vals[i] = rng.choice([-3, 3])
                else:
                    vals[i] = rng.choice([-1, 1])
            # Fix sum to 1
            current = int(np.sum(vals))
            diff = current - 1
            # Change entries to fix sum (each sign flip changes by ±2 or ±6)
            attempts = 0
            while diff != 0 and attempts < 200:
                idx = rng.integers(LEN)
                old = int(vals[idx])
                if diff > 0:
                    # Need to decrease sum
                    if old > -9:
                        new = old - 2
                        if new >= -9:
                            vals[idx] = new
                            diff -= 2
                else:
                    # Need to increase sum
                    if old < 9:
                        new = old + 2
                        if new <= 9:
                            vals[idx] = new
                            diff += 2
                attempts += 1

            if int(np.sum(vals)) == 1:
                return vals

    # Fallback: use general initialization
    return _init_group(LEN, 1, rng)


def _init_group(n, target, rng):
    """
    Generate n odd integers in [-9, 9] summing to target.
    target must have same parity as n (both odd or both even).
    """
    # target parity: n odd ints sum to n mod 2 ≡ target mod 2
    # (odd + odd = even, etc.)
    assert (n % 2) == (target % 2), \
        f"Parity mismatch: {n} entries can't sum to {target}"

    for _ in range(10000):
        # Generate n-1 random odd values, compute the last
        vals = np.array(rng.choice(ODD_VALS, size=n), dtype=np.float64)
        current = int(np.sum(vals))

        # Greedily adjust to hit target
        for attempt in range(n * 10):
            diff = current - target
            if diff == 0:
                return vals

            idx = rng.integers(n)
            old = int(vals[idx])
            # Want to change old to old - diff, but must stay in range and odd
            desired = old - diff
            # Clamp to [-9, 9]
            desired = max(-9, min(9, desired))
            # Make odd
            if desired % 2 == 0:
                desired += 1 if desired < 9 else -1
            if desired < -9 or desired > 9:
                continue

            vals[idx] = desired
            current = int(np.sum(vals))
            if current == target:
                return vals

    # Fallback: deterministic construction
    vals = np.ones(n, dtype=np.float64)  # sum = n
    current = n
    for i in range(n):
        diff = current - target
        if diff == 0:
            break
        # Reduce by changing to more negative values
        change = min(diff, 10)  # max change per entry: 1 - (-9) = 10
        new_val = int(vals[i]) - change
        if new_val < -9:
            new_val = -9
        if new_val % 2 == 0:
            new_val += 1
        actual_change = int(vals[i]) - new_val
        vals[i] = new_val
        current -= actual_change

    if int(np.sum(vals)) != target:
        raise RuntimeError(f"Cannot init group: n={n}, target={target}")
    return vals


def joint_sa(max_iter=5_000_000, seed=None, verbose=True):
    """
    Joint simulated annealing for (r_A, r_B) minimizing:
    E = sum_{k=1}^{18} (PSD_A(k) + PSD_B(k) - 668)^2

    No 3-compression constraint: the 3-compression of the 37-compression
    is NOT the 3-compression of the original (since gcd(3,37)=1 but
    the position groupings differ).

    Uses incremental DFT updates for efficiency.
    """
    rng = np.random.default_rng(seed)

    # Initialize with Parseval-compatible sum of squares
    # sum(d_A²) + sum(d_B²) = 650, so target ~325 each
    sumsq_A = rng.integers(200, 450)
    sumsq_B = 650 - sumsq_A
    r_A = init_sequence(rng, target_sumsq=sumsq_A)
    r_B = init_sequence(rng, target_sumsq=sumsq_B)

    # Compute initial DFTs
    dft_A = compute_dft_all(r_A)
    dft_B = compute_dft_all(r_B)

    psd_A = compute_psd_from_dft(dft_A)
    psd_B = compute_psd_from_dft(dft_B)

    def energy():
        return np.sum(np.abs(psd_A + psd_B - TARGET))

    current_energy = energy()
    best_energy = current_energy
    best_A = r_A.copy()
    best_B = r_B.copy()

    # Estimate typical move dE by doing a few test swaps
    test_dEs = []
    for _ in range(100):
        pi, pj = rng.choice(LEN, size=2, replace=False)
        if r_A[pi] != r_A[pj]:
            di = r_A[pj] - r_A[pi]
            dj = r_A[pi] - r_A[pj]
            nd = dft_A + di * TWIDDLE[:, pi] + dj * TWIDDLE[:, pj]
            np_ = np.real(nd * np.conj(nd))
            ne = np.sum(np.abs(np_ + psd_B - TARGET))
            test_dEs.append(abs(ne - current_energy))
    typical_dE = np.median(test_dEs) if test_dEs else current_energy * 0.1
    T_init = typical_dE * 2.0  # Start hot enough to accept ~50% of moves

    accepts = 0
    t0 = time.time()

    for it in range(max_iter):
        progress = it / max_iter
        T = max(0.1, T_init * (1.0 - progress) ** 1.5)

        # Choose which sequence to modify (A or B)
        if rng.random() < 0.5:
            seq, dft, psd_arr, label = r_A, dft_A, psd_A, 'A'
        else:
            seq, dft, psd_arr, label = r_B, dft_B, psd_B, 'B'

        # Choose move type
        move_type = rng.random()

        if move_type < 0.5:
            # Swap two entries (preserves sum)
            pos_i, pos_j = rng.choice(LEN, size=2, replace=False)

            if seq[pos_i] == seq[pos_j]:
                continue  # no effect

            delta_i = seq[pos_j] - seq[pos_i]
            delta_j = seq[pos_i] - seq[pos_j]

            new_dft = dft + delta_i * TWIDDLE[:, pos_i] + delta_j * TWIDDLE[:, pos_j]
            new_psd = np.real(new_dft * np.conj(new_dft))

            if label == 'A':
                new_energy = np.sum(np.abs(new_psd + psd_B - TARGET))
            else:
                new_energy = np.sum(np.abs(psd_A + new_psd - TARGET))

            dE = new_energy - current_energy
            if dE < 0 or rng.random() < np.exp(-dE / max(T, 0.01)):
                seq[pos_i], seq[pos_j] = seq[pos_j], seq[pos_i]
                dft[:] = new_dft
                psd_arr[:] = new_psd
                current_energy = new_energy
                accepts += 1

        else:
            # Change one entry and adjust another to preserve sum
            pos_i, pos_j = rng.choice(LEN, size=2, replace=False)

            old_i, old_j = int(seq[pos_i]), int(seq[pos_j])
            new_i = int(rng.choice(ODD_VALS))
            new_j = old_i + old_j - new_i  # preserve sum

            if new_j < -9 or new_j > 9 or new_j % 2 == 0:
                continue
            if new_i == old_i and new_j == old_j:
                continue

            delta_i = new_i - old_i
            delta_j = new_j - old_j

            new_dft = dft + delta_i * TWIDDLE[:, pos_i] + delta_j * TWIDDLE[:, pos_j]
            new_psd = np.real(new_dft * np.conj(new_dft))

            if label == 'A':
                new_energy = np.sum(np.abs(new_psd + psd_B - TARGET))
            else:
                new_energy = np.sum(np.abs(psd_A + new_psd - TARGET))

            dE = new_energy - current_energy
            if dE < 0 or rng.random() < np.exp(-dE / max(T, 0.01)):
                seq[pos_i] = new_i
                seq[pos_j] = new_j
                dft[:] = new_dft
                psd_arr[:] = new_psd
                current_energy = new_energy
                accepts += 1

        if current_energy < best_energy:
            best_energy = current_energy
            best_A[:] = r_A
            best_B[:] = r_B

        if best_energy == 0:
            if verbose:
                dt = time.time() - t0
                print(f"  EXACT MATCH at iteration {it+1}! ({dt:.1f}s)")
            return best_A.copy(), best_B.copy(), 0.0

        if verbose and (it + 1) % 500_000 == 0:
            dt = time.time() - t0
            rate = accepts / (it + 1) * 100
            print(f"  iter {it+1:>8,}: E={current_energy:.1f}, "
                  f"best={best_energy:.1f}, T={T:.1f}, "
                  f"accept={rate:.1f}%, {dt:.1f}s", flush=True)

    if verbose:
        dt = time.time() - t0
        print(f"  Final: best_energy={best_energy:.1f}, {dt:.1f}s")

    return best_A.copy(), best_B.copy(), best_energy


def run_sa_campaign(n_trials=20, max_iter=5_000_000, verbose=True):
    """Run multiple SA trials and collect results."""
    results = []
    for trial in range(n_trials):
        if verbose:
            print(f"\n--- Trial {trial+1}/{n_trials} ---", flush=True)

        r_A, r_B, energy = joint_sa(
            max_iter=max_iter,
            seed=trial * 12345 + 42, verbose=verbose)

        results.append({
            'trial': trial,
            'energy': energy,
            'r_A': r_A.astype(int).tolist(),
            'r_B': r_B.astype(int).tolist(),
        })

        if energy == 0:
            if verbose:
                print(f"\n*** FOUND EXACT LP-COMPATIBLE 37-COMPRESSION! ***")
            return results

    # Sort by energy
    results.sort(key=lambda x: x['energy'])

    if verbose:
        print(f"\nBest energies: {[r['energy'] for r in results[:5]]}")

    return results


def search_37(n_trials=50, max_iter=5_000_000):
    """
    Run SA campaign to find PSD-compatible 37-compressed LP pairs.
    No macro-case dependency — the SA directly enforces LP PSD at all
    18 independent row frequencies.
    """
    print(f"37-compression SA search: {n_trials} trials × {max_iter:,} iters")
    results = run_sa_campaign(n_trials=n_trials, max_iter=max_iter)

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    outfile = os.path.join(results_dir, "sa37_results.json")

    best = min(r['energy'] for r in results)
    with open(outfile, 'w') as f:
        json.dump({
            'best_energy': best,
            'n_exact': sum(1 for r in results if r['energy'] == 0),
            'results': sorted(results, key=lambda x: x['energy'])[:50],
        }, f)
    print(f"\nBest energy: {best:.1f}")
    print(f"Exact matches: {sum(1 for r in results if r['energy'] == 0)}")
    print(f"Saved to {outfile}")

    return results


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 5_000_000
    search_37(n_trials=n_trials, max_iter=max_iter)
