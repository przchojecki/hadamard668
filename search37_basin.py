#!/usr/bin/env python3
"""
Basin-hopping + SA for length-37 compressed LP pairs.

Improvements over plain SA:
1. Periodic reheating when stuck
2. Multi-entry perturbation moves (change 3+ entries)
3. Greedy frequency targeting (fix the worst-deviating frequency first)
4. Population of solutions with crossover
"""

import numpy as np
import sys
import time

TARGET = 668
LEN = 37
N_FREQ = 18

TWIDDLE = np.zeros((N_FREQ, LEN), dtype=complex)
for k in range(N_FREQ):
    for j in range(LEN):
        TWIDDLE[k][j] = np.exp(2j * np.pi * (k + 1) * j / LEN)

ODD_VALS = np.array([-9, -7, -5, -3, -1, 1, 3, 5, 7, 9])


def init_seq(rng, target_sumsq=325):
    """Init a length-37 odd-integer sequence with sum=1 and target sum of squares."""
    avg_sq = target_sumsq / LEN
    frac_3 = min(1.0, max(0.0, (avg_sq - 1) / 8))

    for _ in range(10000):
        vals = np.where(rng.random(LEN) < frac_3,
                        rng.choice([-3, 3], LEN),
                        rng.choice([-1, 1], LEN)).astype(np.float64)
        current = int(np.sum(vals))
        for _ in range(500):
            if current == 1:
                return vals
            idx = rng.integers(LEN)
            old = int(vals[idx])
            delta = 2 if current < 1 else -2
            new = old + delta
            if -9 <= new <= 9:
                vals[idx] = new
                current += delta

    raise RuntimeError("Cannot initialize sequence")


def compute_state(seq):
    """Compute DFT and PSD for a sequence."""
    dft = TWIDDLE @ seq.astype(np.float64)
    psd = np.real(dft * np.conj(dft))
    return dft, psd


def energy(psd_A, psd_B):
    """L1 energy: sum of absolute deviations."""
    return np.sum(np.abs(psd_A + psd_B - TARGET))


def basin_hopping_sa(max_iter=50_000_000, seed=None, verbose=True):
    """
    Basin-hopping SA with reheating and multi-entry moves.
    """
    rng = np.random.default_rng(seed)

    # Initialize
    sumsq = rng.integers(200, 450)
    r_A = init_seq(rng, target_sumsq=sumsq)
    r_B = init_seq(rng, target_sumsq=650 - sumsq)

    dft_A, psd_A = compute_state(r_A)
    dft_B, psd_B = compute_state(r_B)
    cur_E = energy(psd_A, psd_B)
    best_E = cur_E
    best_A = r_A.copy()
    best_B = r_B.copy()

    # Temperature calibration
    T_init = cur_E * 0.15
    T = T_init
    accepts = 0
    stale = 0  # iterations since last improvement
    phase_iters = max_iter // 20  # reheating phase length
    t0 = time.time()

    for it in range(max_iter):
        # Reheating: if stuck for too long, reheat
        if stale > phase_iters:
            T = T_init * 0.5
            stale = 0
            # Also do a large perturbation
            which = rng.choice(['A', 'B'])
            seq = r_A if which == 'A' else r_B
            # Shuffle 5 random entries
            idxs = rng.choice(LEN, size=min(5, LEN), replace=False)
            for idx in idxs:
                new_val = int(rng.choice(ODD_VALS))
                seq[idx] = new_val
            # Fix sum
            diff = int(np.sum(seq)) - 1
            for _ in range(100):
                if diff == 0:
                    break
                fix_idx = rng.integers(LEN)
                old = int(seq[fix_idx])
                new = old - (2 if diff > 0 else -2)
                if -9 <= new <= 9:
                    seq[fix_idx] = new
                    diff -= (2 if diff > 0 else -2)
            # Recompute state
            dft_A, psd_A = compute_state(r_A)
            dft_B, psd_B = compute_state(r_B)
            cur_E = energy(psd_A, psd_B)

        # Annealing schedule within phase
        phase_progress = min(1.0, stale / phase_iters)
        T = max(0.1, T_init * 0.5 * (1.0 - phase_progress) ** 1.5)

        # Choose sequence
        if rng.random() < 0.5:
            seq, dft, psd = r_A, dft_A, psd_A
            other_psd = psd_B
            is_A = True
        else:
            seq, dft, psd = r_B, dft_B, psd_B
            other_psd = psd_A
            is_A = False

        # Choose move
        move = rng.random()

        if move < 0.4:
            # Swap two entries
            i, j = rng.choice(LEN, size=2, replace=False)
            if seq[i] == seq[j]:
                stale += 1
                continue
            di, dj = seq[j] - seq[i], seq[i] - seq[j]
            new_dft = dft + di * TWIDDLE[:, i] + dj * TWIDDLE[:, j]

        elif move < 0.8:
            # Change one entry, adjust another
            i, j = rng.choice(LEN, size=2, replace=False)
            old_i, old_j = int(seq[i]), int(seq[j])
            new_i = int(rng.choice(ODD_VALS))
            new_j = old_i + old_j - new_i
            if new_j < -9 or new_j > 9 or new_j % 2 == 0:
                stale += 1
                continue
            if new_i == old_i:
                stale += 1
                continue
            di, dj = new_i - old_i, new_j - old_j
            new_dft = dft + di * TWIDDLE[:, i] + dj * TWIDDLE[:, j]

        else:
            # Targeted: find worst frequency, try to fix it
            devs = psd + other_psd - TARGET
            worst_k = np.argmax(np.abs(devs))

            # Pick a position that contributes most to this frequency
            contributions = np.abs(TWIDDLE[worst_k, :])  # all 1.0 for length-37
            i = rng.integers(LEN)
            j = rng.integers(LEN)
            while j == i:
                j = rng.integers(LEN)

            old_i, old_j = int(seq[i]), int(seq[j])
            # Try to reduce contribution at worst frequency
            # by changing to a value that shifts DFT toward target
            best_new_i = old_i
            best_dft = None
            best_de = float('inf')
            for ni in ODD_VALS:
                nj = old_i + old_j - ni
                if nj < -9 or nj > 9 or nj % 2 == 0:
                    continue
                di_t, dj_t = ni - old_i, nj - old_j
                if di_t == 0 and dj_t == 0:
                    continue
                nd = dft + di_t * TWIDDLE[:, i] + dj_t * TWIDDLE[:, j]
                np_ = np.real(nd * np.conj(nd))
                ne = np.sum(np.abs(np_ + other_psd - TARGET))
                if ne < best_de:
                    best_de = ne
                    best_new_i = ni
                    best_dft = nd

            if best_dft is None or best_de >= cur_E:
                stale += 1
                continue

            new_i = best_new_i
            new_j = old_i + old_j - new_i
            di, dj = new_i - old_i, new_j - old_j
            new_dft = best_dft

        new_psd = np.real(new_dft * np.conj(new_dft))
        new_E = np.sum(np.abs(new_psd + other_psd - TARGET))

        dE = new_E - cur_E
        if dE < 0 or rng.random() < np.exp(-dE / max(T, 0.01)):
            # Accept
            if move < 0.4:
                seq[i], seq[j] = seq[j], seq[i]
            else:
                seq[i] = new_i
                seq[j] = new_j
            dft[:] = new_dft
            psd[:] = new_psd
            cur_E = new_E
            accepts += 1

            if cur_E < best_E:
                best_E = cur_E
                best_A[:] = r_A
                best_B[:] = r_B
                stale = 0

                if best_E < 0.5:
                    if verbose:
                        dt = time.time() - t0
                        print(f"  FOUND E={best_E:.3f} at iter {it+1} ({dt:.1f}s)")
                    return best_A.copy(), best_B.copy(), best_E
            else:
                stale += 1
        else:
            stale += 1

        if verbose and (it + 1) % 2_000_000 == 0:
            dt = time.time() - t0
            rate = accepts / (it + 1) * 100
            print(f"  {it+1:>10,}: E={cur_E:.1f} best={best_E:.1f} "
                  f"T={T:.1f} acc={rate:.1f}% {dt:.0f}s", flush=True)

    if verbose:
        dt = time.time() - t0
        print(f"  Final: best={best_E:.1f}, {dt:.0f}s")

    return best_A.copy(), best_B.copy(), best_E


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000_000

    print(f"Basin-hopping SA: {n_trials} trials × {max_iter:,} iterations")
    best_overall = float('inf')

    for trial in range(n_trials):
        print(f"\n--- Trial {trial} ---", flush=True)
        r_A, r_B, E = basin_hopping_sa(max_iter=max_iter, seed=trial * 9973 + 1)
        print(f"  Trial {trial}: E={E:.1f}", flush=True)
        if E < best_overall:
            best_overall = E
        if E < 0.5:
            print(f"\n*** FOUND EXACT SOLUTION! ***")
            print(f"r_A = {r_A.astype(int).tolist()}")
            print(f"r_B = {r_B.astype(int).tolist()}")
            break

    print(f"\nBest energy: {best_overall:.1f}")
