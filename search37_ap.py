#!/usr/bin/env python3
"""
Alternating-projection search for length-37 compressed LP pairs.

Instead of SA on the energy landscape, alternate between:
1. Spectral projection: enforce PSD_A(k) + PSD_B(k) = 668 at each frequency
   (scale magnitudes, keep phases)
2. Integer projection: round entries to nearest odd integer in [-9,9], fix sum

This is a Gerchberg-Saxton-style algorithm adapted for the LP constraint.
Much faster convergence than SA for this structured problem.
"""

import numpy as np
from numpy.fft import fft, ifft
import sys
import time

TARGET = 668
LEN = 37
N_FREQ = 18  # independent nonzero frequencies
ODD_VALS = np.array([-9, -7, -5, -3, -1, 1, 3, 5, 7, 9])


def project_to_integers(x):
    """
    Project a real sequence to odd integers in [-9, 9] with sum = 1.
    """
    # Round to nearest odd integer in [-9, 9]
    rounded = np.round((x - 1) / 2) * 2 + 1  # nearest odd
    rounded = np.clip(rounded, -9, 9)
    # Make sure they're odd (clip might have made them even at boundaries)
    rounded = np.where(rounded % 2 == 0, rounded + 1, rounded)
    rounded = np.clip(rounded, -9, 9)
    # Ensure odd after final clip
    rounded = np.where(rounded % 2 == 0, rounded - 1, rounded)

    # Fix sum to 1
    current = int(np.sum(rounded))
    while current != 1:
        diff = current - 1
        if diff > 0:
            # Need to decrease: find entry we can decrease by 2
            candidates = np.where(rounded > -9)[0]
            if len(candidates) == 0:
                break
            idx = candidates[np.random.randint(len(candidates))]
            rounded[idx] -= 2
            current -= 2
        else:
            candidates = np.where(rounded < 9)[0]
            if len(candidates) == 0:
                break
            idx = candidates[np.random.randint(len(candidates))]
            rounded[idx] += 2
            current += 2

    return rounded


def spectral_project(F_A, F_B):
    """
    Project DFT vectors to satisfy PSD_A(k) + PSD_B(k) = 668 at each
    nonzero frequency, while keeping phases unchanged.

    F_A, F_B: length-37 complex DFT vectors.
    F[0] = 1 (sum constraint).
    F[k] and F[37-k] = conj(F[k]) for real sequences.
    """
    F_A_new = F_A.copy()
    F_B_new = F_B.copy()

    # Keep DC component (sum = 1)
    F_A_new[0] = 1.0
    F_B_new[0] = 1.0

    for k in range(1, N_FREQ + 1):
        mag_A = np.abs(F_A[k])
        mag_B = np.abs(F_B[k])

        if mag_A < 1e-10 and mag_B < 1e-10:
            # Both near zero — split target evenly
            F_A_new[k] = np.sqrt(TARGET / 2)
            F_B_new[k] = np.sqrt(TARGET / 2)
        else:
            # Scale magnitudes to satisfy |F_A|^2 + |F_B|^2 = TARGET
            total_sq = mag_A ** 2 + mag_B ** 2
            if total_sq < 1e-10:
                scale = 1.0
            else:
                scale = np.sqrt(TARGET / total_sq)
            F_A_new[k] = F_A[k] * scale
            F_B_new[k] = F_B[k] * scale

        # Enforce conjugate symmetry
        F_A_new[LEN - k] = np.conj(F_A_new[k])
        F_B_new[LEN - k] = np.conj(F_B_new[k])

    return F_A_new, F_B_new


def ap_search(max_iter=10000, seed=None, verbose=False):
    """
    Alternating projection search for LP-compatible 37-compressed pairs.

    Returns (r_A, r_B, energy) where energy = sum |PSD_A + PSD_B - 668|.
    """
    rng = np.random.default_rng(seed)

    # Initialize with random odd-integer sequences
    r_A = np.zeros(LEN)
    r_B = np.zeros(LEN)
    for i in range(LEN):
        r_A[i] = rng.choice(ODD_VALS)
        r_B[i] = rng.choice(ODD_VALS)
    r_A = project_to_integers(r_A)
    r_B = project_to_integers(r_B)

    best_energy = float('inf')
    best_A = r_A.copy()
    best_B = r_B.copy()

    for it in range(max_iter):
        # Forward DFT
        F_A = fft(r_A)
        F_B = fft(r_B)

        # Spectral projection: enforce PSD constraint
        F_A, F_B = spectral_project(F_A, F_B)

        # Inverse DFT
        r_A_cont = np.real(ifft(F_A))
        r_B_cont = np.real(ifft(F_B))

        # Integer projection: round to valid entries
        r_A = project_to_integers(r_A_cont)
        r_B = project_to_integers(r_B_cont)

        # Check energy
        psd_A = np.abs(fft(r_A)) ** 2
        psd_B = np.abs(fft(r_B)) ** 2
        energy = np.sum(np.abs(psd_A[1:N_FREQ+1] + psd_B[1:N_FREQ+1] - TARGET))

        if energy < best_energy:
            best_energy = energy
            best_A = r_A.copy()
            best_B = r_B.copy()

        if energy < 0.5:
            return best_A, best_B, energy

    return best_A, best_B, best_energy


def ap_with_phase_perturbation(max_outer=100, max_inner=200, seed=None,
                                verbose=True):
    """
    AP with periodic phase perturbations to escape fixed points.

    The basic AP can get stuck at fixed points of the alternating projection.
    We periodically perturb the DFT phases to escape.
    """
    rng = np.random.default_rng(seed)
    best_overall = float('inf')
    best_A_overall = None
    best_B_overall = None

    for outer in range(max_outer):
        r_A, r_B, energy = ap_search(
            max_iter=max_inner, seed=rng.integers(10**9))

        if energy < best_overall:
            best_overall = energy
            best_A_overall = r_A.copy()
            best_B_overall = r_B.copy()

        if verbose and (outer + 1) % 20 == 0:
            print(f"  AP outer {outer+1}: E={energy:.1f}, best={best_overall:.1f}",
                  flush=True)

        if best_overall < 0.5:
            if verbose:
                print(f"  FOUND at outer iteration {outer+1}!")
            return best_A_overall, best_B_overall, best_overall

    return best_A_overall, best_B_overall, best_overall


def hybrid_ap_sa(max_iter=5_000_000, seed=None, verbose=True):
    """
    Hybrid: use AP to get close, then SA to refine.

    Strategy:
    1. Run many AP restarts to find good starting points
    2. Use the best AP result as SA starting point
    3. SA with small moves to polish
    """
    rng = np.random.default_rng(seed)

    # Phase 1: AP search (fast, many restarts)
    if verbose:
        print("  Phase 1: AP search (1000 restarts × 500 iterations)...",
              flush=True)
    t0 = time.time()

    best_E = float('inf')
    best_A = None
    best_B = None

    for i in range(1000):
        r_A, r_B, E = ap_search(max_iter=500, seed=rng.integers(10**9))
        if E < best_E:
            best_E = E
            best_A = r_A.copy()
            best_B = r_B.copy()

    dt = time.time() - t0
    if verbose:
        print(f"  AP phase done: best E={best_E:.1f} in {dt:.1f}s", flush=True)

    if best_E < 0.5:
        return best_A, best_B, best_E

    # Phase 2: SA refinement from best AP point
    if verbose:
        print(f"  Phase 2: SA refinement from E={best_E:.1f}...", flush=True)

    from hadamard.search37_fast import TWIDDLE, N_FREQ

    r_A = best_A.astype(np.float64)
    r_B = best_B.astype(np.float64)
    dft_A = TWIDDLE @ r_A
    dft_B = TWIDDLE @ r_B
    psd_A = np.real(dft_A * np.conj(dft_A))
    psd_B = np.real(dft_B * np.conj(dft_B))
    cur_E = np.sum(np.abs(psd_A + psd_B - TARGET))

    T_init = cur_E * 0.1
    sa_best_E = cur_E
    sa_best_A = r_A.copy()
    sa_best_B = r_B.copy()
    accepts = 0
    t0 = time.time()

    ODD = [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9]

    for it in range(max_iter):
        progress = it / max_iter
        T = max(0.01, T_init * (1.0 - progress) ** 1.5)

        # Choose sequence
        if rng.random() < 0.5:
            seq, dft, psd, other_psd = r_A, dft_A, psd_A, psd_B
        else:
            seq, dft, psd, other_psd = r_B, dft_B, psd_B, psd_A

        # Move: change one entry, adjust another
        i, j = rng.choice(LEN, size=2, replace=False)
        old_i, old_j = int(seq[i]), int(seq[j])
        new_i = int(rng.choice(ODD))
        new_j = old_i + old_j - new_i
        if new_j < -9 or new_j > 9 or new_j % 2 == 0:
            continue
        if new_i == old_i and new_j == old_j:
            continue

        di = new_i - old_i
        dj = new_j - old_j
        new_dft = dft + di * TWIDDLE[:, i] + dj * TWIDDLE[:, j]
        new_psd = np.real(new_dft * np.conj(new_dft))
        new_E = np.sum(np.abs(new_psd + other_psd - TARGET))

        dE = new_E - cur_E
        if dE < 0 or rng.random() < np.exp(-dE / max(T, 0.01)):
            seq[i] = new_i
            seq[j] = new_j
            dft[:] = new_dft
            psd[:] = new_psd
            cur_E = new_E
            accepts += 1

            if cur_E < sa_best_E:
                sa_best_E = cur_E
                sa_best_A[:] = r_A
                sa_best_B[:] = r_B

                if sa_best_E < 0.5:
                    if verbose:
                        dt = time.time() - t0
                        print(f"  FOUND E={sa_best_E:.3f} at SA iter {it+1} "
                              f"({dt:.1f}s)")
                    return sa_best_A.copy(), sa_best_B.copy(), sa_best_E

        if verbose and (it + 1) % 1_000_000 == 0:
            dt = time.time() - t0
            rate = accepts / (it + 1) * 100
            print(f"  SA {it+1:>8,}: E={cur_E:.1f} best={sa_best_E:.1f} "
                  f"T={T:.1f} acc={rate:.1f}% {dt:.0f}s", flush=True)

    if verbose:
        dt = time.time() - t0
        print(f"  SA final: best={sa_best_E:.1f}, {dt:.0f}s")

    return sa_best_A.copy(), sa_best_B.copy(), sa_best_E


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    mode = sys.argv[2] if len(sys.argv) > 2 else "hybrid"

    print(f"Mode: {mode}, {n_trials} trials", flush=True)
    best = float('inf')

    for trial in range(n_trials):
        print(f"\n=== Trial {trial} ===", flush=True)

        if mode == "ap":
            r_A, r_B, E = ap_with_phase_perturbation(
                max_outer=500, max_inner=500,
                seed=trial * 7919 + 1, verbose=True)
        elif mode == "hybrid":
            r_A, r_B, E = hybrid_ap_sa(
                max_iter=10_000_000,
                seed=trial * 7919 + 1, verbose=True)
        else:
            r_A, r_B, E = ap_search(
                max_iter=50000, seed=trial * 7919 + 1)

        print(f"  Trial {trial}: E={E:.1f}", flush=True)
        if E < best:
            best = E
        if E < 0.5:
            print(f"\n*** EXACT SOLUTION ***")
            print(f"r_A = {r_A.astype(int).tolist()}")
            print(f"r_B = {r_B.astype(int).tolist()}")
            break

    print(f"\nBest energy: {best:.1f}")
