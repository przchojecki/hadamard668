#!/usr/bin/env python3
"""
Direct SA on full length-333 ±1 sequences.

Matrix view: 37×9 ±1 matrix, A[9i+j] = M[i][j].
Move: swap +1 and -1 in the same column (preserves column sum).

This operates on the full LP condition at all 332 nonzero frequencies,
not just the compressed subset. Uses precomputed DFT twiddle factors
for O(333) incremental updates.

Key advantage: column-swap moves preserve the 9-compression (column sums),
so we automatically satisfy the PSD at column frequencies (multiples of 37).
We only need the remaining frequencies to converge.
"""

import numpy as np
import sys
import time

TARGET = 668
ELL = 333
N = ELL  # sequence length

# Precompute twiddle factors for incremental DFT
# For two positions p1, p2 being swapped (delta = ±2):
# ΔF[s] = delta * (ω^{s·p1} - ω^{s·p2}) where ω = e^{-2πi/N}
# We only need frequencies 1..166 (others are conjugates)
N_INDEP = N // 2  # = 166 independent nonzero frequencies


def init_pair(rng):
    """Initialize two 37×9 ±1 matrices with sum(A) = sum(B) = 1."""
    # sum = 1 means 167 ones and 166 minus-ones
    n_ones = (N + 1) // 2  # = 167

    def make_seq():
        seq = np.ones(N, dtype=np.float64)
        seq[:N - n_ones] = -1
        rng.shuffle(seq)
        return seq

    return make_seq(), make_seq()


def compute_full_dft(seq):
    """Compute DFT at frequencies 1..166."""
    s_vals = np.arange(1, N_INDEP + 1)
    k_vals = np.arange(N)
    # DFT[s] = sum_k seq[k] * exp(-2πi·s·k/N)
    phases = np.exp(-2j * np.pi * np.outer(s_vals, k_vals) / N)
    return phases @ seq


def energy_from_psd(psd_A, psd_B):
    """L1 energy over all independent nonzero frequencies."""
    return np.sum(np.abs(psd_A + psd_B - TARGET))


def search_full(max_iter=10_000_000, seed=None, verbose=True):
    """
    SA on full 333-bit LP pair.
    Move: swap +1 and -1 in the same column of the 37×9 matrix.
    """
    rng = np.random.default_rng(seed)
    A, B = init_pair(rng)

    # Precompute DFT twiddle for all positions and frequencies
    s_vals = np.arange(1, N_INDEP + 1)
    # TWID[s_idx, pos] = exp(-2πi·s·pos/N)
    TWID = np.exp(-2j * np.pi * np.outer(s_vals, np.arange(N)) / N)

    # Initial DFT
    dft_A = TWID @ A
    dft_B = TWID @ B
    psd_A = np.real(dft_A * np.conj(dft_A))
    psd_B = np.real(dft_B * np.conj(dft_B))
    cur_E = energy_from_psd(psd_A, psd_B)

    best_E = cur_E
    best_A = A.copy()
    best_B = B.copy()

    # Build column indices: column j contains positions j, j+9, j+18, ...
    col_plus = {}   # col_plus[j] = positions in col j with +1
    col_minus = {}  # col_minus[j] = positions in col j with -1

    def rebuild_col_indices(seq, label):
        for j in range(9):
            positions = np.arange(j, N, 9)
            if label == 'A':
                col_plus[('A', j)] = positions[seq[positions] > 0]
                col_minus[('A', j)] = positions[seq[positions] < 0]
            else:
                col_plus[('B', j)] = positions[seq[positions] > 0]
                col_minus[('B', j)] = positions[seq[positions] < 0]

    rebuild_col_indices(A, 'A')
    rebuild_col_indices(B, 'B')

    T_init = cur_E * 0.05
    accepts = 0
    stale = 0
    t0 = time.time()

    for it in range(max_iter):
        progress = it / max_iter
        T = max(0.01, T_init * (1.0 - progress) ** 1.5)

        # Periodic reheat
        if stale > max_iter // 20:
            T = T_init * 0.3
            stale = 0

        # Choose sequence
        if rng.random() < 0.5:
            seq, dft, psd, other_psd, label = A, dft_A, psd_A, psd_B, 'A'
        else:
            seq, dft, psd, other_psd, label = B, dft_B, psd_B, psd_A, 'B'

        # Choose column
        j = rng.integers(9)
        plus = col_plus[(label, j)]
        minus = col_minus[(label, j)]

        if len(plus) == 0 or len(minus) == 0:
            stale += 1
            continue

        # Pick one +1 and one -1 to swap
        p1 = plus[rng.integers(len(plus))]    # +1 → -1
        p2 = minus[rng.integers(len(minus))]  # -1 → +1

        # Incremental DFT: delta = -2 at p1, +2 at p2
        delta_dft = -2 * TWID[:, p1] + 2 * TWID[:, p2]
        new_dft = dft + delta_dft
        new_psd = np.real(new_dft * np.conj(new_dft))
        new_E = energy_from_psd(new_psd, other_psd)

        dE = new_E - cur_E
        if dE < 0 or rng.random() < np.exp(-dE / max(T, 0.01)):
            # Accept
            seq[p1] = -1
            seq[p2] = 1
            dft[:] = new_dft
            psd[:] = new_psd
            cur_E = new_E
            accepts += 1
            stale = 0

            # Update column indices
            cp = col_plus[(label, j)]
            cm = col_minus[(label, j)]
            col_plus[(label, j)] = np.append(cp[cp != p1], p2)
            col_minus[(label, j)] = np.append(cm[cm != p2], p1)

            if cur_E < best_E:
                best_E = cur_E
                best_A[:] = A
                best_B[:] = B

                if best_E < 0.5:
                    if verbose:
                        dt = time.time() - t0
                        print(f"  FOUND LP! E={best_E:.3f} iter={it+1} "
                              f"({dt:.1f}s)")
                    return best_A.copy(), best_B.copy(), best_E
        else:
            stale += 1

        if verbose and (it + 1) % 500_000 == 0:
            dt = time.time() - t0
            rate = accepts / (it + 1) * 100
            print(f"  {it+1:>10,}: E={cur_E:.1f} best={best_E:.1f} "
                  f"T={T:.1f} acc={rate:.1f}% {dt:.0f}s", flush=True)

    if verbose:
        dt = time.time() - t0
        print(f"  Final: best={best_E:.1f}, {dt:.0f}s")

    return best_A.copy(), best_B.copy(), best_E


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000_000

    print(f"Full LP(333) search: {n_trials} trials × {max_iter:,} iterations",
          flush=True)
    best = float('inf')

    for trial in range(n_trials):
        print(f"\n--- Trial {trial} ---", flush=True)
        A, B, E = search_full(max_iter=max_iter, seed=trial * 104729 + 3)
        print(f"  Trial {trial}: E={E:.1f}", flush=True)
        if E < best:
            best = E
        if E < 0.5:
            print(f"\n*** LP(333) FOUND! ***")
            print(f"A = {A.astype(int).tolist()}")
            print(f"B = {B.astype(int).tolist()}")

            from hadamard.core import check_lp, lp_to_hadamard, verify_hadamard
            ok, dev = check_lp(A, B)
            print(f"LP verified: {ok}, dev={dev:.6e}")
            if ok:
                H, err = lp_to_hadamard(A, B)
                is_h, herr = verify_hadamard(H)
                print(f"Hadamard({H.shape[0]}): {is_h}, err={herr:.6e}")
                np.savez("hadamard/results/LP333_SOLUTION.npz",
                         A=A, B=B, H=H)
            break

    print(f"\nBest energy: {best:.1f}")
