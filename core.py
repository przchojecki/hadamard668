"""
Core mathematics for Legendre pairs and Hadamard matrices.

A Legendre pair (A, B) of length ℓ consists of two sequences of ±1 values
such that PSD(A, s) + PSD(B, s) = 2ℓ + 2 for all nonzero frequencies s.

A Legendre pair of length ℓ yields a Hadamard matrix of order 2(ℓ+1).
For ℓ = 333, this gives a Hadamard matrix of order 668.
"""

import numpy as np
from numpy.fft import fft, ifft


def psd(seq):
    """Power spectral density: |DFT(seq)|^2 at each frequency."""
    s = np.array(seq, dtype=np.float64)
    f = fft(s)
    return np.real(f * np.conj(f))


def check_lp(A, B):
    """
    Check if (A, B) is a Legendre pair of length ℓ.
    Returns (is_valid, max_deviation) where max_deviation is the maximum
    deviation from the constant 2ℓ+2 at nonzero frequencies.
    """
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    ell = len(A)
    assert len(B) == ell
    assert np.all(np.abs(A) == 1), "A must be ±1"
    assert np.all(np.abs(B) == 1), "B must be ±1"

    target = 2 * ell + 2
    pa = psd(A)
    pb = psd(B)
    total = pa + pb
    # Check nonzero frequencies
    dev = np.max(np.abs(total[1:] - target))
    return dev < 1e-6, dev


def lp_to_hadamard(A, B):
    """
    Construct a Hadamard matrix of order n = 2(ℓ+1) from a Legendre pair
    (A, B) of length ℓ, with sum(A) = sum(B) = 1.

    Uses the two-circulant construction with bordered circulant blocks,
    searching over sign conventions for the bordering.
    """
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    ell = len(A)
    n = 2 * (ell + 1)

    def circulant(seq):
        m = len(seq)
        C = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                C[i, j] = seq[(j - i) % m]
        return C

    def back_circulant(seq):
        m = len(seq)
        C = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                C[i, j] = seq[(i + j) % m]
        return C

    CA = circulant(A)
    CB = circulant(B)
    BA = back_circulant(A)
    BB = back_circulant(B)
    e = np.ones((ell, 1))

    # Try many constructions systematically
    best_H = None
    best_err = float('inf')

    # Use prepended circulants: a' = (1, a_0, ..., a_{ℓ-1})
    Ap = np.concatenate([[1], A])
    Bp = np.concatenate([[1], B])
    CAp = circulant(Ap)
    CBp = circulant(Bp)

    # Construction 1: Prepended circulants
    for form_idx, (X, Y) in enumerate([
        (CAp, CBp), (CAp, CBp.T), (CAp.T, CBp), (CAp.T, CBp.T),
    ]):
        for H_form in [
            np.block([[X, Y], [Y, -X]]),
            np.block([[X, Y], [-Y, X]]),
            np.block([[X, -Y], [Y, X]]),
            np.block([[X, Y], [Y.T, -X.T]]),
            np.block([[X, Y.T], [Y, -X.T]]),
            np.block([[X, Y], [-Y.T, X.T]]),
        ]:
            err = np.max(np.abs(H_form @ H_form.T - n * np.eye(n)))
            if err < best_err:
                best_err = err
                best_H = H_form.copy()
            if err < 1e-6:
                return best_H, best_err

    # Construction 2: Bordered core with various signs
    for M1, M2 in [(CA, CB), (CA, BB), (CA, CB.T), (BA, CB)]:
        for s in range(16):
            s1 = 1 if s & 1 else -1
            s2 = 1 if s & 2 else -1
            s3 = 1 if s & 4 else -1
            s4 = 1 if s & 8 else -1
            P = np.block([[np.array([[1.0]]), s1 * e.T],
                          [s2 * e, M1]])
            Q = np.block([[np.array([[1.0]]), s3 * e.T],
                          [s4 * e, M2]])
            for H_form in [
                np.block([[P, Q], [Q, -P]]),
                np.block([[P, Q], [-Q, P]]),
                np.block([[P, Q], [Q.T, -P.T]]),
                np.block([[P, Q], [-Q.T, P.T]]),
                np.block([[P, Q.T], [Q, -P.T]]),
            ]:
                err = np.max(np.abs(H_form @ H_form.T - n * np.eye(n)))
                if err < best_err:
                    best_err = err
                    best_H = H_form.copy()
                if err < 1e-6:
                    return best_H, best_err

    return best_H, best_err


def verify_hadamard(H):
    """Verify that H is a Hadamard matrix: H @ H^T = n*I, entries ±1."""
    H = np.array(H, dtype=np.float64)
    n = H.shape[0]
    assert H.shape == (n, n), "Must be square"
    assert np.all(np.abs(H) == 1), "Entries must be ±1"
    HHT = H @ H.T
    err = np.max(np.abs(HHT - n * np.eye(n)))
    return err < 1e-6, err


def sum_of_two_squares_representations(n):
    """Find all ways to write n = a^2 + b^2 with a >= b >= 0."""
    reps = []
    b = 0
    while 2 * b * b <= n:
        a2 = n - b * b
        a = int(round(a2 ** 0.5))
        if a * a == a2 and a >= b:
            reps.append((a, b))
        b += 1
    return reps


def sum_of_a2_plus_3b2(n):
    """Find all ways to write n = a^2 + 3*b^2 with a, b integers, a >= 0."""
    reps = []
    b = 0
    while 3 * b * b <= n:
        a2 = n - 3 * b * b
        a = int(round(a2 ** 0.5))
        if a * a == a2:
            if a >= 0:
                reps.append((a, b))
                if b > 0:
                    reps.append((a, -b))
        b += 1
    return reps
