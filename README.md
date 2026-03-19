# Hadamard Matrix of Order 668

Computational search for the smallest open case of the Hadamard conjecture: a Hadamard matrix of order 668, via Legendre pairs of length 333.

## The Problem

A **Hadamard matrix** of order $n$ is an $n \times n$ matrix with entries $\pm 1$ satisfying $HH^\top = nI$. The Hadamard conjecture asserts one exists whenever $4 \mid n$. The smallest open case is $n = 668$.

A **Legendre pair** (LP) of odd length $\ell$ is a pair of $\pm 1$ sequences $A, B$ of length $\ell$ with $\sum A = \sum B = 1$ and

$$\text{PSD}(A, s) + \text{PSD}(B, s) = 2\ell + 2 \quad \text{for all } s = 1, \ldots, \ell - 1,$$

where $\text{PSD}(X, s) = |\hat{X}(s)|^2$. An LP of length $\ell$ yields a Hadamard matrix of order $2(\ell + 1)$. For $\ell = 333$, this gives order 668.

## Approach

Since $333 = 9 \times 37$, we view a length-333 sequence as a **37 × 9 sign matrix**. Column sums give the 9-compression (length 9, entries odd in $[-37, 37]$), row sums give the 37-compression (length 37, entries odd in $[-9, 9]$). Each compression preserves PSD at a subset of frequencies.

The search pipeline:

1. **Macro-case split.** The PSD at frequency 111 decomposes into exactly **8 feasible cases**. A mod-3 obstruction (167 is not a sum of two squares) kills all multiplier subgroups with nontrivial image mod 3. An analogous mod-37 obstruction kills subgroups surjecting onto $U_{37}$.

2. **9-compression search (complete).** Exhaustive enumeration of all PSD-compatible column-sum pairs, using the DFT group decomposition ($9 = 3 \times 3$). Found **12,017,243** matching configurations across all 8 macro-cases.

3. **37-compression search (ongoing).** Find row-sum pairs satisfying PSD constraints at all 18 independent frequencies. This is the computational bottleneck.

4. **Intersection.** Match compatible row/column sums via Gale-Ryser feasibility.

5. **SAT decompression.** Reconstruct the full $\pm 1$ matrix from marginals using SAT solving.

6. **Verification.** Check LP condition and construct $H(668)$.

## Results So Far

### 9-Compression (Exhaustive)

| Case | PSD(A,111) | PSD(B,111) | Matching configs |
|------|-----------|-----------|-----------------|
| 0 | 16 | 652 | 1,266,213 |
| 1 | 64 | 604 | 1,317,923 |
| 2 | 76 | 592 | 1,329,962 |
| 3 | 112 | 556 | 1,327,949 |
| 4 | 172 | 496 | 1,401,576 |
| 5 | 256 | 412 | 1,317,121 |
| 6 | 268 | 400 | 1,315,572 |
| 7 | 304 | 364 | 2,740,927 |
| **Total** | | | **12,017,243** |

### 37-Compression (Best: E = 236)

We minimise $E = \sum_{k=1}^{18} |\text{PSD}_A(k) + \text{PSD}_B(k) - 668|$ over pairs of length-37 odd-integer sequences.

| Method | Iter/s | Best E |
|--------|--------|--------|
| Python SA | 65K | 434 |
| Python AP+SA hybrid | 65K | 321 |
| C SA (`sa37.c`) | 28M | 277 |
| C SA, 100 trials | 28M | **236** |

Cascade descent (`descent37.c`) confirmed every SA solution is a true local minimum with respect to 2-, 3-, and 4-entry neighborhoods.

## Proposed Computational Routes

### Route 1: Massive Multi-Trial SA
Run $10^4$–$10^5$ trials of the C SA. Each trial takes ~72s at 2B iterations. On 32 cores, $10^5$ trials ≈ 3 days.

```bash
cc -O3 -march=native -o sa37 sa37.c -lm
for i in $(seq 0 31); do
  ./sa37 3125 2000000000 $((i*3125)) >results/sa37_batch${i}.txt 2>&1 &
done
```

### Route 2: Multiplier-Orbit Reduction
Restrict to sequences invariant under a multiplier subgroup $H \leq \ker(\mathbb{Z}/333\mathbb{Z}^* \to \mathbb{Z}/3\mathbb{Z}^*)$. This collapses 37 variables to ~18–55 orbit values, dramatically narrowing the search.

### Route 3: SAT/SMT Encoding
Encode the 37-entry sequences as bit vectors (4 bits each) and the PSD constraints as integer quadratic equations. Feed to Z3 or a MIQP solver (SCIP, Gurobi).

### Route 4: Lattice-Based Spectral Synthesis
Given a target PSD spectrum, finding a realising sequence is a short-vector problem in an inverse-DFT lattice. Use LLL/BKZ reduction in SageMath.

See `report.tex` for full details.

## Quick Start

```bash
# Verify all mathematical claims
python3 -m hadamard.verify

# Reproduce the 9-compression search (~100 min)
python3 -m hadamard.run_all_cases

# Run C SA for 37-compression
cd hadamard
cc -O3 -march=native -o sa37 sa37.c -lm
./sa37 100 2000000000 42

# Run cascade descent from best known solution
cc -O3 -march=native -o descent37 descent37.c -lm
./descent37 1 100 42 fixed
```

## File Structure

| File | Purpose |
|------|---------|
| `core.py` | PSD, LP verification, LP→Hadamard construction |
| `compression.py` | Compression framework, macro-cases, obstructions |
| `verify.py` | Verify all mathematical claims |
| `search9.py` | Exhaustive 9-compression PSD catalog |
| `sa37.c` | C SA for 37-compression (28M iter/s) |
| `descent37.c` | Exhaustive steepest descent (2/3/4-entry) |
| `phase_search.c` | Phase-space SA |
| `search37_fast.py` | Python SA for 37-compression |
| `search37_ap.py` | Alternating projection + SA hybrid |
| `search37_basin.py` | Basin-hopping SA |
| `search_full.py` | SA on full 333-bit sequences |
| `marginals.py` | Gale-Ryser compatibility |
| `sat_complete.py` | SAT encoder for matrix decompression |
| `pipeline.py` | Full automated pipeline |
| `report.tex` | Detailed report with proofs and compute plan |
| `results/` | Precomputed 9-compression results (12M configs) |
