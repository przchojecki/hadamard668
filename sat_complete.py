"""
SAT-based decompression for LP(333).

Given compatible marginals (row sums and column sums for the 37×9 matrix),
use SAT to find an actual ±1 matrix, and then verify the full LP condition.

Variables: x_{i,j} ∈ {0, 1} for i=0..36, j=0..8
  (representing M[i][j] = 2*x_{i,j} - 1, so x=1 → M=+1, x=0 → M=-1)

Constraints:
1. Row sum constraints: sum_j x_{i,j} = (r_i + 9) / 2 for each row i
2. Column sum constraints: sum_i x_{i,j} = (c_j + 37) / 2 for each column j
3. LP PSD constraints at all frequencies (encoded via auxiliary variables)

The LP PSD constraint at frequency s:
  PSD(A, s) + PSD(B, s) = 668
This is a nonlinear constraint involving DFT values.

For SAT encoding, we use the approach from [arXiv:1907.04987]:
- Encode the cardinality constraints (row/column sums) using standard
  techniques (sequential counter, sorting network, etc.)
- Use CAS (computer algebra) to generate additional cutting planes
  from the PSD constraints
- Iterate: SAT solves → CAS checks → add learned clauses → repeat
"""

import numpy as np
from numpy.fft import fft
import subprocess
import tempfile
import os


class LPSATEncoder:
    """Encode LP(333) decompression as a SAT problem."""

    def __init__(self, col_sums_A, row_sums_A, col_sums_B, row_sums_B):
        """
        col_sums_A: length-9 column sums for matrix A
        row_sums_A: length-37 row sums for matrix A
        col_sums_B: length-9 column sums for matrix B
        row_sums_B: length-37 row sums for matrix B
        """
        self.col_A = np.array(col_sums_A, dtype=int)
        self.row_A = np.array(row_sums_A, dtype=int)
        self.col_B = np.array(col_sums_B, dtype=int)
        self.row_B = np.array(row_sums_B, dtype=int)

        self.nrows = 37
        self.ncols = 9
        self.n_entries = self.nrows * self.ncols  # 333

        # Variable numbering: x_{i,j} for A = i*9 + j + 1  (1-indexed for DIMACS)
        # x_{i,j} for B = 333 + i*9 + j + 1
        self.var_offset_A = 0
        self.var_offset_B = self.n_entries
        self.n_vars = 2 * self.n_entries
        self.next_var = self.n_vars + 1
        self.clauses = []

    def var_A(self, i, j):
        """SAT variable for A[i][j] (1-indexed)."""
        return self.var_offset_A + i * self.ncols + j + 1

    def var_B(self, i, j):
        """SAT variable for B[i][j] (1-indexed)."""
        return self.var_offset_B + i * self.ncols + j + 1

    def new_var(self):
        v = self.next_var
        self.next_var += 1
        return v

    def add_clause(self, lits):
        self.clauses.append(lits)

    def encode_cardinality_eq(self, variables, target):
        """
        Encode: exactly 'target' of the given variables are true.
        Uses sequential counter encoding.
        """
        n = len(variables)
        if target < 0 or target > n:
            # Unsatisfiable
            self.add_clause([])
            return

        if n == 0:
            if target != 0:
                self.add_clause([])
            return

        # Sequential counter: s[i][j] = "at least j of vars[0..i] are true"
        # For efficiency with moderate n, we use the sinz sequential counter
        s = [[0] * (target + 2) for _ in range(n)]
        for i in range(n):
            for j in range(target + 2):
                if j == 0:
                    s[i][j] = None  # always true
                else:
                    s[i][j] = self.new_var()

        for i in range(n):
            x = variables[i]
            for j in range(1, target + 2):
                if i == 0:
                    # s[0][j]: at least j of {x_0}
                    if j == 1:
                        # s[0][1] ↔ x_0
                        self.add_clause([-s[0][1], x])
                        self.add_clause([s[0][1], -x])
                    else:
                        # s[0][j] = false for j > 1
                        self.add_clause([-s[0][j]])
                else:
                    # s[i][j] ↔ s[i-1][j] ∨ (x_i ∧ s[i-1][j-1])
                    # This is: s[i][j] = s[i-1][j] ∨ (x_i ∧ (j==1 or s[i-1][j-1]))
                    prev_j = s[i-1][j]
                    curr = s[i][j]

                    if j == 1:
                        # s[i][1] ↔ s[i-1][1] ∨ x_i
                        self.add_clause([-curr, prev_j, x])
                        self.add_clause([curr, -prev_j])
                        self.add_clause([curr, -x])
                    else:
                        prev_jm1 = s[i-1][j-1]
                        # s[i][j] ↔ s[i-1][j] ∨ (x_i ∧ s[i-1][j-1])
                        # => curr → prev_j ∨ (x_i ∧ prev_jm1)
                        # Tseitin:
                        self.add_clause([-curr, prev_j, x])
                        self.add_clause([-curr, prev_j, prev_jm1])
                        self.add_clause([curr, -prev_j])
                        self.add_clause([curr, -x, -prev_jm1])

        # Exactly target: s[n-1][target] = true, s[n-1][target+1] = false
        self.add_clause([s[n-1][target]])
        self.add_clause([-s[n-1][target + 1]])

    def encode_marginals(self):
        """Encode row and column sum constraints for both A and B."""
        # Matrix A
        for i in range(self.nrows):
            row_vars = [self.var_A(i, j) for j in range(self.ncols)]
            target = (int(self.row_A[i]) + self.ncols) // 2
            self.encode_cardinality_eq(row_vars, target)

        for j in range(self.ncols):
            col_vars = [self.var_A(i, j) for i in range(self.nrows)]
            target = (int(self.col_A[j]) + self.nrows) // 2
            self.encode_cardinality_eq(col_vars, target)

        # Matrix B
        for i in range(self.nrows):
            row_vars = [self.var_B(i, j) for j in range(self.ncols)]
            target = (int(self.row_B[i]) + self.ncols) // 2
            self.encode_cardinality_eq(row_vars, target)

        for j in range(self.ncols):
            col_vars = [self.var_B(i, j) for i in range(self.nrows)]
            target = (int(self.col_B[j]) + self.nrows) // 2
            self.encode_cardinality_eq(col_vars, target)

    def encode(self):
        """Full encoding."""
        self.encode_marginals()
        return self.to_dimacs()

    def to_dimacs(self):
        """Convert to DIMACS CNF format."""
        lines = [f"p cnf {self.next_var - 1} {len(self.clauses)}"]
        for clause in self.clauses:
            lines.append(" ".join(str(l) for l in clause) + " 0")
        return "\n".join(lines)

    def decode_solution(self, assignment):
        """
        Decode a SAT solution into ±1 matrices A and B.
        assignment: dict mapping variable → True/False
        """
        A = np.zeros((self.nrows, self.ncols), dtype=int)
        B = np.zeros((self.nrows, self.ncols), dtype=int)

        for i in range(self.nrows):
            for j in range(self.ncols):
                A[i, j] = 1 if assignment.get(self.var_A(i, j), False) else -1
                B[i, j] = 1 if assignment.get(self.var_B(i, j), False) else -1

        return A.flatten(), B.flatten()


def solve_with_cadical(dimacs_str, timeout=300):
    """
    Solve a SAT instance using CaDiCaL (if available) or any available solver.
    Returns (satisfiable, assignment_dict) or (False, None).
    """
    # Try to find a SAT solver
    solvers = ['cadical', 'minisat', 'glucose', 'picosat']
    solver_cmd = None
    for s in solvers:
        try:
            subprocess.run([s, '--help'], capture_output=True, timeout=5)
            solver_cmd = s
            break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    if solver_cmd is None:
        # Try python-sat
        try:
            from pysat.solvers import Cadical153
            return _solve_with_pysat(dimacs_str)
        except ImportError:
            print("No SAT solver found. Install cadical, minisat, or python-sat.")
            return None, None

    # Write DIMACS to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as f:
        f.write(dimacs_str)
        cnf_file = f.name

    try:
        result = subprocess.run(
            [solver_cmd, cnf_file],
            capture_output=True, text=True, timeout=timeout
        )

        # Parse output
        if result.returncode == 10:  # SAT
            assignment = {}
            for line in result.stdout.split('\n'):
                if line.startswith('v '):
                    for lit in line[2:].split():
                        v = int(lit)
                        if v != 0:
                            assignment[abs(v)] = v > 0
            return True, assignment
        elif result.returncode == 20:  # UNSAT
            return False, None
        else:
            print(f"Solver returned code {result.returncode}")
            return None, None
    except subprocess.TimeoutExpired:
        print(f"Solver timed out after {timeout}s")
        return None, None
    finally:
        os.unlink(cnf_file)


def _solve_with_pysat(dimacs_str):
    """Solve using python-sat library."""
    from pysat.solvers import Cadical153
    from pysat.formula import CNF

    cnf = CNF()
    for line in dimacs_str.split('\n'):
        if line.startswith('p ') or line.startswith('c '):
            continue
        lits = [int(x) for x in line.split() if x != '0']
        if lits:
            cnf.append(lits)

    with Cadical153(bootstrap_with=cnf.clauses) as solver:
        if solver.solve():
            model = solver.get_model()
            assignment = {abs(v): v > 0 for v in model}
            return True, assignment
        else:
            return False, None


def attempt_decompression(col_sums_A, row_sums_A, col_sums_B, row_sums_B,
                          timeout=300):
    """
    Attempt to find ±1 matrices A, B with given marginals that form an LP.

    Returns (A, B) as length-333 ±1 sequences, or None.
    """
    encoder = LPSATEncoder(col_sums_A, row_sums_A, col_sums_B, row_sums_B)
    dimacs = encoder.encode()

    print(f"SAT instance: {encoder.next_var - 1} variables, "
          f"{len(encoder.clauses)} clauses")

    sat, assignment = solve_with_cadical(dimacs, timeout=timeout)

    if sat is None:
        return None

    if not sat:
        print("UNSAT: no ±1 matrix with these marginals exists")
        return None

    A_flat, B_flat = encoder.decode_solution(assignment)

    # Verify the LP condition
    from hadamard.core import check_lp
    is_lp, dev = check_lp(A_flat, B_flat)

    if is_lp:
        print(f"FOUND LEGENDRE PAIR OF LENGTH 333!")
        return A_flat, B_flat
    else:
        print(f"Matrix found but LP condition not satisfied (dev={dev:.6f})")
        print("Need to add PSD constraints and re-solve (SAT+CAS loop)")
        return None
