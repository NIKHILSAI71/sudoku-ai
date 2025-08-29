from __future__ import annotations

from typing import Optional

try:
    from pysat.solvers import Glucose3  # type: ignore
    from pysat.formula import CNF  # type: ignore
except Exception:
    Glucose3 = None  # type: ignore
    CNF = None  # type: ignore

from sudoku_engine.board import Board


def _ensure_sat():
    if Glucose3 is None or CNF is None:
        raise RuntimeError("python-sat not installed. Install with `pip install .[sat]`.")


def _var(r: int, c: int, d: int) -> int:
    # map (r,c,d) to 1..729
    return r * 81 + c * 9 + d


def _encode(board: Board) -> CNF:  # type: ignore[name-defined]
    _ensure_sat()
    cnf = CNF()
    # Cell constraints: exactly one digit per cell
    for r in range(9):
        for c in range(9):
            cnf.append([_var(r, c, d) for d in range(1, 10)])
            for d1 in range(1, 10):
                for d2 in range(d1 + 1, 10):
                    cnf.append([-_var(r, c, d1), -_var(r, c, d2)])
    # Row, col, box uniqueness
    for r in range(9):
        for d in range(1, 10):
            cnf.append([_var(r, c, d) for c in range(9)])
            for c1 in range(9):
                for c2 in range(c1 + 1, 9):
                    cnf.append([-_var(r, c1, d), -_var(r, c2, d)])
    for c in range(9):
        for d in range(1, 10):
            cnf.append([_var(r, c, d) for r in range(9)])
            for r1 in range(9):
                for r2 in range(r1 + 1, 9):
                    cnf.append([-_var(r1, c, d), -_var(r2, c, d)])
    for br in range(3):
        for bc in range(3):
            for d in range(1, 10):
                cells = []
                for r in range(3 * br, 3 * br + 3):
                    for c in range(3 * bc, 3 * bc + 3):
                        cells.append(_var(r, c, d))
                cnf.append(cells)
                for i in range(9):
                    for j in range(i + 1, 9):
                        cnf.append([-cells[i], -cells[j]])
    # Givens
    for r in range(9):
        for c in range(9):
            v = int(board.grid[r, c])
            if v:
                cnf.append([_var(r, c, v)])
    return cnf


def solve_one(board: Board) -> Optional[Board]:
    _ensure_sat()
    cnf = _encode(board)
    with Glucose3(bootstrap_with=cnf.clauses) as solver:  # type: ignore[name-defined]
        if not solver.solve():
            return None
        model = set(solver.get_model())
    b = Board.empty()
    for r in range(9):
        for c in range(9):
            for d in range(1, 10):
                if _var(r, c, d) in model:
                    b.set_cell(r, c, d)
                    break
    return b


def count_solutions(board: Board, limit: int = 2) -> int:
    """Count solutions using SAT by iteratively blocking found models.

    limit: stop after reaching this count for efficiency.
    """
    _ensure_sat()
    cnf = _encode(board)
    count = 0
    with Glucose3(bootstrap_with=cnf.clauses) as solver:  # type: ignore[name-defined]
        while True:
            if not solver.solve():
                break
            model = solver.get_model()
            count += 1
            if count >= limit:
                break
            # Block this exact assignment of all 81 variables
            block = []
            for r in range(9):
                for c in range(9):
                    # find the digit set to true for this cell
                    lit = None
                    for d in range(1, 10):
                        v = _var(r, c, d)
                        if v in model:
                            lit = v
                            break
                    if lit is None:
                        # Should not happen for valid models; be conservative
                        continue
                    block.append(-lit)
            solver.add_clause(block)
    return count
