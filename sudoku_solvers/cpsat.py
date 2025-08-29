from __future__ import annotations

from typing import List, Optional, Generator, Any, cast

try:
    from ortools.sat.python import cp_model
except Exception as e:  # pragma: no cover - optional dependency
    cp_model = None  # type: ignore[assignment]

from sudoku_engine.board import Board, mask_to_digits


def _ensure_ortools():
    if cp_model is None:
        raise RuntimeError("OR-Tools not installed. Install with `pip install .[cpsat]`.")


def _cp() -> Any:
    """Return cp_model with a non-None type for static checkers after ensuring import."""
    _ensure_ortools()
    return cast(Any, cp_model)


def _build_model(board: Board):
    m = _cp()
    model = m.CpModel()
    X = {}
    masks = board.candidates_mask()
    for r in range(9):
        for c in range(9):
            # If cell has a given, restrict to that digit
            given = int(board.grid[r, c])
            if given:
                X[(r, c, given)] = model.NewBoolVar(f"x_{r}_{c}_{given}")
                # Create inactive vars for other digits mapped to false (via fixed constraints) lazily as needed by constraints
                continue
            allowed = int(masks[r, c])
            # Fallback: if masks not initialized for some reason, allow all
            digits = mask_to_digits(allowed) if allowed else list(range(1, 10))
            for d in digits:
                X[(r, c, d)] = model.NewBoolVar(f"x_{r}_{c}_{d}")
    # Each cell has exactly one digit
    for r in range(9):
        for c in range(9):
            given = int(board.grid[r, c])
            if given:
                # Fix given to 1 and skip equality since other vars may be missing
                model.Add(X[(r, c, given)] == 1)
            else:
                allowed = mask_to_digits(int(masks[r, c]))
                # Ensure exactly one allowed digit is chosen (empty -> UNSAT)
                model.Add(sum(X[(r, c, d)] for d in allowed) == 1)
    # Each digit appears once per row/col/box
    for r in range(9):
        for d in range(1, 10):
            # Sum over only cells where variable exists
            terms = [X[(r, c, d)] for c in range(9) if (r, c, d) in X]
            model.Add(sum(terms) == 1)
    for c in range(9):
        for d in range(1, 10):
            terms = [X[(r, c, d)] for r in range(9) if (r, c, d) in X]
            model.Add(sum(terms) == 1)
    for br in range(3):
        for bc in range(3):
            for d in range(1, 10):
                terms = [
                    X[(r, c, d)]
                    for r in range(3 * br, 3 * br + 3)
                    for c in range(3 * bc, 3 * bc + 3)
                    if (r, c, d) in X
                ]
                model.Add(sum(terms) == 1)
    # Givens already enforced in per-cell constraints
    return model, X


def _extract_solution(X, solver) -> Board:
    b = Board.empty()
    for r in range(9):
        for c in range(9):
            for d in range(1, 10):
                if solver.BooleanValue(X[(r, c, d)]):
                    b.set_cell(r, c, d)
                    break
    return b


def solve_one(board: Board) -> Optional[Board]:
    m = _cp()
    model, X = _build_model(board)
    solver = m.CpSolver()
    res = solver.Solve(model)
    if res in (m.OPTIMAL, m.FEASIBLE):
        return _extract_solution(X, solver)
    return None


def count_solutions(board: Board, limit: int = 2) -> int:
    m = _cp()
    model, X = _build_model(board)
    solver = m.CpSolver()
    cnt = 0

    class Cb(m.CpSolverSolutionCallback):  # type: ignore[misc]
        def __init__(self):
            super().__init__()
            self._count = 0

        def on_solution_callback(self):  # type: ignore[override]
            self._count += 1
            if self._count >= limit:
                self.StopSearch()

    cb = Cb()
    solver.SearchForAllSolutions(model, cb)
    return cb._count
