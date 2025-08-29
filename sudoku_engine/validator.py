from __future__ import annotations

from typing import Tuple
import numpy as np

from .board import Board


def _no_duplicates(vals: np.ndarray) -> bool:
    seen = set()
    for v in vals:
        iv = int(v)
        if iv == 0:
            continue
        if iv in seen:
            return False
        seen.add(iv)
    return True


def is_valid_board(board: Board) -> bool:
    g = board.grid
    for r in range(9):
        if not _no_duplicates(g[r, :]):
            return False
    for c in range(9):
        if not _no_duplicates(g[:, c]):
            return False
    for br in range(3):
        for bc in range(3):
            box = g[3 * br : 3 * br + 3, 3 * bc : 3 * bc + 3].ravel()
            if not _no_duplicates(box):
                return False
    return True


def has_unique_solution(board: Board, count_limit: int = 2) -> Tuple[bool, int]:
    """Return (is_unique, count) by counting solutions up to count_limit.

    Uses the DLX solver if available, otherwise falls back to backtracking.
    """
    try:
        from sudoku_solvers.dlx import count_solutions
        cnt = count_solutions(board, limit=count_limit)
        return (cnt == 1, cnt)
    except Exception:
        from sudoku_solvers.backtracking import count_solutions as bt_count
        cnt = bt_count(board, limit=count_limit)
        return (cnt == 1, cnt)
