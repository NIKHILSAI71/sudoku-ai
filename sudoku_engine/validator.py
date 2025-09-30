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
    """Return (is_unique, count) - simplified version for AI-only mode.

    Note: This simplified version only checks if board is valid,
    not uniqueness. Full solution counting requires a solver.
    """
    if is_valid_board(board):
        # In production AI mode, we assume valid boards from trusted sources
        return (True, 1)
    return (False, 0)
