from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple
import numpy as np

from sudoku_engine.board import Board, mask_to_digits, ALL_MASK
from . import heuristics
from .heuristics import Trace, HeuristicStats, Options


@dataclass
class Stats:
    nodes: int = 0
    heuristics: HeuristicStats = field(default_factory=HeuristicStats)


def _select_mrv(board: Board, masks: np.ndarray) -> Optional[Tuple[int, int]]:
    best: Optional[Tuple[int, int, int]] = None  # (count, r, c)
    for r in range(9):
        for c in range(9):
            if board.grid[r, c] == 0:
                m = int(masks[r, c])
                if m == 0:
                    return None  # dead end
                cnt = m.bit_count()
                if best is None or cnt < best[0]:
                    best = (cnt, r, c)
                    if cnt == 1:
                        return (r, c)
    if best is None:
        return None
    return (best[1], best[2])


def _propagate(board: Board, trace: Optional[Trace] = None, hstats: Optional[HeuristicStats] = None, options: Optional[Options] = None) -> Optional[np.ndarray]:
    # Repeatedly run heuristics pipeline; detect contradictions (any empty candidate mask)
    while True:
        masks = board.candidates_mask()
        # contradiction if any empty cell has no candidates
        for r in range(9):
            for c in range(9):
                if board.grid[r, c] == 0 and int(masks[r, c]) == 0:
                    return None
        # choose max_iters from options if provided
        max_iters = 3
        if options is not None and hasattr(options, 'max_iters') and isinstance(options.max_iters, int):
            max_iters = options.max_iters
        delta = heuristics.run_pipeline(board, max_iters=max_iters, trace=trace, stats=hstats, options=options)
        if delta == 0:
            break
    # final check again
    masks = board.candidates_mask()
    for r in range(9):
        for c in range(9):
            if board.grid[r, c] == 0 and int(masks[r, c]) == 0:
                return None
    return masks


def solve_one(board: Board, stats: Optional[Stats] = None, trace: Optional[Trace] = None, options: Optional[Options] = None) -> Optional[Board]:
    b = board.copy()
    masks = _propagate(b, trace=trace, hstats=(stats.heuristics if stats else None), options=options)
    if masks is None:
        return None
    if b.is_complete():
        return b
    if stats is None:
        stats = Stats()
    pos = _select_mrv(b, masks)
    if pos is None:
        return b if b.is_complete() else None
    r, c = pos
    opts = mask_to_digits(int(masks[r, c]))
    for v in opts:
        stats.nodes += 1
        b2 = b.copy()
        b2.set_cell(r, c, v)
        if trace is not None:
            trace.add("guess", f"R{r+1}C{c+1}={v}")
        sol = solve_one(b2, stats, trace, options)
        if sol is not None:
            return sol
    return None


def iterate_solutions(board: Board, limit: Optional[int] = None, options: Optional[Options] = None) -> Generator[Board, None, None]:
    b = board.copy()
    stack: List[Tuple[Board, Optional[np.ndarray]]] = [(b, None)]
    yielded = 0

    while stack:
        cur, masks = stack.pop()
        masks = _propagate(cur, options=options)
        if masks is None:
            continue
        if cur.is_complete():
            yield cur
            yielded += 1
            if limit is not None and yielded >= limit:
                return
            continue
        pos = _select_mrv(cur, masks)
        if pos is None:
            continue
        r, c = pos
        opts = mask_to_digits(int(masks[r, c]))
        # DFS push
        for v in reversed(opts):
            nxt = cur.copy()
            nxt.set_cell(r, c, v)
            stack.append((nxt, None))


def count_solutions(board: Board, limit: int = 2, options: Optional[Options] = None) -> int:
    cnt = 0
    for _ in iterate_solutions(board, limit=limit, options=options):
        cnt += 1
        if cnt >= limit:
            break
    return cnt
