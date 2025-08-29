from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import random

import numpy as np

from sudoku_engine import parse_line, board_to_line, Board


def _line_from_grid(grid: np.ndarray) -> str:
    return "".join(str(int(x)) for x in grid.reshape(-1))


def _make_digit_mapping() -> List[int]:
    mapping = list(range(10))
    digits = list(range(1, 10))
    random.shuffle(digits)
    for src, dst in zip(range(1, 10), digits):
        mapping[src] = dst
    return mapping


def _apply_digit_mapping(line: str, mapping: List[int]) -> str:
    out = []
    for ch in line.strip():
        d = int(ch)
        out.append(str(mapping[d]))
    return "".join(out)


def _rotate90(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, k=1)


def _flip_h(grid: np.ndarray) -> np.ndarray:
    return np.flip(grid, axis=1)


def _flip_v(grid: np.ndarray) -> np.ndarray:
    return np.flip(grid, axis=0)


def random_augment(puzzle: str, solution: str, enable: bool = True) -> Tuple[str, str]:
    if not enable:
        return puzzle, solution
    # digit permutation first (same mapping for puzzle and solution)
    mapping = _make_digit_mapping()
    puzzle = _apply_digit_mapping(puzzle, mapping)
    solution = _apply_digit_mapping(solution, mapping)
    # geometric transforms applied consistently
    g_puz = parse_line(puzzle)
    g_sol = parse_line(solution)
    # choose a sequence of transforms
    ops = []
    if random.random() < 0.5:
        ops.append("rot")
    if random.random() < 0.5:
        ops.append("fliph")
    if random.random() < 0.5:
        ops.append("flipv")
    for op in ops:
        if op == "rot":
            g_puz = _rotate90(g_puz)
            g_sol = _rotate90(g_sol)
        elif op == "fliph":
            g_puz = _flip_h(g_puz)
            g_sol = _flip_h(g_sol)
        elif op == "flipv":
            g_puz = _flip_v(g_puz)
            g_sol = _flip_v(g_sol)
    return _line_from_grid(g_puz), _line_from_grid(g_sol)


def read_puzzles(paths: Iterable[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        if path.suffix.lower() in {".sdk", ".txt"}:
            for ln in path.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if ln:
                    out.append(ln)
        elif path.suffix.lower() in {".jsonl", ".json"}:
            # each line: {"puzzle": "..."}
            import json

            for ln in path.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                    if "puzzle" in rec:
                        out.append(str(rec["puzzle"]))
                except Exception:
                    continue
        else:
            # fallback: treat as raw single-line puzzle
            out.append(path.read_text(encoding="utf-8").strip())
    return out


def make_partial_samples(puzzle: str, solution: str, steps: int = 40) -> List[Tuple[str, np.ndarray]]:
    """
    Create a sequence of (partial_board_line, targets) pairs from a full solution.
    targets is an (81,) int array with -100 for givens (ignore), and 0..8 for solution digit-1 on empties.
    The sequence is created by starting from the solution and restoring givens from the puzzle, then
    progressively removing additional cells in random order up to `steps` samples.
    """
    g_puz = parse_line(puzzle)
    g_sol = parse_line(solution)
    givens_mask = (g_puz > 0)
    # start from current puzzle state
    cur = g_puz.copy()
    samples: List[Tuple[str, np.ndarray]] = []

    # candidate removal order: start from the puzzle and progressively remove original givens
    # (i.e., positions that were filled in the puzzle)
    fill_positions = [i for i in range(81) if givens_mask.reshape(-1)[i]]
    random.shuffle(fill_positions)

    # produce one sample for the original puzzle too
    targets0 = np.full((81,), -100, dtype=np.int64)
    sol_flat = g_sol.reshape(-1)
    cur_flat = cur.reshape(-1)
    for i in range(81):
        if cur_flat[i] == 0:
            targets0[i] = int(sol_flat[i]) - 1
    samples.append((_line_from_grid(cur), targets0))

    for k in range(min(steps, len(fill_positions))):
        idx = fill_positions[k]
        r, c = divmod(idx, 9)
        cur[r, c] = 0
        targets = np.full((81,), -100, dtype=np.int64)
        cur_flat = cur.reshape(-1)
        for i in range(81):
            if cur_flat[i] == 0:
                targets[i] = int(sol_flat[i]) - 1
        samples.append((_line_from_grid(cur), targets))
    return samples


@dataclass
class SupervisedRecord:
    x_line: str
    y_targets: np.ndarray  # (81,) with -100 for ignore


def build_supervised_records(puzzles: List[str], solutions: List[str], max_samples: Optional[int] = None, augment: bool = True) -> List[SupervisedRecord]:
    recs: List[SupervisedRecord] = []
    for pz, sol in zip(puzzles, solutions):
        if augment:
            pz, sol = random_augment(pz, sol, enable=True)
        for line, tgt in make_partial_samples(pz, sol, steps=40):
            recs.append(SupervisedRecord(line, tgt))
            if max_samples is not None and len(recs) >= max_samples:
                return recs
    return recs

