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
    Each sample teaches the model to predict one next move.
    - targets is an (81,) int array with -100 everywhere except for the one cell to be filled,
      which has the target digit (0..8).
    """
    g_puz = parse_line(puzzle)
    g_sol = parse_line(solution)

    # Get indices of non-given cells, in a random order
    sol_indices = [i for i, p_val in enumerate(g_puz.flatten()) if p_val == 0]
    random.shuffle(sol_indices)

    samples: List[Tuple[str, np.ndarray]] = []
    cur = g_puz.copy()

    # Create up to `steps` samples, each teaching one move
    num_to_fill = min(steps, len(sol_indices))
    for i in range(num_to_fill):
        # The current board is the input
        current_line = _line_from_grid(cur)

        # The target is the next cell to fill
        idx_to_fill = sol_indices[i]
        sol_digit = g_sol.flatten()[idx_to_fill]

        targets = np.full((81,), -100, dtype=np.int64)
        targets[idx_to_fill] = sol_digit - 1

        samples.append((current_line, targets))

        # Update the board for the next iteration
        r, c = divmod(idx_to_fill, 9)
        cur[r, c] = sol_digit

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

