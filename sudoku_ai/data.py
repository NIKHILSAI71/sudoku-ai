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
    Create training samples from puzzle-solution pair.

    Strategy: Pick RANDOM valid moves from solution (not always easiest moves).
    This forces the model to learn good move selection, not just trivial deductions.

    Returns list of (board_state, target) pairs where target indicates
    which cell to fill with which digit.
    """
    g_puz = parse_line(puzzle)
    g_sol = parse_line(solution)

    samples: List[Tuple[str, np.ndarray]] = []
    cur = g_puz.copy()

    # Fill cells randomly from solution (force learning, not trivial deduction)
    filled_count = 0
    max_attempts = steps * 3

    for attempt in range(max_attempts):
        if filled_count >= steps:
            break

        # Find all empty cells
        empty_cells = [(r, c) for r in range(9) for c in range(9) if cur[r, c] == 0]
        if not empty_cells:
            break

        # Pick a random empty cell from solution
        r, c = random.choice(empty_cells)
        digit = g_sol[r, c]

        # Verify this is a legal move
        b = Board(cur)
        candidates_mask = b.candidates_mask()
        mask_val = int(candidates_mask[r, c])

        # Check if this move is legal
        if not (mask_val & (1 << (digit - 1))):
            # Not legal, skip (shouldn't happen with valid solution)
            continue

        # Create training sample BEFORE making the move
        current_line = _line_from_grid(cur)
        idx_to_fill = r * 9 + c

        targets = np.full((81,), -100, dtype=np.int64)
        targets[idx_to_fill] = digit - 1

        samples.append((current_line, targets))

        # Apply the move for next iteration
        cur[r, c] = digit
        filled_count += 1

    return samples


@dataclass
class SupervisedRecord:
    x_line: str
    y_targets: np.ndarray  # (81,) with -100 for ignore


def build_supervised_records(
    puzzles: List[str],
    solutions: List[str],
    max_samples: Optional[int] = None,
    augment: bool = True
) -> List[SupervisedRecord]:
    """Build supervised training records from puzzle-solution pairs.

    Args:
        puzzles: List of puzzle strings (81 chars, 0 for empty)
        solutions: List of solution strings (81 chars, all filled)
        max_samples: Maximum number of samples to generate
        augment: Whether to apply data augmentation

    Returns:
        List of SupervisedRecord objects for training
    """
    if len(puzzles) != len(solutions):
        raise ValueError(f"Puzzles ({len(puzzles)}) and solutions ({len(solutions)}) must have same length")

    # Limit puzzles processed if max_samples specified
    puzzles_to_process = puzzles
    solutions_to_process = solutions

    if max_samples is not None:
        # Each puzzle generates ~40 samples, so we only need max_samples/40 puzzles
        max_puzzles = (max_samples // 40) + 1
        if len(puzzles) > max_puzzles:
            print(f"   💡 Limiting to {max_puzzles:,} puzzles (enough for {max_samples:,} samples)")
            puzzles_to_process = puzzles[:max_puzzles]
            solutions_to_process = solutions[:max_puzzles]

    recs: List[SupervisedRecord] = []
    total = len(puzzles_to_process)

    for idx, (pz, sol) in enumerate(zip(puzzles_to_process, solutions_to_process)):
        # Validate solution is complete
        if len(sol) != 81 or '0' in sol:
            continue  # Skip incomplete solutions

        if augment:
            pz, sol = random_augment(pz, sol, enable=True)

        for line, tgt in make_partial_samples(pz, sol, steps=40):
            recs.append(SupervisedRecord(line, tgt))
            if max_samples is not None and len(recs) >= max_samples:
                print(f"   ✅ Reached {len(recs):,} samples (target: {max_samples:,})")
                return recs

        # Progress update every 1000 puzzles or at specific percentages
        if (idx + 1) % 1000 == 0 or (idx + 1) in [int(total * p) for p in [0.25, 0.5, 0.75]]:
            print(f"   📊 Progress: {idx+1:,}/{total:,} puzzles → {len(recs):,} samples", flush=True)

    return recs

