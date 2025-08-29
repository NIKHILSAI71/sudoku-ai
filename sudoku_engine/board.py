from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional
import numpy as np

# Candidates are represented as 9-bit masks (bits 0..8 for digits 1..9)
DIGITS = list(range(1, 10))
ALL_MASK = (1 << 9) - 1  # 0b1_1111_1111


def digit_to_mask(d: int) -> int:
    return 0 if d == 0 else 1 << (d - 1)


def mask_to_digits(mask: int) -> List[int]:
    return [d for d in DIGITS if mask & (1 << (d - 1))]


@dataclass
class Board:
    grid: np.ndarray  # shape (9,9), dtype=int32, values 0..9
    # Maintained candidate masks (9-bit per cell) for empty cells; None means lazy-init on access
    _candidates: Optional[np.ndarray] = None

    @staticmethod
    def empty() -> "Board":
        return Board(np.zeros((9, 9), dtype=np.int32))

    @staticmethod
    def from_list(cells: Iterable[Iterable[int]]) -> "Board":
        arr = np.array(list(list(row) for row in cells), dtype=np.int32)
        assert arr.shape == (9, 9)
        return Board(arr)

    def copy(self) -> "Board":
        # Deep copy grid and candidates (if initialized)
        cand = None if self._candidates is None else self._candidates.copy()
        return Board(self.grid.copy(), cand)

    def rows(self) -> Iterable[np.ndarray]:
        for r in range(9):
            yield self.grid[r, :]

    def cols(self) -> Iterable[np.ndarray]:
        for c in range(9):
            yield self.grid[:, c]

    def box(self, br: int, bc: int) -> np.ndarray:
        r0, c0 = 3 * br, 3 * bc
        return self.grid[r0 : r0 + 3, c0 : c0 + 3]

    def is_complete(self) -> bool:
        return np.all(self.grid > 0)

    def _compute_baseline_candidates(self) -> np.ndarray:
        """Compute baseline candidates from the current grid (row/col/box exclusions)."""
        masks = np.full((9, 9), ALL_MASK, dtype=np.int32)
        for r in range(9):
            row_vals = set(int(x) for x in self.grid[r, :] if x)
            row_mask = sum(digit_to_mask(d) for d in row_vals)
            masks[r, :] &= ~row_mask
        for c in range(9):
            col_vals = set(int(x) for x in self.grid[:, c] if x)
            col_mask = sum(digit_to_mask(d) for d in col_vals)
            masks[:, c] &= ~col_mask
        for br in range(3):
            for bc in range(3):
                vals = set(int(x) for x in self.box(br, bc).ravel() if x)
                box_mask = sum(digit_to_mask(d) for d in vals)
                r0, c0 = 3 * br, 3 * bc
                masks[r0 : r0 + 3, c0 : c0 + 3] &= ~box_mask
        # Occupied cells have mask 0
        occupied = self.grid > 0
        masks[occupied] = 0
        return masks

    def _ensure_candidates(self) -> None:
        if self._candidates is None:
            self._candidates = self._compute_baseline_candidates()

    def candidates_mask(self) -> np.ndarray:
        """Return maintained candidate masks (lazy-initialized)."""
        self._ensure_candidates()
        return self._candidates  # type: ignore[return-value]

    def set_cell(self, r: int, c: int, val: int) -> None:
        if not (0 <= r < 9 and 0 <= c < 9 and 0 <= val <= 9):
            raise ValueError("out of range")
        prev = int(self.grid[r, c])
        self.grid[r, c] = val
        # Update candidates structure if present
        if self._candidates is None:
            return
        if val == 0:
            # Cell cleared; conservative approach: drop candidate cache to recompute lazily
            self._candidates = None
            return
        # Assigning a digit -> remove all candidates in cell and eliminate digit from peers
        self._candidates[r, c] = 0
        bit = digit_to_mask(val)
        # Row and column peers
        for cc in range(9):
            if cc != c and self.grid[r, cc] == 0:
                self._candidates[r, cc] &= ~bit
        for rr in range(9):
            if rr != r and self.grid[rr, c] == 0:
                self._candidates[rr, c] &= ~bit
        # Box peers
        br, bc = r // 3, c // 3
        r0, c0 = 3 * br, 3 * bc
        for dr in range(3):
            for dc in range(3):
                rr, cc = r0 + dr, c0 + dc
                if (rr != r or cc != c) and self.grid[rr, cc] == 0:
                    self._candidates[rr, cc] &= ~bit

    def remove_candidate(self, r: int, c: int, digit: int) -> bool:
        """Eliminate a digit from a cell's candidates. Returns True if changed."""
        if not (1 <= digit <= 9):
            raise ValueError("digit out of range")
        if self.grid[r, c] != 0:
            return False
        self._ensure_candidates()
        bit = digit_to_mask(digit)
        before = int(self._candidates[r, c])  # type: ignore[index]
        if before & bit:
            self._candidates[r, c] = before & ~bit  # type: ignore[index]
            return True
        return False

    def __str__(self) -> str:
        lines: List[str] = []
        for r in range(9):
            row = " ".join(str(int(x)) for x in self.grid[r, :])
            lines.append(row)
        return "\n".join(lines)
