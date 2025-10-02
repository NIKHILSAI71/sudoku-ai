from __future__ import annotations

from typing import List
import numpy as np


def parse_line(s: str) -> np.ndarray:
    """Parse a 81-char string of digits (0 or . for empty) into (9,9) int32 array.

    Accepts spaces/newlines, commas; ignores non-digit except '.'
    """
    chars = [ch for ch in s if ch.isdigit() or ch == "."]
    digits: List[int] = []
    for ch in chars:
        if ch == ".":
            digits.append(0)
        else:
            d = int(ch)
            if 0 <= d <= 9:
                digits.append(d)
    if len(digits) != 81:
        raise ValueError("expected 81 digits")
    arr = np.array(digits, dtype=np.int32).reshape(9, 9)
    return arr


def parse_sdk(s: str) -> np.ndarray:
    """Parse SDK format with optional box separators and formatting.

    Accepts formats like:
    - Simple grid with spaces
    - Grid with | and - separators
    - . or 0 for empty cells
    """
    # Just use parse_line which handles all these cases
    return parse_line(s)


def board_to_line(arr: np.ndarray) -> str:
    if arr.shape != (9, 9):
        raise ValueError("expected (9,9)")
    flat = arr.reshape(-1)
    return "".join(str(int(x)) for x in flat)
