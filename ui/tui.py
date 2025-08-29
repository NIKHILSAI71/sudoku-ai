from __future__ import annotations

from sudoku_engine.board import Board


def render_ascii(board: Board) -> str:
    lines = []
    for r in range(9):
        row = "".join(str(int(x)) if x else "." for x in board.grid[r, :])
        lines.append(row)
    return "\n".join(lines)


def render_pretty(board: Board) -> str:
    sep = "+-------+-------+-------+"
    parts = [sep]
    for r in range(9):
        row = [str(int(board.grid[r, c])) if board.grid[r, c] else "." for c in range(9)]
        parts.append(f"| {' '.join(row[0:3])} | {' '.join(row[3:6])} | {' '.join(row[6:9])} |")
        if r % 3 == 2:
            parts.append(sep)
    return "\n".join(parts)
