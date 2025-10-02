"""Core Sudoku engine: board, parsing/serialization, and validation."""

from .board import Board
from .parser import parse_line, board_to_line
from .validator import is_valid_board, has_unique_solution

__all__ = [
    "Board",
    "parse_line",
    "board_to_line",
    "is_valid_board",
    "has_unique_solution",
]
