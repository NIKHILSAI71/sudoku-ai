"""Tests for Sudoku validation."""
from __future__ import annotations

import numpy as np
import pytest

from sudoku_engine.board import Board
from sudoku_engine.validator import is_valid_board, has_duplicates


class TestHasDuplicates:
    def test_no_duplicates(self):
        """Test array with no duplicates."""
        arr = np.array([1, 2, 3, 0, 0, 0, 4, 5, 6])
        assert not has_duplicates(arr)

    def test_with_duplicates(self):
        """Test array with duplicates."""
        arr = np.array([1, 2, 3, 1, 0, 0, 4, 5, 6])
        assert has_duplicates(arr)

    def test_zeros_allowed(self):
        """Test that zeros (empty cells) don't count as duplicates."""
        arr = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert not has_duplicates(arr)

    def test_single_duplicate(self):
        """Test with single duplicate."""
        arr = np.array([5, 5, 0, 0, 0, 0, 0, 0, 0])
        assert has_duplicates(arr)


class TestIsValidBoard:
    def test_empty_board_valid(self):
        """Test that empty board is valid."""
        board = Board.empty()
        assert is_valid_board(board)

    def test_valid_partial_board(self):
        """Test valid partially filled board."""
        board = Board.empty()
        board.set_cell(0, 0, 5)
        board.set_cell(0, 1, 3)
        board.set_cell(1, 0, 6)
        assert is_valid_board(board)

    def test_invalid_row(self):
        """Test board with duplicate in row."""
        board = Board.empty()
        # Manually set duplicates (bypass set_cell validation)
        board.grid[0, 0] = 5
        board.grid[0, 1] = 5  # Duplicate in row 0
        assert not is_valid_board(board)

    def test_invalid_column(self):
        """Test board with duplicate in column."""
        board = Board.empty()
        board.grid[0, 0] = 5
        board.grid[1, 0] = 5  # Duplicate in column 0
        assert not is_valid_board(board)

    def test_invalid_box(self):
        """Test board with duplicate in 3x3 box."""
        board = Board.empty()
        board.grid[0, 0] = 5
        board.grid[1, 1] = 5  # Duplicate in box (0,0)
        assert not is_valid_board(board)

    def test_valid_complete_board(self):
        """Test a valid complete Sudoku solution."""
        # This is a valid Sudoku solution
        solution = [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9],
        ]
        board = Board.from_list(solution)
        assert is_valid_board(board)

    def test_all_boxes_checked(self):
        """Test that all 9 boxes are validated."""
        board = Board.empty()
        # Put duplicates in bottom-right box (2, 2)
        board.grid[6, 6] = 5
        board.grid[7, 7] = 5  # Duplicate in box (2, 2)
        assert not is_valid_board(board)

    def test_mixed_duplicates(self):
        """Test board with multiple types of duplicates."""
        board = Board.empty()
        board.grid[0, 0] = 5
        board.grid[0, 1] = 5  # Row duplicate
        board.grid[1, 0] = 5  # Column duplicate
        assert not is_valid_board(board)
