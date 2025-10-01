"""Tests for Board class."""
from __future__ import annotations

import numpy as np
import pytest

from sudoku_engine.board import Board, digit_to_mask, mask_to_digits, ALL_MASK


class TestBoardBasics:
    def test_empty_board(self):
        """Test creating an empty board."""
        board = Board.empty()
        assert board.grid.shape == (9, 9)
        assert np.all(board.grid == 0)

    def test_from_list(self):
        """Test creating board from list."""
        cells = [[0] * 9 for _ in range(9)]
        cells[0][0] = 5
        board = Board.from_list(cells)
        assert board.grid[0, 0] == 5
        assert board.grid[1, 1] == 0

    def test_copy(self):
        """Test board copying."""
        b1 = Board.empty()
        b1.set_cell(0, 0, 5)
        b2 = b1.copy()
        b2.set_cell(0, 0, 3)
        assert b1.grid[0, 0] == 5
        assert b2.grid[0, 0] == 3

    def test_is_complete(self):
        """Test completion check."""
        board = Board.empty()
        assert not board.is_complete()

        # Fill the board
        for r in range(9):
            for c in range(9):
                board.grid[r, c] = (r * 3 + c) % 9 + 1
        assert board.is_complete()


class TestBoardCandidates:
    def test_digit_to_mask(self):
        """Test digit to bitmask conversion."""
        assert digit_to_mask(0) == 0
        assert digit_to_mask(1) == 0b000000001
        assert digit_to_mask(5) == 0b000010000
        assert digit_to_mask(9) == 0b100000000

    def test_mask_to_digits(self):
        """Test bitmask to digit list conversion."""
        assert mask_to_digits(0b000000001) == [1]
        assert mask_to_digits(0b000010000) == [5]
        assert mask_to_digits(0b100000000) == [9]
        assert mask_to_digits(0b111111111) == list(range(1, 10))
        assert mask_to_digits(0b000000000) == []

    def test_candidates_empty_board(self):
        """Test candidates on empty board."""
        board = Board.empty()
        masks = board.candidates_mask()
        # All cells should have all candidates
        assert np.all(masks == ALL_MASK)

    def test_candidates_with_filled_cells(self):
        """Test candidates update when cells are filled."""
        board = Board.empty()
        board.set_cell(0, 0, 5)
        masks = board.candidates_mask()

        # Cell (0,0) should have no candidates
        assert masks[0, 0] == 0

        # Row 0 should not have 5 as candidate
        for c in range(1, 9):
            assert not (masks[0, c] & digit_to_mask(5))

        # Column 0 should not have 5 as candidate
        for r in range(1, 9):
            assert not (masks[r, 0] & digit_to_mask(5))

    def test_set_cell_updates_candidates(self):
        """Test that set_cell updates candidates correctly."""
        board = Board.empty()
        board.set_cell(0, 0, 5)

        # Place another digit in same row
        board.set_cell(0, 1, 3)
        masks = board.candidates_mask()

        # Row 0 cells should not have 3 or 5 as candidates
        for c in range(2, 9):
            m = masks[0, c]
            assert not (m & digit_to_mask(3))
            assert not (m & digit_to_mask(5))

    def test_remove_candidate(self):
        """Test manual candidate removal."""
        board = Board.empty()
        masks = board.candidates_mask()

        # Should have candidate 5
        assert masks[0, 0] & digit_to_mask(5)

        # Remove it
        changed = board.remove_candidate(0, 0, 5)
        assert changed

        # Should not have it anymore
        masks = board.candidates_mask()
        assert not (masks[0, 0] & digit_to_mask(5))

        # Try removing again - should return False
        changed = board.remove_candidate(0, 0, 5)
        assert not changed


class TestBoardValidation:
    def test_set_cell_bounds_checking(self):
        """Test that set_cell validates bounds."""
        board = Board.empty()

        with pytest.raises(ValueError):
            board.set_cell(-1, 0, 5)

        with pytest.raises(ValueError):
            board.set_cell(0, -1, 5)

        with pytest.raises(ValueError):
            board.set_cell(9, 0, 5)

        with pytest.raises(ValueError):
            board.set_cell(0, 9, 5)

        with pytest.raises(ValueError):
            board.set_cell(0, 0, 10)

    def test_remove_candidate_validation(self):
        """Test candidate removal validation."""
        board = Board.empty()

        with pytest.raises(ValueError):
            board.remove_candidate(0, 0, 0)

        with pytest.raises(ValueError):
            board.remove_candidate(0, 0, 10)

        # Should not remove from filled cell
        board.set_cell(0, 0, 5)
        changed = board.remove_candidate(0, 0, 3)
        assert not changed


class TestBoardAccessors:
    def test_rows(self):
        """Test row iteration."""
        board = Board.empty()
        board.set_cell(0, 0, 1)
        board.set_cell(1, 0, 2)

        rows = list(board.rows())
        assert len(rows) == 9
        assert rows[0][0] == 1
        assert rows[1][0] == 2

    def test_cols(self):
        """Test column iteration."""
        board = Board.empty()
        board.set_cell(0, 0, 1)
        board.set_cell(0, 1, 2)

        cols = list(board.cols())
        assert len(cols) == 9
        assert cols[0][0] == 1
        assert cols[1][0] == 2

    def test_box(self):
        """Test 3x3 box access."""
        board = Board.empty()
        board.set_cell(0, 0, 1)
        board.set_cell(2, 2, 9)

        box = board.box(0, 0)
        assert box.shape == (3, 3)
        assert box[0, 0] == 1
        assert box[2, 2] == 9

    def test_str_representation(self):
        """Test string representation."""
        board = Board.empty()
        board.set_cell(0, 0, 5)
        s = str(board)
        lines = s.split('\n')
        assert len(lines) == 9
        assert lines[0].startswith('5')
