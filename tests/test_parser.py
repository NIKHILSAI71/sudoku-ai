"""Tests for puzzle parsing."""
from __future__ import annotations

import pytest
import numpy as np

from sudoku_engine.parser import parse_line, parse_sdk
from sudoku_engine import board_to_line


class TestParseSDK:
    def test_parse_sdk_basic(self):
        """Test parsing SDK format."""
        text = """1 2 3 4 5 6 7 8 9
4 5 6 7 8 9 1 2 3
7 8 9 1 2 3 4 5 6
2 3 4 5 6 7 8 9 1
5 6 7 8 9 1 2 3 4
8 9 1 2 3 4 5 6 7
3 4 5 6 7 8 9 1 2
6 7 8 9 1 2 3 4 5
9 1 2 3 4 5 6 7 8"""
        grid = parse_sdk(text)
        assert grid.shape == (9, 9)
        assert grid[0, 0] == 1
        assert grid[0, 8] == 9
        assert grid[8, 0] == 9
        assert grid[8, 8] == 8

    def test_parse_sdk_with_zeros(self):
        """Test parsing SDK with empty cells."""
        text = """0 2 3 . 5 6 7 8 9
. . . . . . . . .
7 8 9 1 2 3 4 5 6
2 3 4 5 6 7 8 9 1
5 6 7 8 9 1 2 3 4
8 9 1 2 3 4 5 6 7
3 4 5 6 7 8 9 1 2
6 7 8 9 1 2 3 4 5
9 1 2 3 4 5 6 7 8"""
        grid = parse_sdk(text)
        assert grid[0, 0] == 0
        assert grid[0, 3] == 0  # '.' should be 0
        assert grid[1, 0] == 0
        assert grid[0, 1] == 2

    def test_parse_sdk_with_box_separators(self):
        """Test parsing SDK with box separators."""
        text = """5 3 . | . 7 . | . . .
6 . . | 1 9 5 | . . .
. 9 8 | . . . | . 6 .
------+-------+------
8 . . | . 6 . | . . 3
4 . . | 8 . 3 | . . 1
7 . . | . 2 . | . . 6
------+-------+------
. 6 . | . . . | 2 8 .
. . . | 4 1 9 | . . 5
. . . | . 8 . | . 7 9"""
        grid = parse_sdk(text)
        assert grid.shape == (9, 9)
        assert grid[0, 0] == 5
        assert grid[0, 1] == 3
        assert grid[0, 2] == 0


class TestParseLine:
    def test_parse_line_basic(self):
        """Test parsing a basic line."""
        line = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        grid = parse_line(line)
        assert grid.shape == (9, 9)
        assert grid[0, 0] == 5
        assert grid[0, 1] == 3
        assert grid[0, 2] == 0

    def test_parse_line_all_zeros(self):
        """Test parsing empty puzzle."""
        line = "0" * 81
        grid = parse_line(line)
        assert np.all(grid == 0)

    def test_parse_line_all_filled(self):
        """Test parsing complete puzzle."""
        line = "123456789" * 9
        grid = parse_line(line)
        assert grid[0, 0] == 1
        assert grid[0, 8] == 9
        assert grid[1, 0] == 1

    def test_parse_line_with_whitespace(self):
        """Test parsing line with whitespace."""
        line = "5 3 0 0 7 0 0 0 0 " + "0" * 72
        grid = parse_line(line)
        assert grid[0, 0] == 5
        assert grid[0, 1] == 3

    def test_parse_line_multiline(self):
        """Test that parse_line can handle SDK format."""
        sdk_text = """5 3 0 0 7 0 0 0 0
6 0 0 1 9 5 0 0 0
0 9 8 0 0 0 0 6 0
8 0 0 0 6 0 0 0 3
4 0 0 8 0 3 0 0 1
7 0 0 0 2 0 0 0 6
0 6 0 0 0 0 2 8 0
0 0 0 4 1 9 0 0 5
0 0 0 0 8 0 0 7 9"""
        grid = parse_line(sdk_text)
        assert grid.shape == (9, 9)
        assert grid[0, 0] == 5


class TestBoardToLine:
    def test_board_to_line(self):
        """Test converting board to line."""
        grid = np.zeros((9, 9), dtype=np.int32)
        grid[0, 0] = 5
        grid[0, 1] = 3
        line = board_to_line(grid)
        assert len(line) == 81
        assert line[0] == '5'
        assert line[1] == '3'
        assert line[2] == '0'

    def test_roundtrip(self):
        """Test parse_line -> board_to_line roundtrip."""
        original = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        grid = parse_line(original)
        result = board_to_line(grid)
        assert result == original
