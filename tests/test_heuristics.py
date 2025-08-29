from sudoku_engine.parser import parse_line
from sudoku_engine.board import Board
from sudoku_solvers.heuristics import apply_naked_singles, apply_hidden_singles, run_pipeline
from sudoku_engine.validator import is_valid_board


def test_hidden_singles_make_progress_on_easy():
    line = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
    b = Board(parse_line(line))
    before_empty = (b.grid == 0).sum()
    # Naked singles alone may or may not progress; hidden singles should add more
    n1 = apply_naked_singles(b)
    n2 = apply_hidden_singles(b)
    assert n1 + n2 >= 1
    assert is_valid_board(b)


def test_pipeline_until_stable_is_valid():
    line = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
    b = Board(parse_line(line))
    assigned = run_pipeline(b, max_iters=20)
    assert assigned >= 1
    assert is_valid_board(b)