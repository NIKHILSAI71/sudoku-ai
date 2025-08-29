from pathlib import Path
from sudoku_engine import Board, parse_line
from sudoku_solvers import backtracking, dlx


def load_example() -> Board:
    p = Path("examples/easy1.sdk").read_text().strip()
    return Board(parse_line(p))


def test_backtracking_solves_easy():
    b = load_example()
    sol = backtracking.solve_one(b)
    assert sol is not None


def test_dlx_solves_easy():
    b = load_example()
    sol = dlx.solve_one(b)
    assert sol is not None
