import pytest

from sudoku_engine.parser import parse_line
from sudoku_engine.board import Board
from sudoku_engine.validator import has_unique_solution
from sudoku_solvers.dlx import solve_one as dlx_solve


line = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"


def _sat_available() -> bool:
    try:
        import sudoku_solvers.sat_adapter as _
        from pysat.solvers import Glucose3  # type: ignore
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _sat_available(), reason="python-sat optional dependency not installed")
def test_sat_solves_like_dlx():
    from sudoku_solvers.sat_adapter import solve_one as sat_solve

    b = Board(parse_line(line))
    s_sat = sat_solve(b)
    s_dlx = dlx_solve(b)
    assert s_sat is not None and s_dlx is not None
    assert (s_sat.grid == s_dlx.grid).all()


@pytest.mark.skipif(not _sat_available(), reason="python-sat optional dependency not installed")
def test_sat_counts_like_validator():
    from sudoku_solvers.sat_adapter import count_solutions as sat_count

    b = Board(parse_line(line))
    unique, _ = has_unique_solution(b, count_limit=2)
    cnt_sat = sat_count(b, limit=2)
    assert (unique and cnt_sat == 1) or (not unique and cnt_sat >= 2)