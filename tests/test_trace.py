from sudoku_engine.parser import parse_line
from sudoku_engine.board import Board
from sudoku_solvers.backtracking import solve_one, Stats
from sudoku_solvers.heuristics import Trace


def test_backtracking_trace_collects_steps():
    line = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
    b = Board(parse_line(line))
    t = Trace(steps=[])
    stats = Stats()
    _ = solve_one(b, stats=stats, trace=t)
    assert isinstance(t.steps, list)
    # Expect at least one step (heuristic or guess) on this puzzle
    assert len(t.steps) >= 1