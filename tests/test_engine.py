from sudoku_engine import Board, parse_line, is_valid_board


def test_parse_and_validate():
    s = "0" * 81
    arr = parse_line(s)
    b = Board(arr)
    assert is_valid_board(b)
