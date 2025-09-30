"""
Dancing Links (DLX) Solver - Training Data Generation Only

This solver is used ONLY for generating training data.
The actual puzzle solving is done by the AI model.
"""

from __future__ import annotations

from typing import Generator, List, Optional, Tuple, Sequence
from dataclasses import dataclass
import logging

from sudoku_engine.board import Board

logger = logging.getLogger(__name__)


@dataclass
class Node:
    left: Node | None = None
    right: Node | None = None
    up: Node | None = None
    down: Node | None = None
    col: Column | None = None
    row_id: int = -1


@dataclass
class Column(Node):
    name: str = ""
    size: int = 0


def _link_lr(nodes: Sequence[Node]) -> None:
    n = len(nodes)
    for i in range(n):
        nodes[i].right = nodes[(i + 1) % n]
        nodes[(i + 1) % n].left = nodes[i]


def _build_exact_cover(board: Board) -> Tuple[Column, List[Node]]:
    """Build exact cover matrix for Sudoku."""
    header = Column(name="header")
    columns: List[Column] = []

    for i in range(324):
        col = Column(name=str(i))
        col.up = col.down = col
        columns.append(col)

    ring = [header] + columns  # type: ignore
    _link_lr(ring)

    def col_index_cell(r: int, c: int) -> int:
        return r * 9 + c

    def col_index_row_digit(r: int, d: int) -> int:
        return 81 + r * 9 + (d - 1)

    def col_index_col_digit(c: int, d: int) -> int:
        return 162 + c * 9 + (d - 1)

    def col_index_box_digit(b: int, d: int) -> int:
        return 243 + b * 9 + (d - 1)

    def box_id(r: int, c: int) -> int:
        return (r // 3) * 3 + (c // 3)

    nodes_pool: List[Node] = []

    def append_to_column(col: Column, node: Node) -> None:
        node.down = col
        node.up = col.up
        col.up.down = node  # type: ignore
        col.up = node
        node.col = col
        col.size += 1

    row_id = 0
    for r in range(9):
        for c in range(9):
            if board.grid[r, c] != 0:
                digits = [int(board.grid[r, c])]
            else:
                digits = list(range(1, 10))
            for d in digits:
                cols = [
                    columns[col_index_cell(r, c)],
                    columns[col_index_row_digit(r, d)],
                    columns[col_index_col_digit(c, d)],
                    columns[col_index_box_digit(box_id(r, c), d)],
                ]
                row_nodes = [Node(row_id=row_id) for _ in range(4)]
                for i, col in enumerate(cols):
                    append_to_column(col, row_nodes[i])
                _link_lr(row_nodes)
                nodes_pool.extend(row_nodes)
                row_id += 1

    return header, nodes_pool


def _cover(col: Column) -> None:
    assert col.right is not None and col.left is not None
    col.right.left = col.left
    col.left.right = col.right
    i = col.down
    while i is not None and i is not col:
        j = i.right
        while j is not None and j is not i:
            assert j.down is not None and j.up is not None and j.col is not None
            j.down.up = j.up
            j.up.down = j.down
            j.col.size -= 1
            j = j.right
        i = i.down


def _uncover(col: Column) -> None:
    i = col.up
    while i is not None and i is not col:
        j = i.left
        while j is not None and j is not i:
            assert j.col is not None and j.down is not None and j.up is not None
            j.col.size += 1
            j.down.up = j
            j.up.down = j
            j = j.left
        i = i.up
    assert col.right is not None and col.left is not None
    col.right.left = col
    col.left.right = col


def _choose_column(header: Column) -> Optional[Column]:
    c = header.right
    best: Optional[Column] = None
    while c is not None and c is not header:
        assert isinstance(c, Column)
        if best is None or c.size < best.size:
            best = c
        c = c.right
    return best


def _iterate_dlx(header: Column, limit: Optional[int] = None) -> Generator[List[Node], None, None]:
    solution_stack: List[Node] = []

    def search(yielded: int) -> int:
        if header.right is header:
            yield_nodes_copy = solution_stack[:]
            nonlocal_gen.append(yield_nodes_copy)
            return yielded + 1
        col = _choose_column(header)
        if col is None or col.size == 0:
            return yielded
        _cover(col)
        r = col.down
        while r is not None and r is not col:
            solution_stack.append(r)
            j = r.right
            while j is not None and j is not r:
                assert j.col is not None
                _cover(j.col)
                j = j.right
            yielded = search(yielded)
            j = r.left
            while j is not None and j is not r:
                assert j.col is not None
                _uncover(j.col)
                j = j.left
            solution_stack.pop()
            if limit is not None and yielded >= limit:
                break
            r = r.down
        _uncover(col)
        return yielded

    nonlocal_gen: List[List[Node]] = []
    yielded = 0
    while True:
        nonlocal_gen.clear()
        yielded = search(yielded)
        if nonlocal_gen:
            yield nonlocal_gen[0]
            if limit is not None and yielded >= limit:
                return
        else:
            return


def _nodes_to_board(sol_nodes: List[Node]) -> Board:
    b = Board.empty()
    for n in sol_nodes:
        j = n
        indices: List[int] = []
        while True:
            assert j.col is not None and j.right is not None
            indices.append(int(j.col.name))
            j = j.right
            if j is n:
                break
        cell_idx = min(x for x in indices if x < 81)
        r, c = divmod(cell_idx, 9)
        rd = [x for x in indices if 81 <= x < 162][0]
        d = (rd - 81) % 9 + 1
        b.set_cell(r, c, d)
    return b


def solve_one(board: Board) -> Optional[Board]:
    """Solve a Sudoku puzzle using DLX algorithm.

    Note: This is used for training data generation only.
    For actual puzzle solving, use the AI model.
    """
    logger.debug("🔧 DLX: Building exact cover matrix...")
    header, _ = _build_exact_cover(board)

    logger.debug("🔧 DLX: Starting search algorithm...")
    for nodes in _iterate_dlx(header, limit=1):
        solution = _nodes_to_board(nodes)
        logger.debug("✅ DLX: Found solution")
        return solution

    logger.warning("❌ DLX: No solution found (puzzle may be invalid)")
    return None
