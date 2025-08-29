from __future__ import annotations

from typing import List, Tuple, Optional, Dict
from itertools import combinations
from dataclasses import dataclass, field
from sudoku_engine.board import Board, mask_to_digits


@dataclass
class Trace:
    steps: List[str]

    def add(self, technique: str, detail: str) -> None:
        self.steps.append(f"{technique}: {detail}")


@dataclass
class HeuristicStats:
    assignments: int = 0
    eliminations: int = 0
    assign_by_tech: Dict[str, int] = field(default_factory=dict)
    elim_by_tech: Dict[str, int] = field(default_factory=dict)


@dataclass
class Options:
    naked_singles: bool = True
    hidden_singles: bool = True
    pointing_pairs: bool = True
    naked_pairs: bool = True
    hidden_pairs: bool = True
    hidden_triples: bool = True
    x_wing: bool = True
    # Controls how many outer iterations run_pipeline performs per propagate
    max_iters: int = 3


def apply_naked_singles(board: Board, trace: Optional[Trace] = None) -> int:
    """Fill all naked singles; return number of assignments."""
    masks = board.candidates_mask()
    applied = 0
    for r in range(9):
        for c in range(9):
            if board.grid[r, c] == 0:
                m = int(masks[r, c])
                if m.bit_count() == 1:
                    v = mask_to_digits(m)[0]
                    board.set_cell(r, c, v)
                    # refresh masks snapshot after assignment
                    masks = board.candidates_mask()
                    if trace is not None:
                        trace.add("naked-single", f"R{r+1}C{c+1}={v}")
                    applied += 1
    return applied


def run_pipeline(
    board: Board,
    max_iters: int = 10,
    trace: Optional[Trace] = None,
    stats: Optional[HeuristicStats] = None,
    options: Optional[Options] = None,
) -> int:
    """Run a simple sequence of heuristics until no progress or max_iters.

    Returns number of assignments performed.
    """
    total = 0
    if options is None:
        options = Options()
    for _ in range(max_iters):
        delta = 0
        if options.naked_singles:
            a = apply_naked_singles(board, trace=trace)
            delta += a
            if stats is not None:
                stats.assignments += a
                stats.assign_by_tech["naked_singles"] = stats.assign_by_tech.get("naked_singles", 0) + a
        if options.hidden_singles:
            a = apply_hidden_singles(board, trace=trace)
            delta += a
            if stats is not None:
                stats.assignments += a
                stats.assign_by_tech["hidden_singles"] = stats.assign_by_tech.get("hidden_singles", 0) + a
        if options.pointing_pairs:
            e = apply_pointing_pairs(board, trace=trace)
            delta += e
            if stats is not None:
                stats.eliminations += e
                stats.elim_by_tech["pointing_pairs"] = stats.elim_by_tech.get("pointing_pairs", 0) + e
        if options.naked_pairs:
            e = apply_naked_pairs(board, trace=trace)
            delta += e
            if stats is not None:
                stats.eliminations += e
                stats.elim_by_tech["naked_pairs"] = stats.elim_by_tech.get("naked_pairs", 0) + e
        if options.hidden_pairs:
            e = apply_hidden_pairs(board, trace=trace)
            delta += e
            if stats is not None:
                stats.eliminations += e
                stats.elim_by_tech["hidden_pairs"] = stats.elim_by_tech.get("hidden_pairs", 0) + e
        if options.hidden_triples:
            e = apply_hidden_triples(board, trace=trace)
            delta += e
            if stats is not None:
                stats.eliminations += e
                stats.elim_by_tech["hidden_triples"] = stats.elim_by_tech.get("hidden_triples", 0) + e
        if options.x_wing:
            e = apply_x_wing(board, trace=trace)
            delta += e
            if stats is not None:
                stats.eliminations += e
                stats.elim_by_tech["x_wing"] = stats.elim_by_tech.get("x_wing", 0) + e
        if delta == 0:
            break
        total += delta
    return total


def apply_hidden_singles(board: Board, trace: Optional[Trace] = None) -> int:
    """Fill all hidden singles in rows, columns, and boxes; return count.

    Hidden single: a digit appears as a candidate in exactly one cell within a unit.
    """
    masks = board.candidates_mask()
    applied = 0

    # Rows
    for r in range(9):
        # For each digit 1..9, find cells in row r where candidate bit is set
        indices_by_digit: List[List[Tuple[int, int]]] = [[] for _ in range(9)]
        for c in range(9):
            if board.grid[r, c] == 0:
                m = int(masks[r, c])
                for d in range(1, 10):
                    if m & (1 << (d - 1)):
                        indices_by_digit[d - 1].append((r, c))
        for d, cells in enumerate(indices_by_digit, start=1):
            if len(cells) == 1:
                rr, cc = cells[0]
                board.set_cell(rr, cc, d)
                masks = board.candidates_mask()
                if trace is not None:
                    trace.add("hidden-single(row)", f"R{rr+1}C{cc+1}={d}")
                applied += 1

    # Columns
    masks = board.candidates_mask()
    for c in range(9):
        indices_by_digit = [[] for _ in range(9)]
        for r in range(9):
            if board.grid[r, c] == 0:
                m = int(masks[r, c])
                for d in range(1, 10):
                    if m & (1 << (d - 1)):
                        indices_by_digit[d - 1].append((r, c))
        for d, cells in enumerate(indices_by_digit, start=1):
            if len(cells) == 1:
                rr, cc = cells[0]
                board.set_cell(rr, cc, d)
                masks = board.candidates_mask()
                if trace is not None:
                    trace.add("hidden-single(col)", f"R{rr+1}C{cc+1}={d}")
                applied += 1

    # Boxes
    masks = board.candidates_mask()
    for br in range(3):
        for bc in range(3):
            indices_by_digit = [[] for _ in range(9)]
            for dr in range(3):
                for dc in range(3):
                    r = br * 3 + dr
                    c = bc * 3 + dc
                    if board.grid[r, c] == 0:
                        m = int(masks[r, c])
                        for d in range(1, 10):
                            if m & (1 << (d - 1)):
                                indices_by_digit[d - 1].append((r, c))
            for d, cells in enumerate(indices_by_digit, start=1):
                if len(cells) == 1:
                    rr, cc = cells[0]
                    board.set_cell(rr, cc, d)
                    masks = board.candidates_mask()
                    if trace is not None:
                        trace.add("hidden-single(box)", f"R{rr+1}C{cc+1}={d}")
                    applied += 1

    return applied


def apply_pointing_pairs(board: Board, trace: Optional[Trace] = None) -> int:
    """Eliminate candidates using pointing pairs/triples (box-line interactions).

    If candidates for a digit in a box are confined to one row (or column),
    eliminate that digit from the rest of that row (or column) outside the box.
    Returns number of eliminations (not assignments). Assignments may result indirectly.
    """
    masks = board.candidates_mask()
    eliminated = 0
    # For each box and digit, check confinement to a row or column
    for br in range(3):
        for bc in range(3):
            # gather positions by digit
            positions_by_digit: List[List[Tuple[int, int]]] = [[] for _ in range(9)]
            for dr in range(3):
                for dc in range(3):
                    r = br * 3 + dr
                    c = bc * 3 + dc
                    if board.grid[r, c] == 0:
                        m = int(masks[r, c])
                        for d in range(1, 10):
                            if m & (1 << (d - 1)):
                                positions_by_digit[d - 1].append((r, c))
            for d, cells in enumerate(positions_by_digit, start=1):
                if len(cells) < 2:
                    continue
                rows = {r for r, _ in cells}
                cols = {c for _, c in cells}
                if len(rows) == 1:
                    rr = next(iter(rows))
                    # eliminate d from row rr outside this box
                    for c in range(9):
                        if c // 3 == bc:
                            continue
                        if board.grid[rr, c] == 0 and (int(masks[rr, c]) & (1 << (d - 1))):
                            if board.remove_candidate(rr, c, d):
                                eliminated += 1
                                if trace is not None:
                                    trace.add("pointing(row)", f"R{rr+1} remove {d} from C{c+1}")
                elif len(cols) == 1:
                    cc = next(iter(cols))
                    # eliminate d from column cc outside this box
                    for r in range(9):
                        if r // 3 == br:
                            continue
                        if board.grid[r, cc] == 0 and (int(masks[r, cc]) & (1 << (d - 1))):
                            if board.remove_candidate(r, cc, d):
                                eliminated += 1
                                if trace is not None:
                                    trace.add("pointing(col)", f"C{cc+1} remove {d} at R{r+1}")
    return eliminated


def apply_naked_pairs(board: Board, trace: Optional[Trace] = None) -> int:
    """Eliminate candidates using naked pairs in rows, columns, and boxes.

    Returns number of eliminations (approximate), not direct assignments.
    """
    masks = board.candidates_mask()
    eliminated = 0

    def process_unit(cells: List[Tuple[int, int]], unit_name: str) -> int:
        nonlocal masks
        elim = 0
        # find cells with exactly two candidates
        pair_map: dict[int, List[Tuple[int, int]]] = {}
        for (r, c) in cells:
            if board.grid[r, c] != 0:
                continue
            m = int(masks[r, c])
            if m.bit_count() == 2:
                pair_map.setdefault(m, []).append((r, c))
        for m, locs in pair_map.items():
            if len(locs) == 2:
                # eliminate these two digits from other cells in unit
                bits = [i + 1 for i in range(9) if (m & (1 << i))]
                for (r, c) in cells:
                    if (r, c) in locs or board.grid[r, c] != 0:
                        continue
                    cm = int(masks[r, c])
                    common = cm & m
                    if common:
                        # remove each common bit via Board API
                        for d in bits:
                            if cm & (1 << (d - 1)):
                                if board.remove_candidate(r, c, d):
                                    elim += 1
                                    if trace is not None:
                                        dig = "/".join(str(b) for b in bits)
                                        trace.add("naked-pair", f"{unit_name} remove {dig} at R{r+1}C{c+1}")
        return elim

    # Rows
    for r in range(9):
        cells = [(r, c) for c in range(9)]
        eliminated += process_unit(cells, f"R{r+1}")
    # Cols
    for c in range(9):
        cells = [(r, c) for r in range(9)]
        eliminated += process_unit(cells, f"C{c+1}")
    # Boxes
    for br in range(3):
        for bc in range(3):
            cells = [(br * 3 + dr, bc * 3 + dc) for dr in range(3) for dc in range(3)]
            eliminated += process_unit(cells, f"B{br+1}{bc+1}")

    return eliminated


def apply_hidden_pairs(board: Board, trace: Optional[Trace] = None) -> int:
    """Hidden pairs: if two digits occupy exactly two cells in a unit, restrict those cells to the pair."""
    masks = board.candidates_mask()
    eliminated = 0

    def process_unit(cells: List[Tuple[int, int]], unit_name: str) -> int:
        nonlocal masks
        elim = 0
        # positions for each digit
        pos_by_digit: List[List[Tuple[int, int]]]=[[] for _ in range(9)]
        empties = [(r,c) for (r,c) in cells if board.grid[r,c]==0]
        for (r,c) in empties:
            m = int(masks[r,c])
            for d in range(1,10):
                if m & (1 << (d-1)):
                    pos_by_digit[d-1].append((r,c))
        for d1, d2 in combinations(range(1,10), 2):
            cells1 = pos_by_digit[d1-1]
            cells2 = pos_by_digit[d2-1]
            union = set(cells1) | set(cells2)
            if len(union) == 2 and len(cells1) >=1 and len(cells2) >=1:
                allowed = {d1, d2}
                for (r,c) in union:
                    m = int(masks[r,c])
                    # remove other digits
                    for d in range(1,10):
                        if d not in allowed and (m & (1 << (d-1))):
                            if board.remove_candidate(r,c,d):
                                elim += 1
                                if trace is not None:
                                    trace.add("hidden-pair", f"{unit_name} R{r+1}C{c+1} restrict to {d1}/{d2} (remove {d})")
        return elim

    # Rows, Cols, Boxes
    for r in range(9):
        eliminated += process_unit([(r,c) for c in range(9)], f"R{r+1}")
    for c in range(9):
        eliminated += process_unit([(r,c) for r in range(9)], f"C{c+1}")
    for br in range(3):
        for bc in range(3):
            cells = [(br*3+dr, bc*3+dc) for dr in range(3) for dc in range(3)]
            eliminated += process_unit(cells, f"B{br+1}{bc+1}")
    return eliminated


def apply_hidden_triples(board: Board, trace: Optional[Trace] = None) -> int:
    """Hidden triples: if three digits occupy exactly three cells in a unit, restrict those cells to the triple."""
    masks = board.candidates_mask()
    eliminated = 0

    def process_unit(cells: List[Tuple[int, int]], unit_name: str) -> int:
        nonlocal masks
        elim = 0
        pos_by_digit: List[List[Tuple[int, int]]]=[[] for _ in range(9)]
        empties = [(r,c) for (r,c) in cells if board.grid[r,c]==0]
        for (r,c) in empties:
            m = int(masks[r,c])
            for d in range(1,10):
                if m & (1 << (d-1)):
                    pos_by_digit[d-1].append((r,c))
        for d1, d2, d3 in combinations(range(1,10), 3):
            union = set(pos_by_digit[d1-1]) | set(pos_by_digit[d2-1]) | set(pos_by_digit[d3-1])
            if len(union) == 3 and all(len(pos_by_digit[d-1])>=1 for d in (d1,d2,d3)):
                allowed = {d1, d2, d3}
                for (r,c) in union:
                    m = int(masks[r,c])
                    for d in range(1,10):
                        if d not in allowed and (m & (1 << (d-1))):
                            if board.remove_candidate(r,c,d):
                                elim += 1
                                if trace is not None:
                                    trace.add("hidden-triple", f"{unit_name} R{r+1}C{c+1} restrict to {sorted(list(allowed))} (remove {d})")
        return elim

    for r in range(9):
        eliminated += process_unit([(r,c) for c in range(9)], f"R{r+1}")
    for c in range(9):
        eliminated += process_unit([(r,c) for r in range(9)], f"C{c+1}")
    for br in range(3):
        for bc in range(3):
            cells = [(br*3+dr, bc*3+dc) for dr in range(3) for dc in range(3)]
            eliminated += process_unit(cells, f"B{br+1}{bc+1}")
    return eliminated


def apply_x_wing(board: Board, trace: Optional[Trace] = None) -> int:
    """X-Wing for rows and columns: eliminate a digit from other lines when two lines share the same two positions."""
    masks = board.candidates_mask()
    eliminated = 0

    # Row-based X-Wing (pair of rows, same two columns)
    for d in range(1,10):
        row_cols: List[List[int]] = [[] for _ in range(9)]
        for r in range(9):
            cols = [c for c in range(9) if board.grid[r,c]==0 and (int(masks[r,c]) & (1 << (d-1)))]
            if len(cols) == 2:
                row_cols[r] = cols
        for r1, r2 in combinations(range(9), 2):
            if row_cols[r1] and row_cols[r1] == row_cols[r2]:
                c1, c2 = row_cols[r1]
                # eliminate d from other rows in these columns
                for rr in range(9):
                    if rr in (r1, r2):
                        continue
                    for cc in (c1, c2):
                        if board.grid[rr, cc]==0 and (int(masks[rr, cc]) & (1 << (d-1))):
                            if board.remove_candidate(rr, cc, d):
                                eliminated += 1
                                if trace is not None:
                                    trace.add("x-wing(row)", f"d={d} rows {r1+1},{r2+1} cols {c1+1},{c2+1} eliminate at R{rr+1}C{cc+1}")

    # Column-based X-Wing (pair of columns, same two rows)
    for d in range(1,10):
        col_rows: List[List[int]] = [[] for _ in range(9)]
        for c in range(9):
            rows = [r for r in range(9) if board.grid[r,c]==0 and (int(masks[r,c]) & (1 << (d-1)))]
            if len(rows) == 2:
                col_rows[c] = rows
        for c1, c2 in combinations(range(9), 2):
            if col_rows[c1] and col_rows[c1] == col_rows[c2]:
                r1, r2 = col_rows[c1]
                for cc in range(9):
                    if cc in (c1, c2):
                        continue
                    for rr in (r1, r2):
                        if board.grid[rr, cc]==0 and (int(masks[rr, cc]) & (1 << (d-1))):
                            if board.remove_candidate(rr, cc, d):
                                eliminated += 1
                                if trace is not None:
                                    trace.add("x-wing(col)", f"d={d} cols {c1+1},{c2+1} rows {r1+1},{r2+1} eliminate at R{rr+1}C{cc+1}")

    return eliminated
