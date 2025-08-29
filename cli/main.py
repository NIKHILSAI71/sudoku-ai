from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List
from rich.console import Console
from rich.table import Table
import logging
from datetime import datetime

from sudoku_engine import Board, parse_line, board_to_line
from sudoku_solvers import backtracking
from sudoku_solvers.heuristics import Trace, run_pipeline, Options
from sudoku_solvers import dlx
from ui.tui import render_pretty

console = Console()


def _init_logger(run_type: str) -> tuple[logging.Logger, Path]:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"{run_type}-{ts}.log"

    logger = logging.getLogger(f"sudoku.{run_type}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Do not add a console StreamHandler to avoid duplicate TUI output
    return logger, log_path


# generator removed


def _load_board(path: str | None, stdin_data: str | None) -> Board:
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        text = stdin_data or ""
    grid = parse_line(text)
    return Board(grid)


def _options_from_args(args: argparse.Namespace) -> Options:
    return Options(
        naked_singles=not args.no_naked_singles,
        hidden_singles=not args.no_hidden_singles,
        pointing_pairs=not args.no_pointing_pairs,
        naked_pairs=not args.no_naked_pairs,
        hidden_pairs=not args.no_hidden_pairs,
        hidden_triples=not args.no_hidden_triples,
        x_wing=not args.no_x_wing,
    )


def cmd_solve(args: argparse.Namespace) -> None:
    logger, log_path = _init_logger("solve")
    logger.info("solve start: input=%s backend=%s", args.input or "<stdin>", args.backend)
    board = _load_board(args.input, args.stdin)
    backend = args.backend
    opts = _options_from_args(args)
    if backend == "dlx":
        sol = dlx.solve_one(board)
    elif backend == "sat":
        try:
            from sudoku_solvers import sat_adapter
        except Exception as e:
            console.print(str(e), style="red")
            logger.exception("sat backend load failed")
            raise SystemExit(3)
        sol = sat_adapter.solve_one(board)
    elif backend == "cpsat":
        try:
            from sudoku_solvers import cpsat
        except Exception as e:
            console.print(str(e), style="red")
            logger.exception("cpsat backend load failed")
            raise SystemExit(3)
        sol = cpsat.solve_one(board)
    else:
        stats = backtracking.Stats()
        if args.trace:
            t = Trace(steps=[])
            b2 = board.copy()
            _ = run_pipeline(b2, max_iters=50, trace=t, options=opts, stats=stats.heuristics)
            if t.steps and not getattr(args, "trace_json", False):
                table = Table(title="Heuristics Trace")
                table.add_column("#")
                table.add_column("step")
                for i, s in enumerate(t.steps, start=1):
                    table.add_row(str(i), s)
                console.print(table)
            if t.steps and getattr(args, "trace_json", False):
                console.print_json(data={"trace": t.steps})
            # also log trace
            for s in (t.steps or []):
                logger.info("trace: %s", s)
            sol = backtracking.solve_one(b2, stats=stats, trace=t, options=opts)
        else:
            sol = backtracking.solve_one(board, stats=stats, options=opts)
    if sol is None:
        msg = "No solution found"
        console.print(msg, style="red")
        logger.error(msg)
        console.print(f"Log saved -> {log_path}")
        raise SystemExit(2)
    line = board_to_line(sol.grid)
    console.print(line)
    logger.info("solution: %s", line)
    if getattr(args, "pretty", False):
        pretty = render_pretty(sol)
        console.print(pretty)
        logger.info("\n%s", pretty)
    if getattr(args, "print_stats", False) and backend in ("backtracking",):
        console.print(f"nodes={stats.nodes}, assignments={stats.heuristics.assignments}, eliminations={stats.heuristics.eliminations}")
        logger.info("stats: nodes=%d assignments=%d eliminations=%d", stats.nodes, stats.heuristics.assignments, stats.heuristics.eliminations)
        if args.verbose_stats:
            table = Table(title="Heuristic Stats")
            table.add_column("type")
            table.add_column("technique")
            table.add_column("count")
            for k, v in sorted(stats.heuristics.assign_by_tech.items()):
                table.add_row("assign", k, str(v))
                logger.info("assign:%s=%d", k, v)
            for k, v in sorted(stats.heuristics.elim_by_tech.items()):
                table.add_row("elim", k, str(v))
                logger.info("elim:%s=%d", k, v)
            console.print(table)
    logger.info("solve end")
    console.print(f"Log saved -> {log_path}")


# difficulty rater removed


def cmd_train(args: argparse.Namespace) -> None:
    logger, log_path = _init_logger("train")
    logger.info(
        "train start: epochs=%d limit=%s out=%s dataset=%s puzzles=%s solutions=%s amp=%s",
        args.epochs,
        getattr(args, "limit", None),
        args.out,
        getattr(args, "dataset", None),
        getattr(args, "puzzles", None),
        getattr(args, "solutions", None),
        getattr(args, "amp", False),
    )
    from sudoku_ai.policy import train_toy, train_supervised

    ckpt = Path(args.out)
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    def _progress(ep: int, loss: float, acc: float) -> None:
        logger.info("epoch %d: loss=%.6f acc=%.4f", ep, loss, acc)

    # If a dataset path or puzzles file is provided, run the supervised pipeline; otherwise keep toy
    if getattr(args, "dataset", None) or getattr(args, "puzzles", None):
        _ = train_supervised(
            out_path=str(ckpt),
            dataset_jsonl=getattr(args, "dataset", None),
            puzzles_path=getattr(args, "puzzles", None),
            solutions_path=getattr(args, "solutions", None),
            epochs=args.epochs,
            batch_size=getattr(args, "batch_size", 64),
            lr=getattr(args, "lr", 1e-3),
            val_split=getattr(args, "val_split", 0.1),
            max_samples=getattr(args, "limit", None),
            augment=not getattr(args, "no_augment", False),
            amp=getattr(args, "amp", False),
            seed=getattr(args, "seed", 42),
            progress_cb=_progress,
        )
    else:
        train_toy(epochs=args.epochs, limit=args.limit, out_path=str(ckpt), progress_cb=_progress)

    msg = f"Saved checkpoint -> {ckpt}"
    console.print(msg)
    logger.info(msg)
    logger.info("train end")
    console.print(f"Log saved -> {log_path}")


def cmd_eval(args: argparse.Namespace) -> None:
    path = Path(args.input)
    lines = path.read_text(encoding="utf-8").splitlines()
    table = Table(title="Eval")
    table.add_column("idx")
    table.add_column("backend")
    table.add_column("ok")
    for i, ln in enumerate(lines[: args.limit]):
        rec = json.loads(ln)
        board = Board(parse_line(rec["puzzle"]))
        for backend in args.backends:
            if backend == "dlx":
                sol = dlx.solve_one(board)
            else:
                sol = backtracking.solve_one(board)
            ok = sol is not None
            table.add_row(str(i), backend, "1" if ok else "0")
    console.print(table)


def cmd_ai_solve(args: argparse.Namespace) -> None:
    logger, log_path = _init_logger("ai-solve")
    logger.info("ai-solve start: input=%s ckpt=%s cpu=%s no-prop=%s max-steps=%d", args.input or "<stdin>", args.ckpt, args.cpu, args.no_prop, args.max_steps)
    try:
        import torch  # type: ignore
    except Exception as e:
        console.print("PyTorch is required for ai-solve. Please install torch.", style="red")
        logger.exception("torch import failed")
        raise SystemExit(3)
    from sudoku_ai.policy import load_policy
    from sudoku_engine import board_to_line as _to_line

    board = _load_board(args.input, args.stdin)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    policy = load_policy(args.ckpt, device=device)
    logger.info("device: %s", device)

    t = Trace(steps=[])
    stats = backtracking.Stats()
    opts = _options_from_args(args)

    def propagate(b: Board) -> Board | None:
        if args.no_prop:
            masks = b.candidates_mask()
            for r in range(9):
                for c in range(9):
                    if b.grid[r, c] == 0 and int(masks[r, c]) == 0:
                        return None
            return b
        else:
            masks = backtracking._propagate(b, trace=t if args.trace else None, hstats=stats.heuristics, options=opts)
            if masks is None:
                return None
            return b

    b = board.copy()
    # keep a pristine copy for fallback
    start_b = b.copy()
    if propagate(b) is None:
        msg = "Unsolvable or contradictory starting board"
        console.print(msg, style="red")
        logger.error(msg)
        console.print(f"Log saved -> {log_path}")
        raise SystemExit(2)

    max_steps = args.max_steps
    fell_back = False
    for step in range(max_steps):
        if b.is_complete():
            break
        line = _to_line(b.grid)
        from sudoku_ai.policy import board_to_tensor as _bt
        x = _bt(line).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = policy(x)
            probs = torch.softmax(logits, dim=-1)[0]
        masks = b.candidates_mask()
        mask_tensor = torch.zeros(81, 9, device=probs.device)
        for idx in range(81):
            r, c = divmod(idx, 9)
            m = int(masks[r, c])
            if b.grid[r, c] != 0 or m == 0:
                continue
            for d in range(1, 10):
                if m & (1 << (d - 1)):
                    mask_tensor[idx, d - 1] = 1.0
        masked = probs * mask_tensor
        flat_vals = masked.view(-1)
        values, indices = torch.sort(flat_vals, descending=True)
        moved = False
        for rank in range(int(indices.numel())):
            val = float(values[rank].item())
            if val <= 0.0:
                break
            idx = int(indices[rank].item())
            cell, dig_idx = divmod(idx, 9)
            r, c = divmod(cell, 9)
            d = dig_idx + 1
            nb = b.copy()
            nb.set_cell(r, c, d)
            logger.info("step %d try R%dC%d=%d (score=%.4f)", step + 1, r + 1, c + 1, d, val)
            if args.trace:
                t.add("policy", f"R{r+1}C{c+1}={d}")
            if propagate(nb) is not None:
                b = nb
                moved = True
                logger.info("accepted R%dC%d=%d", r + 1, c + 1, d)
                break
            if args.trace:
                t.steps[-1] += " (rejected)"
            logger.info("rejected R%dC%d=%d", r + 1, c + 1, d)
        if not moved:
            warn = "AI-only policy got stuck; falling back to backtracking..."
            console.print(warn, style="yellow")
            logger.warning(warn)
            sol_fb = backtracking.solve_one(start_b, stats=stats, trace=(t if args.trace else None), options=opts)
            if sol_fb is None:
                msg = "Fallback backtracking also failed (unsolvable from current state)."
                console.print(msg, style="red")
                logger.error(msg)
                console.print(f"Log saved -> {log_path}")
                raise SystemExit(1)
            b = sol_fb
            fell_back = True
            break

    if not b.is_complete() and not fell_back:
        logger.warning("AI-only solver reached max steps without completion; falling back to backtracking...")
        console.print("AI-only solver reached max steps; falling back to backtracking...", style="yellow")
        sol_fb = backtracking.solve_one(start_b, stats=stats, trace=(t if args.trace else None), options=opts)
        if sol_fb is None:
            msg = "Fallback backtracking failed (unsolvable from current state)."
            console.print(msg, style="red")
            logger.error(msg)
            console.print(f"Log saved -> {log_path}")
            raise SystemExit(1)
        b = sol_fb

    line = board_to_line(b.grid)
    console.print(line)
    logger.info("solution: %s", line)
    if getattr(args, "pretty", False):
        pretty = render_pretty(b)
        console.print(pretty)
        logger.info("\n%s", pretty)
    if args.trace and t.steps:
        table = Table(title="AI Trace")
        table.add_column("#")
        table.add_column("step")
        for i, s in enumerate(t.steps, start=1):
            table.add_row(str(i), s)
            logger.info("trace: %s", s)
        console.print(table)
    logger.info("ai-solve end")
    console.print(f"Log saved -> {log_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="sudoku",
        description=(
        "Sudoku toolkit: solve, evaluate, and AI-solve puzzles.\n\n"
        "Common examples:\n"
        "  sudoku solve -i examples/easy1.sdk --pretty\n"
        "  sudoku eval -i data.jsonl -l 100 -b dlx backtracking\n"
        "  sudoku ai-solve -i examples/easy1.sdk --ckpt checkpoints/policy.pt\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="{solve,train,eval,ai-solve}")

    ap_solve = sub.add_parser(
        "solve",
        aliases=["slove"],
        help="solve puzzle",
        description="Solve a Sudoku from a file or stdin using various backends.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap_solve.add_argument("-i", "--input", type=str, default=None, help="Path to input puzzle (single-line .sdk format)")
    ap_solve.add_argument("--stdin", type=str, default=None, help="Read puzzle from the provided string instead of a file")
    ap_solve.add_argument(
        "-b",
        "--backend",
        type=str,
        default="backtracking",
        choices=["backtracking", "dlx", "sat", "cpsat"],
        help="Solver backend to use",
    )  # sat/cpsat optional
    ap_solve.add_argument("--trace", action="store_true", help="Print heuristics trace before solving (backtracking only)")
    ap_solve.add_argument("--trace-json", action="store_true", help="Emit heuristics trace as JSON (mutually exclusive with table)")
    ap_solve.add_argument("--print-stats", action="store_true", help="Print solver stats like node count (backtracking only)")
    ap_solve.add_argument("--no-naked-singles", action="store_true", help="Disable Naked Singles technique")
    ap_solve.add_argument("--no-hidden-singles", action="store_true", help="Disable Hidden Singles technique")
    ap_solve.add_argument("--no-pointing-pairs", action="store_true", help="Disable Pointing Pairs technique")
    ap_solve.add_argument("--no-naked-pairs", action="store_true", help="Disable Naked Pairs technique")
    ap_solve.add_argument("--no-hidden-pairs", action="store_true", help="Disable Hidden Pairs technique")
    ap_solve.add_argument("--no-hidden-triples", action="store_true", help="Disable Hidden Triples technique")
    ap_solve.add_argument("--no-x-wing", action="store_true", help="Disable X-Wing technique")
    ap_solve.add_argument("--verbose-stats", action="store_true", help="Print per-technique stats in a table")
    ap_solve.add_argument("--pretty", action="store_true", help="Pretty-print the solved grid as a 9x9 board")
    ap_solve.set_defaults(func=cmd_solve)

    # rate subcommand removed

    ap_train = sub.add_parser(
        "train",
        help="train learning agent (toy)",
        description="Train a small policy network on synthetic boards (toy example).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap_train.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    ap_train.add_argument("--limit", type=int, default=500, help="Max samples (toy or supervised)")
    ap_train.add_argument("--out", type=str, default="checkpoints/policy.pt", help="Where to save the checkpoint")
    # Supervised options (optional)
    ap_train.add_argument("--dataset", type=str, default=None, help="Path to JSONL dataset with fields 'puzzle' and optional 'solution'")
    ap_train.add_argument("--puzzles", type=str, default=None, help="Path to file with puzzles, one per line")
    ap_train.add_argument("--solutions", type=str, default=None, help="Optional solutions file aligned with --puzzles")
    ap_train.add_argument("--batch-size", type=int, default=64, help="Batch size for supervised training")
    ap_train.add_argument("--lr", type=float, default=3e-4, help="Learning rate for supervised training")
    ap_train.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction")
    ap_train.add_argument("--seed", type=int, default=42, help="Random seed")
    ap_train.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    ap_train.add_argument("--amp", action="store_true", help="Enable mixed precision (AMP)")
    ap_train.set_defaults(func=cmd_train)

    ap_eval = sub.add_parser(
        "eval",
        help="evaluate backends on dataset",
        description="Evaluate one or more solver backends on an input JSONL dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap_eval.add_argument("-i", "--input", type=str, required=True, help="Path to JSONL with records containing 'puzzle'")
    ap_eval.add_argument("-l", "--limit", type=int, default=50, help="Max number of puzzles to evaluate")
    ap_eval.add_argument(
        "-b",
        "--backends",
        nargs="+",
        default=["dlx", "backtracking"],
        choices=["dlx", "backtracking"],
        help="Backends to evaluate",
    )
    ap_eval.set_defaults(func=cmd_eval)

    ap_ai = sub.add_parser(
        "ai-solve",
        aliases=["ai-slove"],
        help="solve puzzle using a policy checkpoint",
        description="Solve a Sudoku using a trained policy; optionally combine with light propagation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap_ai.add_argument("-i", "--input", type=str, default=None, help="Path to input puzzle (single-line .sdk format)")
    ap_ai.add_argument("--stdin", type=str, default=None, help="Read puzzle from the provided string instead of a file")
    ap_ai.add_argument("--ckpt", type=str, default="checkpoints/policy.pt", help="Path to policy checkpoint (.pt)")
    ap_ai.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    ap_ai.add_argument("--max-steps", type=int, default=200, help="Max policy steps")
    ap_ai.add_argument("--no-prop", action="store_true", help="Disable heuristic propagation (pure policy mode)")
    ap_ai.add_argument("--no-naked-singles", action="store_true", help="Disable Naked Singles technique during propagation")
    ap_ai.add_argument("--no-hidden-singles", action="store_true", help="Disable Hidden Singles technique during propagation")
    ap_ai.add_argument("--no-pointing-pairs", action="store_true", help="Disable Pointing Pairs technique during propagation")
    ap_ai.add_argument("--no-naked-pairs", action="store_true", help="Disable Naked Pairs technique during propagation")
    ap_ai.add_argument("--no-hidden-pairs", action="store_true", help="Disable Hidden Pairs technique during propagation")
    ap_ai.add_argument("--no-hidden-triples", action="store_true", help="Disable Hidden Triples technique during propagation")
    ap_ai.add_argument("--no-x-wing", action="store_true", help="Disable X-Wing technique during propagation")
    ap_ai.add_argument("--pretty", action="store_true", help="Pretty-print the final grid as a 9x9 board")
    ap_ai.add_argument("--trace", action="store_true", help="Log AI move attempts and propagation steps")
    ap_ai.set_defaults(func=cmd_ai_solve)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
