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


def _init_logger(run_type: str, add_stdout: bool = False) -> tuple[logging.Logger, Path]:
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

    # Optionally mirror logs to stdout (useful for notebook training progress)
    if add_stdout:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # For TUI-heavy commands we avoid console handlers by default
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


# commands removed: rater, train, eval


def cmd_ai_solve(args: argparse.Namespace) -> None:
    logger, log_path = _init_logger("ai-solve")
    logger.info(
    "ai-solve start: input=%s ckpt=%s cpu=%s max-steps=%d train=%s temperature=%.3f",
    args.input or "<stdin>", args.ckpt, args.cpu, args.max_steps, getattr(args, "train", False), float(getattr(args, "temperature", 1.0)),
    )
    try:
        import torch  # type: ignore
    except Exception as e:
        console.print("PyTorch is required for ai-solve. Please install torch.", style="red")
        logger.exception("torch import failed")
        raise SystemExit(3)
    from sudoku_ai.policy import load_policy, train_toy, train_supervised
    from sudoku_engine import board_to_line as _to_line

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info("device: %s", device)
    # We'll load policy after any optional training step

    t = Trace(steps=[])

    def valid_state(b: Board) -> bool:
        # Minimal consistency: every empty cell must have at least one legal candidate
        masks = b.candidates_mask()
        for r in range(9):
            for c in range(9):
                if b.grid[r, c] == 0 and int(masks[r, c]) == 0:
                    return False
        return True

    # Optional on-the-fly training: can run with dataset/puzzles without requiring an input board
    board: Board | None = None
    if getattr(args, "train", False):
        ckpt = Path(args.ckpt)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        epochs = int(getattr(args, "train_epochs", 1))
        limit = int(getattr(args, "train_limit", 500))
        dataset = getattr(args, "dataset", None)
        puzzles_path = getattr(args, "puzzles", None)
        solutions_path = getattr(args, "solutions", None)
        try:
            if dataset or puzzles_path:
                logger.info(
                    "supervised training from dataset: epochs=%d max_samples=%d dataset=%s puzzles=%s solutions=%s -> %s",
                    epochs, limit, dataset, puzzles_path, solutions_path, ckpt,
                )
                _ = train_supervised(
                    out_path=str(ckpt),
                    dataset_jsonl=dataset,
                    puzzles_path=puzzles_path,
                    solutions_path=solutions_path,
                    epochs=max(1, epochs),
                    batch_size=64,
                    lr=3e-4,
                    val_split=0.1,
                    max_samples=max(100, limit),
                    augment=True,
                    amp=False,
                    seed=42,
                    overfit=False,
                )
                # If no puzzle provided, this was a training-only run
                if args.input is None and args.stdin is None:
                    policy = load_policy(args.ckpt, device=device)
                    console.print(f"Training complete. Checkpoint saved -> {ckpt}")
                    logger.info("training-only run complete -> %s", ckpt)
                    console.print(f"Log saved -> {log_path}")
                    return
            else:
                # Need a board to derive supervised labels from the single input puzzle
                if args.input is None and args.stdin is None:
                    msg = "No input puzzle provided. Use -i/--stdin to solve, or pass --dataset/--puzzles with --train for training-only."
                    console.print(msg, style="red")
                    logger.error(msg)
                    console.print(f"Log saved -> {log_path}")
                    raise SystemExit(2)
                board = _load_board(args.input, args.stdin)
                logger.info("supervised training from input: epochs=%d max_samples=%d -> %s", epochs, limit, ckpt)
                # Write a temporary puzzles file with the single input puzzle line
                tmp_puzzles = ckpt.parent / "_tmp_puzzles.txt"
                tmp_puzzles.write_text(board_to_line(board.grid) + "\n", encoding="utf-8")
                _ = train_supervised(
                    out_path=str(ckpt),
                    dataset_jsonl=None,
                    puzzles_path=str(tmp_puzzles),
                    solutions_path=None,  # solver will create solutions
                    epochs=max(1, epochs),
                    batch_size=32,
                    lr=3e-4,
                    val_split=0.0,
                    max_samples=max(100, limit),
                    augment=True,
                    amp=False,
                    seed=42,
                    overfit=True,
                    overfit_size=min(1024, max(100, limit)),
                )
        except Exception as e:
            logger.exception("supervised training failed; falling back to toy training: %s", e)
            train_toy(epochs=epochs, limit=limit, out_path=str(ckpt))
        policy = load_policy(args.ckpt, device=device)
    else:
        # Not training, ensure we have an input board
        if args.input is None and args.stdin is None:
            msg = "No input puzzle provided. Use -i/--stdin to solve, or combine with --train and dataset to train."
            console.print(msg, style="red")
            logger.error(msg)
            console.print(f"Log saved -> {log_path}")
            raise SystemExit(2)
        board = _load_board(args.input, args.stdin)
        policy = load_policy(args.ckpt, device=device)

    # At this point we must have a board for solving
    assert board is not None
    b = board.copy()
    if not valid_state(b):
        msg = "Unsolvable or contradictory starting board"
        console.print(msg, style="red")
        logger.error(msg)
        console.print(f"Log saved -> {log_path}")
        raise SystemExit(2)

    max_steps = args.max_steps
    # Pure policy sampling loop (no greedy, no propagation)
    temperature = max(1e-6, float(getattr(args, "temperature", 1.0)))
    for step in range(max_steps):
        if b.is_complete():
            break
        line = _to_line(b.grid)
        from sudoku_ai.policy import board_to_tensor as _bt
        x = _bt(line).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = policy(x)[0]
            # Temperature scaling
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
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
        flat = masked.view(-1)
        total = float(flat.sum().item())
        if total <= 0.0:
            msg = "AI sampling has no legal moves (dead state)."
            console.print(msg, style="red")
            logger.error(msg)
            console.print(f"Log saved -> {log_path}")
            raise SystemExit(1)
        # Resample until a valid move is found or distribution exhausted
        moved = False
        while True:
            dist = flat / flat.sum()
            choice = int(torch.multinomial(dist, num_samples=1).item())
            cell, dig_idx = divmod(choice, 9)
            r, c = divmod(cell, 9)
            d = dig_idx + 1
            nb = b.copy()
            nb.set_cell(r, c, d)
            if valid_state(nb):
                b = nb
                if args.trace:
                    t.add("policy", f"R{r+1}C{c+1}={d}")
                logger.info("accepted R%dC%d=%d", r + 1, c + 1, d)
                moved = True
                break
            # zero out invalid choice and try again
            flat[choice] = 0.0
            if float(flat.sum().item()) <= 0.0:
                if args.trace:
                    t.add("policy", f"R{r+1}C{c+1}={d} (rejected)")
                logger.info("no valid moves remain after rejections at step %d", step + 1)
                break
        if not moved:
            msg = "AI sampling exhausted without a valid move."
            console.print(msg, style="red")
            logger.error(msg)
            console.print(f"Log saved -> {log_path}")
            raise SystemExit(1)

    # No final fallback; AI must have completed by here or exited

    if not b.is_complete():
        msg = "AI did not complete within max-steps."
        console.print(msg, style="red")
        logger.error(msg)
        console.print(f"Log saved -> {log_path}")
        raise SystemExit(1)
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
        "Sudoku toolkit: solve with engine or AI.\n\n"
        "Common examples:\n"
        "  sudoku solve -i examples/easy1.sdk --pretty\n"
    "  sudoku ai-solve -i examples/easy1.sdk --ckpt checkpoints/policy.pt\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="{solve,ai-solve}")

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

    # train/eval subcommands removed

    ap_ai = sub.add_parser(
        "ai-solve",
        aliases=["ai-slove"],
        help="solve puzzle using a policy checkpoint",
    description="Solve a Sudoku using a trained policy (no propagation).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap_ai.add_argument("-i", "--input", type=str, default=None, help="Path to input puzzle (single-line .sdk format)")
    ap_ai.add_argument("--stdin", type=str, default=None, help="Read puzzle from the provided string instead of a file")
    ap_ai.add_argument("--ckpt", type=str, default="checkpoints/policy.pt", help="Path to policy checkpoint (.pt)")
    ap_ai.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    ap_ai.add_argument("--max-steps", type=int, default=1000, help="Max policy steps")
    ap_ai.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (>1 more random, <1 sharper)")
    # On-the-fly training flags
    ap_ai.add_argument("--train", action="store_true", help="Train a tiny policy before solving (toy)")
    ap_ai.add_argument("--train-epochs", type=int, default=1, help="Epochs for on-the-fly training")
    ap_ai.add_argument("--train-limit", type=int, default=500, help="Samples limit for on-the-fly training")
    ap_ai.add_argument("--dataset", type=str, default=None, help="Path to JSONL dataset with 'puzzle' and optional 'solution'")
    ap_ai.add_argument("--puzzles", type=str, default=None, help="Path to text file of puzzles (one per line)")
    ap_ai.add_argument("--solutions", type=str, default=None, help="Path to text file of solutions (one per line; optional)")
    # Propagation flags removed for AI-only mode
    ap_ai.add_argument("--pretty", action="store_true", help="Pretty-print the final grid as a 9x9 board")
    ap_ai.add_argument("--trace", action="store_true", help="Log AI sampling decisions")
    ap_ai.set_defaults(func=cmd_ai_solve)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
