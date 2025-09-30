from __future__ import annotations

import argparse
from pathlib import Path
import logging
from rich.console import Console

from sudoku_engine import Board, parse_line, board_to_line
from ui.tui import render_pretty

console = Console()
logger = logging.getLogger(__name__)


def load_puzzle(path: str | None, stdin_data: str | None) -> Board:
    """Load puzzle from file or stdin."""
    if path:
        text = Path(path).read_text(encoding="utf-8")
    elif stdin_data:
        text = stdin_data
    else:
        raise ValueError("No input provided. Use -i <file> or --stdin <puzzle>")

    grid = parse_line(text)
    return Board(grid)


def cmd_ai_solve(args: argparse.Namespace) -> None:
    """AI-powered Sudoku solver using trained neural network."""
    # Setup logging
    from sudoku_ai.logger_config import setup_logging
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = setup_logging(level=log_level, log_to_file=True)
    if log_file:
        logger.info(f"📝 Logging to: {log_file}")

    try:
        import torch
    except ImportError:
        console.print("❌ PyTorch is required. Install: pip install torch", style="red")
        raise SystemExit(1)

    from sudoku_ai.policy import load_policy, board_to_tensor

    # Load puzzle
    try:
        board = load_puzzle(args.input, args.stdin)
    except Exception as e:
        console.print(f"❌ Failed to load puzzle: {e}", style="red")
        raise SystemExit(1)

    console.print(f"📋 Loaded puzzle from: {args.input or 'stdin'}")
    if args.pretty:
        console.print(render_pretty(board))

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    console.print(f"🔧 Using device: {device}")

    try:
        policy = load_policy(args.ckpt, device=device)
        console.print(f"✅ Loaded model from: {args.ckpt}")
    except Exception as e:
        console.print(f"❌ Failed to load model: {e}", style="red")
        console.print("💡 Train a model first or provide valid checkpoint path", style="yellow")
        raise SystemExit(1)

    # Solve puzzle
    console.print(f"🤖 Solving puzzle (max steps: {args.max_steps}, temperature: {args.temperature})...")
    logger.info(f"Starting AI solving loop (max_steps={args.max_steps}, temperature={args.temperature})")

    b = board.copy()
    temperature = max(0.01, float(args.temperature))
    moves_made = 0

    for step in range(args.max_steps):
        if b.is_complete():
            logger.info(f"✅ Puzzle completed in {step} steps!")
            break

        # Get model prediction
        line = board_to_line(b.grid)
        x = board_to_tensor(line).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = policy(x)[0]  # (81, 9)
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)

        logger.debug(f"Step {step+1}: Got NN predictions")

        # Create legal move mask
        masks = b.candidates_mask()
        mask_tensor = torch.zeros(81, 9, device=probs.device)

        for idx in range(81):
            r, c = divmod(idx, 9)
            if b.grid[r, c] != 0:
                continue  # Already filled

            m = int(masks[r, c])
            if m == 0:
                continue  # No legal moves

            for d in range(1, 10):
                if m & (1 << (d - 1)):
                    mask_tensor[idx, d - 1] = 1.0

        # Apply mask and sample
        masked_probs = probs * mask_tensor
        flat = masked_probs.view(-1)
        total = float(flat.sum().item())

        if total <= 0.0:
            console.print(f"❌ No legal moves at step {step + 1}", style="red")
            raise SystemExit(1)

        # Sample move
        dist = flat / flat.sum()
        choice = int(torch.multinomial(dist, num_samples=1).item())
        cell, dig_idx = divmod(choice, 9)
        r, c = divmod(cell, 9)
        d = dig_idx + 1

        # Make move
        b.set_cell(r, c, d)
        moves_made += 1

        logger.debug(f"Step {step+1}: Placed {d} at R{r+1}C{c+1}")

        if args.verbose:
            console.print(f"  Step {step + 1}: R{r+1}C{c+1}={d}")
        elif (step + 1) % 10 == 0:
            filled = sum(1 for i in range(81) if b.grid[divmod(i, 9)] != 0)
            console.print(f"  Progress: {step+1} steps, {filled}/81 cells filled...")

    # Check completion
    if not b.is_complete():
        filled = sum(1 for i in range(81) if b.grid[divmod(i, 9)] != 0)
        logger.error(f"Failed to complete: {filled}/81 cells filled after {moves_made} moves")
        console.print(f"❌ Did not complete within {args.max_steps} steps ({filled}/81 filled)", style="red")
        raise SystemExit(1)

    # Output solution
    solution = board_to_line(b.grid)
    logger.info(f"✅ Solution found in {moves_made} moves!")
    console.print(f"\n✅ Solution found in {moves_made} moves!", style="green bold")
    console.print(solution)

    if args.pretty:
        console.print("\n📊 Pretty board:")
        console.print(render_pretty(b))


def cmd_train(args: argparse.Namespace) -> None:
    """Train a Sudoku AI model."""
    # Setup logging
    from sudoku_ai.logger_config import setup_logging
    log_level = "DEBUG" if getattr(args, 'verbose', False) else "INFO"
    log_file = setup_logging(level=log_level, log_to_file=True)
    if log_file:
        logger.info(f"📝 Logging to: {log_file}")
        console.print(f"📝 Logging to: {log_file}")

    try:
        import torch
    except ImportError:
        console.print("❌ PyTorch is required. Install: pip install torch", style="red")
        raise SystemExit(1)

    from sudoku_ai.policy import train_supervised

    console.print("🎯 Starting training...")
    console.print(f"  Dataset: {args.dataset or args.puzzles or 'N/A'}")
    console.print(f"  Epochs: {args.epochs}")
    console.print(f"  Batch size: {args.batch_size}")
    console.print(f"  Output: {args.output}")

    # Load puzzles from file if provided
    puzzles_list = None
    if args.puzzles:
        console.print(f"📂 Loading puzzles from: {args.puzzles}")
        puzzle_path = Path(args.puzzles)
        if not puzzle_path.exists():
            console.print(f"❌ Puzzle file not found: {args.puzzles}", style="red")
            raise SystemExit(1)
        puzzles_list = [line.strip() for line in puzzle_path.read_text().splitlines() if line.strip()]
        console.print(f"✅ Loaded {len(puzzles_list)} puzzles")

    try:
        result = train_supervised(
            out_path=args.output,
            dataset_jsonl=args.dataset,
            puzzles=puzzles_list,
            solutions=None,  # Will be generated
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_split=args.val_split,
            max_samples=args.max_samples,
            augment=args.augment,
            seed=args.seed,
        )
        console.print(f"\n✅ Training complete! Model saved to: {args.output}", style="green bold")

        history = result.get("history", {})
        if history.get("val_loss"):
            final_loss = history["val_loss"][-1]
            final_acc = history["val_acc"][-1]
            console.print(f"📊 Final val_loss: {final_loss:.4f}, val_acc: {final_acc:.3f}")

    except Exception as e:
        console.print(f"❌ Training failed: {e}", style="red")
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sudoku",
        description="AI-Powered Sudoku Solver using Neural Networks",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # AI Solve command
    solve_parser = subparsers.add_parser(
        "solve",
        aliases=["ai-solve"],
        help="Solve Sudoku puzzle using AI model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    solve_parser.add_argument(
        "-i", "--input",
        type=str,
        help="Path to puzzle file (.sdk format, 81 chars)"
    )
    solve_parser.add_argument(
        "--stdin",
        type=str,
        help="Puzzle string directly (81 chars, 0 for empty)"
    )
    solve_parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/policy.pt",
        help="Path to model checkpoint"
    )
    solve_parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA available"
    )
    solve_parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum solving steps"
    )
    solve_parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (lower = more deterministic)"
    )
    solve_parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the board"
    )
    solve_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show step-by-step moves"
    )
    solve_parser.set_defaults(func=cmd_ai_solve)

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train AI model on puzzle-solution dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument(
        "--dataset",
        type=str,
        help="Path to JSONL dataset (puzzle+solution or puzzle-only, solutions auto-generated)"
    )
    train_parser.add_argument(
        "--puzzles",
        type=str,
        help="Path to text file with puzzles (one per line, solutions auto-generated)"
    )
    train_parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/policy.pt",
        help="Output checkpoint path"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size"
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    train_parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split fraction"
    )
    train_parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum training samples to use"
    )
    train_parser.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="Disable data augmentation"
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    train_parser.set_defaults(func=cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
