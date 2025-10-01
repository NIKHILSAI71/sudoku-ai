from __future__ import annotations

import argparse
from pathlib import Path
import logging
from rich.console import Console

from sudoku_engine import Board, parse_line, board_to_line
from sudoku_engine.board import mask_to_digits
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
    initial_temperature = max(0.01, float(args.temperature))
    temperature = initial_temperature
    moves_made = 0

    # Track original puzzle cells to ensure we never modify them
    original_cells = set()
    for i in range(81):
        r, c = divmod(i, 9)
        if board.grid[r, c] != 0:
            original_cells.add((r, c))

    logger.info(f"Original puzzle has {len(original_cells)} pre-filled cells")

    for step in range(args.max_steps):
        if b.is_complete():
            logger.info(f"✅ Puzzle completed in {moves_made} steps!")
            break

        # Encode current board state
        line = board_to_line(b.grid)
        x = board_to_tensor(line).to(device)

        with torch.no_grad():
            # Get legal move mask for this board state
            masks = b.candidates_mask()
            mask_tensor = torch.zeros(81, 9, device=device)

            num_legal_moves = 0
            for idx in range(81):
                r, c = divmod(idx, 9)
                if b.grid[r, c] != 0:
                    continue  # Already filled

                m = int(masks[r, c])
                if m == 0:
                    continue  # No legal moves for this cell

                for d in range(1, 10):
                    if m & (1 << (d - 1)):
                        mask_tensor[idx, d - 1] = 1.0
                        num_legal_moves += 1

            # Check if any legal moves exist
            if num_legal_moves == 0:
                filled = sum(1 for i in range(81) if b.grid[divmod(i, 9)] != 0)
                logger.error(f"No legal moves at step {step + 1} ({filled}/81 cells filled)")
                console.print(f"❌ No legal moves available at step {step + 1} ({filled}/81 cells filled)", style="red")
                console.print(f"💡 The puzzle may be unsolvable or the model made an error earlier.", style="yellow")
                raise SystemExit(1)

            # Get model predictions with constraint awareness (pass mask)
            logits = policy(x.unsqueeze(0), mask=mask_tensor.unsqueeze(0))[0]  # (81, 9)

            # Apply temperature and compute probabilities
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)

            # Flatten to get distribution over all (cell, digit) pairs
            flat = probs.view(-1)
            total = float(flat.sum().item())

            # Normalize to valid probability distribution
            if total > 0:
                dist = flat / total
            else:
                # Fallback: uniform over legal moves (shouldn't happen now)
                logger.warning(f"Step {step+1}: Probability sum is zero, using uniform distribution")
                dist = mask_tensor.view(-1) / mask_tensor.sum()

        logger.debug(f"Step {step+1}: Sampling from {num_legal_moves} legal moves (temp={temperature:.3f})")

        # Sample move
        choice = int(torch.multinomial(dist, num_samples=1).item())
        cell, dig_idx = divmod(choice, 9)
        r, c = divmod(cell, 9)
        d = dig_idx + 1

        # Safety checks
        if (r, c) in original_cells:
            logger.error(f"Step {step+1}: CRITICAL - Attempted to overwrite ORIGINAL puzzle cell R{r+1}C{c+1}={board.grid[r,c]}")
            console.print(f"❌ CRITICAL ERROR: Tried to overwrite original puzzle cell!", style="red bold")
            raise SystemExit(1)

        if b.grid[r, c] != 0:
            logger.error(f"Step {step+1}: Attempted to overwrite filled cell R{r+1}C{c+1}={b.grid[r,c]}")
            console.print(f"❌ Error: Tried to overwrite a filled cell", style="red")
            raise SystemExit(1)

        m = int(masks[r, c])
        if not (m & (1 << (d - 1))):
            logger.error(f"Step {step+1}: Illegal move {d} at R{r+1}C{c+1} (candidates: {mask_to_digits(m)})")
            console.print(f"❌ Error: Illegal move attempted", style="red")
            raise SystemExit(1)

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

    # Validate solution
    from sudoku_engine.validator import is_valid_board

    if not is_valid_board(b):
        logger.error("Solution is INVALID - contains duplicates!")
        console.print("❌ Solution is INVALID - contains duplicates!", style="red bold")
        raise SystemExit(1)

    # Verify original cells were not modified
    for r, c in original_cells:
        if b.grid[r, c] != board.grid[r, c]:
            logger.error(f"CRITICAL: Original cell R{r+1}C{c+1} was modified!")
            console.print(f"❌ CRITICAL: Original puzzle was modified!", style="red bold")
            raise SystemExit(1)

    # Output solution
    solution = board_to_line(b.grid)
    logger.info(f"✅ Valid solution found in {moves_made} moves!")
    console.print(f"\n✅ Valid solution found in {moves_made} moves!", style="green bold")
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
    console.print(f"  Max samples: {args.max_samples or 'unlimited (will use all data)'}")
    console.print(f"  Output: {args.output}")

    # Helpful tip for large datasets
    if args.max_samples is None:
        console.print("\n💡 [yellow]Tip: For large datasets, use --max-samples to limit training data (e.g., --max-samples 20000)[/yellow]")

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
        # Training complete message already printed by policy.py
        history = result.get("history", {})
        if history.get("val_loss"):
            final_loss = history["val_loss"][-1]
            final_acc_unmask = history.get("val_acc_unmasked", [0])[-1]
            final_acc_mask = history.get("val_acc_masked", [0])[-1]
            console.print(f"\n📈 Training Summary:", style="bold")
            console.print(f"   Final loss: {final_loss:.4f}")
            console.print(f"   Unmasked accuracy: {final_acc_unmask:.3f} (model learning)")
            console.print(f"   Masked accuracy: {final_acc_mask:.3f} (with constraints)")

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
        default=0.3,
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
        help="Maximum training samples (recommended: 10000-50000 for large datasets)"
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
    train_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging"
    )
    train_parser.set_defaults(func=cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
