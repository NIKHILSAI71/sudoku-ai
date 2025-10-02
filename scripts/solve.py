"""Simple solving script for GNN Sudoku solver.

Quick start solving:
    python scripts/solve.py --puzzle examples/easy1.sdk --checkpoint checkpoints/policy_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import logging
import time

import torch
import numpy as np

from src.utils.logger import setup_logging
from src.models.gnn.sudoku_gnn import load_pretrained_model
from src.inference.hybrid_solver import HybridSolver
from src.core.parser import parse_line
from src.core.board import Board
from src.core.validator import is_valid_board


def print_board(grid: np.ndarray, title: str = ""):
    """Pretty print Sudoku board."""
    if title:
        print(f"\n{title}")
        print("=" * 37)
    
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("------+-------+------")
        
        row = ""
        for j in range(9):
            if j % 3 == 0 and j != 0:
                row += "| "
            
            val = grid[i, j]
            row += f"{val if val != 0 else '.'} "
        
        print(row)


def main():
    """Main solving entry point."""
    parser = argparse.ArgumentParser(
        description="Solve Sudoku puzzles with trained GNN model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input
    parser.add_argument(
        "--puzzle",
        type=str,
        required=True,
        help="Path to puzzle file (.sdk format)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/policy_best.pt",
        help="Path to model checkpoint"
    )
    
    # Inference strategy
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        choices=["auto", "iterative", "beam", "backtrack"],
        help="Solving strategy"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence threshold for iterative solving"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Max iterations for iterative solving"
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=5,
        help="Beam width for beam search"
    )
    
    # Display
    parser.add_argument(
        "--no-pretty",
        action="store_true",
        help="Disable pretty board display"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level, log_to_file=False)
    logger = logging.getLogger(__name__)
    
    print("\n" + "=" * 70)
    print("  GNN Sudoku Solver")
    print("=" * 70)
    
    # Load puzzle
    puzzle_path = Path(args.puzzle)
    if not puzzle_path.exists():
        logger.error(f"Puzzle file not found: {args.puzzle}")
        raise SystemExit(1)
    
    puzzle_str = puzzle_path.read_text().strip()
    puzzle_grid = parse_line(puzzle_str)
    puzzle_tensor = torch.from_numpy(puzzle_grid).long()
    
    if not args.no_pretty:
        print_board(puzzle_grid, "Input Puzzle:")
    
    # Load model
    logger.info(f"Loading model from: {args.checkpoint}")
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        logger.error("Train a model first with: python scripts/train.py --dataset <path>")
        raise SystemExit(1)
    
    model = load_pretrained_model(
        checkpoint_path=str(checkpoint_path),
        device=args.device
    )
    logger.info(f"Model loaded successfully on {args.device}")
    
    # Create solver
    solver = HybridSolver(
        model=model,
        device=args.device,
        confidence_threshold=args.confidence,
        max_iterations=args.max_iterations,
        beam_width=args.beam_width
    )
    
    # Solve
    print(f"\nSolving with strategy: {args.strategy}")
    print(f"Confidence threshold: {args.confidence}")
    
    start_time = time.time()
    solution, info = solver.solve(puzzle_tensor, strategy=args.strategy)
    solve_time = time.time() - start_time
    
    # Display results
    print("\n" + "=" * 70)
    if info['success']:
        print("  ✅ PUZZLE SOLVED!")
    else:
        print("  ⚠️ SOLVING FAILED")
    print("=" * 70)
    
    print(f"\nStrategy used: {info['strategy']}")
    print(f"Solve time: {solve_time*1000:.2f} ms")
    print(f"Iterations: {info.get('iterations', 'N/A')}")
    
    if not args.no_pretty:
        solution_np = solution.cpu().numpy()
        print_board(solution_np, "\nSolution:")
    
    # Validate
    solution_board = Board(solution.cpu().numpy())
    is_valid = is_valid_board(solution_board)
    is_complete = solution_board.is_complete()
    
    print(f"\nValidation:")
    print(f"  Valid: {'✅' if is_valid else '❌'}")
    print(f"  Complete: {'✅' if is_complete else '❌'}")
    
    if is_valid and is_complete:
        print("\n🎉 Perfect solution!")
        return 0
    else:
        print("\n⚠️ Solution has issues")
        return 1


if __name__ == "__main__":
    exit(main())
