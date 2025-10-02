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
from src.inference.neural_solver import load_neural_solver
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
        default="iterative",
        choices=["greedy", "iterative", "careful"],
        help="Solving strategy: greedy (fast), iterative (balanced), careful (accurate)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.90,
        help="Initial confidence threshold for iterative solving"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Max iterations for iterative solving"
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
    
    # Load neural solver
    logger.info(f"Loading neural solver from: {args.checkpoint}")
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        logger.error("Train a model first with: python scripts/train.py --data <path>")
        raise SystemExit(1)
    
    solver = load_neural_solver(
        checkpoint_path=str(checkpoint_path),
        device=args.device,
        verbose=args.verbose
    )
    logger.info(f"🧠 Neural solver loaded successfully on {args.device}")
    
    # Solve
    print("\n" + "=" * 70)
    print("  🧠 PURE NEURAL SOLVING (No Classical Algorithms)")
    print("=" * 70)
    print(f"\nStrategy: {args.strategy}")
    print(f"Initial confidence: {args.confidence}")
    print(f"Max iterations: {args.max_iterations}\n")
    
    solution, info = solver.solve(
        puzzle_tensor,
        strategy=args.strategy,
        max_iterations=args.max_iterations,
        initial_confidence=args.confidence
    )
    
    # Display results
    print("\n" + "=" * 70)
    if info['success']:
        print("  ✅ SOLVED BY NEURAL NETWORK!")
    else:
        print("  ⚠️ INCOMPLETE SOLUTION")
    print("=" * 70)
    
    print(f"\nMethod: {info.get('method', 'unknown')}")
    print(f"Solve time: {info['solve_time_ms']:.2f} ms")
    print(f"Iterations: {info.get('iterations', 1)}")
    print(f"Cells filled: {info.get('cells_filled', 0)}")
    print(f"Avg confidence: {info.get('avg_confidence', 0.0):.3f}")
    
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
