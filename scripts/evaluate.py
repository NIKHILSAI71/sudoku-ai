"""Evaluation script for trained GNN Sudoku solver.

Run from command line:
    python scripts/evaluate.py --model checkpoints/gnn_best.pt --puzzles examples/test.sdk
"""

import argparse
from pathlib import Path
import sys
import time

import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sudoku_ai.gnn_policy import SudokuGNNPolicy
from sudoku_ai.inference import hybrid_solve, iterative_solve, batch_solve
from sudoku_ai.metrics import evaluate_solver, SolverMetrics
from sudoku_ai.graph import create_sudoku_graph
from sudoku_engine import Board, parse_line


def load_puzzles(file_path: str):
    """Load puzzles from file (one per line, 81 characters)."""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return lines


def visualize_puzzle(puzzle_str: str, solution_str: str = None):
    """Pretty print puzzle and optionally solution."""
    def format_grid(s):
        grid = np.array([int(c) for c in s]).reshape(9, 9)
        lines = []
        for i in range(9):
            if i > 0 and i % 3 == 0:
                lines.append("------+-------+------")
            row = []
            for j in range(9):
                if j > 0 and j % 3 == 0:
                    row.append("|")
                val = grid[i, j]
                row.append(str(val) if val != 0 else '.')
                row.append(" ")
            lines.append(''.join(row))
        return '\n'.join(lines)
    
    print("\nPuzzle:")
    print(format_grid(puzzle_str))
    
    if solution_str:
        print("\nSolution:")
        print(format_grid(solution_str))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GNN Sudoku Solver"
    )
    
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--puzzles', type=str, required=True,
        help='Path to puzzle file (one puzzle per line)'
    )
    parser.add_argument(
        '--method', type=str, default='hybrid',
        choices=['iterative', 'hybrid', 'beam_search'],
        help='Solving method'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Visualize each puzzle and solution'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device (cuda/cpu)'
    )
    parser.add_argument(
        '--max-puzzles', type=int, default=None,
        help='Maximum number of puzzles to evaluate'
    )
    
    args = parser.parse_args()
    
    # Setup
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print(f"  GNN Sudoku Solver Evaluation")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Method: {args.method}")
    print(f"{'='*70}\n")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.model, map_location=device)
    
    # Extract model config
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        hidden_dim = config.get('hidden_dim', 96)
        num_iterations = config.get('num_iterations', 32)
    else:
        hidden_dim = 96
        num_iterations = 32
    
    model = SudokuGNNPolicy(
        grid_size=9,
        hidden_dim=hidden_dim,
        num_iterations=num_iterations
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully")
    
    # Load puzzles
    print(f"\nLoading puzzles from: {args.puzzles}")
    puzzles = load_puzzles(args.puzzles)
    
    if args.max_puzzles:
        puzzles = puzzles[:args.max_puzzles]
    
    print(f"Loaded {len(puzzles)} puzzles")
    
    # Create graph structure
    edge_index, n_cells, n_constraints = create_sudoku_graph(9)
    edge_index = edge_index.to(device)
    
    # Solve puzzles
    print(f"\nSolving puzzles using {args.method} method...\n")
    
    results = []
    total_time = 0
    method_counts = {}
    
    for i, puzzle_str in enumerate(puzzles):
        # Convert to tensor
        puzzle_np = np.array([int(c) for c in puzzle_str]).reshape(9, 9)
        puzzle = torch.from_numpy(puzzle_np).to(device).long()
        
        # Solve
        start_time = time.time()
        
        if args.method == 'hybrid':
            solution, method_used, solve_time = hybrid_solve(
                model, puzzle, edge_index, grid_size=9, device=device
            )
            method_counts[method_used] = method_counts.get(method_used, 0) + 1
        elif args.method == 'iterative':
            solution, iters, solve_time = iterative_solve(
                model, puzzle, edge_index, grid_size=9, device=device
            )
            method_used = 'iterative'
        else:  # beam_search
            from sudoku_ai.inference import beam_search_solve
            solution, solve_time = beam_search_solve(
                model, puzzle, edge_index, beam_width=5, grid_size=9, device=device
            )
            method_used = 'beam_search' if solution is not None else 'failed'
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Check if solved
        is_solved = (solution != 0).all()
        
        # Visualize if requested
        if args.visualize and i < 5:  # Show first 5
            solution_str = ''.join(str(x.item()) for x in solution.flatten())
            visualize_puzzle(puzzle_str, solution_str)
            print(f"\nMethod: {method_used}, Time: {elapsed*1000:.2f}ms, Solved: {is_solved}")
            print("-" * 70)
        
        results.append({
            'puzzle': puzzle_str,
            'solved': is_solved,
            'method': method_used if args.method == 'hybrid' else args.method,
            'time': elapsed
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(puzzles)} puzzles...")
    
    # Print summary
    solved_count = sum(1 for r in results if r['solved'])
    solve_rate = 100.0 * solved_count / len(results)
    avg_time = total_time / len(results)
    
    print(f"\n{'='*70}")
    print(f"  Evaluation Summary")
    print(f"{'='*70}")
    print(f"Total puzzles: {len(results)}")
    print(f"Solved: {solved_count} ({solve_rate:.2f}%)")
    print(f"Average time: {avg_time*1000:.2f}ms per puzzle")
    print(f"Total time: {total_time:.2f}s")
    
    if args.method == 'hybrid' and method_counts:
        print(f"\nMethod breakdown:")
        for method, count in method_counts.items():
            pct = 100.0 * count / len(results)
            print(f"  {method}: {count} ({pct:.1f}%)")
    
    print(f"{'='*70}\n")
    
    if solve_rate == 100.0:
        print("🎉 Perfect solve rate achieved!")
    elif solve_rate >= 95.0:
        print("✅ Excellent performance!")
    elif solve_rate >= 90.0:
        print("👍 Good performance")
    else:
        print("⚠️  Performance below target")
    
    print(f"\nExpected performance:")
    print(f"  Iterative: 95-98% solve rate, 30-50ms per puzzle")
    print(f"  Hybrid: 100% solve rate, 10-100ms per puzzle")


if __name__ == '__main__':
    main()
