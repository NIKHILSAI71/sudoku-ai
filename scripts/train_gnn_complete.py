"""Standalone training script for GNN Sudoku solver.

Run from command line:
    python scripts/train_gnn_complete.py --data sudoku.csv --epochs 60 --batch-size 128
"""

import argparse
from pathlib import Path
import sys

import torch
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sudoku_ai.gnn_policy import SudokuGNNPolicy
from sudoku_ai.gnn_trainer import train_gnn_supervised
from sudoku_ai.inference import hybrid_solve, evaluate_solver
from sudoku_ai.metrics import evaluate_solver as eval_metrics
from sudoku_ai.graph import create_sudoku_graph


def main():
    parser = argparse.ArgumentParser(
        description="Train GNN Sudoku Solver with State-of-the-Art Architecture"
    )
    
    # Data arguments
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to CSV file with puzzles and solutions (Kaggle format)'
    )
    parser.add_argument(
        '--max-samples', type=int, default=None,
        help='Maximum number of samples to use (for quick testing)'
    )
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=96, help='Hidden dimension')
    parser.add_argument('--num-iterations', type=int, default=32, help='Message passing iterations')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    
    # Optimization arguments
    parser.add_argument('--no-mixed-precision', action='store_true', help='Disable mixed precision')
    parser.add_argument('--no-curriculum', action='store_true', help='Disable curriculum learning')
    parser.add_argument('--no-augment', action='store_true', help='Disable data augmentation')
    parser.add_argument('--lambda-constraint', type=float, default=0.1, help='Constraint loss weight')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='checkpoints/gnn_best.pt', help='Output model path')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print(f"  GNN Sudoku Solver Training")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*70}\n")
    
    # Load data
    print(f"Loading data from: {args.data}")
    df = pd.read_csv(args.data)
    
    if args.max_samples:
        df = df.head(args.max_samples)
    
    print(f"Dataset size: {len(df):,} puzzles")
    
    # Parse puzzles
    def parse_sudoku(s):
        return np.array([int(c) for c in s]).reshape(9, 9)
    
    puzzles = np.array([parse_sudoku(p) for p in df['quizzes']])
    solutions = np.array([parse_sudoku(s) for s in df['solutions']])
    
    # Convert to strings (expected format)
    puzzle_strs = [''.join(str(x) for x in p.flatten()) for p in puzzles]
    solution_strs = [''.join(str(x) for x in s.flatten()) for s in solutions]
    
    print(f"Puzzles parsed: {len(puzzle_strs):,}")
    
    # Training configuration
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  Message passing iterations: {args.num_iterations}")
    print(f"  Mixed precision: {not args.no_mixed_precision}")
    print(f"  Curriculum learning: {not args.no_curriculum}")
    print(f"  Data augmentation: {not args.no_augment}")
    print(f"  Constraint loss weight: {args.lambda_constraint}")
    print(f"  Output: {args.output}")
    
    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Train
    print(f"\n{'='*70}")
    print("Starting Training...")
    print(f"{'='*70}\n")
    
    result = train_gnn_supervised(
        out_path=args.output,
        dataset_jsonl=None,
        puzzles=puzzle_strs,
        solutions=solution_strs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        max_samples=args.max_samples,
        augment=not args.no_augment,
        use_curriculum=not args.no_curriculum,
        hidden_dim=args.hidden_dim,
        num_iterations=args.num_iterations,
        lambda_constraint=args.lambda_constraint,
        use_mixed_precision=not args.no_mixed_precision,
        checkpoint_dir=args.checkpoint_dir,
        device=device
    )
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Final Results:")
    print(f"  Best validation accuracy: {result.get('best_val_acc', 0):.2f}%")
    print(f"  Training time: {result.get('training_time', 0):.2f} seconds")
    print(f"  Model saved to: {args.output}")
    print(f"{'='*70}\n")
    
    # Evaluate on validation set
    if 'val_metrics' in result:
        print("Validation Metrics:")
        print(result['val_metrics'])
    
    print("\n✅ Training completed successfully!")
    print("\nTo use the model for solving:")
    print(f"  python scripts/evaluate.py --model {args.output} --puzzles examples/test.sdk")


if __name__ == '__main__':
    main()
