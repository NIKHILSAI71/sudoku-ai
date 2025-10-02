"""GNN Sudoku Solver Training Script.

This script trains the GNN model with curriculum learning and mixed precision.

Usage:
    python scripts/train.py --data sudoku.csv --epochs 60

For quick testing:
    python scripts/train.py --data sudoku.csv --max-samples 10000 --epochs 10
"""

import argparse
import sys
from pathlib import Path
import logging

import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gnn.sudoku_gnn import SudokuGNN
from src.training.trainer import GNNTrainer
from src.utils.logger import setup_logging

# Setup logging
setup_logging('logs')
logger = logging.getLogger(__name__)


def parse_sudoku_string(s: str) -> np.ndarray:
    """Parse Kaggle format sudoku string to 9x9 array."""
    return np.array([int(c) for c in s]).reshape(9, 9)


def load_sudoku_csv(file_path: str, max_samples = None):
    """Load Sudoku puzzles from CSV file.
    
    Args:
        file_path: Path to CSV with 'quizzes' and 'solutions' columns
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        puzzles, solutions as numpy arrays
    """
    import csv
    
    puzzles = []
    solutions = []
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break
            puzzles.append(parse_sudoku_string(row['quizzes']))
            solutions.append(parse_sudoku_string(row['solutions']))
    
    return np.array(puzzles), np.array(solutions)


def main():
    parser = argparse.ArgumentParser(
        description="GNN Sudoku Solver Training"
    )
    
    # Data arguments
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to CSV file with puzzles and solutions (Kaggle format)'
    )
    parser.add_argument(
        '--max-samples', type=int, default=None,
        help='Maximum samples to use (for testing)'
    )
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=60,
                       help='Total number of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio')
    
    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=96,
                       help='Hidden dimension')
    parser.add_argument('--num-iterations', type=int, default=32,
                       help='Message passing iterations')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Curriculum learning
    parser.add_argument('--curriculum', type=str, default='20,20,20',
                       help='Curriculum epochs per stage (comma-separated)')
    parser.add_argument('--no-augment', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--no-mixed-precision', action='store_true',
                       help='Disable mixed precision training')
    
    # Output arguments
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of DataLoader workers')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Parse curriculum
    curriculum_epochs = [int(x) for x in args.curriculum.split(',')]
    
    # Setup device
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    logger.info("="*70)
    logger.info("  Starting GNN Sudoku Training")
    logger.info("="*70)
    
    logger.info(f"Device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load dataset
    logger.info(f"Loading dataset from: {args.data}")
    print("Loading dataset with columns: 'quizzes' and 'solutions'")
    
    puzzles, solutions = load_sudoku_csv(args.data, args.max_samples)
    
    logger.info(f"Loaded {len(puzzles)} puzzles")
    logger.info(f"Grid size: 9x9")
    
    # Convert to tensors
    puzzles = torch.from_numpy(puzzles).long()
    solutions = torch.from_numpy(solutions).long()
    
    # Create model
    logger.info("Creating GNN model...")
    logger.info(f"  Hidden dim: {args.hidden_dim}")
    logger.info(f"  Iterations: {args.num_iterations}")
    logger.info(f"  Dropout: {args.dropout}")
    
    model = SudokuGNN(
        hidden_dim=args.hidden_dim,
        num_iterations=args.num_iterations,
        dropout=args.dropout
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    logger.info("Initializing trainer...")
    
    trainer = GNNTrainer(
        model=model,
        device=device,
        lr=args.lr,
        weight_decay=0.01,
        use_amp=not args.no_mixed_precision,
        checkpoint_dir=args.checkpoint_dir,
        gradient_clip=1.0,
        warmup_epochs=3
    )
    
    # Log training configuration
    logger.info("\nStarting training...")
    logger.info(f"  Total epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Validation split: {args.val_split}")
    logger.info(f"  Curriculum: {curriculum_epochs}")
    logger.info(f"  Augmentation: {not args.no_augment}")
    logger.info(f"  Mixed precision: {not args.no_mixed_precision}")
    logger.info(f"  Device: {device}")
    logger.info("")
    
    # Train
    trainer.train(
        puzzles=puzzles,
        solutions=solutions,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
        curriculum_epochs=curriculum_epochs,
        augment=not args.no_augment,
        num_workers=args.num_workers
    )
    
    logger.info("\n" + "="*70)
    logger.info("Training completed successfully!")
    logger.info(f"Best model saved to: {args.checkpoint_dir}/policy_best.pt")
    logger.info("="*70)


if __name__ == '__main__':
    main()
