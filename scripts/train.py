"""Simple training script for GNN Sudoku solver.

Quick start training with Kaggle dataset:
    python scripts/train.py --dataset data/sudoku.csv --epochs 60
"""

from __future__ import annotations

import argparse
from pathlib import Path
import logging

import torch

from src.utils.logger import setup_logging
from src.data.dataset import load_kaggle_dataset
from src.models.gnn.sudoku_gnn import SudokuGNN
from src.training.trainer import GNNTrainer


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train GNN-based Sudoku solver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to Kaggle CSV dataset (puzzle,solution format)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit training samples (useful for testing)"
    )
    
    # Model architecture
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=96,
        help="Hidden dimension for GNN"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=32,
        help="Message passing iterations"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate"
    )
    
    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Total training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--curriculum",
        type=int,
        nargs=3,
        default=[20, 20, 20],
        help="Curriculum epochs [easy, medium, hard]"
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation"
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision training"
    )
    
    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output (DEBUG level)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else args.log_level
    setup_logging(level=log_level, log_to_file=True)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("  Starting GNN Sudoku Training")
    logger.info("=" * 70)
    
    # Load dataset
    logger.info(f"Loading dataset from: {args.dataset}")
    dataset_path = Path(args.dataset)
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {args.dataset}")
        raise SystemExit(1)
    
    puzzles, solutions = load_kaggle_dataset(
        file_path=dataset_path,
        max_samples=args.max_samples,
        grid_size=9
    )
    
    logger.info(f"Loaded {len(puzzles)} puzzles")
    logger.info(f"Grid size: {puzzles.shape[-1]}x{puzzles.shape[-1]}")
    
    # Create model
    logger.info("Creating GNN model...")
    logger.info(f"  Hidden dim: {args.hidden_dim}")
    logger.info(f"  Iterations: {args.num_iterations}")
    logger.info(f"  Dropout: {args.dropout}")
    
    model = SudokuGNN(
        hidden_dim=args.hidden_dim,
        num_iterations=args.num_iterations,
        dropout=args.dropout,
        grid_size=9
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    logger.info("Initializing trainer...")
    trainer = GNNTrainer(
        model=model,
        device=args.device,
        lr=args.lr,
        use_amp=not args.no_amp,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Resume if checkpoint provided
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    logger.info("\nStarting training...")
    logger.info(f"  Total epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Validation split: {args.val_split}")
    logger.info(f"  Curriculum: {args.curriculum}")
    logger.info(f"  Augmentation: {not args.no_augment}")
    logger.info(f"  Mixed precision: {not args.no_amp}")
    logger.info(f"  Device: {args.device}")
    logger.info("")
    
    try:
        trainer.train(
            puzzles=puzzles,
            solutions=solutions,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            val_split=args.val_split,
            curriculum_epochs=args.curriculum,
            augment=not args.no_augment,
            num_workers=args.num_workers
        )
        
        logger.info("\n" + "=" * 70)
        logger.info("  Training Completed Successfully!")
        logger.info("=" * 70)
        logger.info(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
        logger.info(f"Checkpoints saved to: {args.checkpoint_dir}")
        logger.info(f"  - Latest: {args.checkpoint_dir}/policy.pt")
        logger.info(f"  - Best: {args.checkpoint_dir}/policy_best.pt")
        
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint(
            epoch=trainer.current_epoch,
            metrics={'interrupted': True},
            is_best=False
        )
        logger.info("Checkpoint saved. You can resume with --resume")
        raise SystemExit(0)
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
