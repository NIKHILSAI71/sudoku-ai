"""Production-optimized training script with all enhancements.

This script uses the enhanced ProductionGNNTrainer with:
- Gradient accumulation
- Model EMA
- Early stopping
- Enhanced monitoring
- Optimized DataLoader
- Better learning rate scheduling

Usage:
    python scripts/train_production.py --data sudoku.csv --epochs 60

For quick testing:
    python scripts/train_production.py --data sudoku.csv --max-samples 10000 --epochs 10
"""

import argparse
import sys
from pathlib import Path
import logging

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gnn.sudoku_gnn import SudokuGNN
from src.training.trainer_production import ProductionGNNTrainer
from src.utils.logger import setup_logging

# Setup logging
setup_logging('logs')
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'configs/training.yaml') -> dict:
    """Load training configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['training']


def parse_sudoku_string(s: str) -> np.ndarray:
    """Parse Kaggle format sudoku string to 9x9 array."""
    return np.array([int(c) for c in s]).reshape(9, 9)


def main():
    parser = argparse.ArgumentParser(
        description="Production-Grade GNN Sudoku Solver Training"
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
    parser.add_argument('--config', type=str, default='configs/training.yaml',
                       help='Path to training config file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override total epochs from config')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size from config')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate from config')
    
    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=96,
                       help='Hidden dimension')
    parser.add_argument('--num-iterations', type=int, default=32,
                       help='Message passing iterations')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Optimization arguments
    parser.add_argument('--no-ema', action='store_true',
                       help='Disable model EMA')
    parser.add_argument('--no-early-stopping', action='store_true',
                       help='Disable early stopping')
    parser.add_argument('--accumulation-steps', type=int, default=None,
                       help='Override gradient accumulation steps')
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'onecycle'],
                       default=None, help='Learning rate scheduler type')
    
    # Output arguments
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Log directory')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    batch_size = args.batch_size or config['batch_size']
    lr = args.lr or config['optimizer']['lr']
    accumulation_steps = args.accumulation_steps or config.get('gradient_accumulation_steps', 2)
    use_ema = not args.no_ema and config.get('ema', {}).get('enabled', True)
    use_early_stopping = not args.no_early_stopping and config.get('early_stopping', {}).get('enabled', True)
    scheduler_type = args.scheduler or config['scheduler']['type']
    
    # Curriculum epochs
    curr_config = config['curriculum']
    curriculum_epochs = [
        curr_config['stage1_epochs'],
        curr_config['stage2_epochs'],
        curr_config['stage3_epochs']
    ]
    total_epochs = args.epochs or sum(curriculum_epochs)
    
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
    
    df = pd.read_csv(args.data)
    
    if args.max_samples:
        logger.info(f"Limiting to {args.max_samples} samples for testing")
        df = df.head(args.max_samples)
    
    logger.info(f"Loaded {len(df)} puzzles")
    logger.info(f"Grid size: 9x9")
    
    # Parse puzzles
    puzzles = np.array([parse_sudoku_string(p) for p in df['quizzes']])
    solutions = np.array([parse_sudoku_string(s) for s in df['solutions']])
    
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
    logger.info("Initializing production trainer...")
    
    trainer = ProductionGNNTrainer(
        model=model,
        device=device,
        lr=lr,
        weight_decay=config['optimizer']['weight_decay'],
        use_amp=config['mixed_precision'],
        checkpoint_dir=args.checkpoint_dir,
        gradient_clip=config['gradient_clip'],
        warmup_epochs=3,
        # Production features
        gradient_accumulation_steps=accumulation_steps,
        use_ema=use_ema,
        ema_decay=config.get('ema', {}).get('decay', 0.9999),
        early_stopping_patience=config.get('early_stopping', {}).get('patience', 15) if use_early_stopping else 999,
        scheduler_type=scheduler_type,
        max_lr=config['scheduler'].get('max_lr', 0.003)
    )
    
    # Log training configuration
    logger.info("\nStarting training...")
    logger.info(f"  Total epochs: {total_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Gradient accumulation: {accumulation_steps}")
    logger.info(f"  Effective batch size: {batch_size * accumulation_steps}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Scheduler: {scheduler_type}")
    logger.info(f"  Validation split: {config.get('val_split', 0.1)}")
    logger.info(f"  Curriculum: {curriculum_epochs}")
    logger.info(f"  Augmentation: {config.get('augmentation', {}).get('enabled', True)}")
    logger.info(f"  Mixed precision: {config['mixed_precision']}")
    logger.info(f"  Model EMA: {use_ema}")
    logger.info(f"  Early stopping: {use_early_stopping}")
    logger.info(f"  Device: {device}")
    logger.info("")
    
    # Train
    trainer.train(
        puzzles=puzzles,
        solutions=solutions,
        num_epochs=total_epochs,
        batch_size=batch_size,
        val_split=0.1,
        curriculum_epochs=curriculum_epochs,
        augment=True,
        num_workers=config['num_workers'],
        prefetch_factor=config.get('prefetch_factor', 2),
        persistent_workers=config.get('persistent_workers', True)
    )
    
    logger.info("\n" + "="*70)
    logger.info("Training completed successfully!")
    logger.info(f"Best model saved to: {args.checkpoint_dir}/policy_best.pt")
    if use_ema:
        logger.info(f"EMA model saved to: {args.checkpoint_dir}/policy_ema.pt")
    logger.info("="*70)


if __name__ == '__main__':
    main()
