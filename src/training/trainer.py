"""GNN training pipeline with curriculum learning and mixed precision.

Implements training loop with:
- Curriculum learning (easy→medium→hard)
- Mixed precision training (FP16)
- Data augmentation
- Checkpointing
- Validation tracking
"""

from __future__ import annotations

from typing import Optional, Tuple
from pathlib import Path
import logging

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp.autocast_mode import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from ..models.gnn.sudoku_gnn import SudokuGNN
from ..data.dataset import create_curriculum_dataloaders
from .loss import SudokuLoss

logger = logging.getLogger(__name__)


class GNNTrainer:
    """Trainer for Sudoku GNN with curriculum learning."""
    
    def __init__(
        self,
        model: SudokuGNN,
        device: str = 'cuda',
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        use_amp: bool = True,
        checkpoint_dir: str = 'checkpoints'
    ):
        """Initialize trainer.
        
        Args:
            model: SudokuGNN model
            device: Device to train on
            lr: Learning rate
            weight_decay: Weight decay for regularization
            use_amp: Use automatic mixed precision (FP16)
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = SudokuLoss(
            constraint_weight=0.1,
            use_constraint_loss=True
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.best_val_acc = 0.0
        self.current_epoch = 0
        self.curriculum_stage = 0
    
    def train_epoch(
        self,
        train_loader,
        desc: str = "Training"
    ) -> dict:
        """Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            desc: Description for progress bar
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_ce_loss = 0.0
        total_constraint_loss = 0.0
        correct_cells = 0
        total_cells = 0
        
        pbar = tqdm(train_loader, desc=desc)
        for batch_idx, (puzzles, solutions) in enumerate(pbar):
            puzzles = puzzles.to(self.device)
            solutions = solutions.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp and self.scaler is not None:
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = self.model(puzzles)
                    loss, loss_info = self.criterion(logits, solutions, puzzles)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(puzzles)
                loss, loss_info = self.criterion(logits, solutions, puzzles)
                loss.backward()
                self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss_info['total_loss']
            total_ce_loss += loss_info['ce_loss']
            total_constraint_loss += loss_info['constraint_loss']
            
            # Compute accuracy
            mask = (puzzles == 0)
            if mask.any():
                preds = logits.argmax(dim=-1) + 1  # Convert 0-indexed to 1-indexed
                correct = (preds[mask] == solutions[mask]).sum().item()
                total = mask.sum().item()
                correct_cells += correct
                total_cells += total
                
                # Update progress bar
                acc = 100.0 * correct_cells / total_cells
                pbar.set_postfix({
                    'loss': f'{total_loss/(batch_idx+1):.4f}',
                    'acc': f'{acc:.2f}%'
                })
        
        # Average metrics
        num_batches = len(train_loader)
        metrics = {
            'loss': total_loss / num_batches,
            'ce_loss': total_ce_loss / num_batches,
            'constraint_loss': total_constraint_loss / num_batches,
            'accuracy': 100.0 * correct_cells / total_cells if total_cells > 0 else 0.0
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self, val_loader, desc: str = "Validation") -> dict:
        """Validate model.
        
        Args:
            val_loader: DataLoader for validation data
            desc: Description for progress bar
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        correct_cells = 0
        correct_grids = 0
        total_cells = 0
        total_grids = 0
        
        pbar = tqdm(val_loader, desc=desc)
        for puzzles, solutions in pbar:
            puzzles = puzzles.to(self.device)
            solutions = solutions.to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = self.model(puzzles)
                    loss, loss_info = self.criterion(logits, solutions, puzzles)
            else:
                logits = self.model(puzzles)
                loss, loss_info = self.criterion(logits, solutions, puzzles)
            
            total_loss += loss_info['total_loss']
            
            # Compute accuracy
            mask = (puzzles == 0)
            preds = logits.argmax(dim=-1) + 1  # Convert 0-indexed to 1-indexed
            
            for i in range(len(puzzles)):
                if mask[i].any():
                    correct = (preds[i][mask[i]] == solutions[i][mask[i]]).sum().item()
                    total = mask[i].sum().item()
                    correct_cells += correct
                    total_cells += total
                    
                    # Grid accuracy (all cells correct)
                    if correct == total:
                        correct_grids += 1
                
                total_grids += 1
        
        # Average metrics
        metrics = {
            'loss': total_loss / len(val_loader),
            'cell_accuracy': 100.0 * correct_cells / total_cells if total_cells > 0 else 0.0,
            'grid_accuracy': 100.0 * correct_grids / total_grids if total_grids > 0 else 0.0
        }
        
        return metrics
    
    def train(
        self,
        puzzles: torch.Tensor,
        solutions: torch.Tensor,
        num_epochs: int = 60,
        batch_size: int = 128,
        val_split: float = 0.1,
        curriculum_epochs: list = [20, 20, 20],
        augment: bool = True,
        num_workers: int = 4
    ):
        """Full training loop with curriculum learning.
        
        Args:
            puzzles: All training puzzles
            solutions: All training solutions
            num_epochs: Total number of epochs
            batch_size: Batch size
            val_split: Validation split ratio
            curriculum_epochs: Epochs per curriculum stage [easy, medium, hard]
            augment: Apply data augmentation
            num_workers: DataLoader workers
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}, Mixed Precision: {self.use_amp}")
        logger.info(f"Curriculum stages: {curriculum_epochs}")
        
        # Initialize scheduler
        total_epochs = sum(curriculum_epochs)
        scheduler = CosineAnnealingLR(self.optimizer, T_max=total_epochs)
        
        for stage_idx, stage_epochs in enumerate(curriculum_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Curriculum Stage {stage_idx} ({['Easy', 'Medium', 'Hard'][stage_idx]})")
            logger.info(f"{'='*50}")
            
            self.curriculum_stage = stage_idx
            
            # Create dataloaders for this stage
            train_loader, val_loader = create_curriculum_dataloaders(
                puzzles,
                solutions,
                batch_size=batch_size,
                val_split=val_split,
                curriculum_stage=stage_idx,
                augment=augment,
                num_workers=num_workers
            )
            
            # Train for stage epochs
            for epoch in range(stage_epochs):
                self.current_epoch += 1
                
                logger.info(f"\nEpoch {self.current_epoch}/{total_epochs}")
                
                # Train
                train_metrics = self.train_epoch(
                    train_loader,
                    desc=f"Epoch {self.current_epoch} - Training"
                )
                
                logger.info(
                    f"Train - Loss: {train_metrics['loss']:.4f}, "
                    f"Accuracy: {train_metrics['accuracy']:.2f}%"
                )
                
                # Validate
                val_metrics = self.validate(
                    val_loader,
                    desc=f"Epoch {self.current_epoch} - Validation"
                )
                
                logger.info(
                    f"Val - Loss: {val_metrics['loss']:.4f}, "
                    f"Cell Acc: {val_metrics['cell_accuracy']:.2f}%, "
                    f"Grid Acc: {val_metrics['grid_accuracy']:.2f}%"
                )
                
                # Update scheduler
                scheduler.step()
                
                # Save checkpoint
                self.save_checkpoint(
                    epoch=self.current_epoch,
                    metrics=val_metrics,
                    is_best=(val_metrics['cell_accuracy'] > self.best_val_acc)
                )
                
                if val_metrics['cell_accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['cell_accuracy']
                    logger.info(f"New best validation accuracy: {self.best_val_acc:.2f}%")
        
        logger.info(f"\nTraining completed! Best validation accuracy: {self.best_val_acc:.2f}%")
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: dict,
        is_best: bool = False
    ):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_val_acc': self.best_val_acc,
            'curriculum_stage': self.curriculum_stage
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'policy.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'policy_best.pt')
            logger.info(f"Saved best checkpoint at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str | Path):
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.current_epoch = checkpoint.get('epoch', 0)
        self.curriculum_stage = checkpoint.get('curriculum_stage', 0)
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
