"""Loss functions for Sudoku GNN training.

Implements cross-entropy loss with legal move masking
and optional constraint satisfaction penalties.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SudokuLoss(nn.Module):
    """Cross-entropy loss with legal move masking."""
    
    def __init__(
        self,
        constraint_weight: float = 0.1,
        use_constraint_loss: bool = True
    ):
        """Initialize loss function.
        
        Args:
            constraint_weight: Weight for constraint satisfaction loss
            use_constraint_loss: Whether to add constraint loss
        """
        super().__init__()
        self.constraint_weight = constraint_weight
        self.use_constraint_loss = use_constraint_loss
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        puzzles: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Compute loss.
        
        Args:
            logits: Model predictions [B, grid_size, grid_size, num_classes]
            targets: Ground truth values [B, grid_size, grid_size]
            puzzles: Input puzzles [B, grid_size, grid_size]
            
        Returns:
            (loss, info_dict)
        """
        batch_size, grid_size, _, num_classes = logits.shape
        
        # Mask for cells to predict (only empty cells)
        mask = (puzzles == 0).float()
        
        # Flatten for cross-entropy
        logits_flat = logits.reshape(-1, num_classes)
        targets_flat = targets.reshape(-1)
        mask_flat = mask.reshape(-1)
        
        # Cross-entropy loss (only on masked cells)
        ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        ce_loss = (ce_loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        
        total_loss = ce_loss
        
        # Optional constraint loss
        constraint_loss = torch.tensor(0.0, device=logits.device)
        if self.use_constraint_loss:
            constraint_loss = self._compute_constraint_loss(logits, mask)
            total_loss = total_loss + self.constraint_weight * constraint_loss
        
        # Info dict
        info = {
            'ce_loss': ce_loss.item(),
            'constraint_loss': constraint_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, info
    
    def _compute_constraint_loss(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute constraint satisfaction loss.
        
        Penalizes predictions that violate Sudoku constraints.
        """
        batch_size, grid_size, _, num_classes = logits.shape
        block_size = int(math.sqrt(grid_size))
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)  # [B, H, W, C]
        
        total_loss = 0.0
        
        # Row constraint: each digit should appear once per row
        for i in range(grid_size):
            row_probs = probs[:, i, :, :]  # [B, W, C]
            row_mask = mask[:, i, :]  # [B, W]
            
            # Sum probabilities for each digit across row
            digit_sums = (row_probs * row_mask.unsqueeze(-1)).sum(dim=1)  # [B, C]
            
            # Should be close to 1 for each digit (excluding 0)
            target_sum = row_mask.sum(dim=1, keepdim=True) / grid_size
            loss = F.mse_loss(digit_sums[:, 1:], target_sum.expand(-1, num_classes-1))
            total_loss += loss
        
        # Column constraint
        for j in range(grid_size):
            col_probs = probs[:, :, j, :]  # [B, H, C]
            col_mask = mask[:, :, j]  # [B, H]
            
            digit_sums = (col_probs * col_mask.unsqueeze(-1)).sum(dim=1)  # [B, C]
            target_sum = col_mask.sum(dim=1, keepdim=True) / grid_size
            loss = F.mse_loss(digit_sums[:, 1:], target_sum.expand(-1, num_classes-1))
            total_loss += loss
        
        # Box constraint
        for bi in range(block_size):
            for bj in range(block_size):
                box_probs = probs[
                    :,
                    bi*block_size:(bi+1)*block_size,
                    bj*block_size:(bj+1)*block_size,
                    :
                ]  # [B, block_size, block_size, C]
                box_mask = mask[
                    :,
                    bi*block_size:(bi+1)*block_size,
                    bj*block_size:(bj+1)*block_size
                ]  # [B, block_size, block_size]
                
                # Flatten box
                box_probs_flat = box_probs.reshape(batch_size, -1, num_classes)
                box_mask_flat = box_mask.reshape(batch_size, -1)
                
                digit_sums = (box_probs_flat * box_mask_flat.unsqueeze(-1)).sum(dim=1)
                target_sum = box_mask_flat.sum(dim=1, keepdim=True) / grid_size
                loss = F.mse_loss(digit_sums[:, 1:], target_sum.expand(-1, num_classes-1))
                total_loss += loss
        
        # Average across all constraints
        num_constraints = grid_size * 2 + block_size * block_size
        return torch.tensor(total_loss / num_constraints, device=logits.device)
