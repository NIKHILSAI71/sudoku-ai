"""Pure Neural Sudoku Solver - NO classical algorithms, GNN model only!

This solver uses ONLY the trained GNN model with smart iterative refinement,
constraint-aware predictions, and adaptive confidence thresholds.

NO backtracking, NO beam search - Pure neural network solving!
"""

from __future__ import annotations

from typing import Optional, Tuple
import time
import math
import logging

import torch
import torch.nn.functional as F
import numpy as np

from ..models.gnn import SudokuGNN

logger = logging.getLogger(__name__)


class NeuralSolver:
    """Pure neural Sudoku solver using ONLY the GNN model.
    
    Features:
    - Constraint-aware predictions (masks illegal moves)
    - Iterative refinement with confidence thresholding
    - Adaptive confidence (fills easy cells first)
    - Smart cell selection (most confident first)
    - Multiple solving strategies
    
    NO classical algorithms - Pure neural network!
    """
    
    def __init__(
        self,
        model: SudokuGNN,
        device: str = 'cuda',
        verbose: bool = True
    ):
        """Initialize neural solver.
        
        Args:
            model: Trained SudokuGNN model
            device: Device for inference
            verbose: Print solving progress
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.verbose = verbose
        
        logger.info("🧠 Neural Solver initialized (Pure GNN, no classical algorithms)")
    
    def solve(
        self,
        puzzle: torch.Tensor,
        strategy: str = 'iterative',
        max_iterations: int = 50,
        initial_confidence: float = 0.90
    ) -> Tuple[torch.Tensor, dict]:
        """Solve Sudoku puzzle using ONLY the neural model.
        
        Args:
            puzzle: (grid_size, grid_size) puzzle tensor
            strategy: Solving strategy:
                - 'greedy': Fill all cells with top predictions (fast, single pass)
                - 'iterative': Multiple passes with confidence threshold (balanced)
                - 'careful': Very conservative, adaptive confidence (most accurate)
            max_iterations: Maximum iterations for iterative strategies
            initial_confidence: Starting confidence threshold
            
        Returns:
            solution: Solved puzzle tensor
            info: Dict with solve metrics
        """
        start_time = time.time()
        
        # Handle batched input
        if puzzle.dim() == 2:
            puzzle = puzzle.unsqueeze(0)
            single_puzzle = True
        else:
            single_puzzle = False
        
        puzzle = puzzle.to(self.device)
        
        if self.verbose:
            num_empty = (puzzle == 0).sum().item()
            logger.info(f"🎯 Solving puzzle with {num_empty} empty cells...")
            logger.info(f"📊 Strategy: {strategy}, Initial confidence: {initial_confidence:.2f}")
        
        # Choose solving strategy
        if strategy == 'greedy':
            solution, info = self._greedy_solve(puzzle)
        elif strategy == 'iterative':
            solution, info = self._iterative_solve(
                puzzle, max_iterations, initial_confidence
            )
        elif strategy == 'careful':
            solution, info = self._careful_solve(
                puzzle, max_iterations
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'greedy', 'iterative', or 'careful'")
        
        solve_time = time.time() - start_time
        info['solve_time_ms'] = solve_time * 1000
        info['strategy'] = strategy
        
        # Validate solution
        is_complete = (solution > 0).all().item()
        is_valid = self._validate_solution(solution) if is_complete else False
        info['is_complete'] = is_complete
        info['is_valid'] = is_valid
        info['success'] = is_complete and is_valid
        
        if self.verbose:
            if info['success']:
                logger.info(f"✅ SOLVED in {solve_time*1000:.1f}ms ({info.get('iterations', 1)} iterations)")
            else:
                filled = (solution > 0).sum().item()
                total = solution.numel()
                logger.info(f"⚠️ Partial solution: {filled}/{total} cells filled")
        
        if single_puzzle:
            solution = solution.squeeze(0)
        
        return solution, info
    
    def _greedy_solve(self, puzzle: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Greedy single-pass solving: Fill ALL cells with top predictions.
        
        Fast but may violate constraints. Best for well-trained models.
        """
        with torch.no_grad():
            # Get predictions
            logits = self.model(puzzle)
            
            # Apply constraint masking
            masked_logits = self._apply_constraint_mask(logits, puzzle)
            
            # Get top predictions
            probs = F.softmax(masked_logits, dim=-1)
            predictions = probs.argmax(dim=-1) + 1  # Convert to 1-indexed
            
            # Fill empty cells
            solution = puzzle.clone()
            empty_mask = (puzzle == 0)
            solution[empty_mask] = predictions[empty_mask]
            
            # Compute average confidence
            max_probs = probs.max(dim=-1)[0]
            avg_confidence = max_probs[empty_mask].mean().item()
        
        info = {
            'iterations': 1,
            'cells_filled': empty_mask.sum().item(),
            'avg_confidence': avg_confidence,
            'method': 'greedy_single_pass'
        }
        
        return solution, info
    
    def _iterative_solve(
        self,
        puzzle: torch.Tensor,
        max_iterations: int,
        confidence_threshold: float
    ) -> Tuple[torch.Tensor, dict]:
        """Iterative refinement: Fill high-confidence cells progressively.
        
        Fills most confident cells first, allowing the model to refine
        its predictions for remaining cells.
        """
        current = puzzle.clone()
        iteration = 0
        total_filled = 0
        confidence_history = []
        
        with torch.no_grad():
            for iteration in range(max_iterations):
                # Get predictions
                logits = self.model(current)
                
                # Apply constraint masking
                masked_logits = self._apply_constraint_mask(logits, current)
                
                # Get probabilities and predictions
                probs = F.softmax(masked_logits, dim=-1)
                max_probs, max_values = probs.max(dim=-1)
                max_values = max_values + 1  # Convert to 1-indexed
                
                # Find high-confidence empty cells
                empty_mask = (current == 0)
                confident_mask = (max_probs >= confidence_threshold) & empty_mask
                
                if not confident_mask.any():
                    # No more confident predictions - lower threshold
                    confidence_threshold *= 0.95
                    
                    if confidence_threshold < 0.5:
                        # Fill remaining with best guesses
                        if empty_mask.any():
                            current[empty_mask] = max_values[empty_mask]
                        break
                    continue
                
                # Fill confident cells
                num_filled = confident_mask.sum().item()
                current[confident_mask] = max_values[confident_mask]
                total_filled += num_filled
                
                avg_conf = max_probs[confident_mask].mean().item()
                confidence_history.append(avg_conf)
                
                if self.verbose and num_filled > 0:
                    logger.info(f"  Iter {iteration+1}: Filled {num_filled} cells "
                              f"(conf={avg_conf:.3f}, threshold={confidence_threshold:.3f})")
                
                # Check if complete
                if (current > 0).all():
                    break
        
        info = {
            'iterations': iteration + 1,
            'cells_filled': total_filled,
            'avg_confidence': np.mean(confidence_history) if confidence_history else 0.0,
            'final_threshold': confidence_threshold,
            'method': 'iterative_refinement'
        }
        
        return current, info
    
    def _careful_solve(
        self,
        puzzle: torch.Tensor,
        max_iterations: int
    ) -> Tuple[torch.Tensor, dict]:
        """Careful solving: Adaptive confidence with strategic cell selection.
        
        Selects the single most confident cell each iteration for maximum accuracy.
        Slower but most accurate for challenging puzzles.
        """
        current = puzzle.clone()
        iteration = 0
        total_filled = 0
        confidence_history = []
        
        with torch.no_grad():
            for iteration in range(max_iterations):
                # Check if complete
                empty_mask = (current == 0)
                if not empty_mask.any():
                    break
                
                # Get predictions
                logits = self.model(current)
                
                # Apply constraint masking
                masked_logits = self._apply_constraint_mask(logits, current)
                
                # Get probabilities
                probs = F.softmax(masked_logits, dim=-1)
                max_probs, max_values = probs.max(dim=-1)
                max_values = max_values + 1  # Convert to 1-indexed
                
                # Find most confident empty cell
                masked_probs = max_probs.clone()
                masked_probs[~empty_mask] = 0.0
                
                # Get the single most confident cell
                flat_probs = masked_probs.view(-1)
                best_idx = int(flat_probs.argmax().item())
                
                if flat_probs[best_idx] < 0.3:
                    # Very low confidence - fill remaining with best guesses
                    if empty_mask.any():
                        current[empty_mask] = max_values[empty_mask]
                    break
                
                # Convert flat index to 2D coordinates
                batch_idx = int(best_idx // (current.size(1) * current.size(2)))
                cell_idx = int(best_idx % (current.size(1) * current.size(2)))
                row = int(cell_idx // current.size(2))
                col = int(cell_idx % current.size(2))
                
                # Fill the most confident cell
                confidence = flat_probs[best_idx].item()
                value = int(max_values.view(-1)[best_idx].item())
                current[batch_idx, row, col] = value
                
                total_filled += 1
                confidence_history.append(confidence)
                
                if self.verbose and iteration % 10 == 0:
                    remaining = empty_mask.sum().item() - total_filled
                    logger.info(f"  Iter {iteration+1}: Filled cell ({row},{col})={value} "
                              f"(conf={confidence:.3f}, {remaining} remaining)")
        
        info = {
            'iterations': iteration + 1,
            'cells_filled': total_filled,
            'avg_confidence': np.mean(confidence_history) if confidence_history else 0.0,
            'method': 'careful_strategic'
        }
        
        return current, info
    
    def _apply_constraint_mask(
        self,
        logits: torch.Tensor,
        puzzle: torch.Tensor
    ) -> torch.Tensor:
        """Apply constraint masking: Set logits to -inf for illegal moves.
        
        This forces the model to only predict legal values based on Sudoku rules.
        HUGE improvement for accuracy!
        
        Args:
            logits: [B, H, W, C] prediction logits
            puzzle: [B, H, W] current puzzle state
            
        Returns:
            masked_logits: Logits with illegal moves masked out
        """
        batch_size, grid_size, _, num_classes = logits.shape
        block_size = int(math.sqrt(grid_size))
        device = logits.device
        
        # Create mask for legal moves (True = legal, False = illegal)
        legal_mask = torch.ones_like(logits, dtype=torch.bool)
        
        # For each cell, mask values that already exist in its row/col/box
        for b in range(batch_size):
            for i in range(grid_size):
                for j in range(grid_size):
                    # Skip if cell is already filled
                    if puzzle[b, i, j] > 0:
                        continue
                    
                    # Get values in row, column, and box
                    row_values = puzzle[b, i, :]
                    col_values = puzzle[b, :, j]
                    
                    block_row = (i // block_size) * block_size
                    block_col = (j // block_size) * block_size
                    box_values = puzzle[b, 
                                       block_row:block_row+block_size,
                                       block_col:block_col+block_size].flatten()
                    
                    # Mask illegal values (convert 1-indexed to 0-indexed)
                    for val in row_values:
                        if val > 0:
                            legal_mask[b, i, j, val-1] = False
                    for val in col_values:
                        if val > 0:
                            legal_mask[b, i, j, val-1] = False
                    for val in box_values:
                        if val > 0:
                            legal_mask[b, i, j, val-1] = False
        
        # Apply mask: Set illegal moves to -inf
        masked_logits = logits.clone()
        masked_logits[~legal_mask] = float('-inf')
        
        return masked_logits
    
    def _validate_solution(self, solution: torch.Tensor) -> bool:
        """Validate complete solution using Sudoku rules."""
        try:
            for b in range(solution.size(0)):
                grid = solution[b].cpu().numpy()
                grid_size = grid.shape[0]
                block_size = int(math.sqrt(grid_size))
                
                # Check all values are in valid range
                if not np.all((grid >= 1) & (grid <= grid_size)):
                    return False
                
                # Check rows
                for i in range(grid_size):
                    if len(set(grid[i, :])) != grid_size:
                        return False
                
                # Check columns
                for j in range(grid_size):
                    if len(set(grid[:, j])) != grid_size:
                        return False
                
                # Check blocks
                for block_i in range(block_size):
                    for block_j in range(block_size):
                        block = grid[
                            block_i * block_size:(block_i + 1) * block_size,
                            block_j * block_size:(block_j + 1) * block_size
                        ]
                        if len(set(block.flatten())) != grid_size:
                            return False
            
            return True
        except Exception:
            return False


def load_neural_solver(
    checkpoint_path: str,
    device: str = 'cuda',
    verbose: bool = True
) -> NeuralSolver:
    """Convenience function to load a neural solver from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device for inference
        verbose: Enable verbose logging
        
    Returns:
        NeuralSolver instance
    """
    from ..models.gnn.sudoku_gnn import load_pretrained_model
    
    model = load_pretrained_model(checkpoint_path, device=device)
    solver = NeuralSolver(model, device=device, verbose=verbose)
    
    return solver


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧠 Pure Neural Sudoku Solver")
    print("="*70)
    print("\nThis solver uses ONLY the GNN model - no classical algorithms!")
    print("\nFeatures:")
    print("  ✅ Constraint-aware predictions (masks illegal moves)")
    print("  ✅ Iterative refinement with confidence thresholds")
    print("  ✅ Adaptive solving strategies")
    print("  ✅ Smart cell selection (most confident first)")
    print("\nStrategies:")
    print("  • greedy: Single pass, fill all cells (fastest)")
    print("  • iterative: Multiple passes with confidence (balanced)")
    print("  • careful: Strategic cell-by-cell (most accurate)")
    print("\nUsage:")
    print("  solver = load_neural_solver('checkpoints/policy_best.pt')")
    print("  solution, info = solver.solve(puzzle, strategy='iterative')")
    print("="*70 + "\n")
