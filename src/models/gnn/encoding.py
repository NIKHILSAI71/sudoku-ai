"""Size-independent feature encoding for Sudoku puzzles.

This module creates relative position encodings that work across any grid size,
enabling the model to generalize from 9×9 to 4×4, 16×16, etc.

Key insight: Use normalized [0,1] positions instead of absolute coordinates.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


class SudokuEncoder:
    """Encodes Sudoku puzzles into size-agnostic node features.
    
    Feature vector per cell node (5 dimensions):
    1. Normalized value: value / grid_size  [0, 1]
    2. Is given mask: 1 if clue, 0 if blank  {0, 1}
    3. Relative row position: row / grid_size  [0, 1]
    4. Relative column position: col / grid_size  [0, 1]
    5. Relative block position: block_idx / num_blocks  [0, 1]
    
    This encoding is identical for any grid size - a cell at (8,8) in 9×9
    has the same relative encoding as (15,15) in 16×16.
    """
    
    @staticmethod
    def encode_cells(
        puzzle: torch.Tensor,
        grid_size: int,
        n_cells: int
    ) -> torch.Tensor:
        """Encode cell nodes with size-independent features.
        
        Args:
            puzzle: (batch_size, grid_size, grid_size) puzzle tensor
            grid_size: Size of the grid
            n_cells: Number of cell nodes (grid_size²)
            
        Returns:
            features: (batch_size, n_cells, 5) feature tensor
        """
        batch_size = puzzle.size(0)
        device = puzzle.device
        block_size = int(math.sqrt(grid_size))
        
        # Flatten puzzle: (batch, grid_size, grid_size) -> (batch, n_cells)
        values = puzzle.view(batch_size, n_cells)
        
        # Feature 1: Normalized values [0, 1]
        # 0 for empty cells, normalized for given values
        norm_values = (values.float() / grid_size).unsqueeze(-1)
        
        # Feature 2: Is given mask
        is_given = (values > 0).float().unsqueeze(-1)
        
        # Features 3-5: Relative positions (VECTORIZED - no Python loops!)
        # Create row and column indices: (grid_size, grid_size)
        rows = torch.arange(grid_size, device=device).float()
        cols = torch.arange(grid_size, device=device).float()
        
        # Create meshgrid for all positions at once
        row_idx, col_idx = torch.meshgrid(rows, cols, indexing='ij')
        
        # Relative row and column [0, 1]: (grid_size, grid_size)
        rel_row = row_idx / grid_size
        rel_col = col_idx / grid_size
        
        # Relative block position [0, 1]: (grid_size, grid_size)
        block_row = (row_idx / block_size).long()
        block_col = (col_idx / block_size).long()
        block_idx = block_row * block_size + block_col
        rel_block = block_idx.float() / (block_size * block_size)
        
        # Stack and reshape: (grid_size, grid_size, 3) -> (n_cells, 3)
        pos_tensor = torch.stack([rel_row, rel_col, rel_block], dim=-1)
        pos_tensor = pos_tensor.reshape(n_cells, 3)
        
        # Expand for batch: (batch_size, n_cells, 3)
        pos_features = pos_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate all features: (batch_size, n_cells, 5)
        cell_features = torch.cat([
            norm_values,
            is_given,
            pos_features
        ], dim=-1)
        
        return cell_features
    
    @staticmethod
    def encode_constraints(
        batch_size: int,
        n_constraints: int,
        feature_dim: int,
        device: torch.device
    ) -> torch.Tensor:
        """Encode constraint nodes (initially zero features).
        
        Constraint nodes learn their features through message passing.
        Start with zeros and let the network learn what to represent.
        
        Args:
            batch_size: Number of puzzles in batch
            n_constraints: Number of constraint nodes
            feature_dim: Dimension of feature vectors
            device: Device to place tensor on
            
        Returns:
            features: (batch_size, n_constraints, feature_dim) zero tensor
        """
        return torch.zeros(
            batch_size, 
            n_constraints, 
            feature_dim,
            dtype=torch.float32,
            device=device
        )
    
    @staticmethod
    def create_node_features(
        puzzle: torch.Tensor,
        n_cells: int,
        n_constraints: int,
        grid_size: int
    ) -> torch.Tensor:
        """Create complete node feature matrix for Sudoku graph.
        
        Combines cell features and constraint features into a single tensor
        suitable for GNN processing.
        
        Args:
            puzzle: (batch_size, grid_size, grid_size) puzzle tensor
            n_cells: Number of cell nodes
            n_constraints: Number of constraint nodes
            grid_size: Size of the grid
            
        Returns:
            features: (batch_size, n_cells + n_constraints, 5) tensor
        """
        batch_size = puzzle.size(0)
        device = puzzle.device
        
        # Encode cell nodes (first n_cells nodes)
        cell_features = SudokuEncoder.encode_cells(puzzle, grid_size, n_cells)
        
        # Encode constraint nodes (next n_constraints nodes)
        constraint_features = SudokuEncoder.encode_constraints(
            batch_size, n_constraints, cell_features.size(-1), device
        )
        
        # Concatenate: cells first, then constraints
        # Shape: (batch_size, n_cells + n_constraints, feature_dim)
        all_features = torch.cat([cell_features, constraint_features], dim=1)
        
        return all_features


class CandidateEncoder:
    """Extended encoder that includes candidate value tracking.
    
    For advanced models, we can track which values are still possible
    for each cell based on current constraints.
    """
    
    @staticmethod
    def compute_candidates(
        puzzle: torch.Tensor,
        grid_size: int
    ) -> torch.Tensor:
        """Compute candidate values for each empty cell.
        
        A candidate is valid if it doesn't violate row, column, or box constraints.
        
        Args:
            puzzle: (batch_size, grid_size, grid_size) puzzle tensor
            grid_size: Size of the grid
            
        Returns:
            candidates: (batch_size, grid_size, grid_size, grid_size) binary tensor
                        candidates[b, i, j, v] = 1 if value v+1 is valid for cell (i,j)
        """
        batch_size = puzzle.size(0)
        device = puzzle.device
        block_size = int(math.sqrt(grid_size))
        
        # Initialize all candidates as possible
        candidates = torch.ones(
            batch_size, grid_size, grid_size, grid_size,
            dtype=torch.float32,
            device=device
        )
        
        for b in range(batch_size):
            for i in range(grid_size):
                for j in range(grid_size):
                    if puzzle[b, i, j] > 0:
                        # Given cell - no candidates needed
                        candidates[b, i, j, :] = 0
                        continue
                    
                    # Check row constraints
                    row_values = puzzle[b, i, :]
                    for v in row_values:
                        if v > 0:
                            candidates[b, i, j, int(v.item()) - 1] = 0
                    
                    # Check column constraints
                    col_values = puzzle[b, :, j]
                    for v in col_values:
                        if v > 0:
                            candidates[b, i, j, int(v.item()) - 1] = 0
                    
                    # Check box constraints
                    box_row = (i // block_size) * block_size
                    box_col = (j // block_size) * block_size
                    box_values = puzzle[b, box_row:box_row+block_size, box_col:box_col+block_size]
                    for v in box_values.flatten():
                        if v > 0:
                            candidates[b, i, j, int(v.item()) - 1] = 0
        
        return candidates
    
    @staticmethod
    def encode_with_candidates(
        puzzle: torch.Tensor,
        grid_size: int,
        n_cells: int
    ) -> torch.Tensor:
        """Encode cells with candidate information.
        
        Feature vector per cell (5 + grid_size dimensions):
        - First 5: standard encoding (value, mask, positions)
        - Next grid_size: binary candidate vector
        
        Args:
            puzzle: (batch_size, grid_size, grid_size) puzzle tensor
            grid_size: Size of the grid
            n_cells: Number of cell nodes
            
        Returns:
            features: (batch_size, n_cells, 5 + grid_size) feature tensor
        """
        # Get standard features
        standard_features = SudokuEncoder.encode_cells(puzzle, grid_size, n_cells)
        
        # Compute candidates
        candidates = CandidateEncoder.compute_candidates(puzzle, grid_size)
        
        # Flatten candidates: (batch, grid_size, grid_size, grid_size) 
        #                  -> (batch, n_cells, grid_size)
        batch_size = puzzle.size(0)
        candidate_features = candidates.view(batch_size, n_cells, grid_size)
        
        # Concatenate
        extended_features = torch.cat([standard_features, candidate_features], dim=-1)
        
        return extended_features


def create_node_features(
    puzzle: torch.Tensor,
    grid_size: int,
    n_cells: int,
    n_constraints: int,
    use_candidates: bool = False
) -> torch.Tensor:
    """Convenience function to create node features.
    
    Args:
        puzzle: (batch_size, grid_size, grid_size) puzzle tensor
        grid_size: Size of the grid
        n_cells: Number of cell nodes
        n_constraints: Number of constraint nodes
        use_candidates: Whether to include candidate tracking
        
    Returns:
        features: (batch_size, n_nodes, feature_dim) tensor
        
    Example:
        >>> puzzle = torch.randint(0, 10, (4, 9, 9))
        >>> features = create_node_features(puzzle, 9, 81, 27)
        >>> print(features.shape)
        torch.Size([4, 108, 5])
    """
    if use_candidates:
        # Extended encoding with candidates
        cell_features = CandidateEncoder.encode_with_candidates(
            puzzle, grid_size, n_cells
        )
        constraint_features = SudokuEncoder.encode_constraints(
            puzzle.size(0), n_constraints, cell_features.size(-1), puzzle.device
        )
    else:
        # Standard encoding
        cell_features = SudokuEncoder.encode_cells(puzzle, grid_size, n_cells)
        constraint_features = SudokuEncoder.encode_constraints(
            puzzle.size(0), n_constraints, cell_features.size(-1), puzzle.device
        )
    
    # Combine cell and constraint features
    all_features = torch.cat([cell_features, constraint_features], dim=1)
    
    return all_features


if __name__ == "__main__":
    # Demo: Create features for different sizes
    print("Size-Independent Encoding Demo")
    print("=" * 50)
    
    for size in [4, 9, 16]:
        print(f"\nGrid size: {size}×{size}")
        
        # Create sample puzzle
        puzzle = torch.randint(0, size + 1, (2, size, size))
        n_cells = size * size
        n_constraints = 3 * size
        
        # Create features
        features = create_node_features(puzzle, size, n_cells, n_constraints)
        
        print(f"Feature shape: {features.shape}")
        print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"Memory: {features.numel() * 4 / 1024:.2f} KB")
