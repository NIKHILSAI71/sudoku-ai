"""Size-agnostic graph construction for Sudoku puzzles.

This module creates bipartite graphs representing Sudoku constraints
that work for any grid size (4×4, 9×9, 16×16, 25×25, etc.).

Time Complexity: O(n²) for n×n grid
Space Complexity: O(n²) edges
"""

from __future__ import annotations

import math
from typing import Tuple
from dataclasses import dataclass

import torch
import numpy as np


@dataclass
class SudokuGraph:
    """Container for Sudoku graph structure.
    
    Attributes:
        edge_index: (2, num_edges) tensor of bidirectional edges
        n_cells: Number of cell nodes (grid_size²)
        n_constraints: Number of constraint nodes (3 × grid_size)
        grid_size: Size of the Sudoku grid
        block_size: Size of each block (√grid_size)
    """
    edge_index: torch.Tensor
    n_cells: int
    n_constraints: int
    grid_size: int
    block_size: int
    
    @property
    def num_nodes(self) -> int:
        """Total number of nodes in graph."""
        return self.n_cells + self.n_constraints
    
    @property
    def num_edges(self) -> int:
        """Total number of edges in graph."""
        return self.edge_index.size(1)


class GraphBuilder:
    """Factory for creating size-agnostic Sudoku graphs.
    
    The bipartite graph structure:
    - Cell nodes (n²): One node per grid cell
    - Constraint nodes (3n): One per row, column, and box
    - Edges: Each cell connects to its 3 constraints (bidirectional)
    
    This representation is optimal for:
    - Size generalization (same code for any grid size)
    - Constraint propagation via message passing
    - Efficient GPU computation
    """
    
    # Cache for pre-built graphs (avoids reconstruction)
    _cache: dict[tuple[int, str], SudokuGraph] = {}
    
    @classmethod
    def create(cls, grid_size: int = 9, device: str = 'cpu') -> SudokuGraph:
        """Create bipartite graph for Sudoku of any size.
        
        Args:
            grid_size: Size of the Sudoku grid (must be perfect square)
            device: Device to place tensors on ('cpu' or 'cuda')
            
        Returns:
            SudokuGraph object with edge structure and metadata
            
        Raises:
            ValueError: If grid_size is not a perfect square
            
        Example:
            >>> graph = GraphBuilder.create(grid_size=9)
            >>> print(f"Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")
            Nodes: 108, Edges: 486
        """
        # Check cache first
        cache_key = (grid_size, device)
        if cache_key in cls._cache:
            return cls._cache[cache_key]
        
        # Validate grid size
        block_size = int(math.sqrt(grid_size))
        if block_size * block_size != grid_size:
            raise ValueError(
                f"Grid size {grid_size} must be a perfect square. "
                f"Valid sizes: 4, 9, 16, 25, 36, 49, 64, 81, 100, ..."
            )
        
        n_cells = grid_size * grid_size
        n_constraints = 3 * grid_size  # rows + cols + boxes
        
        # Build edge list
        edges = cls._build_edges(grid_size, block_size, n_cells)
        
        # Convert to tensor
        edge_index = torch.tensor(edges, dtype=torch.long, device=device)
        edge_index = edge_index.t().contiguous()  # Shape: (2, num_edges)
        
        graph = SudokuGraph(
            edge_index=edge_index,
            n_cells=n_cells,
            n_constraints=n_constraints,
            grid_size=grid_size,
            block_size=block_size
        )
        
        # Cache for reuse
        cls._cache[cache_key] = graph
        
        return graph
    
    @staticmethod
    def _build_edges(
        grid_size: int,
        block_size: int,
        n_cells: int
    ) -> list[list[int]]:
        """Build bidirectional edge list for Sudoku graph.
        
        Args:
            grid_size: Size of the grid
            block_size: Size of each block
            n_cells: Number of cell nodes
            
        Returns:
            List of [source, target] edges
        """
        edges = []
        
        for row in range(grid_size):
            for col in range(grid_size):
                cell_idx = row * grid_size + col
                
                # Calculate constraint node indices
                # Constraint nodes come after cell nodes in the graph
                row_constraint = n_cells + row
                col_constraint = n_cells + grid_size + col
                
                # Box constraint: which block does this cell belong to?
                block_row = row // block_size
                block_col = col // block_size
                box_idx = block_row * block_size + block_col
                box_constraint = n_cells + 2 * grid_size + box_idx
                
                # Add bidirectional edges (cell ↔ constraint)
                for constraint in [row_constraint, col_constraint, box_constraint]:
                    edges.append([cell_idx, constraint])
                    edges.append([constraint, cell_idx])
        
        return edges
    
    @classmethod
    def create_batch(
        cls,
        grid_size: int,
        batch_size: int,
        device: str = 'cpu'
    ) -> SudokuGraph:
        """Create graph structure for batched processing.
        
        For batched inference, we replicate the graph structure
        but keep it as a single graph (PyTorch Geometric handles batching).
        
        Args:
            grid_size: Size of each puzzle
            batch_size: Number of puzzles to process together
            device: Device to place tensors on
            
        Returns:
            SudokuGraph suitable for batch processing
        """
        # Single graph structure works for all puzzles in batch
        # PyTorch Geometric will handle the batching via node offsets
        return cls.create(grid_size, device)
    
    @classmethod
    def clear_cache(cls):
        """Clear the graph cache (useful for memory management)."""
        cls._cache.clear()
    
    @classmethod
    def get_cache_info(cls) -> dict:
        """Get information about cached graphs."""
        return {
            'cached_sizes': [size for size, _ in cls._cache.keys()],
            'cache_size': len(cls._cache),
            'memory_estimate_mb': len(cls._cache) * 0.1  # Rough estimate
        }


def visualize_graph(graph: SudokuGraph) -> str:
    """Generate ASCII visualization of graph structure.
    
    Args:
        graph: SudokuGraph to visualize
        
    Returns:
        String representation of the graph
    """
    lines = [
        f"Sudoku Graph ({graph.grid_size}×{graph.grid_size})",
        f"=" * 50,
        f"Grid Size: {graph.grid_size} ({graph.block_size}×{graph.block_size} blocks)",
        f"Cell Nodes: {graph.n_cells}",
        f"Constraint Nodes: {graph.n_constraints}",
        f"  - Row constraints: {graph.grid_size}",
        f"  - Column constraints: {graph.grid_size}",
        f"  - Box constraints: {graph.grid_size}",
        f"Total Nodes: {graph.num_nodes}",
        f"Total Edges: {graph.num_edges} (bidirectional)",
        f"",
        f"Each cell connects to 3 constraints:",
        f"  Cell[0,0] → Row[0], Col[0], Box[0]",
        f"  Cell[i,j] → Row[i], Col[j], Box[block_idx]",
        f"",
        f"Memory: ~{graph.num_edges * 8 / 1024:.2f} KB"
    ]
    
    return "\n".join(lines)


# Convenience function
def create_sudoku_graph(grid_size: int = 9, device: str = 'cpu') -> SudokuGraph:
    """Convenience function to create a Sudoku graph.
    
    Args:
        grid_size: Size of the grid (default: 9 for standard Sudoku)
        device: Device to place tensors on
        
    Returns:
        SudokuGraph object
        
    Example:
        >>> graph = create_sudoku_graph(9)
        >>> print(visualize_graph(graph))
    """
    return GraphBuilder.create(grid_size, device)


if __name__ == "__main__":
    # Demo: Create graphs for different sizes
    for size in [4, 9, 16]:
        graph = create_sudoku_graph(size)
        print(visualize_graph(graph))
        print()
