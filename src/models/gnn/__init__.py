"""GNN models for size-agnostic Sudoku solving."""

from .graph_builder import (
    GraphBuilder,
    SudokuGraph,
    create_sudoku_graph,
    visualize_graph
)
from .encoding import (
    SudokuEncoder,
    CandidateEncoder,
    create_node_features
)
from .message_passing import (
    MessagePassingLayer,
    RecurrentMessagePassing,
    AttentionMessagePassing
)
from .sudoku_gnn import (
    SudokuGNN,
    LightweightSudokuGNN,
    HybridSudokuGNN,
    load_pretrained_model
)

__all__ = [
    # Graph construction
    'GraphBuilder',
    'SudokuGraph',
    'create_sudoku_graph',
    'visualize_graph',
    
    # Feature encoding
    'SudokuEncoder',
    'CandidateEncoder',
    'create_node_features',
    
    # Message passing
    'MessagePassingLayer',
    'RecurrentMessagePassing',
    'AttentionMessagePassing',
    
    # Models
    'SudokuGNN',
    'LightweightSudokuGNN',
    'HybridSudokuGNN',
    'load_pretrained_model'
]
