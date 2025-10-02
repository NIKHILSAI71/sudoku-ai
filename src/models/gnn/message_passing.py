"""Message passing layer for constraint propagation in Sudoku.

Implements the core reasoning mechanism via iterative message passing
between cell nodes and constraint nodes.

Time Complexity: O(num_edges × hidden_dim²) per iteration
Space Complexity: O(num_nodes × hidden_dim)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MessagePassingLayer(nn.Module):
    """Single message passing layer for Sudoku constraint propagation.
    
    This layer implements one iteration of:
    1. Message computation: Compute messages from neighbors
    2. Message aggregation: Sum/mean messages at each node
    3. Node update: Update node embeddings with aggregated messages
    
    The architecture is size-agnostic - works for any graph structure.
    """
    
    def __init__(
        self,
        hidden_dim: int = 96,
        dropout: float = 0.3,
        activation: str = 'relu'
    ):
        """Initialize message passing layer.
        
        Args:
            hidden_dim: Dimension of node embeddings
            dropout: Dropout probability for regularization
            activation: Activation function ('relu', 'gelu', 'elu')
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Message computation: combine source and target features
        self.message_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node update: combine current state with aggregated messages
        self.update_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Layer normalization for training stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(inplace=True),
            'gelu': nn.GELU(),
            'elu': nn.ELU(inplace=True),
            'silu': nn.SiLU(inplace=True)
        }
        return activations.get(name.lower(), nn.ReLU(inplace=True))
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Perform one iteration of message passing.
        
        Args:
            node_features: (batch_size, num_nodes, hidden_dim) node embeddings
            edge_index: (2, num_edges) edge connectivity
            
        Returns:
            updated_features: (batch_size, num_nodes, hidden_dim) updated embeddings
        """
        batch_size, num_nodes, hidden_dim = node_features.shape
        device = node_features.device
        
        # Flatten batch dimension for message passing
        # (batch_size, num_nodes, hidden_dim) -> (batch_size * num_nodes, hidden_dim)
        flat_features = node_features.view(-1, hidden_dim)
        
        # Compute messages for each edge
        messages = self._compute_messages(flat_features, edge_index, batch_size)
        
        # Aggregate messages at each node
        aggregated = self._aggregate_messages(messages, edge_index, num_nodes, batch_size)
        
        # Reshape back to batch format
        aggregated = aggregated.view(batch_size, num_nodes, hidden_dim)
        
        # Update node features
        combined = torch.cat([node_features, aggregated], dim=-1)
        updated = self.update_net(combined)
        
        # Residual connection + layer norm
        output = self.layer_norm(node_features + updated)
        
        return output
    
    def _compute_messages(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """Compute messages for all edges.
        
        Args:
            node_features: (batch_size * num_nodes, hidden_dim) flat features
            edge_index: (2, num_edges) edge connectivity
            batch_size: Number of puzzles in batch
            
        Returns:
            messages: (batch_size * num_edges, hidden_dim) messages
        """
        num_edges = edge_index.size(1)
        
        # Get source and target node features for each edge
        # For batched processing, we need to offset node indices
        messages_list = []
        
        for b in range(batch_size):
            offset = b * (node_features.size(0) // batch_size)
            source_idx = edge_index[0] + offset
            target_idx = edge_index[1] + offset
            
            source_features = node_features[source_idx]
            target_features = node_features[target_idx]
            
            # Concatenate source and target features
            combined = torch.cat([source_features, target_features], dim=-1)
            
            # Compute message
            batch_messages = self.message_net(combined)
            messages_list.append(batch_messages)
        
        # Stack messages from all batches
        messages = torch.cat(messages_list, dim=0)
        
        return messages
    
    def _aggregate_messages(
        self,
        messages: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        batch_size: int
    ) -> torch.Tensor:
        """Aggregate messages at each node.
        
        Args:
            messages: (batch_size * num_edges, hidden_dim) messages
            edge_index: (2, num_edges) edge connectivity
            num_nodes: Number of nodes per graph
            batch_size: Number of puzzles in batch
            
        Returns:
            aggregated: (batch_size * num_nodes, hidden_dim) aggregated messages
        """
        num_edges = edge_index.size(1)
        hidden_dim = messages.size(-1)
        device = messages.device
        dtype = messages.dtype  # Match dtype for mixed precision compatibility
        
        # Initialize aggregation buffer with same dtype as messages
        aggregated = torch.zeros(
            batch_size * num_nodes,
            hidden_dim,
            device=device,
            dtype=dtype
        )
        
        # Count incoming messages for averaging
        counts = torch.zeros(
            batch_size * num_nodes,
            device=device,
            dtype=dtype
        )
        
        # Aggregate messages
        for b in range(batch_size):
            offset = b * num_nodes
            msg_offset = b * num_edges
            
            target_idx = edge_index[1] + offset
            batch_messages = messages[msg_offset:msg_offset + num_edges]
            
            # Use scatter_add for efficient aggregation
            aggregated.index_add_(0, target_idx, batch_messages)
            counts.index_add_(0, target_idx, torch.ones(num_edges, device=device, dtype=dtype))
        
        # Average aggregation (avoid division by zero)
        counts = counts.clamp(min=1).unsqueeze(-1)
        aggregated = aggregated / counts
        
        return aggregated


class RecurrentMessagePassing(nn.Module):
    """Recurrent message passing with shared parameters.
    
    Performs multiple iterations of message passing using the same layer.
    This is memory-efficient and enables deep reasoning without parameter explosion.
    
    Research shows 32 iterations is optimal for Sudoku.
    """
    
    def __init__(
        self,
        hidden_dim: int = 96,
        num_iterations: int = 32,
        dropout: float = 0.3,
        activation: str = 'relu'
    ):
        """Initialize recurrent message passing.
        
        Args:
            hidden_dim: Dimension of node embeddings
            num_iterations: Number of message passing iterations
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        self.num_iterations = num_iterations
        
        # Single shared layer used recurrently
        self.mp_layer = MessagePassingLayer(hidden_dim, dropout, activation)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Perform multiple iterations of message passing.
        
        Args:
            node_features: (batch_size, num_nodes, hidden_dim) initial embeddings
            edge_index: (2, num_edges) edge connectivity
            
        Returns:
            refined_features: (batch_size, num_nodes, hidden_dim) refined embeddings
        """
        h = node_features
        
        for iteration in range(self.num_iterations):
            h = self.mp_layer(h, edge_index)
        
        return h


class AttentionMessagePassing(nn.Module):
    """Message passing with attention mechanism.
    
    Enhanced version that learns to weight messages based on importance.
    Useful for hard puzzles where some constraints are more critical.
    """
    
    def __init__(
        self,
        hidden_dim: int = 96,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        """Initialize attention-based message passing.
        
        Args:
            hidden_dim: Dimension of node embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Attention components
        self.query_net = nn.Linear(hidden_dim, hidden_dim)
        self.key_net = nn.Linear(hidden_dim, hidden_dim)
        self.value_net = nn.Linear(hidden_dim, hidden_dim)
        
        self.output_net = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Perform attention-based message passing.
        
        Args:
            node_features: (batch_size, num_nodes, hidden_dim) embeddings
            edge_index: (2, num_edges) edge connectivity
            
        Returns:
            updated_features: (batch_size, num_nodes, hidden_dim) updated embeddings
        """
        batch_size, num_nodes, hidden_dim = node_features.shape
        
        # Compute queries, keys, values
        queries = self.query_net(node_features)
        keys = self.key_net(node_features)
        values = self.value_net(node_features)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # Compute attention scores (simplified - full version would respect edge_index)
        scores = torch.einsum('bnhd,bmhd->bnmh', queries, keys) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=2)
        attention = self.dropout(attention)
        
        # Apply attention to values
        output = torch.einsum('bnmh,bmhd->bnhd', attention, values)
        output = output.reshape(batch_size, num_nodes, hidden_dim)
        
        # Output projection
        output = self.output_net(output)
        
        # Residual connection + layer norm
        output = self.layer_norm(node_features + output)
        
        return output


if __name__ == "__main__":
    # Demo: Test message passing layer
    print("Message Passing Layer Demo")
    print("=" * 50)
    
    # Create dummy data
    batch_size = 4
    num_nodes = 108  # 81 cells + 27 constraints for 9×9
    hidden_dim = 96
    num_edges = 486  # Each cell connects to 3 constraints, bidirectional
    
    # Random node features and edges
    node_features = torch.randn(batch_size, num_nodes, hidden_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Test standard message passing
    mp_layer = MessagePassingLayer(hidden_dim)
    output = mp_layer(node_features, edge_index)
    
    print(f"Input shape: {node_features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in mp_layer.parameters()):,}")
    
    # Test recurrent message passing
    print("\nRecurrent Message Passing (32 iterations)")
    rmp = RecurrentMessagePassing(hidden_dim, num_iterations=32)
    output = rmp(node_features, edge_index)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in rmp.parameters()):,}")
