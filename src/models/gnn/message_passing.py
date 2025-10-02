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
    """Enhanced message passing layer with residual connections and GELU.
    
    This layer implements one iteration of:
    1. Message computation: Compute messages from neighbors
    2. Message aggregation: Sum/mean messages at each node
    3. Node update: Update node embeddings with aggregated messages
    4. Residual connection: Add skip connection for better gradient flow
    
    The architecture is size-agnostic - works for any graph structure.
    Improvements over basic version:
    - GELU activation (better than ReLU for pattern learning)
    - Stronger residual connections
    - Pre-layer normalization
    - Gated updates for selective information flow
    """
    
    def __init__(
        self,
        hidden_dim: int = 96,
        dropout: float = 0.3,
        activation: str = 'gelu',
        use_gates: bool = True
    ):
        """Initialize enhanced message passing layer.
        
        Args:
            hidden_dim: Dimension of node embeddings
            dropout: Dropout probability for regularization
            activation: Activation function ('gelu', 'silu', 'relu', 'elu')
            use_gates: Use gated updates for selective information flow
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_gates = use_gates
        
        # Pre-layer normalization for better training dynamics
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Message computation: combine source and target features
        self.message_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim * 2),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Node update: combine current state with aggregated messages
        self.update_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim * 2),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Gated update mechanism (like GRU)
        if use_gates:
            self.gate_net = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
    
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
        
        # Pre-normalization
        normed_features = self.norm1(node_features)
        
        # Update node features
        combined = torch.cat([normed_features, aggregated], dim=-1)
        updated = self.update_net(combined)
        
        # Gated residual connection if enabled
        if self.use_gates:
            gate = self.gate_net(combined)
            output = node_features + gate * updated
        else:
            output = node_features + updated
        
        # Post-normalization
        output = self.norm2(output)
        
        return output
    
    def _compute_messages(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """VECTORIZED message computation - no Python loops!
        
        Args:
            node_features: (batch_size * num_nodes, hidden_dim) flat features
            edge_index: (2, num_edges) edge connectivity
            batch_size: Number of puzzles in batch
            
        Returns:
            messages: (batch_size * num_edges, hidden_dim) messages
        """
        num_edges = edge_index.size(1)
        num_nodes = node_features.size(0) // batch_size
        
        # Create batched edge indices in one operation (vectorized!)
        # Shape: (batch_size, num_edges)
        batch_offsets = torch.arange(batch_size, device=edge_index.device) * num_nodes
        batch_offsets = batch_offsets.view(-1, 1)  # (batch_size, 1)
        
        # Expand edge indices for all batches simultaneously
        # (2, num_edges) + (batch_size, 1) -> (batch_size, 2, num_edges)
        source_idx = edge_index[0].unsqueeze(0) + batch_offsets  # (batch_size, num_edges)
        target_idx = edge_index[1].unsqueeze(0) + batch_offsets  # (batch_size, num_edges)
        
        # Flatten to get all indices at once
        source_idx_flat = source_idx.reshape(-1)  # (batch_size * num_edges)
        target_idx_flat = target_idx.reshape(-1)  # (batch_size * num_edges)
        
        # Gather all source and target features in parallel (fully vectorized!)
        source_features = node_features[source_idx_flat]  # (batch_size * num_edges, hidden_dim)
        target_features = node_features[target_idx_flat]  # (batch_size * num_edges, hidden_dim)
        
        # Concatenate and compute messages for ALL edges at once
        combined = torch.cat([source_features, target_features], dim=-1)
        messages = self.message_net(combined)  # Single forward pass for entire batch!
        
        return messages
    
    def _aggregate_messages(
        self,
        messages: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        batch_size: int
    ) -> torch.Tensor:
        """VECTORIZED message aggregation - no Python loops!
        
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
        dtype = messages.dtype
        
        # Create batched target indices (vectorized!)
        batch_offsets = torch.arange(batch_size, device=device) * num_nodes
        batch_offsets = batch_offsets.view(-1, 1)  # (batch_size, 1)
        
        # Expand target indices for all batches
        target_idx = edge_index[1].unsqueeze(0) + batch_offsets  # (batch_size, num_edges)
        target_idx_flat = target_idx.reshape(-1)  # (batch_size * num_edges)
        
        # Initialize aggregation buffer
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
        
        # Aggregate ALL messages at once using scatter_add (fully vectorized!)
        aggregated.index_add_(0, target_idx_flat, messages)
        counts.index_add_(
            0, 
            target_idx_flat, 
            torch.ones(batch_size * num_edges, device=device, dtype=dtype)
        )
        
        # Average aggregation (avoid division by zero)
        counts = counts.clamp(min=1).unsqueeze(-1)
        aggregated = aggregated / counts
        
        return aggregated


class RecurrentMessagePassing(nn.Module):
    """Enhanced recurrent message passing with shared parameters and gating.
    
    Performs multiple iterations of message passing using the same layer.
    This is memory-efficient and enables deep reasoning without parameter explosion.
    
    Improvements:
    - GELU activation for better pattern learning
    - Gated updates for selective information flow
    - Better gradient flow through residual connections
    
    Research shows 32 iterations is optimal for Sudoku.
    """
    
    def __init__(
        self,
        hidden_dim: int = 96,
        num_iterations: int = 32,
        dropout: float = 0.3,
        activation: str = 'gelu'
    ):
        """Initialize enhanced recurrent message passing.
        
        Args:
            hidden_dim: Dimension of node embeddings
            num_iterations: Number of message passing iterations
            dropout: Dropout probability
            activation: Activation function (default 'gelu')
        """
        super().__init__()
        self.num_iterations = num_iterations
        
        # Single shared layer used recurrently with enhanced features
        self.mp_layer = MessagePassingLayer(
            hidden_dim, 
            dropout, 
            activation,
            use_gates=True  # Enable gated updates
        )
    
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
