"""Tests for AI policy and training."""
from __future__ import annotations

import pytest
import torch
import numpy as np

from sudoku_ai.policy import (
    board_to_tensor,
    legal_mask_from_line,
    SimplePolicyNet,
)


class TestBoardToTensor:
    def test_empty_board(self):
        """Test encoding empty board."""
        line = "0" * 81
        tensor = board_to_tensor(line)
        assert tensor.shape == (10, 9, 9)
        # All should be in channel 9 (empty)
        assert torch.all(tensor[9, :, :] == 1.0)
        assert torch.all(tensor[0:9, :, :] == 0.0)

    def test_single_digit(self):
        """Test encoding board with single digit."""
        line = "5" + "0" * 80
        tensor = board_to_tensor(line)
        # First cell should be in channel 4 (digit 5 -> index 4)
        assert tensor[4, 0, 0] == 1.0
        # Rest should be empty
        assert tensor[9, 0, 1] == 1.0

    def test_full_board(self):
        """Test encoding complete board."""
        line = "123456789" * 9
        tensor = board_to_tensor(line)
        # No empty cells
        assert torch.all(tensor[9, :, :] == 0.0)
        # Each row should have digits 1-9
        for r in range(9):
            for c in range(9):
                digit = (c % 9) + 1
                ch = digit - 1
                assert tensor[ch, r, c] == 1.0

    def test_one_hot_property(self):
        """Test that encoding is one-hot (exactly one 1 per cell)."""
        line = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        tensor = board_to_tensor(line)
        # Sum across channels should be 1 for all cells
        channel_sum = tensor.sum(dim=0)
        assert torch.allclose(channel_sum, torch.ones(9, 9))


class TestLegalMask:
    def test_empty_board_all_legal(self):
        """Test that empty board has all moves legal."""
        line = "0" * 81
        mask = legal_mask_from_line(line)
        assert mask.shape == (81, 9)
        # All moves should be legal
        assert torch.all(mask == 1.0)

    def test_filled_cell_no_legal_moves(self):
        """Test that filled cells have no legal moves."""
        line = "5" + "0" * 80
        mask = legal_mask_from_line(line)
        # First cell is filled, should have no legal moves
        assert torch.all(mask[0, :] == 0.0)

    def test_row_constraint(self):
        """Test row constraint eliminates moves."""
        line = "5" + "0" * 80
        mask = legal_mask_from_line(line)
        # Rest of row 0 should not have 5 as legal move
        for c in range(1, 9):
            idx = c  # Row 0
            assert mask[idx, 4] == 0.0  # Digit 5 -> index 4

    def test_column_constraint(self):
        """Test column constraint eliminates moves."""
        line = "5" + "0" * 80
        mask = legal_mask_from_line(line)
        # Rest of column 0 should not have 5 as legal move
        for r in range(1, 9):
            idx = r * 9  # Column 0
            assert mask[idx, 4] == 0.0  # Digit 5 -> index 4

    def test_box_constraint(self):
        """Test box constraint eliminates moves."""
        line = "5" + "0" * 80
        mask = legal_mask_from_line(line)
        # Rest of top-left box should not have 5 as legal move
        for r in range(3):
            for c in range(3):
                if r == 0 and c == 0:
                    continue
                idx = r * 9 + c
                assert mask[idx, 4] == 0.0  # Digit 5 -> index 4


class TestSimplePolicyNet:
    def test_model_creation(self):
        """Test creating model."""
        model = SimplePolicyNet(width=64, drop=0.1, n_blocks=2)
        assert model is not None

    def test_forward_pass(self):
        """Test forward pass."""
        model = SimplePolicyNet(width=64, drop=0.1, n_blocks=2)
        model.eval()

        # Create input
        x = torch.zeros(2, 10, 9, 9)  # Batch of 2
        x[:, 9, :, :] = 1.0  # Empty board

        # Forward pass
        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (2, 81, 9)

    def test_forward_with_mask(self):
        """Test forward pass with legal move mask."""
        model = SimplePolicyNet(width=64, drop=0.1, n_blocks=2)
        model.eval()

        # Create input
        x = torch.zeros(1, 10, 9, 9)
        x[:, 9, :, :] = 1.0

        # Create mask (only allow first move)
        mask = torch.zeros(1, 81, 9)
        mask[0, 0, 0] = 1.0  # Only (0, 0, 1) is legal

        # Forward pass
        with torch.no_grad():
            logits = model(x, mask=mask)

        # Illegal moves should have very negative logits
        assert logits[0, 0, 1] < -1e8
        assert logits[0, 0, 0] > -1e8

    def test_output_shape_consistency(self):
        """Test that output shape is consistent."""
        model = SimplePolicyNet(width=128, drop=0.0, n_blocks=4)

        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 10, 9, 9)
            logits = model(x)
            assert logits.shape == (batch_size, 81, 9)

    def test_gradient_flow(self):
        """Test that gradients flow through model."""
        model = SimplePolicyNet(width=64, drop=0.1, n_blocks=2)
        model.train()

        x = torch.randn(2, 10, 9, 9)
        target = torch.randint(0, 9, (2, 81))

        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 9), target.view(-1)
        )
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestModelProperties:
    def test_model_is_deterministic_in_eval(self):
        """Test model is deterministic in eval mode."""
        model = SimplePolicyNet(width=64, drop=0.1, n_blocks=2)
        model.eval()

        x = torch.randn(1, 10, 9, 9)

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.allclose(out1, out2)

    def test_model_uses_dropout_in_train(self):
        """Test that dropout is active in training mode."""
        model = SimplePolicyNet(width=64, drop=0.5, n_blocks=2)
        model.train()

        x = torch.randn(1, 10, 9, 9)

        # Run multiple times - should get different outputs due to dropout
        outputs = []
        for _ in range(5):
            out = model(x)
            outputs.append(out)

        # At least some outputs should be different
        all_same = all(torch.allclose(outputs[0], out) for out in outputs[1:])
        assert not all_same, "Dropout should cause variation in train mode"
