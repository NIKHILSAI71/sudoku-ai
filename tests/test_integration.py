"""Integration tests for end-to-end workflows."""
from __future__ import annotations

import pytest
import torch
import tempfile
from pathlib import Path

from sudoku_engine import Board, parse_line, board_to_line
from sudoku_engine.validator import is_valid_board
from sudoku_ai.policy import (
    board_to_tensor,
    legal_mask_from_line,
    SimplePolicyNet,
    train_supervised,
    load_policy,
)


class TestEndToEnd:
    def test_parse_encode_cycle(self):
        """Test parsing and encoding pipeline."""
        puzzle_str = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"

        # Parse
        grid = parse_line(puzzle_str)
        board = Board(grid)

        # Encode
        tensor = board_to_tensor(puzzle_str)
        mask = legal_mask_from_line(puzzle_str)

        # Verify shapes
        assert tensor.shape == (10, 9, 9)
        assert mask.shape == (81, 9)

        # Verify board properties
        assert not board.is_complete()

    def test_model_inference_pipeline(self):
        """Test complete inference pipeline."""
        model = SimplePolicyNet(width=64, drop=0.0, n_blocks=2)
        model.eval()

        puzzle_str = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"

        # Encode puzzle
        x = board_to_tensor(puzzle_str).unsqueeze(0)
        mask = legal_mask_from_line(puzzle_str).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            logits = model(x, mask=mask)
            probs = torch.softmax(logits, dim=-1)

        # Verify output
        assert logits.shape == (1, 81, 9)
        assert probs.shape == (1, 81, 9)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1, 81))


class TestTrainingPipeline:
    def test_minimal_training(self):
        """Test minimal training pipeline."""
        # Create temporary dataset
        puzzles = [
            "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
            "003020600900305001001806400008102900700000008006708200002609500800203009005010300",
        ]
        solutions = [
            "534678912672195348198342567859761423426853791713924856961537284287419635345286179",
            "483921657967345821251876493548132976729564138136798245372689514814253769695417382",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_model.pt")

            # Train
            result = train_supervised(
                out_path=output_path,
                puzzles=puzzles,
                solutions=solutions,
                epochs=2,
                batch_size=2,
                lr=0.001,
                val_split=0.0,
                augment=False,
                seed=42,
            )

            # Verify checkpoint exists
            assert Path(output_path).exists()

            # Verify history
            assert "history" in result
            history = result["history"]
            assert len(history["train_loss"]) == 2
            assert len(history["train_acc"]) == 2

            # Load and test model
            model = load_policy(output_path)
            assert model is not None

            # Test inference
            x = board_to_tensor(puzzles[0]).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
            assert logits.shape == (1, 81, 9)

    def test_training_with_auto_solution_generation(self):
        """Test training with automatic solution generation."""
        puzzles = [
            "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_model.pt")

            # Train without providing solutions
            result = train_supervised(
                out_path=output_path,
                puzzles=puzzles,
                solutions=None,  # Will auto-generate
                epochs=1,
                batch_size=2,
                val_split=0.0,
                augment=False,
                seed=42,
            )

            assert Path(output_path).exists()


class TestValidation:
    def test_validation_catches_duplicates(self):
        """Test that validation catches invalid boards."""
        board = Board.empty()

        # Create invalid board
        board.grid[0, 0] = 5
        board.grid[0, 1] = 5  # Duplicate

        assert not is_valid_board(board)

    def test_validation_accepts_valid(self):
        """Test that validation accepts valid boards."""
        board = Board.empty()
        board.set_cell(0, 0, 5)
        board.set_cell(0, 1, 3)
        board.set_cell(1, 0, 6)

        assert is_valid_board(board)

    def test_complete_valid_solution(self):
        """Test validation of complete solution."""
        solution = [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9],
        ]
        board = Board.from_list(solution)
        assert is_valid_board(board)
        assert board.is_complete()


class TestRobustness:
    def test_mask_consistency(self):
        """Test that legal mask is consistent with board state."""
        puzzle_str = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        board = Board(parse_line(puzzle_str))

        # Get masks
        board_masks = board.candidates_mask()
        tensor_mask = legal_mask_from_line(puzzle_str)

        # Verify consistency
        for idx in range(81):
            r, c = divmod(idx, 9)
            board_mask = int(board_masks[r, c])

            for d in range(1, 10):
                is_legal_board = bool(board_mask & (1 << (d - 1)))
                is_legal_tensor = bool(tensor_mask[idx, d - 1] > 0.5)

                assert is_legal_board == is_legal_tensor, \
                    f"Inconsistency at R{r+1}C{c+1} digit {d}"

    def test_model_respects_mask(self):
        """Test that masked logits are very negative."""
        model = SimplePolicyNet(width=64, drop=0.0, n_blocks=2)
        model.eval()

        x = torch.zeros(1, 10, 9, 9)
        x[:, 9, :, :] = 1.0

        # Create restrictive mask
        mask = torch.zeros(1, 81, 9)
        mask[0, 0, 0] = 1.0  # Only one legal move

        with torch.no_grad():
            logits = model(x, mask=mask)

        # Masked positions should have very negative logits
        assert logits[0, 0, 1] < -1e8
        assert logits[0, 1, 0] < -1e8
        # Unmasked position should be normal
        assert logits[0, 0, 0] > -1e8
