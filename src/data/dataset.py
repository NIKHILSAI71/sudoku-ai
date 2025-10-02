"""Training utilities and dataset classes.

Curriculum learning dataset that progressively increases difficulty
during training for better convergence.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
from pathlib import Path
import math

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class SudokuDataset(Dataset):
    """Basic Sudoku dataset."""
    
    def __init__(
        self,
        puzzles: torch.Tensor,
        solutions: torch.Tensor,
        augment: bool = False
    ):
        """Initialize dataset.
        
        Args:
            puzzles: Puzzle grids [N, size, size]
            solutions: Solution grids [N, size, size]
            augment: Apply data augmentation
        """
        assert puzzles.shape == solutions.shape
        self.puzzles = puzzles
        self.solutions = solutions
        self.augment = augment
        self.grid_size = puzzles.size(-1)
    
    def __len__(self) -> int:
        return len(self.puzzles)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        puzzle = self.puzzles[idx].clone()
        solution = self.solutions[idx].clone()
        
        if self.augment:
            puzzle, solution = self._augment(puzzle, solution)
        
        return puzzle, solution
    
    def _augment(
        self,
        puzzle: torch.Tensor,
        solution: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentation.
        
        Applies digit permutation, rotation, and transpose randomly.
        """
        # Digit permutation (40% chance)
        if torch.rand(1).item() < 0.4:
            perm = torch.randperm(self.grid_size) + 1
            perm_dict = {i: perm[i-1].item() for i in range(1, self.grid_size+1)}
            perm_dict[0] = 0
            
            puzzle_new = puzzle.clone()
            solution_new = solution.clone()
            for old_val, new_val in perm_dict.items():
                puzzle_new[puzzle == old_val] = new_val
                solution_new[solution == old_val] = new_val
            puzzle, solution = puzzle_new, solution_new
        
        # Rotation (30% chance)
        if torch.rand(1).item() < 0.3:
            k = int(torch.randint(1, 4, (1,)).item())  # Rotate 90, 180, or 270 degrees
            puzzle = torch.rot90(puzzle, k=k, dims=(0, 1))
            solution = torch.rot90(solution, k=k, dims=(0, 1))
        
        # Transpose (20% chance)
        if torch.rand(1).item() < 0.2:
            puzzle = puzzle.T
            solution = solution.T
        
        return puzzle, solution


class CurriculumDataset(SudokuDataset):
    """Dataset with curriculum learning based on difficulty.
    
    Progressively increases difficulty by sorting puzzles by
    number of empty cells (more empty = harder).
    """
    
    def __init__(
        self,
        puzzles: torch.Tensor,
        solutions: torch.Tensor,
        curriculum_stage: int = 0,
        num_stages: int = 3,
        augment: bool = False
    ):
        """Initialize curriculum dataset.
        
        Args:
            puzzles: Puzzle grids
            solutions: Solution grids
            curriculum_stage: Current stage (0=easy, 1=medium, 2=hard)
            num_stages: Total number of curriculum stages
            augment: Apply data augmentation
        """
        super().__init__(puzzles, solutions, augment)
        
        self.curriculum_stage = curriculum_stage
        self.num_stages = num_stages
        
        # Calculate difficulty for each puzzle (number of empty cells)
        self.difficulties = (puzzles == 0).sum(dim=(1, 2))
        
        # Sort by difficulty
        sorted_indices = torch.argsort(self.difficulties)
        self.puzzles = self.puzzles[sorted_indices]
        self.solutions = self.solutions[sorted_indices]
        self.difficulties = self.difficulties[sorted_indices]
        
        # Determine which subset to use based on curriculum stage
        self._set_curriculum_subset()
    
    def _set_curriculum_subset(self):
        """Set the active subset based on curriculum stage."""
        total_samples = len(self.puzzles)
        
        if self.curriculum_stage == 0:
            # Easy stage: first 40%
            end_idx = int(0.4 * total_samples)
            self.active_indices = range(0, end_idx)
        elif self.curriculum_stage == 1:
            # Medium stage: middle 40%
            start_idx = int(0.3 * total_samples)
            end_idx = int(0.7 * total_samples)
            self.active_indices = range(start_idx, end_idx)
        else:
            # Hard stage: all data
            self.active_indices = range(0, total_samples)
    
    def __len__(self) -> int:
        return len(self.active_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        actual_idx = self.active_indices[idx]
        return super().__getitem__(actual_idx)
    
    def advance_stage(self):
        """Advance to next curriculum stage."""
        if self.curriculum_stage < self.num_stages - 1:
            self.curriculum_stage += 1
            self._set_curriculum_subset()


def load_kaggle_dataset(
    file_path: str | Path,
    max_samples: Optional[int] = None,
    grid_size: int = 9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load Kaggle Sudoku dataset.
    
    Supports multiple Kaggle dataset formats:
    - 'puzzle' and 'solution' columns (3M dataset)
    - 'quizzes' and 'solutions' columns (9M dataset)
    - First two columns as fallback
    
    Args:
        file_path: Path to CSV file with puzzle/solution data
        max_samples: Maximum samples to load (None = all)
        grid_size: Grid size (default 9x9)
        
    Returns:
        (puzzles, solutions) tensors of shape [N, grid_size, grid_size]
    """
    df = pd.read_csv(file_path)
    
    # Auto-detect column names
    columns = df.columns.tolist()
    
    # Try different common column name combinations
    if 'puzzle' in columns and 'solution' in columns:
        puzzle_col, solution_col = 'puzzle', 'solution'
    elif 'quizzes' in columns and 'solutions' in columns:
        puzzle_col, solution_col = 'quizzes', 'solutions'
    elif len(columns) >= 2:
        # Fallback to first two columns
        puzzle_col, solution_col = columns[0], columns[1]
        print(f"Warning: Using columns '{puzzle_col}' and '{solution_col}' as puzzle/solution")
    else:
        raise ValueError(
            f"Cannot find puzzle/solution columns. Available columns: {columns}\n"
            f"Expected: 'puzzle' & 'solution' OR 'quizzes' & 'solutions'"
        )
    
    print(f"Loading dataset with columns: '{puzzle_col}' and '{solution_col}'")
    
    if max_samples:
        df = df.head(max_samples)
    
    puzzles = []
    solutions = []
    
    for _, row in df.iterrows():
        puzzle_str = row[puzzle_col]
        solution_str = row[solution_col]
        
        # Convert string to grid
        puzzle_grid = torch.tensor([int(c) for c in puzzle_str], dtype=torch.long)
        solution_grid = torch.tensor([int(c) for c in solution_str], dtype=torch.long)
        
        puzzle_grid = puzzle_grid.reshape(grid_size, grid_size)
        solution_grid = solution_grid.reshape(grid_size, grid_size)
        
        puzzles.append(puzzle_grid)
        solutions.append(solution_grid)
    
    return torch.stack(puzzles), torch.stack(solutions)


def create_curriculum_dataloaders(
    puzzles: torch.Tensor,
    solutions: torch.Tensor,
    batch_size: int = 128,
    val_split: float = 0.1,
    curriculum_stage: int = 0,
    augment: bool = True,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders with curriculum learning.
    
    Args:
        puzzles: All puzzle grids
        solutions: All solution grids
        batch_size: Batch size
        val_split: Validation split ratio
        curriculum_stage: Current curriculum stage
        augment: Apply augmentation to training data
        num_workers: Number of dataloader workers
        
    Returns:
        (train_loader, val_loader)
    """
    # Split data
    num_val = int(len(puzzles) * val_split)
    num_train = len(puzzles) - num_val
    
    train_puzzles = puzzles[:num_train]
    train_solutions = solutions[:num_train]
    val_puzzles = puzzles[num_train:]
    val_solutions = solutions[num_train:]
    
    # Create datasets
    train_dataset = CurriculumDataset(
        train_puzzles,
        train_solutions,
        curriculum_stage=curriculum_stage,
        augment=augment
    )
    
    val_dataset = SudokuDataset(
        val_puzzles,
        val_solutions,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
