"""Comprehensive evaluation metrics for Sudoku solvers.

Tracks cell accuracy, grid accuracy, constraint satisfaction,
solving time, and difficulty-based performance breakdown.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time
import math

import numpy as np
import torch


@dataclass
class SolverMetrics:
    """Container for solver performance metrics."""
    
    # Accuracy metrics
    cell_accuracy: float = 0.0  # Percentage of correctly filled cells
    grid_accuracy: float = 0.0  # Percentage of completely solved puzzles
    
    # Constraint satisfaction
    row_satisfaction: float = 0.0
    col_satisfaction: float = 0.0
    box_satisfaction: float = 0.0
    
    # Timing metrics
    avg_solve_time: float = 0.0  # Average time per puzzle (seconds)
    median_solve_time: float = 0.0
    min_solve_time: float = 0.0
    max_solve_time: float = 0.0
    
    # Method breakdown (for hybrid solver)
    method_counts: Dict[str, int] = field(default_factory=dict)
    
    # Difficulty breakdown
    difficulty_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Additional stats
    total_puzzles: int = 0
    total_cells_evaluated: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/saving."""
        return {
            'cell_accuracy': self.cell_accuracy,
            'grid_accuracy': self.grid_accuracy,
            'row_satisfaction': self.row_satisfaction,
            'col_satisfaction': self.col_satisfaction,
            'box_satisfaction': self.box_satisfaction,
            'avg_solve_time': self.avg_solve_time,
            'median_solve_time': self.median_solve_time,
            'min_solve_time': self.min_solve_time,
            'max_solve_time': self.max_solve_time,
            'method_counts': self.method_counts,
            'difficulty_metrics': self.difficulty_metrics,
            'total_puzzles': self.total_puzzles,
            'total_cells_evaluated': self.total_cells_evaluated
        }
    
    def __str__(self) -> str:
        """Formatted string representation."""
        lines = [
            "═══════════════════════════════════════",
            "         Sudoku Solver Metrics         ",
            "═══════════════════════════════════════",
            f"Total Puzzles Evaluated: {self.total_puzzles}",
            "",
            "Accuracy Metrics:",
            f"  Cell Accuracy:  {self.cell_accuracy:6.2f}%",
            f"  Grid Accuracy:  {self.grid_accuracy:6.2f}%",
            "",
            "Constraint Satisfaction:",
            f"  Row Constraints: {self.row_satisfaction:6.2f}%",
            f"  Col Constraints: {self.col_satisfaction:6.2f}%",
            f"  Box Constraints: {self.box_satisfaction:6.2f}%",
            "",
            "Timing Statistics:",
            f"  Avg Time:    {self.avg_solve_time*1000:7.2f} ms",
            f"  Median Time: {self.median_solve_time*1000:7.2f} ms",
            f"  Min Time:    {self.min_solve_time*1000:7.2f} ms",
            f"  Max Time:    {self.max_solve_time*1000:7.2f} ms",
        ]
        
        if self.method_counts:
            lines.extend([
                "",
                "Method Breakdown:",
            ])
            for method, count in self.method_counts.items():
                pct = 100.0 * count / self.total_puzzles if self.total_puzzles > 0 else 0
                lines.append(f"  {method:12s}: {count:5d} ({pct:5.1f}%)")
        
        if self.difficulty_metrics:
            lines.extend([
                "",
                "Performance by Difficulty:",
            ])
            for difficulty, metrics in self.difficulty_metrics.items():
                lines.append(f"  {difficulty}:")
                for key, value in metrics.items():
                    if 'time' in key:
                        lines.append(f"    {key}: {value*1000:.2f} ms")
                    else:
                        lines.append(f"    {key}: {value:.2f}%")
        
        lines.append("═══════════════════════════════════════")
        return "\n".join(lines)


def evaluate_solver(
    solver,
    puzzles: List[torch.Tensor],
    solutions: List[torch.Tensor],
    difficulty_labels: Optional[List[str]] = None,
    verbose: bool = True
) -> SolverMetrics:
    """Evaluate solver performance on a dataset.
    
    Args:
        solver: Solver instance with solve() method
        puzzles: List of puzzle tensors
        solutions: List of solution tensors
        difficulty_labels: Optional difficulty labels for each puzzle
        verbose: Print progress
        
    Returns:
        SolverMetrics with comprehensive evaluation results
    """
    metrics = SolverMetrics()
    metrics.total_puzzles = len(puzzles)
    
    solve_times = []
    correct_cells = 0
    correct_grids = 0
    total_cells = 0
    grid_size = 9  # Will be updated from first puzzle
    
    row_violations = 0
    col_violations = 0
    box_violations = 0
    
    difficulty_stats: Dict[str, Dict[str, List[float]]] = {}
    
    for i, (puzzle, solution) in enumerate(zip(puzzles, solutions)):
        if verbose and i % 100 == 0:
            print(f"Evaluating puzzle {i+1}/{len(puzzles)}...")
        
        # Solve puzzle
        start_time = time.time()
        try:
            pred_solution, info = solver.solve(puzzle)
            solve_time = time.time() - start_time
        except Exception as e:
            if verbose:
                print(f"Error solving puzzle {i}: {e}")
            solve_time = 0.0
            pred_solution = puzzle
            info = {'strategy': 'error', 'success': False}
        
        solve_times.append(solve_time)
        
        # Track method used
        method = info.get('strategy', 'unknown')
        metrics.method_counts[method] = metrics.method_counts.get(method, 0) + 1
        
        # Evaluate accuracy
        grid_size = solution.size(-1)
        blank_mask = (puzzle == 0)
        
        cell_correct = 0
        cell_total = 0
        
        if blank_mask.any():
            cell_correct = (pred_solution[blank_mask] == solution[blank_mask]).sum().item()
            cell_total = blank_mask.sum().item()
            correct_cells += cell_correct
            total_cells += cell_total
            
            # Grid accuracy (all cells correct)
            if cell_correct == cell_total:
                correct_grids += 1
        else:
            # Already solved
            correct_grids += 1
        
        # Evaluate constraints
        pred_np = pred_solution.cpu().numpy()
        row_violations += count_row_violations(pred_np)
        col_violations += count_col_violations(pred_np)
        box_violations += count_box_violations(pred_np)
        
        # Track by difficulty
        if difficulty_labels:
            diff = difficulty_labels[i]
            if diff not in difficulty_stats:
                difficulty_stats[diff] = {'times': [], 'accuracies': []}
            difficulty_stats[diff]['times'].append(solve_time)
            if blank_mask.any():
                acc = 100.0 * cell_correct / cell_total
                difficulty_stats[diff]['accuracies'].append(acc)
    
    # Calculate final metrics
    if total_cells > 0:
        metrics.cell_accuracy = 100.0 * correct_cells / total_cells
    
    metrics.grid_accuracy = 100.0 * correct_grids / len(puzzles)
    
    # Constraint satisfaction (% of puzzles with no violations)
    max_violations = len(puzzles) * grid_size
    metrics.row_satisfaction = 100.0 * (1 - row_violations / max_violations)
    metrics.col_satisfaction = 100.0 * (1 - col_violations / max_violations)
    metrics.box_satisfaction = 100.0 * (1 - box_violations / max_violations)
    
    # Timing stats
    if solve_times:
        metrics.avg_solve_time = sum(solve_times) / len(solve_times)
        metrics.median_solve_time = sorted(solve_times)[len(solve_times) // 2]
        metrics.min_solve_time = min(solve_times)
        metrics.max_solve_time = max(solve_times)
    
    # Difficulty breakdown
    for diff, stats in difficulty_stats.items():
        metrics.difficulty_metrics[diff] = {
            'avg_time': sum(stats['times']) / len(stats['times']) if stats['times'] else 0,
            'avg_accuracy': sum(stats['accuracies']) / len(stats['accuracies']) if stats['accuracies'] else 0
        }
    
    metrics.total_cells_evaluated = int(total_cells)
    
    return metrics


def count_row_violations(grid: np.ndarray) -> int:
    """Count number of row constraint violations."""
    violations = 0
    grid_size = grid.shape[0]
    for row in range(grid_size):
        unique_vals = len(set(grid[row, :]))
        violations += grid_size - unique_vals
    return violations


def count_col_violations(grid: np.ndarray) -> int:
    """Count number of column constraint violations."""
    violations = 0
    grid_size = grid.shape[0]
    for col in range(grid_size):
        unique_vals = len(set(grid[:, col]))
        violations += grid_size - unique_vals
    return violations


def count_box_violations(grid: np.ndarray) -> int:
    """Count number of box constraint violations."""
    violations = 0
    grid_size = grid.shape[0]
    block_size = int(math.sqrt(grid_size))
    
    for block_i in range(block_size):
        for block_j in range(block_size):
            block = grid[
                block_i * block_size:(block_i + 1) * block_size,
                block_j * block_size:(block_j + 1) * block_size
            ]
            unique_vals = len(set(block.flatten()))
            violations += grid_size - unique_vals
    
    return violations
