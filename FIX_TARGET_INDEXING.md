# Critical Fixes - CUDA Assertion Error & Deprecation Warning

## Issues Resolved

### 1. CUDA Assertion Error (Primary Issue)

The training was failing with a CUDA assertion error:
```
nll_loss_forward_no_reduce_cuda_kernel: block: [10,0,0], thread: [44,0,0] 
Assertion `cur_target >= 0 && cur_target < n_classes` failed.
```

This error occurred during cross-entropy loss computation in `loss.py:59`.

## Root Cause

The issue was a **mismatch between model output indexing and target value range**:

### Model Output
- The model outputs `grid_size` classes (9 classes for 9x9 Sudoku)
- Output indices: **0-8** representing digits **1-9**
- See `sudoku_gnn.py:105`: `nn.Linear(hidden_dim, grid_size)`

### Target Values
- Dataset provides targets with values **0-9**
  - `0` = empty cell
  - `1-9` = Sudoku digits
- Target range was **1-indexed**, but model expected **0-indexed**

### The Problem
When computing loss:
- Model predicts indices `0-8` (for digits 1-9)
- CrossEntropyLoss expects targets in range `[0, num_classes-1]` = `[0, 8]`
- But targets contained values `1-9`, causing target `9` to be out of bounds for 9 classes

## Solution

### 1. Loss Function Fix (`src/training/loss.py`)

Added target conversion before loss computation:

```python
# Convert targets from 1-indexed (1-9) to 0-indexed (0-8)
# Model outputs grid_size classes for digits 1 to grid_size
# Targets contain 0 for empty cells and 1-grid_size for digits
# We subtract 1 to get 0-based indices, clamping to handle empty cells
targets_flat = (targets_flat - 1).clamp(0, num_classes - 1)
```

**Mapping:**
- Target `1` → Index `0` (digit 1)
- Target `2` → Index `1` (digit 2)
- ...
- Target `9` → Index `8` (digit 9)
- Target `0` → Index `0` (clamped, but masked out anyway)

### 2. Accuracy Computation Fix (`src/training/trainer.py`)

Updated both training and validation accuracy to convert predictions back:

```python
# Training epoch (line 137)
preds = logits.argmax(dim=-1) + 1  # Convert 0-indexed to 1-indexed

# Validation (line 198)
preds = logits.argmax(dim=-1) + 1  # Convert 0-indexed to 1-indexed
```

This ensures predictions (0-8) are converted to Sudoku digits (1-9) before comparing with solutions.

### 3. Documentation Updates

Updated:
- Model docstring to clarify output format
- Documentation examples to include `+ 1` conversion

## Files Modified

1. **src/training/loss.py**
   - Line 61: Added target indexing conversion

2. **src/training/trainer.py**
   - Line 137: Fixed training accuracy computation
   - Line 198: Fixed validation accuracy computation

3. **src/models/gnn/sudoku_gnn.py**
   - Line 128: Clarified docstring about output format

4. **docs/ARCHITECTURE_COMPARISON.md**
   - Lines 171, 358: Updated code examples

## Verification

The model's `predict()` method already handled this correctly:
```python
predictions = probs.max(dim=-1) + 1  # Convert to 1-indexed
```

This confirms our fix is consistent with the intended design.

## Key Takeaway

**Always ensure target values match the model's output indexing:**
- Model outputs `N` classes → expects target indices `[0, N-1]`
- If your targets are 1-indexed, subtract 1 before loss computation
- If you need 1-indexed predictions, add 1 after argmax

### 2. Deprecated GradScaler Warning (Secondary Issue)

The training was showing a deprecation warning:
```
FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. 
Please use `torch.amp.GradScaler('cuda', args...)` instead.
```

**Fix:**
Updated imports and usage to the new PyTorch AMP API:

```python
# Old import
from torch.cuda.amp import GradScaler

# New import
from torch.amp.grad_scaler import GradScaler

# Usage
self.scaler = GradScaler('cuda') if self.use_amp else None
```

## Testing Recommendations

After these fixes, the training should proceed without errors. Monitor:
1. ✅ No CUDA assertion errors
2. ✅ No deprecation warnings
3. ✅ Loss decreases smoothly
4. ✅ Accuracy metrics look reasonable (60-70% early in training, 96-98% final)
5. ✅ No NaN losses or gradient explosions
