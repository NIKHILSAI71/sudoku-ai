# 🎯 Training Monitoring Guide

## Quick Reference: What to Expect

### ⚡ Immediate Improvements (First Batch)
```
Before: 4.19s/batch with UserWarning
After:  0.2-0.5s/batch, no warnings

✅ Should see 10-50x speedup immediately
✅ No more tensor construction warnings
```

### 📊 Loss Trajectory (Expected)

#### Epoch 1 (Warmup - LR: 10% of base)
```
Train Loss: 2.0-1.8  →  CE: 1.5-1.3,  Constraint: 0.8-0.7
Accuracy: 35-45%
Time/Epoch: 5-15 minutes (was ~3 hours)
```

#### Epoch 3 (Warmup End - LR: 100%)
```
Train Loss: 1.3-1.1  →  CE: 0.9-0.7,  Constraint: 0.7-0.6
Accuracy: 65-75%
```

#### Epoch 10 (Full Training)
```
Train Loss: 0.9-0.7  →  CE: 0.5-0.4,  Constraint: 0.6-0.5
Accuracy: 85-90%
Cell Acc: 88-92%, Grid Acc: 55-70%
```

#### Epoch 20 (Convergence)
```
Train Loss: 0.6-0.5  →  CE: 0.3-0.2,  Constraint: 0.5-0.4
Accuracy: 92-95%
Cell Acc: 94-96%, Grid Acc: 75-85%
```

#### Epoch 40 (Target)
```
Train Loss: 0.4-0.3  →  CE: 0.1-0.05,  Constraint: 0.4-0.3
Accuracy: 96-98%
Cell Acc: 97-98%, Grid Acc: 85-92%
```

---

## 🚨 Troubleshooting

### If Training is Still Slow (>1s/batch):
1. Check GPU is being used: `Device: cuda` in logs
2. Verify mixed precision: `Mixed Precision: True`
3. Check batch size: Should be 128
4. Ensure CUDA is available: `torch.cuda.is_available()`

### If Loss Not Decreasing:
```
Symptom: Loss stays ~2.0 after 5 epochs
Solution: 
  1. Increase constraint_weight from 0.5 to 1.0
  2. Decrease learning_rate from 1e-3 to 5e-4
  3. Check data quality (puzzles have solutions)
```

### If Gradients Exploding:
```
Symptom: Loss suddenly jumps to inf or nan
Solution:
  1. Gradient clipping is already enabled (1.0)
  2. Reduce learning_rate to 5e-4
  3. Increase warmup_epochs to 5
```

### If Accuracy Plateaus Early:
```
Symptom: Stuck at 70-80% accuracy
Solution:
  1. Increase num_iterations from 32 to 48
  2. Increase hidden_dim from 96 to 128
  3. Train longer (more epochs)
```

---

## 📈 Monitoring Checklist

### Every Epoch:
- [ ] Batch time: Should be 0.2-0.5s (not 4s)
- [ ] Loss decreasing: Each component going down
- [ ] Accuracy increasing: Should gain 2-5% per epoch early on
- [ ] No warnings: Clean training logs

### Every 5 Epochs:
- [ ] Learning rate: Should be decreasing (cosine schedule)
- [ ] CE Loss: Should be < Constraint Loss initially
- [ ] Constraint Loss: Should stabilize around 0.4-0.6
- [ ] Grid Accuracy: Should be improving steadily

### Red Flags:
- ❌ Batch time > 1s → Check GPU usage
- ❌ Loss increasing → Reduce learning rate
- ❌ Loss = NaN → Gradients exploded, restart with lower LR
- ❌ Accuracy stuck → Increase model capacity or training time

---

## 🎯 Success Metrics

### Excellent Training (Target):
```
By Epoch 10: 85%+ cell accuracy
By Epoch 20: 92%+ cell accuracy
By Epoch 40: 96%+ cell accuracy
Final: 97-98% cell accuracy, 85-92% grid accuracy
```

### Training Speed:
```
Time per epoch: 5-20 minutes (was 2-3 hours)
Time to 90% accuracy: 2-3 hours (was 1-2 days)
Total training time: 4-8 hours (was 2-3 days)
```

### Loss Values:
```
Start: Total ~2.2, CE ~1.8, Constraint ~0.8
Middle: Total ~0.7, CE ~0.4, Constraint ~0.6
End: Total ~0.3, CE ~0.1, Constraint ~0.4
```

---

## 💡 Pro Tips

1. **First 3 epochs are warmup** - Loss drops slower, this is intentional
2. **CE loss should drop faster than constraint loss** - Model learns digits first
3. **Constraint loss stabilizes around 0.4-0.6** - This is normal and expected
4. **Cell accuracy >> Grid accuracy initially** - Grid accuracy catches up later
5. **Watch for constraint violations** - If high, increase constraint_weight

---

## 🔬 Understanding the Metrics

### CE Loss (Cross-Entropy):
- Measures: How well model predicts correct digits
- Target: <0.1 (very confident predictions)
- Interpretation: Lower = better digit predictions

### Constraint Loss:
- Measures: How well model follows Sudoku rules
- Target: 0.3-0.5 (some constraint satisfaction)
- Interpretation: Lower = fewer rule violations

### Cell Accuracy:
- Measures: % of individual cells predicted correctly
- Target: 96-98%
- Interpretation: Overall prediction quality

### Grid Accuracy:
- Measures: % of complete puzzles solved perfectly
- Target: 80-92%
- Interpretation: End-to-end solving ability

---

## 📞 Quick Fixes

### Training too slow?
```python
# In trainer initialization:
batch_size=256  # Increase from 128
num_workers=8   # Increase from 4
```

### Want faster convergence?
```python
# In loss initialization:
constraint_weight=1.0  # Increase from 0.5
focal_gamma=3.0       # Increase from 2.0
```

### Model not learning patterns?
```python
# In model initialization:
hidden_dim=128       # Increase from 96
num_iterations=48    # Increase from 32
```

### Need more stability?
```python
# In trainer initialization:
gradient_clip=0.5    # Decrease from 1.0
lr=5e-4             # Decrease from 1e-3
warmup_epochs=5      # Increase from 3
```

---

## ✅ Checklist Before Reporting Issues

- [ ] GPU is being used (`nvidia-smi` shows GPU usage)
- [ ] Mixed precision enabled (logs show "Mixed Precision: True")
- [ ] CUDA version compatible with PyTorch
- [ ] Batch size appropriate for GPU memory
- [ ] Dataset loaded correctly (1M puzzles)
- [ ] Trained for at least 10 epochs
- [ ] No NaN or Inf in losses
- [ ] Learning rate in reasonable range (1e-5 to 1e-2)

---

*The model should now learn patterns exactly and train 10-50x faster!* 🚀
