# 🚀 QUICK START - Pure Neural Solver

## ✅ What You Have Now

Your Sudoku solver now uses **ONLY the GNN model** - no backtracking, no classical algorithms!

---

## 📋 Quick Commands

### 1. **Test the Neural Solver** (After Training)

```bash
# Basic solving (iterative strategy - recommended)
python scripts/solve.py --puzzle examples/easy1.sdk

# Fast solving (greedy - single pass)
python scripts/solve.py --puzzle examples/easy1.sdk --strategy greedy

# Accurate solving (careful - best for hard puzzles)
python scripts/solve.py --puzzle examples/easy1.sdk --strategy careful

# With verbose logging to see what's happening
python scripts/solve.py --puzzle examples/easy1.sdk --verbose
```

### 2. **Train the Model First** (If Not Trained Yet)

```bash
# Quick test training (10k samples, 10 epochs)
python scripts/train.py --data sudoku.csv --max-samples 10000 --epochs 10

# Full training (recommended - all data, 60 epochs)
python scripts/train.py --data sudoku.csv --epochs 60
```

---

## 🎯 Expected Output

### ✅ Success (Pure Neural):
```
======================================================================
  🧠 PURE NEURAL SOLVING (No Classical Algorithms)
======================================================================

Strategy: iterative
Initial confidence: 0.9
Max iterations: 50

🎯 Solving puzzle with 51 empty cells...
📊 Strategy: iterative, Initial confidence: 0.90

  Iter 1: Filled 15 cells (conf=0.947, threshold=0.900)
  Iter 2: Filled 18 cells (conf=0.891, threshold=0.855)
  Iter 3: Filled 12 cells (conf=0.823, threshold=0.812)
  Iter 4: Filled 6 cells (conf=0.756, threshold=0.771)

✅ SOLVED in 145.3ms (4 iterations)

======================================================================
  ✅ SOLVED BY NEURAL NETWORK!
======================================================================

Method: iterative_refinement
Solve time: 145.23 ms
Iterations: 4
Cells filled: 51
Avg confidence: 0.854
```

### ❌ Old Output (Was Using Backtracking):
```
Strategy used: backtracking  ← BAD! This was classical algorithm
Solve time: 1024.56 ms
```

---

## 🔧 Adjusting Parameters

### If model isn't filling cells:
```bash
# Lower confidence threshold
python scripts/solve.py --puzzle examples/easy1.sdk --confidence 0.80

# Use greedy (forces predictions)
python scripts/solve.py --puzzle examples/easy1.sdk --strategy greedy
```

### If getting incomplete solutions:
```bash
# More iterations
python scripts/solve.py --puzzle examples/easy1.sdk --max-iterations 100

# Try careful strategy
python scripts/solve.py --puzzle examples/easy1.sdk --strategy careful
```

---

## 📊 Strategy Guide

| Strategy | When to Use | Command |
|----------|-------------|---------|
| **greedy** | Fast inference, easy puzzles | `--strategy greedy` |
| **iterative** | Most puzzles (DEFAULT) | `--strategy iterative` |
| **careful** | Hard puzzles, maximum accuracy | `--strategy careful` |

---

## 🎓 Full Example Session

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model (quick test)
python scripts/train.py --data sudoku.csv --max-samples 10000 --epochs 10

# 3. Test solving
python scripts/solve.py --puzzle examples/easy1.sdk --verbose

# 4. Expected output:
#    ✅ SOLVED BY NEURAL NETWORK!
#    Method: iterative_refinement
#    Solve time: 145.23 ms
```

---

## 🎯 Key Features

1. **✅ Pure Neural** - No backtracking fallback
2. **✅ Constraint-Aware** - Masks illegal moves
3. **✅ Iterative Refinement** - Fills easiest cells first
4. **✅ Adaptive Confidence** - Adjusts threshold automatically
5. **✅ Multiple Strategies** - greedy, iterative, careful

---

## 🔥 Performance Tips

1. **Train for 60 epochs** - Better accuracy
2. **Use our optimized training** - 10-30x faster
3. **Start with iterative** - Good balance
4. **Try careful for hard puzzles** - More accurate
5. **Lower confidence if stuck** - 0.80 or 0.85

---

## 💡 Troubleshooting

| Problem | Solution |
|---------|----------|
| Model not confident | Train longer (60+ epochs) |
| Incomplete solutions | Lower confidence: `--confidence 0.80` |
| Want faster solving | Use greedy: `--strategy greedy` |
| Want more accuracy | Use careful: `--strategy careful` |
| Model not trained | Run: `python scripts/train.py --data sudoku.csv --epochs 60` |

---

## 🎉 Success Indicators

✅ You'll know it's working when you see:
- **"SOLVED BY NEURAL NETWORK!"** (not "backtracking")
- **"Method: iterative_refinement"** (or greedy/careful)
- **Fast solve times** (10-500ms depending on strategy)
- **High confidence** (0.80-0.95 average)
- **Valid solutions** (all constraints satisfied)

---

**Enjoy your pure neural Sudoku solver! 🧠✨**
