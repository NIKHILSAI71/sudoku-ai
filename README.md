# Sudoku AI 🧠

**Production-grade AI-powered Sudoku solver using deep learning.**

This project uses a convolutional neural network to solve Sudoku puzzles entirely through learned patterns—no algorithmic solvers, no backtracking, just pure AI inference.

---

## ✨ Features

- **Pure AI Solving**: Neural network trained to predict next moves
- **No Algorithmic Fallbacks**: 100% learned solution strategy
- **Clean Architecture**: Minimal codebase focused on core functionality
- **Easy Training**: Train on your own puzzle datasets
- **Production Ready**: Simplified, maintainable code structure

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd sudoku

# Install dependencies
pip install -e .

# Or with specific torch version
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118
```

### Requirements
- Python 3.11+
- PyTorch 2.0+
- NumPy 1.26+
- Rich (for CLI display)

---

## 📖 Usage

### 1. Train a Model

First, you need a dataset in JSONL format where each line contains a puzzle-solution pair:

```jsonl
{"puzzle": "530070000600195000098000060800060003400803001700020006060000280000419005000080079", "solution": "534678912672195348198342567859761423426853791713924856961537284287419635345286179"}
{"puzzle": "000000000000000000000000000000000000000000000000000000000000000000000000000000000", "solution": "123456789456789123789123456214365897365897214897214365531642978642978531978531642"}
```

Then train:

```bash
sudoku train --dataset data/puzzles.jsonl --output checkpoints/my_model.pt --epochs 20
```

**Training Options:**
```bash
sudoku train --help

Options:
  --dataset PATH        Path to JSONL dataset (required)
  --output PATH         Output checkpoint path (default: checkpoints/policy.pt)
  --epochs N            Number of epochs (default: 10)
  --batch-size N        Batch size (default: 64)
  --lr FLOAT            Learning rate (default: 0.001)
  --val-split FLOAT     Validation split (default: 0.1)
  --max-samples N       Limit training samples
  --no-augment          Disable data augmentation
  --seed N              Random seed (default: 42)
```

### 2. Solve Puzzles

Once you have a trained model:

```bash
# Solve from file
sudoku solve -i examples/easy1.sdk --ckpt checkpoints/my_model.pt --pretty

# Solve from string
sudoku solve --stdin "530070000600195000098000060800060003400803001700020006060000280000419005000080079" --pretty

# Verbose mode (show each step)
sudoku solve -i examples/easy1.sdk --verbose
```

**Solving Options:**
```bash
sudoku solve --help

Options:
  -i, --input PATH      Puzzle file path
  --stdin STRING        Puzzle string (81 chars, 0 for empty)
  --ckpt PATH           Model checkpoint (default: checkpoints/policy.pt)
  --cpu                 Force CPU
  --max-steps N         Max solving steps (default: 200)
  --temperature FLOAT   Sampling temperature (default: 1.0)
  --pretty              Pretty-print board
  --verbose             Show step-by-step moves
```

---

## 📂 Project Structure

```
sudoku/
├── sudoku_engine/          # Core Sudoku logic
│   ├── board.py           # Board representation & candidate tracking
│   ├── parser.py          # Parse puzzle strings
│   ├── validator.py       # Basic validation
│   └── __init__.py
├── sudoku_ai/             # Neural network components
│   ├── policy.py          # SimplePolicyNet model & training
│   ├── data.py            # Dataset loading & augmentation
│   └── __init__.py
├── cli/                   # Command-line interface
│   └── main.py            # Solve & train commands
├── ui/                    # Display utilities
│   └── tui.py             # Pretty board rendering
├── examples/              # Sample puzzles
│   ├── easy1.sdk
│   └── test.sdk
├── pyproject.toml         # Project configuration
└── README.md
```

---

## 🧪 How It Works

### Model Architecture

**SimplePolicyNet** - A lightweight CNN:
- **Input**: 10-channel one-hot encoding (9 digits + empty)
- **Backbone**: 3x Conv2d layers (64 filters, 3×3 kernels)
- **Head**: Flattened → Dense → 81×9 logits
- **Output**: Probability distribution over (cell, digit) pairs

### Training Process

1. **Data Augmentation**: Random digit permutations & geometric transforms
2. **Partial Boards**: Creates training samples from intermediate solving states
3. **Supervised Learning**: Cross-entropy loss on next-move prediction
4. **Validation**: Tracks accuracy and loss on held-out set

### Inference Strategy

1. Load puzzle into board representation
2. For each step:
   - Encode board as one-hot tensor
   - Run forward pass through network
   - Mask illegal moves using Sudoku constraints
   - Sample next move from probability distribution
   - Update board state
3. Repeat until complete (or max steps reached)

---

## 📊 Example Session

```bash
$ sudoku solve -i examples/easy1.sdk --pretty --verbose

📋 Loaded puzzle from: examples/easy1.sdk
┌───────┬───────┬───────┐
│ 5 3 · │ · 7 · │ · · · │
│ 6 · · │ 1 9 5 │ · · · │
│ · 9 8 │ · · · │ · 6 · │
├───────┼───────┼───────┤
│ 8 · · │ · 6 · │ · · 3 │
│ 4 · · │ 8 · 3 │ · · 1 │
│ 7 · · │ · 2 · │ · · 6 │
├───────┼───────┼───────┤
│ · 6 · │ · · · │ 2 8 · │
│ · · · │ 4 1 9 │ · · 5 │
│ · · · │ · 8 · │ · 7 9 │
└───────┴───────┴───────┘

🔧 Using device: cuda
✅ Loaded model from: checkpoints/policy.pt
🤖 Solving puzzle (max steps: 200)...
  Step 1: R1C3=4
  Step 2: R1C4=6
  Step 3: R2C2=7
  ...
  Step 49: R9C7=3

✅ Solution found!
534678912672195348198342567859761423426853791713924856961537284287419635345286179

📊 Pretty board:
┌───────┬───────┬───────┐
│ 5 3 4 │ 6 7 8 │ 9 1 2 │
│ 6 7 2 │ 1 9 5 │ 3 4 8 │
│ 1 9 8 │ 3 4 2 │ 5 6 7 │
├───────┼───────┼───────┤
│ 8 5 9 │ 7 6 1 │ 4 2 3 │
│ 4 2 6 │ 8 5 3 │ 7 9 1 │
│ 7 1 3 │ 9 2 4 │ 8 5 6 │
├───────┼───────┼───────┤
│ 9 6 1 │ 5 3 7 │ 2 8 4 │
│ 2 8 7 │ 4 1 9 │ 6 3 5 │
│ 3 4 5 │ 2 8 6 │ 1 7 9 │
└───────┴───────┴───────┘
```

---

## 🎯 Creating Training Data

Your dataset should be in JSONL format (one JSON object per line):

```python
import json

# Example: Generate dataset
data = []
for puzzle, solution in your_puzzle_pairs:
    data.append({
        "puzzle": puzzle,      # 81 chars, 0 for empty
        "solution": solution   # 81 chars, complete solution
    })

# Save to file
with open("dataset.jsonl", "w") as f:
    for record in data:
        f.write(json.dumps(record) + "\n")
```

**Puzzle Format:**
- String of 81 characters
- Digits 1-9 for filled cells
- `0` (zero) for empty cells
- Row-major order (left→right, top→bottom)

---

## 🔧 Advanced

### Temperature Control

Control sampling randomness:
- `--temperature 0.5` → More deterministic (greedy-like)
- `--temperature 1.0` → Balanced (default)
- `--temperature 2.0` → More exploratory

### GPU Usage

The solver automatically uses CUDA if available:
```bash
# Check GPU usage
nvidia-smi

# Force CPU
sudoku solve -i puzzle.sdk --cpu
```

---

## 🤝 Contributing

This is a clean, production-focused codebase. Contributions welcome:
1. Keep code simple and focused
2. No algorithmic solver dependencies
3. Maintain pure AI approach

---

## 📝 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

Built with:
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computing
- **Rich** - Beautiful terminal formatting

---

## 📚 Citation

If you use this project in research:

```bibtex
@software{sudoku_ai_2024,
  title={Sudoku AI: Pure Neural Network Sudoku Solver},
  author={Sudoku AI Team},
  year={2024},
  url={https://github.com/yourusername/sudoku}
}
```

---

**Happy Solving! 🧩**
