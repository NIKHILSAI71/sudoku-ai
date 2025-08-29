# Sudoku Pro

Production-grade Sudoku engine with multiple solver backends (heuristics+backtracking, DLX/Algorithm X, optional SAT/CP-SAT) and a PyTorch learning agent.

## Quickstart

- Python 3.11+
- Optional extras: `[sat]` for python-sat and `[cpsat]` for OR-Tools

Install (editable for dev):

```bash
pip install -e .[dev]
# optional extras
pip install -e .[sat]
pip install -e .[cpsat]
```

Solve a puzzle:

```bash
sudoku solve -i examples/easy1.sdk -b dlx --trace
```


Train the learning agent (toy):

```bash
sudoku train --epochs 1 --limit 500 --out checkpoints/policy.pt
```

Evaluate:

```bash
sudoku eval -i data/out.jsonl -b dlx heur backtracking policy
```

## Design overview

- `sudoku_engine/`: Core domain: `Board`, parsing/serialization, and validation.
- `sudoku_solvers/`: Multiple backends: heuristics, CP+backtracking, DLX, SAT/CP-SAT adapters.
- `sudoku_ai/`: Data pipelines, PyTorch models, training/eval, self-play hooks.
- `cli/`: Unified CLI for solve/train/eval/benchmark.
- `tests/`: Unit, property, and integration tests.
- `docs/`: Design notes, tutorials, API docs.

Key acceptance:
- Three independent solver backends: heuristics+backtracking, DLX, and CP/SAT (optional).
- Learning agent improves search ordering and reduces expansions vs unguided search.

## Datasets

- Optional public datasets (Kaggle/17M Sudoku) via `scripts/` ingestion (small samples included).

## Benchmarks

Run: `sudoku benchmark -s 100 -b dlx backtracking heur` to get nodes/time metrics and CSV output.

## License

MIT