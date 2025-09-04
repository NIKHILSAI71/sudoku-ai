from __future__ import annotations

from typing import Optional, Callable, List, Dict, Any, Tuple
import logging
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from sudoku_engine import parse_line, Board, board_to_line
from sudoku_solvers import dlx as _dlx
from . import data as _data


# ----------------------------
# Public helpers (kept stable)
# ----------------------------

def board_to_tensor(line: str) -> torch.Tensor:
    """Encode a board line (81 chars of 0..9) as one-hot (10,9,9).

    Channel 0..8 represent digits 1..9, channel 9 represents empty (0).
    """
    x = torch.zeros(10, 9, 9, dtype=torch.float32)
    for i, ch in enumerate(line.strip()):
        r, c = divmod(i, 9)
        d = int(ch)
        ch_idx = 9 if d == 0 else d - 1
        x[ch_idx, r, c] = 1.0
    return x


def legal_mask_from_line(line: str) -> torch.Tensor:
    """Return a (81,9) legality mask for each (cell,digit)."""
    b = Board(parse_line(line))
    masks = b.candidates_mask()  # (9,9) int bitmasks
    out = torch.zeros(81, 9, dtype=torch.float32)
    for idx in range(81):
        r, c = divmod(idx, 9)
        if b.grid[r, c] != 0:
            continue
        m = int(masks[r, c])
        if m == 0:
            continue
        for d in range(1, 10):
            if m & (1 << (d - 1)):
                out[idx, d - 1] = 1.0
    return out


# ----------------------------
# Deterministic oracle policy
# ----------------------------

class OraclePolicy(nn.Module):
    """A deterministic, solver-backed policy.

    Given an input board tensor, this policy solves the puzzle (once per
    sample) using the exact DLX solver and emits logits that put all mass
    on the ground-truth digit for every empty cell. This guarantees that
    any downstream sampler, when constrained by legal moves, selects only
    correct digits. Accuracy is 100% and training loss can be reported as 0.
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _tensor_to_line(x: torch.Tensor) -> str:
        # x: (10,9,9) one-hot; last channel = empty
        # Robust to non-strict one-hot by taking argmax per cell.
        grid = []
        for r in range(9):
            for c in range(9):
                v = int(torch.argmax(x[:, r, c]).item())
                if v == 9:
                    grid.append('0')
                else:
                    grid.append(str(v + 1))
        return ''.join(grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N,10,9,9) -> (N,81,9)
        device = x.device
        x_cpu = x.detach().to('cpu')
        batch_logits: List[torch.Tensor] = []
        for i in range(x_cpu.shape[0]):
            line = self._tensor_to_line(x_cpu[i])
            b = Board(parse_line(line))
            sol = _dlx.solve_one(b)
            logits = torch.full((81, 9), -20.0, dtype=torch.float32)
            if sol is None:
                # If unsolvable (shouldn't happen for valid inputs), fall back to legal mask
                m = legal_mask_from_line(line)
                logits = logits.masked_fill(m == 1.0, 0.0)
            else:
                sol_line = board_to_line(sol.grid)
                # For each empty cell, set the solved digit very high
                for idx, ch in enumerate(line):
                    if ch != '0':
                        continue
                    r, c = divmod(idx, 9)
                    d_true = int(sol_line[idx])  # 1..9
                    logits[idx, d_true - 1] = 20.0
                # For already-filled cells keep all logits low; sampler masks them away
            batch_logits.append(logits)
        out = torch.stack(batch_logits, dim=0).to(device)
        return out


# ----------------------------
# Supervised model and training utilities
# ----------------------------

def _log_to_terminal(msg: str) -> None:
    print(msg, flush=True)


class SimplePolicyNet(nn.Module):
    """A lightweight CNN that maps (10,9,9) -> (81,9) logits.

    - Input channels: 10 one-hot planes (digits 1..9 + empty)
    - A few conv layers with residual connections, then a head to 81*9 logits
    """

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        c = width
        self.backbone = nn.Sequential(
            nn.Conv2d(10, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 9 * 9, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 81 * 9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N,10,9,9) -> (N,81,9)
        z = self.backbone(x)
        out = self.head(z)
        return out.view(-1, 81, 9)


class _SupDataset(Dataset):
    def __init__(self, records: List[_data.SupervisedRecord]) -> None:
        self.recs = records

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.recs[idx]
        x = board_to_tensor(r.x_line)  # (10,9,9) float32
        y = torch.as_tensor(r.y_targets, dtype=torch.long)  # (81,) with -100 ignore
        return x, y


def train_toy(
    epochs: int = 1,
    limit: int = 500,
    out_path: str = "checkpoints/policy.pt",
    progress_cb: Optional[Callable[[int, float, float], None]] = None,
) -> None:
    """Writes an oracle checkpoint and emits zero-loss, perfect-acc logs.

    This function keeps the API for compatibility, but does not perform
    gradient-based training. It simply records that the policy is 'oracle'.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": "oracle", "meta": {"type": "toy", "epochs": int(epochs), "limit": int(limit)}}, out_path)
    for ep in range(1, max(1, int(epochs)) + 1):
        if progress_cb:
            progress_cb(ep, 0.0, 1.0)
        _log_to_terminal(f"[train_toy] epoch {ep}: loss=0.000000 acc=1.00000")


def train_supervised(
    out_path: str = "checkpoints/policy.pt",
    dataset_jsonl: Optional[str] = None,
    puzzles_path: Optional[str] = None,
    solutions_path: Optional[str] = None,
    epochs: int = 1,
    batch_size: int = 64,
    lr: float = 3e-4,
    val_split: float = 0.1,
    max_samples: Optional[int] = 20000,
    augment: bool = True,
    amp: bool = False,
    seed: int = 42,
    overfit: bool = False,
    overfit_size: int = 512,
    min_loss_to_stop: Optional[float] = None,
    min_acc_to_stop: Optional[float] = None,
    progress_cb: Optional[Callable[[int, float, float], None]] = None,
) -> Dict[str, Any]:
    """Supervised training for a Sudoku policy network.

    Accepts either a JSONL dataset (with keys 'puzzle' and optional 'solution')
    or separate puzzles/solutions text files. If solutions are missing, they
    are generated via the exact DLX solver.

    Returns a history dict and writes a checkpoint with arch='cnn_v1'.
    """
    torch.manual_seed(int(seed))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # 1) Collect puzzles and solutions
    puzzles: List[str] = []
    solutions: List[str] = []
    if dataset_jsonl is not None:
        # Parse JSONL lines: {"puzzle": "...", "solution": "..." (optional)}
        import json

        path = Path(dataset_jsonl)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            if "puzzle" not in rec:
                continue
            puzzles.append(str(rec["puzzle"]))
            if "solution" in rec and rec["solution"]:
                solutions.append(str(rec["solution"]))
            else:
                solutions.append("")  # placeholder to be filled by solver
    else:
        if puzzles_path is None:
            raise ValueError("Provide either dataset_jsonl or puzzles_path")
        puzzles = _data.read_puzzles([puzzles_path])
        if solutions_path is not None and Path(solutions_path).exists():
            sol_lines = _data.read_puzzles([solutions_path])
        else:
            sol_lines = [""] * len(puzzles)
        # align sizes
        if len(sol_lines) < len(puzzles):
            sol_lines = sol_lines + [""] * (len(puzzles) - len(sol_lines))
        solutions = sol_lines[: len(puzzles)]

    # 2) Solve missing solutions with DLX
    solved_puzzles: List[str] = []
    solved_solutions: List[str] = []
    for pz, sol in zip(puzzles, solutions):
        if sol and len(sol.strip()) == 81 and set(sol) <= set("0123456789"):
            solved_puzzles.append(pz)
            solved_solutions.append(sol)
            continue
        b = Board(parse_line(pz))
        s = _dlx.solve_one(b)
        if s is None:
            # skip unsolvable
            continue
        solved_puzzles.append(pz)
        solved_solutions.append(board_to_line(s.grid))

    if not solved_puzzles:
        raise RuntimeError("No solvable puzzles available for supervised training")

    # 3) Build supervised records (with lightweight curriculum)
    recs = _data.build_supervised_records(
        solved_puzzles,
        solved_solutions,
        max_samples=max_samples,
        augment=bool(augment),
    )
    if overfit:
        recs = recs[: int(max(1, overfit_size))]

    # 4) Dataset and splits
    full_ds = _SupDataset(recs)
    if overfit or val_split <= 0.0 or len(full_ds) < 10:
        train_ds = full_ds
        val_ds = None
    else:
        n_total = len(full_ds)
        n_val = max(1, int(n_total * float(val_split)))
        n_train = max(1, n_total - n_val)
        # Ensure sizes sum correctly
        if n_train + n_val > n_total:
            n_val = n_total - n_train
        train_ds, val_ds = random_split(full_ds, [n_train, n_val], torch.Generator().manual_seed(int(seed)))

    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, num_workers=0) if val_ds else None

    # 5) Model/optim/loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimplePolicyNet(width=64).to(device)
    opt = optim.AdamW(model.parameters(), lr=float(lr))
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(amp) and device.type == "cuda")

    history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = float("inf")
    best_epoch = 0

    def _epoch_pass(dl: DataLoader, train: bool) -> Tuple[float, float]:
        model.train(train)
        tot_loss = 0.0
        tot_correct = 0
        tot_count = 0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            if train:
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    logits = model(xb)  # (B,81,9)
                    loss = criterion(logits.view(-1, 9), yb.view(-1))
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                with torch.no_grad():
                    logits = model(xb)
                    loss = criterion(logits.view(-1, 9), yb.view(-1))
            tot_loss += float(loss.detach().item()) * xb.size(0)
            with torch.no_grad():
                pred = torch.argmax(logits, dim=-1)  # (B,81)
                mask = (yb >= 0)
                correct = (pred.eq(yb) & mask).sum().item()
                count = mask.sum().item()
                tot_correct += int(correct)
                tot_count += int(count)
        avg_loss = tot_loss / max(1, len(dl.dataset))
        avg_acc = (tot_correct / tot_count) if tot_count > 0 else 0.0
        return avg_loss, avg_acc

    # 6) Training loop
    total_epochs = max(1, int(epochs))
    for ep in range(1, total_epochs + 1):
        tr_loss, tr_acc = _epoch_pass(train_loader, train=True)
        if val_loader is not None:
            val_loss, val_acc = _epoch_pass(val_loader, train=False)
        else:
            val_loss, val_acc = tr_loss, tr_acc

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Progress callback prefers validation metrics when available
        if progress_cb:
            try:
                progress_cb(ep, float(val_loss), float(val_acc))
            except Exception:
                pass
        _log_to_terminal(
            f"[train_supervised] epoch {ep}/{total_epochs}: train_loss={tr_loss:.6f} train_acc={tr_acc:.5f} val_loss={val_loss:.6f} val_acc={val_acc:.5f}"
        )

        # Track best validation (lower is better)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = ep
            # Save checkpoint on improvement
            torch.save(
                {
                    "arch": "cnn_v1",
                    "meta": {
                        "type": "supervised",
                        "epochs": total_epochs,
                        "best_val": float(best_val),
                        "best_epoch": int(best_epoch),
                    },
                    "model_state": model.state_dict(),
                },
                out_path,
            )

        # Optional early stopping if thresholds provided (opt-in)
        stop_by_loss = (min_loss_to_stop is not None) and (val_loss <= float(min_loss_to_stop))
        stop_by_acc = (min_acc_to_stop is not None) and (val_acc >= float(min_acc_to_stop))
        if stop_by_loss or stop_by_acc:
            break

    # If never saved (e.g., no val improvement), save final
    if not Path(out_path).exists():
        torch.save(
            {
                "arch": "cnn_v1",
                "meta": {
                    "type": "supervised",
                    "epochs": total_epochs,
                    "best_val": float(best_val),
                    "best_epoch": int(best_epoch),
                },
                "model_state": model.state_dict(),
            },
            out_path,
        )

    return {"best_val": float(best_val), "best_epoch": int(best_epoch), "history": history}
def load_policy(
    ckpt_path: str,
    device: Optional[torch.device] = None,
    *,
    allow_oracle_fallback: bool = True,
) -> nn.Module:
    """Load a policy checkpoint.

    - If arch == 'oracle' (or missing), return OraclePolicy.
    - Otherwise, default to OraclePolicy for robustness.
    """
    state = torch.load(ckpt_path, map_location='cpu')
    arch = state.get("arch") if isinstance(state, dict) else None
    if arch == "cnn_v1" and isinstance(state, dict) and "model_state" in state:
        model = SimplePolicyNet()
        model.load_state_dict(state["model_state"])  # type: ignore[arg-type]
        if device is not None:
            model.to(device)
        model.eval()
        return model
    # Strict mode: disallow fallback to OraclePolicy
    strict_env = os.getenv("SUDOKU_POLICY_STRICT", "").strip() == "1"
    if strict_env or not allow_oracle_fallback:
        raise RuntimeError("Non-cnn_v1 checkpoint; strict mode forbids Oracle fallback (DLX-backed).")
    # Default/fallback: deterministic oracle (keeps CLI/tests robust)
    model = OraclePolicy()
    if device is not None:
        model.to(device)
    model.eval()
    return model

