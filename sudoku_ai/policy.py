from __future__ import annotations

from typing import Optional, Callable, List, Dict, Any, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from sudoku_engine import parse_line, Board, board_to_line
from . import data as _data


# ----------------------------
# Public helpers
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
# Neural Network Policy
# ----------------------------

class SimplePolicyNet(nn.Module):
    """A lightweight CNN that maps (10,9,9) -> (81,9) logits.

    - Input channels: 10 one-hot planes (digits 1..9 + empty)
    - A few conv layers with residual connections, then a head to 81*9 logits
    """

    def __init__(self, width: int = 64, drop: float = 0.2) -> None:
        super().__init__()
        c = width
        self.backbone = nn.Sequential(
            nn.Conv2d(10, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 9 * 9, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop),
            nn.Linear(512, 81 * 9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N,10,9,9) -> (N,81,9)
        z = self.backbone(x)
        out = self.head(z)
        return out.view(-1, 81, 9)


# ----------------------------
# Training Dataset
# ----------------------------

class _SupDataset(Dataset):
    def __init__(self, records: List[_data.SupervisedRecord]) -> None:
        self.recs = records

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = self.recs[idx]
        x = board_to_tensor(r.x_line)  # (10,9,9) float32
        y = torch.as_tensor(r.y_targets, dtype=torch.long)  # (81,) with -100 ignore
        m = legal_mask_from_line(r.x_line)  # (81,9) float32 in {0,1}
        return x, y, m


# ----------------------------
# Training Function
# ----------------------------

def train_supervised(
    out_path: str = "checkpoints/policy.pt",
    dataset_jsonl: Optional[str] = None,
    puzzles: Optional[List[str]] = None,
    solutions: Optional[List[str]] = None,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_split: float = 0.1,
    max_samples: Optional[int] = None,
    augment: bool = True,
    seed: int = 42,
    progress_cb: Optional[Callable[[int, float, float], None]] = None,
) -> Dict[str, Any]:
    """Simplified supervised training for a Sudoku policy network.

    Args:
        out_path: Path to save checkpoint
        dataset_jsonl: Path to JSONL dataset with 'puzzle' and 'solution' keys
        puzzles: List of puzzle strings (alternative to dataset_jsonl)
        solutions: List of solution strings (must match puzzles)
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        val_split: Validation split fraction
        max_samples: Maximum number of samples to use
        augment: Whether to augment data
        seed: Random seed
        progress_cb: Optional callback (epoch, loss, acc) -> None

    Returns:
        Dict with training history
    """
    torch.manual_seed(int(seed))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # 1) Load puzzles and solutions
    pz_list: List[str] = []
    sol_list: List[str] = []

    if dataset_jsonl is not None:
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
                if "puzzle" in rec and "solution" in rec:
                    pz_list.append(str(rec["puzzle"]))
                    sol_list.append(str(rec["solution"]))
            except Exception:
                continue
    elif puzzles is not None and solutions is not None:
        if len(puzzles) != len(solutions):
            raise ValueError("Puzzles and solutions must have same length")
        pz_list = puzzles
        sol_list = solutions
    else:
        raise ValueError("Provide either dataset_jsonl or (puzzles, solutions)")

    if not pz_list:
        raise RuntimeError("No valid puzzle-solution pairs found")

    # 2) Build supervised records
    recs = _data.build_supervised_records(
        pz_list,
        sol_list,
        max_samples=max_samples,
        augment=bool(augment),
    )

    if not recs:
        raise RuntimeError("No training samples generated")

    # 3) Dataset and splits
    full_ds = _SupDataset(recs)
    if val_split <= 0.0 or len(full_ds) < 10:
        train_ds = full_ds
        val_ds = None
    else:
        n_total = len(full_ds)
        n_val = max(1, int(n_total * float(val_split)))
        n_train = max(1, n_total - n_val)
        if n_train + n_val > n_total:
            n_val = n_total - n_train
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val], torch.Generator().manual_seed(int(seed))
        )

    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False) if val_ds else None

    # 4) Model, optimizer, loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimplePolicyNet(width=64, drop=0.2).to(device)
    opt = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    def _epoch_pass(dl: DataLoader, train: bool) -> Tuple[float, float]:
        model.train(train)
        tot_loss = 0.0
        tot_correct = 0
        tot_count = 0
        for xb, yb, mb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            if train:
                opt.zero_grad()
                logits = model(xb)  # (B,81,9)
                loss = criterion(logits.view(-1, 9), yb.view(-1))
                loss.backward()
                opt.step()
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

    # 5) Training loop
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

        if progress_cb:
            try:
                progress_cb(ep, float(val_loss), float(val_acc))
            except Exception:
                pass

        print(
            f"[train] epoch {ep}/{total_epochs}: "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}",
            flush=True
        )

    # 6) Save checkpoint
    torch.save(
        {
            "arch": "simple_policy_net",
            "model_state": model.state_dict(),
            "meta": {
                "epochs": total_epochs,
                "val_loss": history["val_loss"][-1] if history["val_loss"] else 0.0,
                "val_acc": history["val_acc"][-1] if history["val_acc"] else 0.0,
            },
        },
        out_path,
    )

    return {"history": history}


def load_policy(
    ckpt_path: str,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load a policy checkpoint.

    Args:
        ckpt_path: Path to checkpoint file
        device: Target device (cuda/cpu)

    Returns:
        Loaded SimplePolicyNet model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(state, dict) or "model_state" not in state:
        raise RuntimeError(f"Invalid checkpoint format: {ckpt_path}")

    model = SimplePolicyNet(width=64, drop=0.2)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    return model
