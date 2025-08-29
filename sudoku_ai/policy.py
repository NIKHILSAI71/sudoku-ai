from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Callable, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sudoku_engine import parse_line, Board, board_to_line
from sudoku_solvers import backtracking
from .data import build_supervised_records, make_partial_samples, random_augment


def board_to_tensor(line: str) -> torch.Tensor:
    # line: 81 chars of 0..9
    x = torch.zeros(10, 9, 9, dtype=torch.float32)
    for i, ch in enumerate(line.strip()):
        r, c = divmod(i, 9)
        d = int(ch)
        ch_idx = 9 if d == 0 else d - 1
        x[ch_idx, r, c] = 1.0
    return x  # (10,9,9)


def legal_mask_from_line(line: str) -> torch.Tensor:
    """Build a (81,9) mask of legal digits for each cell from a board line.

    1 for allowed digit, 0 otherwise. Filled cells will be all zeros.
    """
    b = Board(parse_line(line))
    masks = b.candidates_mask()  # (9,9) int bitmasks
    out = torch.zeros(81, 9, dtype=torch.float32)
    for idx in range(81):
        r, c = divmod(idx, 9)
        m = int(masks[r, c])
        if m == 0:
            continue
        for d in range(1, 10):
            if m & (1 << (d - 1)):
                out[idx, d - 1] = 1.0
    return out


class SmallPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(10, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.head = nn.Conv2d(64, 9, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,10,9,9) -> logits over 9 digits per cell -> (N,81,9)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.head(x)  # (N,9,9,9digits) if we permute
        x = x.permute(0, 2, 3, 1).contiguous()  # (N,9,9,9)
        x = x.view(x.size(0), 81, 9)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + x)
        return out


class ResNetPolicy(nn.Module):
    def __init__(self, channels: int = 256, depth: int = 20) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(10, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.Conv2d(channels, 9, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)  # (N,9,9,9digits before permute)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N,9,9,9)
        x = x.view(x.size(0), 81, 9)
        return x


@dataclass
class TrainConfig:
    epochs: int = 1
    limit: int = 500
    lr: float = 1e-3


def train_toy(
    epochs: int = 1,
    limit: int = 500,
    out_path: str = "checkpoints/policy.pt",
    progress_cb: Optional[Callable[[int, float, float], None]] = None,
) -> None:
    # Simple synthetic dataset: use identity targets (prefer lower digits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallPolicy().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # create random boards with mask of legal actions (all digits allowed on empties)
    xs = torch.zeros(limit, 10, 9, 9)
    targets = torch.zeros(limit, 81, dtype=torch.long)
    for i in range(limit):
        # empty board with few givens
        board = [0] * 81
        for k in range(20):
            idx = torch.randint(0, 81, (1,)).item()
            board[idx] = torch.randint(1, 10, (1,)).item()
        line = "".join(str(d) for d in board)
        xs[i] = board_to_tensor(line)
        # dummy target: digit 1 everywhere (class 0)
        targets[i] = 0

    batch_size = 32
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        total_acc = 0.0
        total_items = 0
        for i in range(0, limit, batch_size):
            xb = xs[i : i + batch_size].to(device)
            yb = targets[i : i + batch_size].to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, 9), yb.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                preds = logits.view(-1, 9).argmax(dim=-1)
                acc = (preds == yb.view(-1)).float().mean().item()
            bsz = xb.size(0) * 81
            total_loss += loss.item() * bsz
            total_acc += acc * bsz
            total_items += bsz
        avg_loss = total_loss / max(1, total_items)
        avg_acc = total_acc / max(1, total_items)
        if progress_cb is not None:
            progress_cb(ep, avg_loss, avg_acc)

    torch.save({"model": model.state_dict()}, out_path)


class _SupDataset(Dataset):
    def __init__(self, items: List[Tuple[str, torch.Tensor]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        line, tgt = self.items[idx]
        x = board_to_tensor(line)
        m = legal_mask_from_line(line)
        return x, tgt, m


def _prepare_supervised_items(
    puzzles: List[str],
    solutions: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    augment: bool = True,
) -> List[Tuple[str, torch.Tensor]]:
    # If solutions missing, solve puzzles first
    if solutions is None:
        sols: List[str] = []
        for ln in puzzles:
            b = backtracking.solve_one(Board(parse_line(ln)))
            if b is None:
                continue
            sols.append(board_to_line(b.grid))
        solutions = sols
    # Align lengths
    n = min(len(puzzles), len(solutions))
    puzzles = puzzles[:n]
    solutions = solutions[:n]
    recs = build_supervised_records(puzzles, solutions, max_samples=max_samples, augment=augment)
    items: List[Tuple[str, torch.Tensor]] = []
    for r in recs:
        items.append((r.x_line, torch.from_numpy(r.y_targets.astype('int64'))))
    return items


def train_supervised(
    out_path: str = "checkpoints/policy.pt",
    dataset_jsonl: Optional[str] = None,
    puzzles_path: Optional[str] = None,
    solutions_path: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 3e-4,
    val_split: float = 0.1,
    max_samples: Optional[int] = 20000,
    augment: bool = True,
    amp: bool = False,
    seed: int = 42,
    progress_cb: Optional[Callable[[int, float, float], None]] = None,
) -> Dict[str, Any]:
    import json
    import math
    from pathlib import Path
    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)

    puzzles: List[str] = []
    solutions: Optional[List[str]] = None
    if dataset_jsonl is not None:
        path = Path(dataset_jsonl)
        if path.exists():
            puzzles = []
            solutions = []
            for ln in path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                if "puzzle" in rec:
                    puzzles.append(str(rec["puzzle"]))
                    solutions.append(str(rec.get("solution", "")))
            # If solutions missing (empty strings), treat as None
            if solutions and all(s == "" for s in solutions):
                solutions = None
    else:
        if puzzles_path is not None:
            puzzles = [ln.strip() for ln in Path(puzzles_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        if solutions_path is not None and Path(solutions_path).exists():
            sols = [ln.strip() for ln in Path(solutions_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
            solutions = sols

    if not puzzles:
        raise ValueError("No training puzzles found. Provide --dataset or --puzzles")

    # Build grouped samples per puzzle to avoid leakage between train/val
    groups: List[List[Tuple[str, torch.Tensor]]] = []
    total_added = 0
    for pz, sol in zip(puzzles, solutions or []):
        if augment:
            pz, sol = random_augment(pz, sol, enable=True)
        samples = make_partial_samples(pz, sol, steps=40)
        group_items: List[Tuple[str, torch.Tensor]] = []
        for line, tgt_np in samples:
            t = torch.from_numpy(tgt_np.astype('int64'))
            group_items.append((line, t))
            total_added += 1
            if max_samples is not None and total_added >= max_samples:
                break
        if group_items:
            groups.append(group_items)
        if max_samples is not None and total_added >= max_samples:
            break
    if not groups:
        raise ValueError("No supervised items prepared. Check dataset or solver availability.")
    # Shuffle groups deterministically and split
    gperm = torch.randperm(len(groups), generator=g).tolist()
    groups = [groups[i] for i in gperm]
    val_gn = max(1, int(len(groups) * val_split)) if len(groups) > 1 and val_split > 0 else 0
    val_groups = groups[:val_gn]
    train_groups = groups[val_gn:]
    # Flatten
    val_items = [it for grp in val_groups for it in grp]
    train_items = [it for grp in train_groups for it in grp]

    train_ds = _SupDataset(train_items)
    val_ds = _SupDataset(val_items) if val_items else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetPolicy().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    # AMP: prefer torch.amp API with fallback for older torch
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
        autocast_cm = lambda: torch.amp.autocast('cuda', enabled=amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        autocast_cm = lambda: torch.cuda.amp.autocast(enabled=amp)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    def _loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=g)

    best_val = float('inf')
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_cnt = 0
        train_correct = 0.0
        train_total_labels = 0.0
        for xb, yb, mb in _loader(train_ds, shuffle=True):
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            with autocast_cm():
                logits = model(xb)
                # Ensure ground-truth digit is legal (in case of degenerate masks)
                with torch.no_grad():
                    y_flat = yb.view(-1)
                    m_flat = mb.view(-1, 9)
                    valid_pos = (y_flat >= 0)
                    if valid_pos.any():
                        idxs = torch.nonzero(valid_pos, as_tuple=False).squeeze(1)
                        m_flat[idxs, y_flat[valid_pos]] = 1.0
                masked_logits = logits.masked_fill(mb == 0, -1e9)
                loss = loss_fn(masked_logits.view(-1, 9), yb.view(-1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # Gradient clipping for stability
            if amp:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            # Count only non-ignored labels for averaging
            with torch.no_grad():
                valid = (yb != -100)
                n_valid = int(valid.sum().item())
            total_loss += float(loss.item()) * n_valid
            total_cnt += n_valid
            # Train accuracy on non-masked labels
            with torch.no_grad():
                preds = masked_logits.argmax(dim=-1)  # (B,81)
                mask = (yb != -100)
                train_correct += (preds[mask] == yb[mask]).float().sum().item()
                train_total_labels += float(mask.sum().item())
        sched.step()
        train_loss = total_loss / max(1, total_cnt)
        train_acc = (train_correct / train_total_labels) if train_total_labels > 0 else 0.0

        # Validation
        val_loss = 0.0
        val_cnt = 0
        val_correct = 0.0
        val_total_labels = 0.0
        if val_ds is not None and len(val_ds) > 0:
            model.eval()
            with torch.no_grad():
                for xb, yb, mb in _loader(val_ds, shuffle=False):
                    xb = xb.to(device)
                    yb = yb.to(device)
                    mb = mb.to(device)
                    with autocast_cm():
                        logits = model(xb)
                        masked_logits = logits.masked_fill(mb == 0, -1e9)
                    loss = loss_fn(masked_logits.view(-1, 9), yb.view(-1))
                    valid = (yb != -100)
                    n_valid = int(valid.sum().item())
                    val_loss += float(loss.item()) * n_valid
                    val_cnt += n_valid
                    preds = masked_logits.argmax(dim=-1)  # (N,81)
                    mask = valid
                    val_correct += (preds[mask] == yb[mask]).float().sum().item()
                    val_total_labels += float(mask.sum().item())
        val_acc = (val_correct / val_total_labels) if val_total_labels > 0 else 0.0
        val_loss = val_loss / max(1, val_cnt)

        history["train_loss"].append(train_loss)
        if val_cnt > 0:
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

        if progress_cb is not None:
            progress_cb(ep, val_loss if val_cnt > 0 else train_loss, float(val_acc) if val_cnt > 0 else float(train_acc))

        # Save best
        cur_val = val_loss if val_cnt > 0 else train_loss
        if cur_val < best_val:
            best_val = cur_val
            torch.save({
                "arch": "resnet",
                "model": model.state_dict(),
                "meta": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "val_split": val_split,
                },
            }, out_path)

    return {"best_val": best_val, "history": history}


def load_policy(ckpt_path: str, device: Optional[torch.device] = None) -> nn.Module:
    """Load a policy checkpoint (arch-aware) and set to eval mode."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_path, map_location=device)
    arch = None
    if isinstance(state, dict):
        arch = state.get("arch")
    if arch == "resnet":
        model = ResNetPolicy().to(device)
        state = state.get("model", state)
    else:
        model = SmallPolicy().to(device)
    # support both raw state_dict and wrapped dict
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model
