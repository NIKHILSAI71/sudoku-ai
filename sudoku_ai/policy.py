from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from sudoku_engine import parse_line, Board, board_to_line
from . import data as _data

logger = logging.getLogger(__name__)


# ----------------------------
# Encoding/Decoding
# ----------------------------

def board_to_tensor(line: str) -> torch.Tensor:
    """Encode board as (10, 9, 9) one-hot tensor."""
    x = torch.zeros(10, 9, 9, dtype=torch.float32)
    for i, ch in enumerate(line.strip()):
        r, c = divmod(i, 9)
        d = int(ch)
        ch_idx = 9 if d == 0 else d - 1
        x[ch_idx, r, c] = 1.0
    return x


def legal_mask_from_line(line: str) -> torch.Tensor:
    """Return (81, 9) binary mask of legal moves."""
    b = Board(parse_line(line))
    masks = b.candidates_mask()
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
# Lightweight CNN Policy
# ----------------------------

class SudokuPolicyNet(nn.Module):
    """Lightweight CNN for Sudoku move prediction.

    Simple, fast, and effective architecture:
    - Input: (10, 9, 9) one-hot encoding
    - CNN backbone for pattern extraction
    - Direct output to (81, 9) logits
    """

    def __init__(self, channels: int = 256):
        super().__init__()

        # Convolutional backbone - extract Sudoku patterns
        self.conv1 = nn.Conv2d(10, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(channels)

        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(channels)

        # Output head - predict digit for each cell
        self.head = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 9, kernel_size=1),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: board -> logits.

        Args:
            x: (B, 10, 9, 9) one-hot board encoding

        Returns:
            logits: (B, 81, 9) unnormalized scores for each (cell, digit)
        """
        # CNN feature extraction with residual connections
        z = self.relu(self.bn1(self.conv1(x)))  # (B, C, 9, 9)

        identity = z
        z = self.relu(self.bn2(self.conv2(z)))
        z = z + identity  # Residual

        identity = z
        z = self.relu(self.bn3(self.conv3(z)))
        z = z + identity  # Residual

        identity = z
        z = self.relu(self.bn4(self.conv4(z)))
        z = z + identity  # Residual

        # Generate logits for each cell
        logits = self.head(z)  # (B, 9, 9, 9)

        # Reshape to (B, 81, 9)
        B = logits.size(0)
        logits = logits.permute(0, 2, 3, 1).reshape(B, 81, 9)

        return logits


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
        x = board_to_tensor(r.x_line)
        y = torch.as_tensor(r.y_targets, dtype=torch.long)
        m = legal_mask_from_line(r.x_line)
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
    progress_cb: Optional[Any] = None,
) -> Dict[str, Any]:
    """Train Sudoku policy network with supervised learning.

    Key improvements:
    - Simpler architecture for faster training
    - Proper loss masking without breaking gradients
    - Better learning rate schedule
    """
    torch.manual_seed(seed)
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
                if "puzzle" in rec:
                    pz_list.append(str(rec["puzzle"]))
                    sol_list.append(str(rec.get("solution", "")))
            except Exception:
                continue
    elif puzzles is not None:
        pz_list = puzzles
        if solutions is not None:
            if len(puzzles) != len(solutions):
                raise ValueError("Puzzles and solutions must have same length")
            sol_list = solutions
        else:
            sol_list = [""] * len(puzzles)
    else:
        raise ValueError("Provide either dataset_jsonl or puzzles")

    if not pz_list:
        raise RuntimeError("No valid puzzles found")

    # Generate missing solutions
    missing_count = sum(1 for s in sol_list if not s or len(s) != 81)
    if missing_count > 0:
        logger.info(f"⚙️ Generating {missing_count} solutions...")
        print(f"⚙️ Generating {missing_count} solutions...")
        from sudoku_solvers.dlx import solve_one as dlx_solve

        for i, (pz, sol) in enumerate(zip(pz_list, sol_list)):
            if not sol or len(sol) != 81 or '0' in sol:
                b = Board(parse_line(pz))
                solved = dlx_solve(b)
                if solved:
                    sol_list[i] = board_to_line(solved.grid)
                else:
                    sol_list[i] = ""

        valid_pairs = [(pz, sol) for pz, sol in zip(pz_list, sol_list) if sol and '0' not in sol]
        pz_list, sol_list = [p[0] for p in valid_pairs], [p[1] for p in valid_pairs]
        logger.info(f"✅ {len(pz_list)} solvable puzzles ready")
        print(f"✅ {len(pz_list)} solvable puzzles ready")

    if not pz_list:
        raise RuntimeError("No solvable puzzles available")

    # 2) Build training records
    logger.info(f"📦 Building dataset from {len(pz_list)} puzzles...")
    print(f"📦 Building dataset from {len(pz_list)} puzzles...")
    recs = _data.build_supervised_records(pz_list, sol_list, max_samples=max_samples, augment=augment)

    if not recs:
        raise RuntimeError("No training samples generated")

    logger.info(f"✅ Created {len(recs)} training samples")
    print(f"✅ Created {len(recs)} training samples")

    # 3) Dataset splits
    full_ds = _SupDataset(recs)
    if val_split <= 0.0 or len(full_ds) < 10:
        train_ds = full_ds
        val_ds = None
    else:
        n_val = max(1, int(len(full_ds) * val_split))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(full_ds, [n_train, n_val], torch.Generator().manual_seed(seed))

    # 4) Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import sys
    num_workers = 0 if sys.platform == "win32" else 4

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda"
    ) if val_ds else None

    logger.info(f"🔧 Device: {device}")
    print(f"🔧 Device: {device}")

    # Initialize model
    model = SudokuPolicyNet(channels=256).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"🧠 SudokuPolicyNet ({param_count:,} parameters)")
    print(f"🧠 SudokuPolicyNet ({param_count:,} parameters)")

    # Optimizer with weight decay
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.1)

    # Loss function - ignore_index handles invalid positions
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    def _epoch_pass(dl: DataLoader, train: bool) -> Tuple[float, float]:
        """Run one epoch - returns (loss, accuracy)."""
        model.train(train)
        tot_loss = 0.0
        tot_correct = 0
        tot_count = 0

        for xb, yb, mb in dl:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)

            if train:
                opt.zero_grad()

            # Forward pass
            logits = model(xb)  # (B, 81, 9)

            # Apply legal move mask to logits (in-place for memory efficiency)
            logits = logits.masked_fill(mb == 0, -1e9)

            # Compute loss
            loss = criterion(logits.view(-1, 9), yb.view(-1))

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            # Track metrics
            tot_loss += loss.item() * xb.size(0)

            with torch.no_grad():
                pred = torch.argmax(logits, dim=-1)
                mask = (yb >= 0)
                tot_correct += (pred.eq(yb) & mask).sum().item()
                tot_count += mask.sum().item()

        avg_loss = tot_loss / len(dl.dataset)
        avg_acc = tot_correct / tot_count if tot_count > 0 else 0.0
        return avg_loss, avg_acc

    # 5) Sanity check - verify training isn't trivial
    logger.info("🔍 Running sanity check on sample batch...")
    sample_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    xb, yb, mb = next(iter(sample_loader))
    xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)

    # Check that masks have multiple legal moves (not trivial)
    num_legal_per_sample = mb.sum(dim=[1, 2]).cpu().numpy()
    avg_legal_moves = num_legal_per_sample.mean()
    logger.info(f"   Average legal moves per sample: {avg_legal_moves:.1f}")

    if avg_legal_moves < 5:
        logger.warning("⚠️ Very few legal moves per sample - training may be too easy!")

    # Check untrained model accuracy
    with torch.no_grad():
        logits = model(xb)
        logits_masked = logits.masked_fill(mb == 0, -1e9)
        pred = torch.argmax(logits_masked, dim=-1)
        mask = (yb >= 0)
        untrained_acc = (pred.eq(yb) & mask).sum().item() / mask.sum().item()
        logger.info(f"   Untrained model accuracy: {untrained_acc:.3f} (should be ~0.11 for 9-class random)")

    if untrained_acc > 0.5:
        logger.error("❌ Untrained model has >50% accuracy - task is too easy or data is leaking!")
        raise RuntimeError("Training data is too easy - model doesn't need to learn")

    logger.info("✅ Sanity check passed")

    # 6) Training loop
    logger.info(f"🎯 Training {epochs} epochs...")
    print(f"\n{'='*60}")
    print(f"🎯 Training: {epochs} epochs, {len(train_ds)} samples")
    print(f"{'='*60}\n")

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = _epoch_pass(train_loader, train=True)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        if val_loader is not None:
            val_loss, val_acc = _epoch_pass(val_loader, train=False)
        else:
            val_loss, val_acc = tr_loss, tr_acc

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_ckpt = out_path.replace('.pt', '_best.pt')
            torch.save({
                "model_state": model.state_dict(),
                "meta": {"epoch": ep, "val_loss": val_loss, "val_acc": val_acc}
            }, best_ckpt)
            logger.info(f"💾 Best model saved (acc={val_acc:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"⚠️ Early stopping at epoch {ep}")
                print(f"\n⚠️ Early stopping at epoch {ep}")
                break

        log_msg = (
            f"Epoch {ep}/{epochs}: "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
            f"lr={current_lr:.2e}"
        )
        logger.info(log_msg)
        print(f"📊 {log_msg}")

    # 6) Save final checkpoint
    logger.info(f"💾 Saving final model to {out_path}")
    print(f"\n💾 Saving model to {out_path}")

    torch.save({
        "model_state": model.state_dict(),
        "meta": {"epochs": epochs, "val_loss": history["val_loss"][-1], "val_acc": history["val_acc"][-1]}
    }, out_path)

    logger.info("✅ Training complete!")
    print(f"\n✅ Training complete!")
    print(f"📊 Final accuracy: {history['val_acc'][-1]:.3f}")
    print(f"📊 Final loss: {history['val_loss'][-1]:.4f}")

    return {"history": history}


def load_policy(ckpt_path: str, device: Optional[torch.device] = None) -> nn.Module:
    """Load trained policy from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(state, dict) or "model_state" not in state:
        raise RuntimeError(f"Invalid checkpoint: {ckpt_path}")

    model = SudokuPolicyNet(channels=256)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    return model
