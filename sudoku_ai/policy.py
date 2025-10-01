from __future__ import annotations

from typing import Optional, Callable, List, Dict, Any, Tuple
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

class SudokuAttentionBlock(nn.Module):
    """Attention block that understands Sudoku constraints (row, col, box)."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 81, C)
        B, N, C = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Self-attention with constraint awareness
        attn = (q @ k.transpose(-2, -1)) / (C ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = self.proj(out)
        return x + out


class SimplePolicyNet(nn.Module):
    """Sudoku-aware policy network with attention and constraint encoding.

    - Input: (10,9,9) one-hot planes
    - Architecture: CNN backbone + transformer blocks + output head
    - Output: (81,9) logits for next move prediction
    - Masking is applied externally, NOT in forward pass
    """

    def __init__(self, width: int = 128, drop: float = 0.1, n_blocks: int = 4) -> None:
        super().__init__()
        c = width

        # Input encoding with constraint awareness
        self.input_conv = nn.Sequential(
            nn.Conv2d(10, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        # Residual conv blocks for pattern extraction
        conv_blocks = []
        for _ in range(3):
            conv_blocks.extend([
                nn.Conv2d(c, c, kernel_size=3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=drop),
            ])
        self.conv_backbone = nn.Sequential(*conv_blocks)

        # Transformer blocks for global reasoning
        self.attn_blocks = nn.ModuleList([
            SudokuAttentionBlock(c) for _ in range(n_blocks)
        ])

        # Output head
        self.head = nn.Sequential(
            nn.Linear(c, c // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop),
            nn.Linear(c // 2, 9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without masking.

        Args:
            x: (B, 10, 9, 9) one-hot encoded board

        Returns:
            logits: (B, 81, 9) raw logits for all (cell, digit) pairs
        """
        # Extract local features
        z = self.input_conv(x)  # (B, C, 9, 9)
        z = self.conv_backbone(z) + z  # Residual connection

        # Reshape to sequence: (B, C, 9, 9) -> (B, 81, C)
        z = z.flatten(start_dim=2).transpose(1, 2)  # (B, 81, C)

        # Apply attention blocks for global reasoning
        for attn_block in self.attn_blocks:
            z = attn_block(z)

        # Generate logits
        logits = self.head(z)  # (B, 81, 9)

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
                if "puzzle" in rec:
                    pz_list.append(str(rec["puzzle"]))
                    # Solution is optional - we can generate it
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
            sol_list = [""] * len(puzzles)  # Will generate solutions
    else:
        raise ValueError("Provide either dataset_jsonl or puzzles")

    if not pz_list:
        raise RuntimeError("No valid puzzles found")

    # Generate missing solutions using DLX solver (training data prep only)
    missing_count = sum(1 for s in sol_list if not s or len(s) != 81)
    if missing_count > 0:
        logger.info(f"⚙️ Generating {missing_count} solutions using DLX solver (training data prep only)...")
        print(f"⚙️ Generating {missing_count} solutions using DLX solver (training data prep)...")
        from sudoku_solvers.dlx import solve_one as dlx_solve

        solved_count = 0
        for i, (pz, sol) in enumerate(zip(pz_list, sol_list)):
            if not sol or len(sol) != 81 or '0' in sol:
                logger.debug(f"   Solving puzzle {i+1}/{len(pz_list)}...")
                b = Board(parse_line(pz))
                solved = dlx_solve(b)
                if solved:
                    sol_list[i] = board_to_line(solved.grid)
                    solved_count += 1
                    if (i + 1) % 10 == 0:
                        print(f"   Progress: {i+1}/{missing_count} puzzles solved...")
                else:
                    sol_list[i] = ""  # Mark as unsolvable
                    logger.warning(f"   ⚠️ Puzzle {i+1} is unsolvable")

        # Filter out unsolvable puzzles
        valid_pairs = [(pz, sol) for pz, sol in zip(pz_list, sol_list) if sol and '0' not in sol]
        pz_list, sol_list = [p[0] for p in valid_pairs], [p[1] for p in valid_pairs]
        logger.info(f"✅ Generated {solved_count} solutions. {len(pz_list)} solvable puzzles ready for training.")
        print(f"✅ Generated solutions. {len(pz_list)} solvable puzzles ready for training.")

    if not pz_list:
        raise RuntimeError("No solvable puzzles available for training")

    # 2) Build supervised records
    logger.info(f"📦 Building training dataset from {len(pz_list)} puzzles...")
    print(f"📦 Building training dataset from {len(pz_list)} puzzles...")
    recs = _data.build_supervised_records(
        pz_list,
        sol_list,
        max_samples=max_samples,
        augment=bool(augment),
    )

    if not recs:
        raise RuntimeError("No training samples generated")

    logger.info(f"✅ Created {len(recs)} training samples")
    print(f"✅ Created {len(recs)} training samples")

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

    # 4) Model, optimizer, loss (define device first)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use larger batch size and multiple workers for faster training
    effective_batch_size = int(batch_size) * 2  # Double batch size for efficiency

    # Windows has issues with multiprocessing in DataLoader
    import sys
    num_workers = 0 if sys.platform == "win32" else (4 if device.type == "cuda" else 2)

    train_loader = DataLoader(
        train_ds,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if num_workers > 0 else False
    ) if val_ds else None
    logger.info(f"🔧 Initializing model on device: {device}")
    print(f"🔧 Device: {device}")

    # Use improved architecture with attention
    model = SimplePolicyNet(width=128, drop=0.1, n_blocks=4).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"🧠 SudokuPolicyNet initialized with {param_count:,} parameters")
    print(f"🧠 Model: SudokuPolicyNet with Attention ({param_count:,} parameters)")

    # Use learning rate schedule for better convergence
    opt = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.1)

    # Label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    logger.info(f"⚙️ Optimizer: AdamW (lr={lr}, weight_decay=0.01, cosine schedule)")

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc_unmasked": [],
        "train_acc_masked": [],
        "val_loss": [],
        "val_acc_unmasked": [],
        "val_acc_masked": []
    }

    def _epoch_pass(dl: DataLoader, train: bool) -> Tuple[float, float, float]:
        """Run one epoch of training or validation.

        Key insight: Train on RAW logits, only mask for accuracy computation.
        CrossEntropyLoss with ignore_index=-100 handles invalid positions automatically.

        Returns:
            (avg_loss, avg_acc_unmasked, avg_acc_masked)
        """
        model.train(train)
        tot_loss = 0.0
        tot_correct_masked = 0
        tot_correct_unmasked = 0
        tot_count = 0

        for xb, yb, mb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)  # Legal move mask (B, 81, 9)

            if train:
                opt.zero_grad()
                # Get raw logits - NO MASKING during training!
                logits = model(xb)  # (B, 81, 9)

                # Compute loss on raw logits
                # CrossEntropyLoss ignores positions where yb == -100
                # This is the correct way - let the model learn naturally
                loss = criterion(logits.view(-1, 9), yb.view(-1))

                # Check for invalid loss
                if not torch.isfinite(loss):
                    logger.error(f"Non-finite loss detected: {loss.item()}")
                    continue

                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                opt.step()
            else:
                with torch.no_grad():
                    logits = model(xb)
                    loss = criterion(logits.view(-1, 9), yb.view(-1))

            tot_loss += float(loss.detach().item()) * xb.size(0)

            # Compute accuracy: Both masked and unmasked
            with torch.no_grad():
                mask = (yb >= 0)
                count = mask.sum().item()
                tot_count += int(count)

                # Unmasked accuracy: True measure of learning
                pred_unmasked = torch.argmax(logits, dim=-1)  # (B,81)
                correct_unmasked = (pred_unmasked.eq(yb) & mask).sum().item()
                tot_correct_unmasked += int(correct_unmasked)

                # Masked accuracy: What we use at inference
                masked_logits = logits.clone()
                masked_logits[mb == 0] = -1e9
                pred_masked = torch.argmax(masked_logits, dim=-1)  # (B,81)
                correct_masked = (pred_masked.eq(yb) & mask).sum().item()
                tot_correct_masked += int(correct_masked)

        avg_loss = tot_loss / max(1, len(dl.dataset))
        avg_acc_unmasked = (tot_correct_unmasked / tot_count) if tot_count > 0 else 0.0
        avg_acc_masked = (tot_correct_masked / tot_count) if tot_count > 0 else 0.0
        return avg_loss, avg_acc_unmasked, avg_acc_masked

    # 5) Pre-training validation: Check a sample batch
    logger.info("🔍 Running pre-training validation on sample batch...")
    sample_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    xb, yb, mb = next(iter(sample_loader))
    xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)

    with torch.no_grad():
        logits = model(xb)
        logger.info(f"   Sample logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
        logger.info(f"   Sample targets: {(yb >= 0).sum().item()} valid positions")
        logger.info(f"   Legal moves per sample: {mb.sum(dim=[1,2]).cpu().numpy()}")

        # Test loss on RAW logits (correct approach)
        test_loss = criterion(logits.view(-1, 9), yb.view(-1))
        logger.info(f"   Sample loss (raw logits): {test_loss.item():.4f}")

        if not torch.isfinite(test_loss):
            logger.error("❌ Pre-training check failed: non-finite loss!")
            raise RuntimeError("Model initialization produced invalid loss")

        # Also check masked accuracy
        masked_logits = logits.clone()
        masked_logits[mb == 0] = -1e9
        pred = torch.argmax(masked_logits, dim=-1)
        acc = (pred.eq(yb) & (yb >= 0)).sum().item() / (yb >= 0).sum().item()
        logger.info(f"   Sample accuracy (masked): {acc:.3f}")

    logger.info("✅ Pre-training validation passed")

    # 6) Training loop
    total_epochs = max(1, int(epochs))
    logger.info(f"🎯 Starting training for {total_epochs} epochs...")
    print(f"\n{'='*60}")
    print(f"🎯 Training: {total_epochs} epochs, {len(train_ds)} samples")
    print(f"{'='*60}\n")

    # Early stopping parameters
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

    for ep in range(1, total_epochs + 1):
        logger.info(f"Epoch {ep}/{total_epochs}: Training...")
        tr_loss, tr_acc_unmask, tr_acc_mask = _epoch_pass(train_loader, train=True)

        # Step learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        if val_loader is not None:
            logger.info(f"Epoch {ep}/{total_epochs}: Validating...")
            val_loss, val_acc_unmask, val_acc_mask = _epoch_pass(val_loader, train=False)
        else:
            val_loss, val_acc_unmask, val_acc_mask = tr_loss, tr_acc_unmask, tr_acc_mask

        history["train_loss"].append(tr_loss)
        history["train_acc_unmasked"].append(tr_acc_unmask)
        history["train_acc_masked"].append(tr_acc_mask)
        history["val_loss"].append(val_loss)
        history["val_acc_unmasked"].append(val_acc_unmask)
        history["val_acc_masked"].append(val_acc_mask)

        # Early stopping check - use UNMASKED accuracy as true measure
        if val_acc_unmask > best_val_acc:
            best_val_acc = val_acc_unmask
            best_val_loss = val_loss
            patience_counter = 0
            # Save best checkpoint
            best_ckpt_path = out_path.replace('.pt', '_best.pt')
            torch.save(
                {
                    "arch": "sudoku_policy_net",
                    "model_state": model.state_dict(),
                    "meta": {
                        "epoch": ep,
                        "val_loss": val_loss,
                        "val_acc_unmasked": val_acc_unmask,
                        "val_acc_masked": val_acc_mask,
                    },
                },
                best_ckpt_path,
            )
            logger.info(f"💾 Saved best model (unmasked_acc={val_acc_unmask:.3f}, masked_acc={val_acc_mask:.3f}) to: {best_ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"⚠️ Early stopping triggered after {ep} epochs (no improvement for {patience} epochs)")
                print(f"\n⚠️ Early stopping: No improvement for {patience} epochs")
                break

        if progress_cb:
            try:
                progress_cb(ep, float(val_loss), float(val_acc_unmask))
            except Exception:
                pass

        # Log to both logger and console with unmasked and masked accuracy
        log_msg = (
            f"Epoch {ep}/{total_epochs}: "
            f"train_loss={tr_loss:.4f} "
            f"train_acc={tr_acc_unmask:.3f}→{tr_acc_mask:.3f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc_unmask:.3f}→{val_acc_mask:.3f} | "
            f"lr={current_lr:.2e}"
        )
        if val_acc_unmask == best_val_acc:
            log_msg += " 🌟"
        logger.info(log_msg)
        print(f"📊 {log_msg}", flush=True)

    # 7) Save checkpoint
    logger.info(f"💾 Saving model checkpoint to: {out_path}")
    print(f"\n💾 Saving model to: {out_path}")

    torch.save(
        {
            "arch": "simple_policy_net",
            "model_state": model.state_dict(),
            "meta": {
                "epochs": total_epochs,
                "val_loss": history["val_loss"][-1] if history["val_loss"] else 0.0,
                "val_acc_unmasked": history["val_acc_unmasked"][-1] if history["val_acc_unmasked"] else 0.0,
                "val_acc_masked": history["val_acc_masked"][-1] if history["val_acc_masked"] else 0.0,
            },
        },
        out_path,
    )

    logger.info("✅ Training complete!")
    print(f"\n✅ Training complete!")
    print(f"📊 Final metrics:")
    print(f"   Unmasked accuracy: {history['val_acc_unmasked'][-1]:.3f} (true learning measure)")
    print(f"   Masked accuracy: {history['val_acc_masked'][-1]:.3f} (with constraints)")
    print(f"   Loss: {history['val_loss'][-1]:.4f}")
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

    # Load with new architecture parameters
    model = SimplePolicyNet(width=128, drop=0.1, n_blocks=4)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    return model
