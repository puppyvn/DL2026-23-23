"""
src/train.py — Sequential training of 5 width-scaled ResNet variants on CIFAR-10
==================================================================================

Usage:
    python -m src.train                          # use default config
    python -m src.train --config config_train.yaml
    python -m src.train --config config_train.yaml --device cuda

What this script does
---------------------
1. Load config_train.yaml
2. Build a torchvision DataLoader from data/data.py (original CIFAR-10, no shift)
3. Call Model.resize_model() to produce 5 scaled variants (0.5× … 2×)
4. Train each variant sequentially:
   - SGD/Adam/AdamW optimizer
   - Cosine / StepLR / no-op scheduler with optional linear warmup
   - Save best checkpoint per scale to checkpoints/
5. Print per-epoch train loss + val accuracy for every model
6. Save a summary CSV to results/train/summary.csv
"""

import argparse
import csv
import math
import os
import random
import time
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure project root is in sys.path so imports work regardless of how script is executed
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

import yaml

from data.data import Data_loader
from src.model import Model

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class _TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def build_dataloaders(cfg: dict, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    """
    Use Data_loader from data/data.py to load the original (unshifted) CIFAR-10
    training split, then wrap it in train/val DataLoaders.
    """
    aug_cfg  = cfg["augmentation"]
    data_cfg = cfg["data"]

    # ── Transforms ──────────────────────────────────────────────────────
    resize = aug_cfg.get("resize", 224)
    mean   = aug_cfg["mean"]
    std    = aug_cfg["std"]

    train_transforms = T.Compose([
        T.Resize((resize, resize)),
        *(  [T.RandomCrop(resize, padding=aug_cfg.get("crop_padding", 4))]
            if aug_cfg.get("random_crop", True) else []  ),
        *(  [T.RandomHorizontalFlip()]
            if aug_cfg.get("random_horizontal_flip", True) else []  ),
        T.ToTensor(),
        *(  [T.Normalize(mean=mean, std=std)]
            if aug_cfg.get("normalize", True) else []  ),
    ])

    val_transforms = T.Compose([
        T.Resize((resize, resize)),
        T.ToTensor(),
        *(  [T.Normalize(mean=mean, std=std)]
            if aug_cfg.get("normalize", True) else []  ),
    ])

    # ── Load raw dataset via Data_loader ────────────────────────────────
    dl = Data_loader()
    dl.load_data()           # loads self.data (PIL, no transform)
    full_dataset = dl.data   # torchvision CIFAR10 object

    # ── Train / Val split ────────────────────────────────────────────────
    n_total  = len(full_dataset)
    n_train  = int(n_total * data_cfg.get("train_split", 0.9))
    n_val    = n_total - n_train

    train_subset, val_subset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
    )

    train_ds = _TransformedSubset(train_subset, train_transforms)
    val_ds   = _TransformedSubset(val_subset,   val_transforms)

    pin = data_cfg.get("pin_memory", True) and device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["train"]["batch_size"],
        shuffle     = True,
        num_workers = data_cfg.get("num_workers", 4),
        pin_memory  = pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["train"]["batch_size"] * 2,
        shuffle     = False,
        num_workers = data_cfg.get("num_workers", 4),
        pin_memory  = pin,
    )

    print(f"[data] train={n_train:,}  val={n_val:,}  "
          f"resize={resize}px  batch={cfg['train']['batch_size']}")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Optimizer & Scheduler builders
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: dict) -> optim.Optimizer:
    tcfg = cfg["train"]
    name = tcfg["optimizer"].lower()
    lr   = tcfg["lr"]
    wd   = tcfg.get("weight_decay", 5e-4)

    if name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=tcfg.get("momentum", 0.9),
            weight_decay=wd,
            nesterov=tcfg.get("nesterov", True),
        )
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer '{name}'. Choose sgd | adam | adamw.")


def build_scheduler(optimizer: optim.Optimizer, cfg: dict):
    tcfg     = cfg["train"]
    name     = tcfg.get("scheduler", "cosine").lower()
    epochs   = tcfg["epochs"]
    warmup_e = tcfg.get("warmup_epochs", 0)

    # Base scheduler
    if name == "cosine":
        base_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = max(1, epochs - warmup_e),
            eta_min = tcfg.get("lr_min", 1e-5),
        )
    elif name == "step":
        base_sched = optim.lr_scheduler.StepLR(
            optimizer,
            step_size = tcfg.get("lr_step_size", 20),
            gamma     = tcfg.get("lr_step_gamma", 0.1),
        )
    else:  # "none"
        base_sched = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1.0)

    if warmup_e <= 0:
        return base_sched

    # Linear warmup then hand off to base scheduler
    warmup_start = tcfg.get("warmup_lr_start", 1e-3)
    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = warmup_start / tcfg["lr"],
        end_factor   = 1.0,
        total_iters  = warmup_e,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers  = [warmup_sched, base_sched],
        milestones  = [warmup_e],
    )


# ---------------------------------------------------------------------------
# Train / Eval one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    log_interval: int,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0
    n_batches    = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0 or batch_idx == n_batches - 1:
            avg = running_loss / (batch_idx + 1)
            print(f"    epoch {epoch:3d}  [{batch_idx+1:4d}/{n_batches}]  "
                  f"loss={avg:.4f}", end="\r", flush=True)

    print()  # newline after \r
    return running_loss / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss    += loss.item() * images.size(0)
        preds          = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Train one scaled model
# ---------------------------------------------------------------------------

def train_model(
    scale: float,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    ckpt_dir: Path,
) -> dict:
    """Train a single scaled model and return its best metrics."""
    tcfg       = cfg["train"]
    epochs     = tcfg["epochs"]
    log_ivl    = cfg["output"].get("log_interval", 10)
    save_best  = cfg["output"].get("save_best_only", True)
    ckpt_path  = ckpt_dir / f"resnet50_scale{scale:.2f}.pth"

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    best_val_acc  = 0.0
    best_epoch    = 0
    history       = []
    t0            = time.time()

    print(f"\n{'═'*60}")
    print(f"  Training scale={scale}×   "
          f"params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'═'*60}")

    for epoch in range(1, epochs + 1):
        train_loss            = train_one_epoch(
            model, train_loader, criterion, optimizer, device, log_ivl, epoch
        )
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") \
                     else optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "lr": round(current_lr, 8),
        })

        elapsed = time.time() - t0
        eta     = elapsed / epoch * (epochs - epoch)
        print(f"  epoch {epoch:3d}/{epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  "
              f"lr={current_lr:.2e}  "
              f"ETA {format_time(eta)}")

        # ── Checkpoint ──────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            if save_best:
                torch.save({
                    "scale":     scale,
                    "epoch":     epoch,
                    "val_acc":   val_acc,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, ckpt_path)
                print(f"  ✓ checkpoint saved → {ckpt_path}  (val_acc={val_acc:.4f})")

    elapsed_total = time.time() - t0
    print(f"\n  ✓ scale={scale}×  best val_acc={best_val_acc:.4f} "
          f"at epoch {best_epoch}  total time={format_time(elapsed_total)}")

    return {
        "scale":        scale,
        "best_val_acc": best_val_acc,
        "best_epoch":   best_epoch,
        "total_time_s": round(elapsed_total, 1),
        "history":      history,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train width-scaled ResNets on CIFAR-10")
    parser.add_argument(
        "--config", default="config_train.yaml",
        help="Path to the YAML config file (default: config_train.yaml)"
    )
    parser.add_argument(
        "--device", default=None,
        help="Override device: 'cpu' | 'cuda' | 'cuda:0' etc."
    )
    args = parser.parse_args()

    # ── Config ──────────────────────────────────────────────────────────
    cfg    = load_config(args.config)
    device = resolve_device(args.device or cfg.get("device", "auto"))
    set_seed(cfg.get("seed", 42))

    print(f"[config] loaded '{args.config}'")
    print(f"[device] using {device}")

    # ── Output directories ───────────────────────────────────────────────
    results_dir = Path(cfg["output"]["results_dir"])
    ckpt_dir    = Path(cfg["output"]["checkpoint_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg, device)

    # ── Build scaled models ──────────────────────────────────────────────
    m_cfg   = cfg["model"]
    wrapper = Model(arch=m_cfg["arch"], device=str(device))
    variants: List[Tuple[float, nn.Module]] = wrapper.resize_model(
        scales      = m_cfg["scales"],
        num_classes = m_cfg["num_classes"],
    )

    # ── Sequential training ──────────────────────────────────────────────
    all_results = []
    for scale, model in variants:
        result = train_model(
            scale        = scale,
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            cfg          = cfg,
            device       = device,
            ckpt_dir     = ckpt_dir,
        )
        all_results.append(result)

    # ── Summary CSV ─────────────────────────────────────────────────────
    summary_path = results_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scale", "best_val_acc", "best_epoch", "total_time_s"
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    print(f"\n{'═'*60}")
    print(f"  TRAINING COMPLETE — Summary")
    print(f"{'═'*60}")
    print(f"  {'Scale':>8}  {'Best Val Acc':>14}  {'Best Epoch':>12}  {'Time':>10}")
    print(f"  {'─'*8}  {'─'*14}  {'─'*12}  {'─'*10}")
    for r in all_results:
        print(f"  {r['scale']:>7.2f}×  {r['best_val_acc']:>14.4f}  "
              f"{r['best_epoch']:>12}  {format_time(r['total_time_s']):>10}")
    print(f"\n  Summary CSV → {summary_path}")
    print(f"  Checkpoints → {ckpt_dir}/")


if __name__ == "__main__":
    main()
