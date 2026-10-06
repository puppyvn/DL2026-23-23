"""
experiments.py — Confidence-calibration experiments for width-scaled ResNet-50 models.
=======================================================================================

Experiment A — 5 model sizes × original CIFAR-10 test set
    Loads each trained checkpoint (0.5×, 0.75×, 1.0×, 1.5×, 2.0×) and evaluates
    accuracy, ECE, ACE and confidence gap on the clean test set.

Experiment B — Original (1.0×) model × 5 distribution shifts
    Loads the 1.0× checkpoint and evaluates it on each of the 5 shifted datasets
    defined in data/data.py (Camera, Lighting, Quality, Appearance, Mixed).

Outputs (saved to results/experiments/):
    - fig_exp_a_reliability.png     : reliability diagrams for all 5 model sizes
    - fig_exp_a_conf_hist.png       : confidence histograms for all 5 model sizes
    - fig_exp_a_metrics.png         : ECE / ACE vs model scale bar chart
    - fig_exp_b_reliability.png     : reliability diagrams for 1.0× across 5 shifts
    - fig_exp_b_conf_hist.png       : confidence histograms across 5 shifts
    - fig_exp_b_metrics.png         : ECE / ACE per shift bar chart
    - summary_exp_a.csv             : per-scale numeric results
    - summary_exp_b.csv             : per-shift  numeric results

Usage:
    python experiments.py
    python experiments.py --config config_train.yaml --device cuda
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Make sure project root is on sys.path ──────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.data import Data_loader
from src.model import Model

# ---------------------------------------------------------------------------
# Global plot style — matching DeepLearning_Part1.ipynb
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi":  110,
    "axes.grid":   True,
    "grid.alpha":  0.25,
    "font.size":   10,
})

# Colour palette — one colour per entity
# Experiment A (5 model scales)
SCALE_COLORS = {
    0.50: "#1b7837",   # green
    0.75: "#2166ac",   # blue
    1.00: "#333333",   # dark grey (original / baseline)
    1.50: "#d6600a",   # orange
    2.00: "#c0392b",   # red
}

# Experiment B (5 shifts + original)
SHIFT_COLORS = {
    "Original":     "#333333",   # dark grey
    "Camera":       "#1b7837",   # green
    "Lighting":     "#2166ac",   # blue
    "Quality":      "#d6600a",   # orange
    "Appearance":   "#c0392b",   # red
    "Mixed":        "#6a3d9a",   # purple
}

N_BINS  = 15   # ECE / ACE bins
N_BOOT  = 1000 # bootstrap resamples
BATCH   = 64

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

# ---------------------------------------------------------------------------
# Metric helpers (identical logic to DeepLearning_Part1.ipynb)
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z.astype(np.float64))
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def ece_score(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    idx = np.digitize(conf, np.linspace(0, 1, n_bins + 1)[1:-1], right=True)
    sc  = np.bincount(idx, weights=conf,    minlength=n_bins)
    sa  = np.bincount(idx, weights=correct, minlength=n_bins)
    return float(np.abs(sa - sc).sum() / len(conf))


def ace_score(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    o = np.argsort(conf, kind="stable")
    c = conf[o]; a = correct[o].astype(float)
    parts = np.array_split(np.arange(len(c)), n_bins)
    return float(np.mean([abs(a[p].mean() - c[p].mean()) for p in parts]))


def reliability_bins(conf: np.ndarray, correct: np.ndarray,
                     n_bins: int = N_BINS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.digitize(conf, np.linspace(0, 1, n_bins + 1)[1:-1], right=True)
    cnt = np.bincount(idx, minlength=n_bins).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        bc = np.bincount(idx, weights=conf,    minlength=n_bins) / cnt
        ba = np.bincount(idx, weights=correct, minlength=n_bins) / cnt
    return bc, ba, cnt


def all_metrics(conf: np.ndarray, correct: np.ndarray) -> dict:
    # Note: scalar mean confidence is stored as 'mean_conf' to avoid
    # colliding with the 'conf' numpy array stored directly on each record.
    return dict(
        acc=float(correct.mean()),
        mean_conf=float(conf.mean()),
        gap=float(conf.mean() - correct.mean()),
        ece=ece_score(conf, correct),
        ace=ace_score(conf, correct),
    )


def bootstrap_ci(conf: np.ndarray, correct: np.ndarray,
                 B: int = N_BOOT, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    N   = len(conf)
    acc_b = np.empty(B); ece_b = np.empty(B); ace_b = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, N, N)
        c, k = conf[idx], correct[idx]
        acc_b[b] = k.mean()
        ece_b[b] = ece_score(c, k)
        ace_b[b] = ace_score(c, k)
    return dict(
        acc_lo=float(np.percentile(acc_b,  2.5)),
        acc_hi=float(np.percentile(acc_b, 97.5)),
        ece_lo=float(np.percentile(ece_b,  2.5)),
        ece_hi=float(np.percentile(ece_b, 97.5)),
        ace_lo=float(np.percentile(ace_b,  2.5)),
        ace_hi=float(np.percentile(ace_b, 97.5)),
    )


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_logits(model: nn.Module, dataset: torchvision.datasets.VisionDataset,
               device: torch.device, batch_size: int = BATCH) -> np.ndarray:
    """Run forward pass in batches; return raw logits (N, C)."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    model.eval().to(device)
    all_logits = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images).float().cpu()
        all_logits.append(logits)
    del loader
    return torch.cat(all_logits, dim=0).numpy()


# ---------------------------------------------------------------------------
# Plotting helpers — styled like DeepLearning_Part1.ipynb
# ---------------------------------------------------------------------------

def plot_reliability_diagrams(records: List[dict], out_path: str,
                               suptitle: str = "") -> None:
    """
    records: list of dicts, each with keys:
        label, color, conf (np array), correct (np array),
        acc, ece, ace
    """
    n = len(records)
    ncol = min(n, 4)
    nrow = int(np.ceil(n / ncol))

    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow))
    axs_flat = axs.ravel() if n > 1 else [axs]
    w = 1 / N_BINS

    for ax, rec in zip(axs_flat, records):
        bc, ba, cnt = reliability_bins(rec["conf"], rec["correct"])
        centers = (np.arange(N_BINS) + 0.5) * w
        ok = cnt > 0

        # Accuracy in bin — blue bar (same as Part1)
        ax.bar(centers[ok], ba[ok], width=w * 0.95,
               color="tab:blue", edgecolor="k", lw=0.4, label="accuracy in bin")
        # Over-confidence gap — red overlay
        over = np.where(bc[ok] > ba[ok], bc[ok] - ba[ok], 0)
        ax.bar(centers[ok], over, bottom=ba[ok], width=w * 0.95,
               color="tab:red", alpha=0.35, label="over-conf gap")
        # Under-confidence gap — green overlay
        under = np.where(ba[ok] > bc[ok], ba[ok] - bc[ok], 0)
        ax.bar(centers[ok], under, bottom=bc[ok], width=w * 0.95,
               color="tab:green", alpha=0.35, label="under-conf gap")

        # Perfect-calibration diagonal
        ax.plot([0, 1], [0, 1], "k--", lw=1)

        ax.set_title(
            f"{rec['label']}\n"
            f"acc {100*rec['acc']:.1f}%  ECE {100*rec['ece']:.2f}%  ACE {100*rec['ace']:.2f}%",
            fontsize=9,
        )
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")

    # Hide unused axes
    for ax in axs_flat[n:]:
        ax.axis("off")

    axs_flat[0].legend(loc="upper left", fontsize=7)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


def plot_confidence_histograms(records: List[dict], out_path: str,
                                suptitle: str = "") -> None:
    n = len(records)
    ncol = min(n, 4)
    nrow = int(np.ceil(n / ncol))

    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow))
    axs_flat = axs.ravel() if n > 1 else [axs]

    for ax, rec in zip(axs_flat, records):
        c, k = rec["conf"], rec["correct"]
        ax.hist(c[k == 1], bins=30, range=(0, 1), alpha=0.6,
                label="correct", color="tab:blue")
        ax.hist(c[k == 0], bins=30, range=(0, 1), alpha=0.7,
                label="wrong", color="tab:red")
        ax.set_yscale("log")
        ax.set_title(rec["label"], fontsize=9)
        ax.set_xlabel("confidence")

    for ax in axs_flat[n:]:
        ax.axis("off")

    axs_flat[0].legend(fontsize=7)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


def plot_metrics_bar(records: List[dict], out_path: str,
                     x_label: str = "", suptitle: str = "") -> None:
    """Bar chart comparing ECE and ACE with 95% bootstrap CI error bars."""
    labels = [r["label"] for r in records]
    colors = [r["color"] for r in records]
    eces   = [100 * r["ece"] for r in records]
    aces   = [100 * r["ace"] for r in records]
    ece_errs = [
        [100 * (r["ece"] - r.get("ece_lo", r["ece"])),
         100 * (r.get("ece_hi", r["ece"]) - r["ece"])]
        for r in records
    ]
    ace_errs = [
        [100 * (r["ace"] - r.get("ace_lo", r["ace"])),
         100 * (r.get("ace_hi", r["ace"]) - r["ace"])]
        for r in records
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, 2.5 * len(labels)), 4.5))

    for ax, vals, errs, metric_name in [
        (ax1, eces, ece_errs, "ECE (%)"),
        (ax2, aces, ace_errs, "ACE (%)"),
    ]:
        bars = ax.bar(x, vals, width=0.6, color=colors,
                      edgecolor="k", linewidth=0.5)
        # Error bars
        lo = [e[0] for e in errs]
        hi = [e[1] for e in errs]
        ax.errorbar(x, vals, yerr=[lo, hi], fmt="none",
                    color="black", capsize=3, linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(metric_name)
        if x_label:
            ax.set_xlabel(x_label)
        ax.set_title(metric_name)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


def plot_acc_vs_calibration(records: List[dict], out_path: str,
                             suptitle: str = "") -> None:
    """Scatter: Accuracy vs ECE and Accuracy vs ACE (matching Part1 Cell 12)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, metric, ylabel in [(ax1, "ece", "ECE (%)"), (ax2, "ace", "ACE (%)")]:
        for rec in records:
            ax.errorbar(
                100 * rec["acc"], 100 * rec[metric],
                yerr=[[100 * (rec[metric] - rec.get(f"{metric}_lo", rec[metric]))],
                      [100 * (rec.get(f"{metric}_hi", rec[metric]) - rec[metric])]],
                fmt="o", color=rec["color"], capsize=2,
            )
            ax.annotate(rec["label"],
                        (100 * rec["acc"], 100 * rec[metric]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Accuracy (%)")
        ax.set_ylabel(ylabel)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------

def save_csv(records: List[dict], path: str) -> None:
    fields = ["label", "acc", "acc_lo", "acc_hi",
              "mean_conf", "gap", "ece", "ece_lo", "ece_hi",
              "ace", "ace_lo", "ace_hi"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_test_dataset(resize: int) -> torchvision.datasets.VisionDataset:
    """
    Load the CIFAR-10 *test* split (10 000 held-out samples) via
    Data_loader.load_validation_data() and apply standard normalisation.

    This is the correct dataset for evaluation — completely separate from
    the 50 000-sample training split used by src/train.py.
    """
    dl = Data_loader()
    dl.load_validation_data()   # loads self.val_data as PIL images

    transform = T.Compose([
        T.Resize((resize, resize)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    # Attach the inference transform to val_data
    ds = torchvision.datasets.CIFAR10.__new__(torchvision.datasets.CIFAR10)
    ds.__dict__.update(dl.val_data.__dict__)
    ds.transform = transform
    return ds


def load_shifted_datasets(resize: int) -> Dict[str, torchvision.datasets.VisionDataset]:
    """
    Build 5 distribution-shifted versions of the CIFAR-10 *test* split using
    Data_loader.load_validation_data() + Data_loader.augmentation_validation().

    Each shifted dataset applies its corruption pipeline then resizes and
    normalises for inference — keeping the same 10 000 test images as the
    unshifted baseline so results are directly comparable.
    """
    normalise_tail = [
        T.Resize((resize, resize)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ]

    # ── Use Data_loader to build the 5 shifted test datasets ────────────
    dl = Data_loader()
    dl.load_validation_data()      # loads 10 000-sample test split as PIL
    dl.augmentation_validation()   # builds shift1…shift5 on the test split

    # Helper: append resize+normalise after the shift pipeline's ToTensor.
    # Each shift pipeline already ends with ToTensor(); we replace that with
    # ToTensor → Resize → Normalize so spatial resolution and pixel range are
    # correct for the trained model.
    def _make_tfm(shift_ds) -> T.Compose:
        tfms = list(shift_ds.transform.transforms)
        # Drop the trailing ToTensor added by the shift pipeline
        if isinstance(tfms[-1], T.ToTensor):
            tfms = tfms[:-1]
        return T.Compose(tfms + normalise_tail)

    def _apply_tfm(shift_ds):
        new_ds = torchvision.datasets.CIFAR10.__new__(torchvision.datasets.CIFAR10)
        new_ds.__dict__.update(dl.val_data.__dict__)   # base = test split
        new_ds.transform = _make_tfm(shift_ds)
        return new_ds

    return {
        "Camera":     _apply_tfm(dl.shift1),
        "Lighting":   _apply_tfm(dl.shift2),
        "Quality":    _apply_tfm(dl.shift3),
        "Appearance": _apply_tfm(dl.shift4),
        "Mixed":      _apply_tfm(dl.shift5),
    }



# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: str, arch: str, scale: float,
                    num_classes: int, device: torch.device) -> nn.Module:
    """Load a trained scaled ResNet from a .pth checkpoint on CPU first, then transfer."""
    wrapper = Model(arch=arch, device="cpu")
    variants = wrapper.resize_model(scales=[scale], num_classes=num_classes)
    _, model = variants[0]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    del ckpt, state, wrapper, variants
    model.eval().to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model


# ---------------------------------------------------------------------------
# Main experiments
# ---------------------------------------------------------------------------

def run_experiment_a(cfg: dict, device: torch.device, out_dir: Path,
                     batch_size: int = BATCH) -> None:
    """5 model scales × original CIFAR-10 test set."""
    print("\n" + "═" * 60)
    print("  Experiment A: Model Size vs. Calibration (original test set)")
    print("═" * 60)

    m_cfg   = cfg["model"]
    arch    = m_cfg["arch"]
    scales  = m_cfg["scales"]
    resize  = cfg["augmentation"].get("resize", 224)
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])

    test_ds = load_test_dataset(resize)
    y_test  = np.array(test_ds.targets)

    records = []
    for scale in scales:
        ckpt_name = f"resnet50_scale{scale:.2f}.pth"
        ckpt_path = ckpt_dir / ckpt_name
        if not ckpt_path.exists():
            print(f"  [SKIP] checkpoint not found: {ckpt_path}")
            continue

        print(f"  Loading scale={scale}× from {ckpt_path} …")
        model = load_checkpoint(str(ckpt_path), arch, scale,
                                m_cfg["num_classes"], device)

        logits  = get_logits(model, test_ds, device, batch_size=batch_size)
        probs   = _softmax(logits)
        conf    = probs.max(1)
        correct = (probs.argmax(1) == y_test).astype(float)

        # Deload model from GPU immediately before computing metrics
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        metrics = all_metrics(conf, correct)
        ci      = bootstrap_ci(conf, correct)

        rec = dict(
            label=f"scale {scale:.2f}×",
            color=SCALE_COLORS.get(scale, "#888888"),
            conf=conf, correct=correct,
            **metrics, **ci,
        )
        records.append(rec)
        print(f"    acc={100*metrics['acc']:.2f}%  "
              f"ECE={100*metrics['ece']:.2f}%  ACE={100*metrics['ace']:.2f}%")

    if not records:
        print("  No checkpoints found — skipping Experiment A plots.")
        return

    suptitle_a = "Experiment A — Width-Scaled ResNet-50 on Clean CIFAR-10"

    plot_reliability_diagrams(
        records,
        str(out_dir / "fig_exp_a_reliability.png"),
        suptitle=suptitle_a,
    )
    plot_confidence_histograms(
        records,
        str(out_dir / "fig_exp_a_conf_hist.png"),
        suptitle=suptitle_a,
    )
    plot_metrics_bar(
        records,
        str(out_dir / "fig_exp_a_metrics.png"),
        x_label="Model scale (width multiplier)",
        suptitle=suptitle_a,
    )
    plot_acc_vs_calibration(
        records,
        str(out_dir / "fig_exp_a_acc_vs_cal.png"),
        suptitle=suptitle_a,
    )
    save_csv(records, str(out_dir / "summary_exp_a.csv"))


def run_experiment_b(cfg: dict, device: torch.device, out_dir: Path,
                     batch_size: int = BATCH) -> None:
    """Original (1.0×) model × original + 5 shifted test sets."""
    print("\n" + "═" * 60)
    print("  Experiment B: Distribution Shift Effect on ResNet-50 (1.0×)")
    print("═" * 60)

    m_cfg    = cfg["model"]
    arch     = m_cfg["arch"]
    resize   = cfg["augmentation"].get("resize", 224)
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])

    scale_1x  = 1.0
    ckpt_name = f"resnet50_scale{scale_1x:.2f}.pth"
    ckpt_path = ckpt_dir / ckpt_name

    if not ckpt_path.exists():
        print(f"  [SKIP] 1.0× checkpoint not found: {ckpt_path}")
        return

    print(f"  Loading scale=1.0× from {ckpt_path} …")
    model = load_checkpoint(str(ckpt_path), arch, scale_1x,
                            m_cfg["num_classes"], device)

    # ── Original test set ────────────────────────────────────────────────
    test_ds = load_test_dataset(resize)
    y_test  = np.array(test_ds.targets)

    records = []

    def _eval_dataset(ds, label: str, color: str) -> dict:
        logits  = get_logits(model, ds, device, batch_size=batch_size)
        probs   = _softmax(logits)
        conf    = probs.max(1)
        correct = (probs.argmax(1) == y_test).astype(float)
        metrics = all_metrics(conf, correct)
        ci      = bootstrap_ci(conf, correct)
        print(f"    [{label}]  acc={100*metrics['acc']:.2f}%  "
              f"ECE={100*metrics['ece']:.2f}%  ACE={100*metrics['ace']:.2f}%")
        return dict(label=label, color=color,
                    conf=conf, correct=correct, **metrics, **ci)

    print("  Evaluating on original test set …")
    records.append(_eval_dataset(test_ds, "Original", SHIFT_COLORS["Original"]))

    # ── Shifted test sets ────────────────────────────────────────────────
    print("  Building shifted datasets …")
    shifted = load_shifted_datasets(resize)
    shift_order = ["Camera", "Lighting", "Quality", "Appearance", "Mixed"]
    for name in shift_order:
        ds = shifted[name]
        print(f"  Evaluating on '{name}' shift …")
        records.append(_eval_dataset(ds, name, SHIFT_COLORS[name]))

    # Deload model after all Experiment B evaluations
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    suptitle_b = "Experiment B — ResNet-50 (1.0×) under Distribution Shift"

    plot_reliability_diagrams(
        records,
        str(out_dir / "fig_exp_b_reliability.png"),
        suptitle=suptitle_b,
    )
    plot_confidence_histograms(
        records,
        str(out_dir / "fig_exp_b_conf_hist.png"),
        suptitle=suptitle_b,
    )
    plot_metrics_bar(
        records,
        str(out_dir / "fig_exp_b_metrics.png"),
        x_label="Dataset",
        suptitle=suptitle_b,
    )
    plot_acc_vs_calibration(
        records,
        str(out_dir / "fig_exp_b_acc_vs_cal.png"),
        suptitle=suptitle_b,
    )
    save_csv(records, str(out_dir / "summary_exp_b.csv"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(
        description="Run calibration experiments on trained ResNet-50 variants"
    )
    parser.add_argument(
        "--config", default="config_train.yaml",
        help="Path to the YAML training config (default: config_train.yaml)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Override device: 'cpu' | 'cuda' | 'cuda:0' etc.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH,
        help=f"Inference batch size (default: {BATCH})",
    )
    parser.add_argument(
        "--out_dir", default="results/experiments",
        help="Directory to save plots and CSVs (default: results/experiments)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Resolve device
    raw_device = args.device or cfg.get("device", "auto")
    if raw_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(raw_device)

    print(f"[device] using {device}")
    print(f"[batch_size] using {args.batch_size}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_experiment_a(cfg, device, out_dir, batch_size=args.batch_size)
    run_experiment_b(cfg, device, out_dir, batch_size=args.batch_size)

    print("\n" + "═" * 60)
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"  Output → {out_dir}/")
    print("═" * 60)


if __name__ == "__main__":
    main()
