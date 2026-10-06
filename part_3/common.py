"""Shared settings, metrics and plot helpers for Part 3 (raw confidence under CIFAR-10-C shift)."""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
# Dataset constants and all downloads live in shared/ (one copy for every part).
from shared.data import K, CORRUPTIONS, SEVERITIES, MEAN, STD   # noqa: E402,F401

BINS = 15

PALETTE = {"charcoal": "#333333", "blue": "#2166AC", "orange": "#D6600A",
           "green": "#1B7837", "red": "#C0392B", "purple": "#6A3D9A"}
# Clean = charcoal, then one colour per severity (same colours as the Part 4 figures).
SEVERITY_COLORS = dict(zip((0,) + SEVERITIES,
                           [PALETTE[c] for c in ("charcoal", "green", "blue", "orange", "red", "purple")]))
CORRUPTION_COLORS = dict(zip(CORRUPTIONS, [PALETTE[c] for c in ("orange", "purple", "blue", "red", "green")]))


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def reliability_bins(p, y, n_bins=BINS):
    """(mean confidence, accuracy, count) of every non-empty equal-width bin."""
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    bin_id = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    points = []
    for b in range(n_bins):
        selected = bin_id == b
        if selected.any():
            points.append((conf[selected].mean(), correct[selected].mean(), selected.sum()))
    return np.asarray(points, dtype=float)


def adaptive_top_label(p, y, n_bins=BINS):
    """ACE: top-label calibration error over equal-mass bins (Nixon et al., 2019)."""
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    groups = np.array_split(np.argsort(conf, kind="stable"), n_bins)
    return sum(len(g) / len(y) * abs(conf[g].mean() - correct[g].mean()) for g in groups if len(g))


def ace_classwise(p, y, n_bins=BINS):
    errors = []
    for k in range(p.shape[1]):
        groups = np.array_split(np.argsort(p[:, k], kind="stable"), n_bins)
        errors.extend(abs(p[g, k].mean() - (y[g] == k).mean()) for g in groups if len(g))
    return float(np.mean(errors))


def risk_coverage(p, y):
    """Accept the k most confident predictions, k = 1..N; returns (coverage, risk)."""
    order = np.argsort(-p.max(1), kind="stable")
    errors = (p.argmax(1)[order] != y[order]).astype(float)
    accepted = np.arange(1, len(y) + 1)
    return accepted / len(y), np.cumsum(errors) / accepted


def auroc_correct(conf, correct):
    """P(a random correct prediction is more confident than a random wrong one); 0.5 = no information."""
    correct = np.asarray(correct, dtype=bool)
    n1, n0 = correct.sum(), (~correct).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(conf)
    return float((r[correct].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(p, y):
    p, y = np.asarray(p), np.asarray(y)
    assert p.shape == (len(y), K) and len(y) > 0
    assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1)
    points = reliability_bins(p, y)
    _, risk = risk_coverage(p, y)
    conf, correct = p.max(1), (p.argmax(1) == y)
    return {
        "Accuracy": float(correct.mean()),
        "ECE": float(np.sum(points[:, 2] / len(y) * np.abs(points[:, 0] - points[:, 1]))),
        "AdaptiveTopLabel": float(adaptive_top_label(p, y)),
        "ACE_classwise": ace_classwise(p, y),
        "NLL": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-300, 1)).mean()),
        "Risk@80": float(risk[max(0, int(np.ceil(.8 * len(y))) - 1)]),
        "Risk@60": float(risk[max(0, int(np.ceil(.6 * len(y))) - 1)]),
        "MeanConf": float(conf.mean()),
        "Gap": float(conf.mean() - correct.mean()),      # > 0 overconfident, < 0 underconfident
        "AUROC": auroc_correct(conf, correct),
    }
