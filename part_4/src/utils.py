"""Shared configuration, calibration methods, metrics, and plotting helpers."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax, log_softmax
from scipy.optimize import minimize, minimize_scalar
from sklearn.model_selection import StratifiedKFold

K, BINS = 10, 15
DEFAULT_ROOT = Path(__file__).resolve().parent / "artifacts"
LAMBDAS = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
CORRUPTIONS = ("gaussian_noise", "motion_blur", "brightness", "contrast", "pixelate")
SEVERITIES = (1, 2, 3, 4, 5)
# Categorical palette sampled from the report's comparison figures (pages 11-16).
REPORT_PALETTE = {
    "charcoal": "#333333", "blue": "#2166AC", "orange": "#D6600A",
    "green": "#1B7837", "red": "#C0392B", "purple": "#6A3D9A",
}
COLORS = {"Baseline": REPORT_PALETTE["charcoal"],
          "Temperature Scaling": REPORT_PALETTE["blue"],
          "Dirichlet": REPORT_PALETTE["orange"]}
METHOD_NAMES = tuple(COLORS)
SEVERITY_COLORS = dict(zip((0,) + SEVERITIES,
    [REPORT_PALETTE[name] for name in ("charcoal", "green", "blue", "orange", "red", "purple")]))
TEMPERATURE_COLORS = {0.5: REPORT_PALETTE["red"], 1.0: REPORT_PALETTE["charcoal"],
                      2.0: REPORT_PALETTE["blue"]}
CORRUPTION_COLORS = dict(zip(CORRUPTIONS,
    [REPORT_PALETTE[name] for name in ("orange", "purple", "blue", "red", "green")]))
REFERENCE_COLOR = REPORT_PALETTE["charcoal"]
GUIDE_COLOR = "#999999"


def style_boxplot(artists, colors):
    """Use the same severity colors as the reliability and risk plots."""
    for box, color in zip(artists["boxes"], colors):
        box.set(facecolor=color, edgecolor=color, alpha=0.65)
    for key in ("whiskers", "caps", "medians"):
        for artist in artists[key]:
            artist.set_color(REFERENCE_COLOR)

def project_paths(root):
    root = Path(root).expanduser().resolve()
    cache, results = root / "cache", root / "results"
    figures = results / "figures"
    for folder in (root, cache, results, figures):
        folder.mkdir(parents=True, exist_ok=True)
    return root, cache, results, figures

def display(table):
    """Print a DataFrame in a terminal without an IPython dependency."""
    print(table.to_string(index=not isinstance(table.index, pd.RangeIndex)))

def reliability_bins(p, y, n_bins=BINS):
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
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    groups = np.array_split(np.argsort(conf, kind="stable"), n_bins)
    return sum(len(g) / len(y) * abs(conf[g].mean() - correct[g].mean())
               for g in groups if len(g))

def ace_classwise(p, y, n_bins=BINS):
    errors = []
    for k in range(p.shape[1]):
        groups = np.array_split(np.argsort(p[:, k], kind="stable"), n_bins)
        errors.extend(abs(p[g, k].mean() - (y[g] == k).mean()) for g in groups if len(g))
    return float(np.mean(errors))

def risk_coverage(p, y):
    order = np.argsort(-p.max(1), kind="stable")
    errors = (p.argmax(1)[order] != y[order]).astype(float)
    accepted = np.arange(1, len(y) + 1)
    return accepted / len(y), np.cumsum(errors) / accepted

def evaluate(p, y):
    p, y = np.asarray(p), np.asarray(y)
    assert p.shape == (len(y), K) and len(y) > 0
    assert np.isfinite(p).all() and (p >= 0).all()
    assert np.allclose(p.sum(1), 1)
    points = reliability_bins(p, y)
    coverage, risk = risk_coverage(p, y)
    return {
        "Accuracy": float((p.argmax(1) == y).mean()),
        "ECE": float(np.sum(points[:, 2] / len(y) * np.abs(points[:, 0] - points[:, 1]))),
        "AdaptiveTopLabel": float(adaptive_top_label(p, y)),
        "ACE_classwise": ace_classwise(p, y),
        "NLL": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-300, 1)).mean()),
        "Risk@80": float(risk[max(0, int(np.ceil(.8 * len(y))) - 1)]),
        "Risk@60": float(risk[max(0, int(np.ceil(.6 * len(y))) - 1)]),
    }

def finish_figure(fig, filename, figures, show=True):
    fig.tight_layout()
    fig.savefig(figures / f"{filename}.png", dpi=180, bbox_inches="tight")
    fig.savefig(figures / f"{filename}.pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def draw_reliability(ax, p, y, title, color=REPORT_PALETTE["blue"]):
    points = reliability_bins(p, y)
    ax.plot([0, 1], [0, 1], "--", color=REFERENCE_COLOR, lw=1)
    ax.plot(points[:, 0], points[:, 1], "-", color=color, alpha=.6)
    # Keep all nonempty bins; marker size represents the sample count.
    ax.scatter(points[:, 0], points[:, 1],
               s=12 + 90 * points[:, 2] / points[:, 2].max(), color=color)
    score = evaluate(p, y)
    ax.set(title=f"{title}\nECE={score['ECE']:.4f}", xlabel="Mean confidence",
           ylabel="Observed accuracy", xlim=(0, 1), ylim=(0, 1))

def draw_risk(ax, p, y, label, color=REPORT_PALETTE["blue"], linestyle="-"):
    coverage, risk = risk_coverage(p, y)
    ax.plot(coverage, risk, label=label, color=color, linestyle=linestyle)
    ax.set(xlabel="Coverage", ylabel="Risk (error rate)", xlim=(0, 1))

def make_split(y, seed):
    rng = np.random.RandomState(seed)
    calibration, final_test = [], []
    for k in range(K):
        indices = rng.permutation(np.flatnonzero(y == k))
        assert len(indices) == 1000
        calibration.extend(indices[:500])
        final_test.extend(indices[500:])
    calibration, final_test = np.sort(calibration), np.sort(final_test)
    assert len(calibration) == len(final_test) == 5000
    assert np.intersect1d(calibration, final_test).size == 0
    return calibration, final_test

def fit_temperature(logits, labels):
    # T = exp(log_T) is always positive; use log_softmax to avoid underflow.
    def objective(log_T):
        return -log_softmax(logits / np.exp(log_T), axis=1)[np.arange(len(labels)), labels].mean()
    result = minimize_scalar(objective, bounds=(np.log(.05), np.log(20)), method="bounded")
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"TS optimization failed: {result.message}")
    temperature = float(np.exp(result.x))
    if temperature < .051 or temperature > 19.9:
        warnings.warn("T is close to a search boundary; check the optimization range.")
    return temperature

def dirichlet_objective(theta, lp, labels, regularization):
    n, k = lp.shape
    W, b = theta[:k*k].reshape(k, k), theta[k*k:]
    off_diag = 1 - np.eye(k)
    scores = lp @ W.T + b
    log_q = log_softmax(scores, axis=1)
    loss = (-log_q[np.arange(n), labels].mean()
            + regularization * np.sum((W * off_diag)**2) / (k * (k-1))
            + regularization * np.sum(b**2) / k)
    delta = np.exp(log_q)
    delta[np.arange(n), labels] -= 1
    delta /= n
    grad_W = delta.T @ lp + 2 * regularization * W * off_diag / (k * (k-1))
    grad_b = delta.sum(0) + 2 * regularization * b / k
    return float(loss), np.concatenate([grad_W.ravel(), grad_b])

def fit_dirichlet(logits, labels, regularization):
    k = logits.shape[1]
    lp = log_softmax(logits, axis=1)
    initial = np.concatenate([np.eye(k).ravel(), np.zeros(k)])
    result = minimize(dirichlet_objective, initial, args=(lp, labels, regularization),
                      jac=True, method="L-BFGS-B",
                      options={"maxiter": 2000, "ftol": 1e-10, "maxls": 50})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Dirichlet failed, lambda={regularization}: {result.message}")
    return result.x[:k*k].reshape(k, k), result.x[k*k:]

def dirichlet_prob(logits, W, b):
    return softmax(log_softmax(logits, axis=1) @ W.T + b, axis=1)

def select_regularization(logits, labels, seed):
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(logits, labels))
    rows = []
    for lam in LAMBDAS:
        losses = []
        for train, valid in folds:
            W_cv, b_cv = fit_dirichlet(logits[train], labels[train], lam)
            scores = log_softmax(logits[valid], axis=1) @ W_cv.T + b_cv
            losses.append(float(-log_softmax(scores, axis=1)[np.arange(len(valid)), labels[valid]].mean()))
        rows.append({"lambda": lam, "CV_NLL": np.mean(losses), "fold_std": np.std(losses, ddof=1)})
    table = pd.DataFrame(rows)
    best = float(table.loc[table.CV_NLL.idxmin(), "lambda"])
    return best, table

def calibrated_predictions(logits, T, W, b):
    return {"Baseline": softmax(logits, axis=1),
            "Temperature Scaling": softmax(logits / T, axis=1),
            "Dirichlet": dirichlet_prob(logits, W, b)}
