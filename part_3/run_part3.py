"""Part 3 - Can we trust the model forever? Raw confidence under increasing distribution shift.

Same frozen ResNet-18 as Parts 2 and 4, raw confidence (T = 1), clean CIFAR-10 and CIFAR-10-C
severities 1-5 for five corruptions (gaussian_noise, motion_blur, brightness, contrast, pixelate).

  3.1  confidence distribution vs accuracy of fixed 500-image batches (boxplots)
  3.2  reliability diagrams per severity (+ ECE / ACE)
  3.3  Risk-Coverage per severity (+ Risk@60%, Risk@80%, AUROC)

Run prepare_data.py first, then:
    python run_part3.py                    # reads ./artifacts/cache, writes ./artifacts/results/part3_<time>/
    python run_part3.py --root /path/to/artifacts
"""
import argparse
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import (K, BINS, CORRUPTIONS, SEVERITIES, DEFAULT_ROOT, PALETTE, SEVERITY_COLORS,
                    project_paths, softmax, evaluate, reliability_bins, risk_coverage)


def load_logits(cache):
    with np.load(cache / "clean_logits.npz", allow_pickle=False) as saved:
        Z, Y, provenance = saved["logits"].astype(np.float64), saved["labels"].astype(int), str(saved["provenance"])
    assert Z.shape == (10000, K) and np.all(np.bincount(Y, minlength=K) == 1000)
    C = {}
    for name in CORRUPTIONS:
        with np.load(cache / f"corruption_{name}.npz", allow_pickle=False) as saved:
            if str(saved["provenance"]) != provenance or not np.array_equal(saved["labels"], Y):
                raise ValueError(f"corruption_{name}.npz was made with a different checkpoint or label order.")
            C[name] = saved["logits"].astype(np.float64)
        assert C[name].shape == (5, 10000, K) and np.isfinite(C[name]).all()
    return Z, Y, C, provenance


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_reliability(ax, p, y, title, color):
    pts = reliability_bins(p, y)
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["charcoal"], lw=1)
    ax.plot(pts[:, 0], pts[:, 1], "-", color=color, alpha=.6)
    ax.scatter(pts[:, 0], pts[:, 1], s=12 + 90 * pts[:, 2] / pts[:, 2].max(), color=color)
    ax.set(title=f"{title}\nECE={evaluate(p, y)['ECE']:.4f}", xlabel="Mean confidence",
           ylabel="Observed accuracy", xlim=(0, 1), ylim=(0, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="Same folder used by prepare_data.py")
    ap.add_argument("--seed", type=int, default=42, help="Seed of the fixed 500-image batches")
    args = ap.parse_args()
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True, "grid.alpha": .22})
    _, cache, results = project_paths(args.root)
    out = results / f"part3_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    Z, Y, C, provenance = load_logits(cache)

    # Per corruption x severity, all 10,000 images.
    rows = [{"Corruption": c, "Severity": s, **evaluate(softmax(C[c][s - 1]), Y)}
            for c in CORRUPTIONS for s in SEVERITIES]
    per_corruption = pd.DataFrame(rows)
    per_corruption.to_csv(out / "part3_per_corruption.csv", index=False)

    # Pool the five corruptions at each severity.
    P = {0: (softmax(Z), Y)}
    for s in SEVERITIES:
        P[s] = (np.concatenate([softmax(C[c][s - 1]) for c in CORRUPTIONS]), np.tile(Y, len(CORRUPTIONS)))

    # 3.1: the same 20 batches of 500 indices for every corruption and severity.
    batches = np.random.RandomState(args.seed).permutation(len(Y)).reshape(-1, 500)
    conf_groups, acc_groups = [], []
    for s in (0,) + SEVERITIES:
        probs, _ = P[s]
        conf_groups.append(probs.max(1))
        parts = [probs] if s == 0 else np.split(probs, len(CORRUPTIONS))
        acc_groups.append(np.concatenate([(p.argmax(1)[batches] == Y[batches]).mean(axis=1) for p in parts]))

    pooled = pd.DataFrame([{"Severity": s, **evaluate(p, y)} for s, (p, y) in P.items()])
    pooled.to_csv(out / "part3_pooled_metrics.csv", index=False)
    summary = pooled[["Severity", "Accuracy", "MeanConf", "Gap", "ECE", "AdaptiveTopLabel",
                      "Risk@60", "Risk@80", "AUROC"]].copy()
    summary.insert(2, "MedianConf", [float(np.median(c)) for c in conf_groups])
    summary.insert(3, "MedianBatchAcc", [float(np.median(a)) for a in acc_groups])
    summary.insert(4, "MedianConf_minus_MedianBatchAcc", summary.MedianConf - summary.MedianBatchAcc)
    summary["ShareConf>=0.9"] = [float((c >= .9).mean()) for c in conf_groups]
    summary.to_csv(out / "part3_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))

    # Figure 3.1: boxplots.
    labels = ["Clean"] + [f"Sev {s}" for s in SEVERITIES]
    colors = [SEVERITY_COLORS[s] for s in (0,) + SEVERITIES]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, groups in zip(axes, (conf_groups, acc_groups)):
        bp = ax.boxplot(groups, showfliers=False, patch_artist=True)
        for box, color in zip(bp["boxes"], colors):
            box.set(facecolor=color, edgecolor=color, alpha=.65)
        for key in ("whiskers", "caps", "medians"):
            for art in bp[key]:
                art.set_color(PALETTE["charcoal"])
        ax.set_xticks(range(1, 7), labels)
        ax.set(xlabel="Shift intensity", ylim=(0, 1.02))
    axes[0].set(title="Sample confidence", ylabel="Max softmax probability")
    axes[1].set(title="Fixed batches of 500, separately per corruption", ylabel="Batch accuracy")
    fig.suptitle("Part 3 — Raw confidence; outliers hidden for readability")
    save(fig, figs, "part3_boxplots")

    # Figure 3.2: reliability diagrams.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, s in zip(axes.flat, (0,) + SEVERITIES):
        draw_reliability(ax, *P[s], "Clean" if s == 0 else f"Severity {s}", SEVERITY_COLORS[s])
    fig.suptitle("Part 3 — Raw-confidence reliability; pooled corruptions")
    save(fig, figs, "part3_reliability")

    # Figure 3.3: Risk-Coverage.
    fig, ax = plt.subplots(figsize=(8, 5))
    for s, (p, y) in P.items():
        coverage, risk = risk_coverage(p, y)
        ax.plot(coverage, risk, color=SEVERITY_COLORS[s], label="Clean" if s == 0 else f"Severity {s}")
    ax.set(xlabel="Coverage", ylabel="Risk (error rate)", xlim=(0, 1))
    ax.legend()
    ax.set_title("Part 3 — Raw-confidence Risk–Coverage; pooled corruptions")
    save(fig, figs, "part3_risk_coverage")

    manifest = {"part": 3, "batch_seed": args.seed, "bins": BINS, "corruptions": CORRUPTIONS,
                "severities": SEVERITIES, "provenance": json.loads(provenance),
                "aggregation": "Per-corruption metrics on all 10K images; trends pool the five corruptions per severity."}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Results:", out)


if __name__ == "__main__":
    main()
