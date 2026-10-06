"""The stress test repeated on ten pretrained checkpoints from the same repository.

data/clean_logits.npz also holds the test-set logits of ResNet-34/50, VGG-11/13/16/19 and DenseNet-121/161/169.
For each model: signed gap, ECE, Risk@80%, Risk@60% and AUROC at T = 0.5, 1, 2 with 95% paired bootstrap
intervals, and the change versus T = 1. The ten models share one training recipe and were each trained once, so
they are ten related checkpoints, not ten independent repetitions.

    python experiments/robustness_10_models.py   -> results/figures/fig_p2_robustness_10_models.png
                                                     results/tables/p2_robustness_10_models.csv
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import TS, config, load
from part2.metrics import METRICS, ci, paired_bootstrap, predict, spearman
from part2.plots import PAL, T_COLOR, T_NAME

SHOWN = ("gap", "ece", "risk80", "risk60", "auroc")


def main(B=1000):
    _, labels, _ = load()
    archive = np.load(config.RAW_LOGITS)
    models = [k for k in archive.files if k != "labels"]
    assert np.array_equal(archive["labels"], labels)
    rows, t0 = [], time.time()
    for name in models:
        rec = {T: predict(archive[name], labels, T) for T in TS}
        boot = paired_bootstrap(rec, metrics={m: METRICS[m] for m in SHOWN}, B=B, seed=config.SEED)
        for T in TS:
            row = {"model": name.replace("_", "-"), "T": T, "accuracy": rec[T]["correct"].mean()}
            for m in SHOWN:
                lo, hi = ci(boot[T][m])
                row.update({m: METRICS[m](rec[T]), f"{m}_lo": lo, f"{m}_hi": hi})
                if T != 1.0:
                    dlo, dhi = ci(boot[T][m] - boot[1.0][m])
                    row.update({f"d_{m}": METRICS[m](rec[T]) - METRICS[m](rec[1.0]), f"d_{m}_lo": dlo, f"d_{m}_hi": dhi})
            row["spearman_vs_T1"] = spearman(rec[1.0]["score"], rec[T]["score"])
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(config.TABLES / "p2_robustness_10_models.csv", index=False)
    print(f"{len(models)} models x {B} paired bootstrap resamples: {time.time() - t0:.0f}s")

    for T in (0.5, 2.0):
        sub = table[table["T"] == T]
        right = (sub["gap"] > 0).sum() if T == 0.5 else (sub["gap"] < 0).sum()
        sig = lambda m: int(((sub[f"d_{m}_lo"] > 0) | (sub[f"d_{m}_hi"] < 0)).sum())
        print(f"T={T:g}: created direction in {right}/10 models | ECE change {100 * sub['d_ece'].min():+.2f}..{100 * sub['d_ece'].max():+.2f} pp "
              f"(CI excludes 0 in {sig('ece')}/10) | Risk@80% change {100 * sub['d_risk80'].min():+.3f}..{100 * sub['d_risk80'].max():+.3f} pp "
              f"(CI excludes 0 in {sig('risk80')}/10) | AUROC change {sub['d_auroc'].min():+.4f}..{sub['d_auroc'].max():+.4f} "
              f"(CI excludes 0 in {sig('auroc')}/10) | min Spearman {sub['spearman_vs_T1'].min():.4f}")

    # grouped bars: one group per model, one bar per T; top = signed gap, bottom = Risk@80%
    names = [m.replace("_", "-") for m in models]
    x = np.arange(len(names)) + np.repeat([0, .6, 1.2], [3, 4, 3])       # gaps between ResNet / VGG / DenseNet
    w = .26
    fig, axs = plt.subplots(2, 1, figsize=(13, 8.2), sharex=True, gridspec_kw={"height_ratios": [1.25, 1]})
    for ax, m, ylab in [(axs[0], "gap", "signed gap = mean conf − accuracy (%)"), (axs[1], "risk80", "Risk@80% (%)")]:
        for k, T in enumerate(TS):
            sub = table[table["T"] == T].set_index("model").loc[names]
            v = 100 * sub[m].values
            err = np.vstack([v - 100 * sub[f"{m}_lo"].values, 100 * sub[f"{m}_hi"].values - v])
            ax.bar(x + (k - 1) * w, v, w, yerr=err, color=T_COLOR[T], alpha=.9, edgecolor="k", lw=.5,
                   error_kw={"elinewidth": .8, "capsize": 2.5}, label=T_NAME[T])
        ax.set_ylabel(ylab); ax.axhline(0, color="k", lw=.8)
        for b in ((x[2] + x[3]) / 2, (x[6] + x[7]) / 2):
            ax.axvline(b, color="#bbbbbb", lw=.8)
    lo_y, hi_y = 100 * table["gap_lo"].min() - 3, 100 * table["gap_hi"].max() + 4
    axs[0].axhspan(0, hi_y, color=PAL["red"], alpha=.08, lw=0); axs[0].axhspan(lo_y, 0, color=PAL["blue"], alpha=.08, lw=0)
    axs[0].set_ylim(lo_y, hi_y)
    axs[0].text(x[-1] + .5, hi_y - .8, "above 0: overconfident", ha="right", va="top", fontsize=9, color=PAL["red"])
    axs[0].text(x[-1] + .5, lo_y + .8, "below 0: underconfident", ha="right", va="bottom", fontsize=9, color=PAL["blue"])
    axs[0].set_title("Calibration: changing T pushes every model far, from overconfident (T = 0.5) to underconfident (T = 2)", pad=24)
    axs[1].set_title("Selective risk at 80% coverage: the three bars of each model are almost the same height")
    axs[1].set_ylim(0, 100 * table["risk80_hi"].max() * 1.15)
    for xm, fam in ((x[1], "ResNet"), (x[4] + .5, "VGG"), (x[8], "DenseNet")):
        axs[0].text(xm, hi_y + .3, fam, ha="center", va="bottom", fontsize=10, fontweight="bold", color="#444")
    axs[1].set_xticks(x); axs[1].set_xticklabels(names, rotation=30, ha="right")
    axs[0].legend(loc="lower left", ncol=3, fontsize=8)
    fig.suptitle("Part 2 — The stress test on ten public CIFAR-10 checkpoints (bars = value at each T, error bars = 95% bootstrap CI)", y=1.0)
    plt.tight_layout(); fig.savefig(config.FIGURES / "fig_p2_robustness_10_models.png"); plt.close(fig)


if __name__ == "__main__":
    main()
