"""Report Experiments 3 (depth) and 4 (architecture): raw confidence (T = 1) of ten public CIFAR-10 checkpoints.

Uses the test-set logits of ResNet-18/34/50, VGG-11/13/16/19 and DenseNet-121/161/169 stored in
data/clean_logits.npz (huyvnphan/PyTorch_CIFAR10, evaluated unchanged). For every model: accuracy, ECE
(15 equal-width bins), ACE (15 equal-mass bins) and signed gap, each with a 95% percentile bootstrap interval
(1,000 resamples, seed 0, the same resampled images for every model). Within each family, consecutive depths are
compared with paired bootstrap intervals of the difference.

    python experiments/depth_architecture.py   -> results/tables/p2_depth_architecture.csv
                                                   results/tables/p2_depth_differences.csv
                                                   results/figures/fig_p2_depth_architecture.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import config, load
from part2.metrics import METRICS, ci, paired_bootstrap, predict
from part2.plots import PAL

SHOWN = ("acc", "ece", "ace", "gap")
FAMILIES = {"ResNet": ["ResNet_18", "ResNet_34", "ResNet_50"],
            "VGG": ["VGG_11", "VGG_13", "VGG_16", "VGG_19"],
            "DenseNet": ["DenseNet_121", "DenseNet_161", "DenseNet_169"]}
FAMILY_COLOR = {"ResNet": PAL["blue"], "VGG": PAL["orange"], "DenseNet": PAL["green"]}


def main(B=1000):
    _, labels, _ = load()
    archive = np.load(config.RAW_LOGITS)
    assert np.array_equal(archive["labels"], labels)
    rec = {name: predict(archive[name], labels, 1.0) for models in FAMILIES.values() for name in models}
    boot = paired_bootstrap(rec, metrics={m: METRICS[m] for m in SHOWN}, B=B, seed=config.SEED)

    rows = []
    for family, models in FAMILIES.items():
        for name in models:
            row = {"family": family, "model": name.replace("_", "-")}
            for m in SHOWN:
                lo, hi = ci(boot[name][m])
                row.update({m: METRICS[m](rec[name]), f"{m}_lo": lo, f"{m}_hi": hi})
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(config.TABLES / "p2_depth_architecture.csv", index=False)

    diffs = []
    for family, models in FAMILIES.items():
        for shallow, deep in zip(models, models[1:]):
            row = {"family": family, "comparison": f"{deep} - {shallow}".replace("_", "-")}
            for m in SHOWN:
                d = boot[deep][m] - boot[shallow][m]
                lo, hi = ci(d)
                row.update({f"d_{m}": METRICS[m](rec[deep]) - METRICS[m](rec[shallow]),
                            f"d_{m}_lo": lo, f"d_{m}_hi": hi, f"d_{m}_ci_excludes_0": bool(lo > 0 or hi < 0)})
            diffs.append(row)
    pd.DataFrame(diffs).to_csv(config.TABLES / "p2_depth_differences.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, m, title in zip(axes, ("ece", "ace"), ("ECE (%)", "ACE (%)")):
        for family in FAMILIES:
            t = table[table.family == family]
            ax.errorbar(100 * t.acc, 100 * t[m],
                        xerr=[100 * (t.acc - t.acc_lo), 100 * (t.acc_hi - t.acc)],
                        yerr=[100 * (t[m] - t[f"{m}_lo"]), 100 * (t[f"{m}_hi"] - t[m])],
                        fmt="o", ms=6, capsize=2, lw=1, color=FAMILY_COLOR[family], label=family)
            for _, r in t.iterrows():
                ax.annotate(r.model.split("-")[1], (100 * r.acc, 100 * r[m]), textcoords="offset points",
                            xytext=(4, 4), fontsize=7, color="#555555")
        ax.set(xlabel="Accuracy (%)", ylabel=title)
    axes[0].legend(fontsize=8)
    fig.suptitle("Depth and architecture on clean CIFAR-10 (T = 1); error bars: 95% bootstrap intervals")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig_p2_depth_architecture.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    show = table.copy()
    for m in SHOWN:
        show[m] = (100 * show[m]).round(2)
    print(show[["family", "model", *SHOWN]].to_string(index=False))
    print("tables: p2_depth_architecture.csv, p2_depth_differences.csv; figure: fig_p2_depth_architecture.png")


if __name__ == "__main__":
    main()
