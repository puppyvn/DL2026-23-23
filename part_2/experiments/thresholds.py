"""Practical meaning: a fixed confidence threshold versus a fixed coverage.

"Accept the 80% most confident predictions" depends only on the order of the confidence, so it behaves almost
the same at every T. "Accept if confidence >= 0.9" depends on the size of the confidence, so the same rule gives
very different systems when only T changes.

    python experiments/thresholds.py         -> results/figures/fig_p2_thresholds.png
                                                 results/tables/p2_thresholds.csv
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import TS, config, load
from part2.plots import T_COLOR, T_NAME


def main():
    _, _, rec = load()
    corr, n = rec[1.0]["correct"], len(rec[1.0]["correct"])
    n_err = (1 - corr).sum()
    rows = []
    for T in TS:
        c = rec[T]["conf"]
        top80 = np.argsort(-rec[T]["score"], kind="stable")[:int(.8 * n)]
        keep = c >= 0.9
        rows.append({"T": T, "errors rejected at 80% coverage %": 100 * (1 - (1 - corr[top80]).sum() / n_err),
                     "accepted if conf >= 0.9 %": 100 * keep.mean(),
                     "risk if conf >= 0.9 %": 100 * (1 - corr[keep].mean()) if keep.any() else np.nan,
                     "threshold giving 80% coverage": float(np.sort(c)[::-1][int(.8 * n) - 1]),
                     "max confidence": float(c.max())})
    table = pd.DataFrame(rows).set_index("T")
    table.to_csv(config.TABLES / "p2_thresholds.csv")
    print(table.round(4).to_string())

    grid = np.linspace(0.1, 1.0, 400)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.3))
    for T in TS:
        c = rec[T]["conf"]
        axs[0].plot(grid, [(c >= t).mean() for t in grid], color=T_COLOR[T], label=T_NAME[T])
        axs[1].plot(grid, [100 * (1 - corr[c >= t].mean()) if (c >= t).any() else np.nan for t in grid],
                    color=T_COLOR[T], label=T_NAME[T])
    for ax in axs:
        ax.axvline(0.9, color="k", lw=.7, ls=":"); ax.set_xlim(0.1, 1)
        ax.set_xlabel("confidence threshold (accept if conf >= threshold)")
    axs[0].set_ylabel("coverage (fraction accepted)"); axs[0].set_title("Same ranking, but a fixed threshold accepts very different amounts")
    axs[1].set_ylabel("risk on accepted (%)"); axs[1].set_title("Risk at a fixed threshold depends on T")
    axs[0].legend(fontsize=8)
    fig.suptitle("Part 2 — A fixed confidence threshold behaves differently at each T")
    plt.tight_layout(); fig.savefig(config.FIGURES / "fig_p2_thresholds.png"); plt.close(fig)


if __name__ == "__main__":
    main()
