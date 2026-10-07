"""Positive control: is Risk-Coverage able to see a broken ranking?

Gaussian noise of size sigma is added to the T = 1 confidence (20 repetitions per sigma), which scrambles the order
of the predictions while leaving their average level unchanged. A sensitive tool must report a rising risk as sigma
grows, towards the random ranking. The oracle (every correct prediction first) is the lower bound.

    python experiments/positive_control.py   -> results/figures/fig_p2_positive_control.png
                                                 results/tables/p2_positive_control.csv
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import TS, config, load
from part2.metrics import auroc, e_aurc, oracle_score, risk_at, risk_coverage, spearman
from part2.plots import PAL, T_COLOR, T_NAME, plot_rc, shades

SIGMAS, REPS = [0.003, 0.01, 0.03, 0.1, 0.3], 20


def main():
    _, _, rec = load()
    corr, n = rec[1.0]["correct"], len(rec[1.0]["correct"])
    rng = np.random.default_rng(config.SEED)
    runs = []

    def add(condition, score):
        runs.append({"condition": condition, "Risk@80%": risk_at(score, corr, .8), "Risk@60%": risk_at(score, corr, .6),
                     "E-AURC": e_aurc(score, corr), "AUROC": auroc(score, corr),
                     "Spearman vs T=1": spearman(rec[1.0]["score"], score)})

    add("oracle", oracle_score(corr))
    for T in TS:
        add(f"T = {T:g}", rec[T]["score"])
    example = {}
    for s in SIGMAS:
        for rep in range(REPS):
            score = rec[1.0]["conf"] + s * rng.standard_normal(n)
            add(f"T = 1 + noise sigma={s}", score)
            example.setdefault(s, score)
    for _ in range(REPS):
        add("random", rng.permutation(n).astype(float))

    runs = pd.DataFrame(runs)
    table = runs.groupby("condition", sort=False).mean()
    table.loc["oracle", "Spearman vs T=1"] = np.nan          # the oracle score is mostly ties: rho means nothing
    table.to_csv(config.TABLES / "p2_positive_control.csv")
    print(table.round(4).to_string())

    err = 1 - corr.mean()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    cov_o, risk_o = risk_coverage(oracle_score(corr), corr)
    ax.plot(cov_o, risk_o, ":", color=PAL["green"], lw=1.8, label="oracle")
    ax.axhline(err, ls=":", color=PAL["gray"], lw=1.8, label="random")
    for color, s in zip(shades(PAL["orange"], len(SIGMAS)), SIGMAS):
        plot_rc(ax, example[s], corr, color=color, lw=1.3, label=f"T = 1 + noise sigma={s}")
    for T in TS:
        plot_rc(ax, rec[T]["score"], rec[T]["correct"], color=T_COLOR[T], lw=1.2, label=T_NAME[T])
    for c in config.COVERAGES:
        ax.axvline(c, color="k", lw=.6, alpha=.5)
    ax.set_ylim(0, err * 1.25); ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.suptitle("Part 2 — Positive control: breaking the ranking moves Risk–Coverage, changing T barely does")
    plt.tight_layout(); fig.savefig(config.FIGURES / "fig_p2_positive_control.png"); plt.close(fig)


if __name__ == "__main__":
    main()
