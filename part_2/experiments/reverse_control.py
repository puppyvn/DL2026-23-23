"""Reverse control: a perfectly calibrated confidence whose ranking is useless.

The temperature experiment shows calibration changing while ranking barely moves. This script shows the opposite
direction with two confidences built from the test labels (diagnostics, not usable methods):

- constant = accuracy: every image gets 93.07%. Perfectly calibrated, no information; every score is tied, so the
  order is set by tie-breaking alone (200 random tie-breaks are also reported).
- precision of the predicted class: a prediction of class k gets the accuracy of class k. Also exactly calibrated,
  with a little information left.

ECE is exactly 0 because each bin holds whole groups of identical confidence. ACE is not 0: equal-mass bins cut
across those groups, so each bin's accuracy fluctuates around the true value.

    python experiments/reverse_control.py    -> results/figures/fig_p2_reverse_control.png
                                                 results/tables/p2_reverse_control.csv
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import config, load
from part2.metrics import ace_score, auroc, e_aurc, ece_score, oracle_score, reliability_bins, risk_at, \
    risk_coverage, signed_gap
from part2.plots import PAL, T_COLOR, plot_rc


def main():
    _, _, rec = load()
    corr, n = rec[1.0]["correct"], len(rec[1.0]["correct"])
    acc = corr.mean()
    pred = rec[1.0]["pred"]
    precision = np.array([corr[pred == k].mean() for k in range(10)])
    controls = {"model, T = 1": (rec[1.0]["conf"], rec[1.0]["score"]),
                "constant = accuracy": (np.full(n, acc), np.full(n, acc)),
                "precision of predicted class": (precision[pred], precision[pred])}

    rows = [{"confidence": name, "ECE %": 100 * ece_score(c, corr), "ACE %": 100 * ace_score(c, corr),
             "signed gap %": 100 * signed_gap(c, corr), "Risk@80% %": 100 * risk_at(s, corr, .8),
             "Risk@60% %": 100 * risk_at(s, corr, .6), "E-AURC %": 100 * e_aurc(s, corr), "AUROC": auroc(s, corr)}
            for name, (c, s) in controls.items()]
    table = pd.DataFrame(rows).set_index("confidence")
    table.to_csv(config.TABLES / "p2_reverse_control.csv")
    print(table.round(3).to_string())
    tie = [100 * (1 - corr[np.random.default_rng(config.SEED + i).permutation(n)[:int(.8 * n)]].mean()) for i in range(200)]
    print(f"\nconstant confidence, 200 random tie-breaks: Risk@80% = {np.mean(tie):.2f}% "
          f"(95% of tie-breaks between {np.percentile(tie, 2.5):.2f}% and {np.percentile(tie, 97.5):.2f}%)")

    color = {"model, T = 1": T_COLOR[1.0], "constant = accuracy": PAL["purple"], "precision of predicted class": PAL["orange"]}
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    axs[0].plot([0, 1], [0, 1], "--", color="black", lw=1, label="y = x (perfect calibration)")
    for name, (c, s) in controls.items():
        bc, ba, cnt = reliability_bins(c, corr)
        ok = cnt >= 20                                                  # bins with at least 20 images
        axs[0].scatter(bc[ok], ba[ok], s=10 + 140 * cnt[ok] / cnt.max(), color=color[name], alpha=.8, edgecolor="white",
                       label=f"{name}: ECE {100 * ece_score(c, corr):.2f}%")
        plot_rc(axs[1], s, corr, color=color[name], lw=1.8, label=f"{name}: Risk@80% {100 * risk_at(s, corr, .8):.2f}%")
    axs[0].set_xlim(.5, 1); axs[0].set_ylim(.5, 1)
    axs[0].set_xlabel("mean confidence in bin"); axs[0].set_ylabel("accuracy in bin")
    axs[0].set_title("Calibration: both controls sit exactly on y = x (dot size = samples)"); axs[0].legend(fontsize=7, loc="upper left")
    cov_o, risk_o = risk_coverage(oracle_score(corr), corr)
    axs[1].plot(cov_o, risk_o, ":", color=PAL["green"], lw=1.4, label="oracle")
    axs[1].axhline(1 - acc, ls=":", color=PAL["gray"], lw=1.4, label="random")
    axs[1].axvline(.8, color="k", lw=.6, alpha=.5); axs[1].set_ylim(0, (1 - acc) * 1.25)
    axs[1].set_title("Ranking: the calibrated controls are (almost) as bad as random")
    axs[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2, frameon=False)
    fig.suptitle("Part 2 — Reverse control: perfect calibration does not imply useful ranking", y=1.02)
    plt.tight_layout(); fig.savefig(config.FIGURES / "fig_p2_reverse_control.png"); plt.close(fig)


if __name__ == "__main__":
    main()
