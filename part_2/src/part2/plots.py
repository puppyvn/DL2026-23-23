"""Figures for Part 2, in the team style: bold figure titles, light grid, reliability bars in blue with pink
(overconfidence) and teal (underconfidence) gaps."""
import matplotlib
matplotlib.use("Agg")                               # scripts save figures; no window needed
import matplotlib.pyplot as plt
import numpy as np

from .metrics import N_BINS, reliability_bins, risk_coverage

STYLE = {"figure.dpi": 110, "axes.grid": True, "grid.alpha": .25, "font.size": 10, "savefig.dpi": 150,
         "savefig.bbox": "tight", "figure.titlesize": 13, "figure.titleweight": "bold", "axes.titlesize": 11,
         "axes.edgecolor": "black", "legend.fontsize": 8}
COLOR = {"acc": "#1f77b4", "over": "#f1b3b4", "under": "#238584", "diag": "black"}
T_COLOR = {0.5: "#c0392b", 1.0: "#333333", 2.0: "#2166ac"}
T_NAME = {0.5: "T = 0.5 (sharper)", 1.0: "T = 1 (baseline)", 2.0: "T = 2 (softer)"}


def apply_style():
    plt.rcParams.update(STYLE)


def plot_reliability(ax, conf, correct, title=None, n_bins=N_BINS):
    """Blue bar = accuracy of the bin. Pink = overconfidence gap (bar below y = x).
    Teal = underconfidence gap (bar above y = x). Black dot = (mean confidence, accuracy) of the bin."""
    bc, ba, cnt = reliability_bins(conf, correct, n_bins)
    w = 1 / n_bins; centers = (np.arange(n_bins) + .5) * w; ok = cnt > 0
    ax.bar(centers[ok], ba[ok], width=w * .95, color=COLOR["acc"], edgecolor="k", lw=.4, label="accuracy in bin")
    ax.bar(centers[ok], np.where(bc[ok] > ba[ok], bc[ok] - ba[ok], 0), bottom=ba[ok], width=w * .95,
           color=COLOR["over"], edgecolor="white", lw=.4, label="over-conf gap (conf > acc)")
    ax.bar(centers[ok], np.where(ba[ok] > bc[ok], ba[ok] - bc[ok], 0), bottom=bc[ok], width=w * .95,
           color=COLOR["under"], edgecolor="white", lw=.4, label="under-conf gap (acc > conf)")
    ax.plot(bc[ok], ba[ok], "o", color="k", ms=3, label="(mean conf, acc) of bin")
    ax.plot([0, 1], [0, 1], "--", color=COLOR["diag"], lw=1, label="y = x")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    if title:
        ax.set_title(title, fontsize=9)


def reliability_figure(records, acc, ece, ace, path):
    """One panel per temperature, same axes."""
    temps = list(records)
    fig, axs = plt.subplots(1, len(temps), figsize=(15, 4.8), sharey=True)
    for ax, T in zip(axs, temps):
        r = records[T]
        plot_reliability(ax, r["conf"], r["correct"],
                         title=f"{T_NAME[T]}\nacc {100 * acc:.2f}%   ECE {100 * ece[T]:.2f}%   ACE {100 * ace[T]:.2f}%")
    axs[0].legend(loc="upper left", fontsize=7)
    for ax in axs[1:]:
        ax.set_ylabel("")
    fig.suptitle("Part 2 — Reliability Diagram after controlled temperature stress (clean CIFAR-10, ResNet-18)", y=1.02)
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


def risk_coverage_figure(records, acc, coverages, path):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for T, r in records.items():
        cov, risk = risk_coverage(r["score"], r["correct"])
        ax.plot(cov, risk, color=T_COLOR[T], lw=2.4 if T == 1 else 1.6, ls="--" if T == 1 else "-", label=T_NAME[T])
    for c in coverages:
        ax.axvline(c, color="k", lw=.6, alpha=.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, (1 - acc) * 1.25)
    ax.set_xlabel("coverage (fraction of predictions accepted)"); ax.set_ylabel("risk (error rate on accepted)")
    ax.legend()
    fig.suptitle("Part 2 — Risk–Coverage (clean CIFAR-10, ResNet-18)")
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


PAL = {"green": "#1b7837", "blue": "#2166ac", "dark": "#333333", "orange": "#d6600a", "red": "#c0392b",
       "purple": "#6a3d9a", "gray": "#8c8c8c"}
T_MARKER = {0.5: ">", 1.0: "o", 2.0: "<"}


def shades(hex_color, n, lo=.35, hi=1.0):
    """n shades of one hue, light to dark, for ordered series such as noise levels."""
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return [tuple(1 - a * (1 - rgb)) for a in np.linspace(lo, hi, n)]


def plot_rc(ax, score, correct, **kw):
    cov, risk = risk_coverage(score, correct)
    ax.plot(cov, risk, **kw)
    ax.set_xlabel("coverage (fraction of predictions accepted)"); ax.set_ylabel("risk (error rate on accepted)")
    ax.set_xlim(0, 1)
