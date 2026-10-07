"""Rebuild the two figures made only for the report (run from the repository root):
    python report/make_report_figures.py  -> report/figures/fig_rq2_shift_summary.pdf, fig_depth_families.pdf
Inputs: part_4/output/csv, part_4/output/shift_run/csv, part_2/results/tables/p2_depth_architecture.csv."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("report/figures"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
C = {"Baseline": "#333333", "Temperature Scaling": "#2166AC", "Dirichlet": "#D6600A"}
M = {"Baseline": "o", "Temperature Scaling": "s", "Dirichlet": "^"}
L = {"Baseline": "Raw ($T=1$)", "Temperature Scaling": "Temperature Scaling", "Dirichlet": "Dirichlet"}

clean = pd.read_csv("part_4/output/csv/part4_clean_seed42.csv").set_index("Method")
pc = pd.read_csv("part_4/output/shift_run/csv/part4_shift_per_corruption.csv")
fig, axes = plt.subplots(1, 3, figsize=(10, 3.1))
for ax, (m, title, scale) in zip(axes, [("Accuracy", "Accuracy (%)", 100), ("ECE", "ECE (%)", 100), ("NLL", "NLL (nats)", 1)]):
    for meth, off in zip(C, (-0.08, 0, 0.08)):
        g = pc[pc.Method == meth].groupby("Severity")[m]
        x = np.arange(6) + off
        mean = np.r_[clean.loc[meth, m], g.mean().values] * scale
        ax.plot(x, mean, marker=M[meth], color=C[meth], lw=1.6, ms=5, label=L[meth])
        ax.vlines(x[1:], g.min().values * scale, g.max().values * scale, color=C[meth], lw=0.9, alpha=.45)
    ax.set_xticks(range(6), ["Clean", "1", "2", "3", "4", "5"]); ax.set_xlabel("Corruption severity"); ax.set_title(title)
axes[0].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig_rq2_shift_summary.pdf", bbox_inches="tight")

d = pd.read_csv("part_2/results/tables/p2_depth_architecture.csv")
FC = {"ResNet": "#2166AC", "VGG": "#D6600A", "DenseNet": "#1B7837"}
fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
pos, ticks, labs, xs = 0, [], [], {}
for fam in FC:
    t = d[d.family == fam]; xs[fam] = np.arange(pos, pos + len(t)); pos += len(t) + 1
    ticks += list(xs[fam]); labs += [m.replace("DenseNet", "DN").replace("ResNet", "RN") for m in t.model]
for ax, (m, title) in zip(axes, [("acc", "Accuracy (%)"), ("ece", "ECE (%)"), ("ace", "ACE (%)")]):
    for fam in FC:
        t = d[d.family == fam]
        ax.errorbar(xs[fam], 100 * t[m], yerr=[100 * (t[m] - t[m + "_lo"]), 100 * (t[m + "_hi"] - t[m])],
                    color=FC[fam], marker="o", ms=5, lw=1.4, capsize=2, label=fam)
    ax.set_xticks(ticks, labs, rotation=45, ha="right", fontsize=7.5); ax.set_title(title)
axes[0].legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig_depth_families.pdf", bbox_inches="tight")
print("written to", OUT)
