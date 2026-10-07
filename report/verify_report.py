"""Check every number in the final report against the repository outputs.

Run from the repository root (after README Steps 1 and 5):
    python report/verify_report.py                                   # committed results
    python report/verify_report.py part_4/src/artifacts/results/all_seed42_<time>   # also your fresh Part 4 run
Prints a pass count per table and every FAIL with the report value and the value recomputed from the repo.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "part_2" / "src"))
from part2.metrics import predict, ece_score, ace_score, reliability_bins, risk_at, auroc  # noqa: E402

P4RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else None
results = []


def check(where, label, reported, value, decimals=None, tol=None):
    """reported: number as printed in the report; value: recomputed. Pass if it rounds to the reported value."""
    if tol is None:
        d = decimals if decimals is not None else (len(str(reported).split(".")[1]) if "." in str(reported) else 0)
        tol = 0.5 * 10 ** (-d) + 1e-9
    ok = abs(float(reported) - float(value)) <= tol
    results.append((where, label, reported, value, ok))


def show():
    bad = [r for r in results if not r[4]]
    by = {}
    for r in results:
        by.setdefault(r[0], [0, 0])
        by[r[0]][0] += 1
        by[r[0]][1] += r[4]
    for k, (n, good) in by.items():
        print(f"{k:42s} {good:3d}/{n:3d} OK")
    print(f"\nTOTAL {len(results) - len(bad)}/{len(results)} checks pass")
    for r in bad:
        print(f"FAIL {r[0]} | {r[1]}: report {r[2]} vs repo {r[3]}")


pc = lambda x: 100 * x

# ------------------------------------------------------------------ Part 1: Table 4, Exp 1 text
a = pd.read_csv(ROOT / "part_1/results/experiments/summary_exp_a.csv")
T4 = {"0.50": (92.36, 2.16, 2.05, 2.05), "0.75": (92.71, 2.04, 2.02, 2.02), "1.00": (92.35, 2.40, 2.32, 2.32),
      "1.50": (92.92, 2.16, 2.11, 2.11), "2.00": (91.51, 1.94, 1.93, 1.93)}
for s, (acc, ece, ace, gap) in T4.items():
    r = a[a.label.str.contains(s)].iloc[0]
    for name, rep, val in (("acc", acc, r.acc), ("ECE", ece, r.ece), ("ACE", ace, r.ace), ("gap", gap, r.gap)):
        check("Table 4 (Exp 1, width)", f"{s}x {name}", f"{rep:.2f}", pc(val))

# Table 2: parameter counts of the width-scaled ResNet-50
try:
    sys.path.insert(0, str(ROOT / "part_1"))
    import torch  # noqa: F401
    from src.model import _build_scaled_resnet
    for s, rep in ((0.5, 5.9), (0.75, 13.3), (1.0, 23.5), (1.5, 52.9), (2.0, 93.9)):
        m = _build_scaled_resnet("resnet50", s, 10) if _build_scaled_resnet.__code__.co_argcount >= 3 else _build_scaled_resnet(s)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        check("Table 2 (width params, M)", f"{s}x", f"{rep:.1f}", n)
except Exception as e:  # noqa: BLE001
    print("Table 2 parameter check skipped:", repr(e))

# ------------------------------------------------------------------ Part 2: Table 6, Exp 2 text, Table 9, Table 11, Section 7
arch = np.load(ROOT / "part_2/data/clean_logits.npz")
labels = arch["labels"]
rec = {T: predict(arch["ResNet_18"], labels, T) for T in (0.5, 1.0, 2.0)}
T6 = {0.5: (98.44, 5.37, 5.37, 5.37, 0.70, 1.36, 0.9014), 1.0: (94.85, 1.78, 2.02, 2.41, 0.82, 1.44, 0.8959),
      2.0: (68.85, -24.22, 24.22, 24.22, 0.88, 1.46, 0.8914)}
for T, (mc, gap, ece, ace, r60, r80, au) in T6.items():
    r = rec[T]
    c, k = r["conf"], r["correct"]
    check("Table 6 (Exp 2)", f"T={T} accuracy", "93.07", pc(k.mean()))
    check("Table 6 (Exp 2)", f"T={T} mean conf", f"{mc:.2f}", pc(c.mean()))
    check("Table 6 (Exp 2)", f"T={T} gap", f"{gap:.2f}", pc(c.mean() - k.mean()))
    check("Table 6 (Exp 2)", f"T={T} ECE", f"{ece:.2f}", pc(ece_score(c, k)))
    check("Table 6 (Exp 2)", f"T={T} ACE", f"{ace:.2f}", pc(ace_score(c, k)))
    check("Table 6 (Exp 2)", f"T={T} Risk@60", f"{r60:.2f}", pc(risk_at(c, k, 0.6)))
    check("Table 6 (Exp 2)", f"T={T} Risk@80", f"{r80:.2f}", pc(risk_at(c, k, 0.8)))
    check("Table 6 (Exp 2)", f"T={T} AUROC", f"{au:.4f}", auroc(c, k))

# exchanged images / accepted errors in the top-80% sets (Exp 2 text)
def top(c, frac=0.8):
    return set(np.argsort(-c, kind="stable")[: int(round(frac * len(c)))])
base = top(rec[1.0]["conf"])
for T, exch, errs in ((0.5, 44, 109), (2.0, 35, 117)):
    s = top(rec[T]["conf"])
    check("Exp 2 text", f"T={T} images exchanged in top-80%", exch, len(s - base), tol=0)
    check("Exp 2 text", f"T={T} accepted errors", errs, int((~rec[T]["correct"].astype(bool))[list(s)].sum()), tol=0)
check("Exp 2 text", "T=1 accepted errors", 115, int((~rec[1.0]["correct"].astype(bool))[list(base)].sum()), tol=0)
d = pd.read_csv(ROOT / "part_2/results/tables/p2_diff_vs_T1.csv")
for T in (0.5, 2.0):
    x = d[d["T vs T=1"] == T].set_index("metric")
    check("Exp 2 text", f"T={T} Risk@80 change CI includes 0", 0, int(x.loc["Risk@80%", "ci_excludes_0"]), tol=0)
    check("Exp 2 text", f"T={T} Risk@60 change CI excludes 0", 1, int(x.loc["Risk@60%", "ci_excludes_0"]), tol=0)

# Section 7: reliability bins at T = 0.5 (top bin) and T = 2 (two most populated bins)
bc, ba, cnt = reliability_bins(rec[0.5]["conf"], rec[0.5]["correct"])
check("Section 7 (bins)", "T=0.5 top-bin count", 9461, cnt[-1], tol=0)
check("Section 7 (bins)", "T=0.5 top-bin conf", "99.85", pc(bc[-1]))
check("Section 7 (bins)", "T=0.5 top-bin acc", "95.4", pc(ba[-1]))
bc, ba, cnt = reliability_bins(rec[2.0]["conf"], rec[2.0]["correct"])
i1, i2 = sorted(np.argsort(cnt)[-2:])
for i, (n, c, a_) in zip((i1, i2), ((3679, "71.1", "98.1"), (3993, "75.9", "99.0"))):
    check("Section 7 (bins)", f"T=2 bin {i} count", n, cnt[i], tol=0)
    check("Section 7 (bins)", f"T=2 bin {i} conf", c, pc(bc[i]))
    check("Section 7 (bins)", f"T=2 bin {i} acc", a_, pc(ba[i]))

# Table 9 (depth/architecture)
da = pd.read_csv(ROOT / "part_2/results/tables/p2_depth_architecture.csv")
T9 = {"ResNet-18": (93.07, 2.02, 2.41, 1.78, 1.32, 2.22), "ResNet-34": (93.33, 2.63, 3.26, 2.59, 2.17, 3.03),
      "ResNet-50": (93.65, 2.23, 2.91, 2.22, 1.81, 2.62), "VGG-11": (92.39, 1.50, 1.92, 1.24, 0.79, 1.70),
      "VGG-13": (94.21, 1.11, 1.74, 0.59, 0.20, 0.97), "VGG-16": (94.00, 1.52, 2.70, 1.47, 1.09, 1.91),
      "VGG-19": (93.95, 2.13, 2.88, 2.06, 1.62, 2.48), "DenseNet-121": (94.06, 2.02, 2.82, 0.53, 0.14, 0.93),
      "DenseNet-161": (94.07, 2.10, 2.90, 0.84, 0.43, 1.26), "DenseNet-169": (94.05, 2.37, 2.96, 0.68, 0.28, 1.09)}
for m, vals in T9.items():
    r = da[da.model == m].iloc[0]
    for name, rep, val in zip(("acc", "ECE", "ACE", "gap", "gap CI lo", "gap CI hi"), vals,
                              (r.acc, r.ece, r.ace, r.gap, r.gap_lo, r.gap_hi)):
        check("Table 9 (Exp 3-4)", f"{m} {name}", f"{rep:.2f}", pc(val))
dd = pd.read_csv(ROOT / "part_2/results/tables/p2_depth_differences.csv")
r = dd[dd.comparison == "ResNet-34 - ResNet-18"].iloc[0]
check("Exp 3 text", "ResNet-34 - 18 dECE", "0.60", pc(r.d_ece))
check("Exp 3 text", "dECE CI lo", "0.22", pc(r.d_ece_lo))
check("Exp 3 text", "dECE CI hi", "1.07", pc(r.d_ece_hi))
check("Exp 3 text", "dAcc CI excludes 0 (False)", 0, int(r.d_acc_ci_excludes_0), tol=0)
check("Exp 3 text", "all 10 gap CIs exclude 0", 10, int((da.gap_lo > 0).sum()), tol=0)

# Table 11 + Section 7 RQ1 profile
cr = pd.read_csv(ROOT / "part_2/results/tables/p2_confidence_ranges.csv")
for i, (share, mc, acc, gap) in enumerate(((88.8, 98.11, 97.24, 0.87), (11.2, 69.07, 60.11, 8.97), (100.0, 94.85, 93.07, 1.78))):
    r = cr.iloc[i]
    check("Table 11 (RQ1 profile)", f"{r['range']} share", f"{share:.1f}", pc(r.share))
    check("Table 11 (RQ1 profile)", f"{r['range']} mean conf", f"{mc:.2f}", pc(r.mean_conf))
    check("Table 11 (RQ1 profile)", f"{r['range']} accuracy", f"{acc:.2f}", pc(r.accuracy))
    check("Table 11 (RQ1 profile)", f"{r['range']} gap", f"{gap:.2f}", pc(r.gap))
cg = pd.read_csv(ROOT / "part_2/results/tables/p2_class_gap.csv").set_index("predicted_class")
for c, v in (("cat", 4.9), ("dog", 4.4), ("bird", 3.2)):
    check("Section 7 (class gaps)", c, f"{v:.1f}", pc(cg.loc[c, "gap"]))
for c in ("automobile", "frog", "horse"):
    check("Section 7 (class gaps)", f"{c} in [-0.8,-0.4]", -0.6, pc(cg.loc[c, "gap"]), tol=0.25)
nf = pd.read_csv(ROOT / "part_2/results/tables/p2_ece_noise_floor.csv").set_index("metric")
check("Section 7 (noise floor)", "perfect mean ECE", "0.47", pc(nf.loc["ECE", "perfect_mean"]))
check("Section 7 (noise floor)", "perfect p95 ECE", "0.67", pc(nf.loc["ECE", "perfect_p95"]))
rc = pd.read_csv(ROOT / "part_2/results/tables/p2_reverse_control.csv")
r = rc[rc.confidence == "constant = accuracy"].iloc[0]
check("Section 7 (constant conf.)", "ECE ~0", 0, r["ECE %"], tol=1e-6)
check("Section 7 (constant conf.)", "AUROC", "0.5", r["AUROC"])
check("Section 7 (constant conf.)", "Risk@80", "6.98", r["Risk@80% %"])

# ------------------------------------------------------------------ Part 3: Table 7, Exp 6 text
s3 = pd.read_csv(ROOT / "part_3/outputs/tables/part3_summary.csv")
T7 = [(93.07, 98.39, 93.0, 1.78, 2.02, 1.44, 0.896), (90.32, 98.34, 91.0, 3.42, 3.43, 2.62, 0.890),
      (84.88, 98.18, 86.8, 6.70, 6.70, 6.62, 0.865), (77.80, 97.90, 79.6, 11.49, 11.49, 13.01, 0.847),
      (71.03, 97.24, 76.5, 16.47, 16.47, 20.21, 0.826), (56.63, 95.22, 67.9, 29.05, 29.05, 36.70, 0.774)]
for sev, (acc, mc, mba, gap, ece, r80, au) in enumerate(T7):
    r = s3[s3.Severity == sev].iloc[0]
    check("Table 7 (Exp 6 raw)", f"sev{sev} acc", f"{acc:.2f}", pc(r.Accuracy))
    check("Table 7 (Exp 6 raw)", f"sev{sev} median conf", f"{mc:.2f}", pc(r.MedianConf))
    check("Table 7 (Exp 6 raw)", f"sev{sev} median batch acc", f"{mba:.1f}", pc(r.MedianBatchAcc))
    check("Table 7 (Exp 6 raw)", f"sev{sev} gap", f"{gap:.2f}", pc(r.Gap))
    check("Table 7 (Exp 6 raw)", f"sev{sev} ECE", f"{ece:.2f}", pc(r.ECE))
    check("Table 7 (Exp 6 raw)", f"sev{sev} Risk@80", f"{r80:.2f}", pc(r["Risk@80"]))
    check("Table 7 (Exp 6 raw)", f"sev{sev} AUROC", f"{au:.3f}", r.AUROC)
r5 = s3[s3.Severity == 5].iloc[0]
check("Exp 6 / Sec 7 text", "share conf>=0.9 at sev5", "62.2", pc(r5["ShareConf>=0.9"]))
check("Exp 6 / Sec 7 text", "accuracy sev5", "56.6", pc(r5.Accuracy))
for sev in range(2, 6):
    r = s3[s3.Severity == sev].iloc[0]
    check("Exp 6 text", f"gap == ECE at sev{sev}", f"{pc(r.ECE):.2f}", pc(r.Gap))

# ------------------------------------------------------------------ Part 4: Table 5, Table 8, Table 10, Exp 5/6/7 text
committed_sens = pd.read_csv(ROOT / "part_4/output/csv/part4_split_sensitivity.csv")
pool = pd.read_csv(ROOT / "part_4/output/shift_run/csv/part4_shift_pooled_metrics.csv")
mean = pd.read_csv(ROOT / "part_4/output/shift_run/csv/part4_shift_mean.csv")
per = pd.read_csv(ROOT / "part_4/output/shift_run/csv/part4_shift_per_corruption.csv")
M = ("Baseline", "Temperature Scaling", "Dirichlet")
runs = [("committed", committed_sens)]
if P4RUN is not None:
    runs.append(("your run", pd.read_csv(P4RUN / "part4_split_sensitivity.csv")))
for src_name, S in runs:
    s42 = S[S.Seed == 42].set_index("Method")
    T5 = {"Baseline": (92.78, 2.30, 2.70, 0.260, 1.60), "Temperature Scaling": (92.78, 1.78, 2.62, 0.259, 1.60),
          "Dirichlet": (92.76, 1.60, 2.26, 0.248, 1.68)}
    for m, (acc, ece, ace, nll, r80) in T5.items():
        r = s42.loc[m]
        w = f"Table 5 (Exp 5, {src_name})"
        check(w, f"{m} acc", f"{acc:.2f}", pc(r.Accuracy))
        check(w, f"{m} ECE", f"{ece:.2f}", pc(r.ECE))
        check(w, f"{m} ACE", f"{ace:.2f}", pc(r.AdaptiveTopLabel))
        check(w, f"{m} NLL", f"{nll:.3f}", r.NLL)
        check(w, f"{m} Risk@80", f"{r80:.2f}", pc(r["Risk@80"]), tol=0.0051)
    for m, (mu, sd) in {"Baseline": (2.03, 0.25), "Temperature Scaling": (1.64, 0.28), "Dirichlet": (1.52, 0.29)}.items():
        e = S[S.Method == m].ECE
        check(w, f"{m} ECE mean 3 splits", f"{mu:.2f}", pc(e.mean()))
        check(w, f"{m} ECE std 3 splits", f"{sd:.2f}", pc(e.std(ddof=1)))
    check(w, "fitted T seed 42", "1.037", s42.loc["Baseline", "T"])
    Ts = S.groupby("Seed").T.first()
    check(w, "T min over splits", "1.037", Ts.min())
    check(w, "T max over splits", "1.052", Ts.max())
    check(w, "lambda = 1 for all splits", 3, int((S.groupby("Seed")["lambda"].first() == 1.0).sum()), tol=0)
    for seed, ts, dr in ((42, 0.52, 0.70), (1, 0.51, 0.64), (2, 0.13, 0.20)):
        x = S[S.Seed == seed].set_index("Method").ECE
        check(f"Exp 7 text ({src_name})", f"seed {seed} TS reduction", f"{ts:.2f}", pc(x["Baseline"] - x["Temperature Scaling"]))
        check(f"Exp 7 text ({src_name})", f"seed {seed} Dir reduction", f"{dr:.2f}", pc(x["Baseline"] - x["Dirichlet"]))
    check(f"Exp 5 text ({src_name})", "TS relative reduction 23%", "23", 100 * (1 - s42.loc["Temperature Scaling", "ECE"] / s42.loc["Baseline", "ECE"]))

if P4RUN is not None:
    p = np.load(P4RUN / "part4_predictions_seed42.npz")
    y, b, t, d = p["labels"], p["Baseline"].argmax(1), p["Temperature_Scaling"].argmax(1), p["Dirichlet"].argmax(1)
    ch = b != d
    check("Exp 5 text", "Dirichlet changes predictions", 22, int(ch.sum()), tol=0)
    check("Exp 5 text", "... become correct", 9, int((ch & (d == y) & (b != y)).sum()), tol=0)
    check("Exp 5 text", "... become wrong", 10, int((ch & (b == y) & (d != y)).sum()), tol=0)
    check("Exp 5 text", "TS changes no prediction", 0, int((b != t).sum()), tol=0)

# Table 8
pool0 = pool[pool.Severity == 0].set_index("Method")
T8 = {0: (92.78, 2.30, 1.78, 1.60, 0.260, 0.259, 0.248, None, None, -22.7),
      1: (90.14, 3.64, 3.11, 2.98, 0.352, 0.349, 0.338, 5, 5, -14.5),
      2: (84.62, 6.88, 6.26, 6.20, 0.562, 0.554, 0.550, 5, 5, -9.0),
      3: (77.55, 11.76, 11.01, 11.14, 0.843, 0.827, 0.841, 5, 4, -6.4),
      4: (70.82, 16.55, 15.75, 15.85, 1.115, 1.092, 1.113, 5, 4, -4.8),
      5: (56.32, 29.39, 28.49, 28.28, 1.851, 1.804, 1.839, 5, 4, -3.0)}
for sev, (acc, er, et, edr, nr, nt, nd, ct, cd, rel) in T8.items():
    x = pool0 if sev == 0 else mean[mean.Severity == sev].set_index("Method")
    check("Table 8 (Exp 6 calibrated)", f"sev{sev} acc raw", f"{acc:.2f}", pc(x.loc["Baseline", "Accuracy"]))
    for m, e, n in zip(M, (er, et, edr), (nr, nt, nd)):
        check("Table 8 (Exp 6 calibrated)", f"sev{sev} {m} ECE", f"{e:.2f}", pc(x.loc[m, "ECE"]))
        check("Table 8 (Exp 6 calibrated)", f"sev{sev} {m} NLL", f"{n:.3f}", x.loc[m, "NLL"])
    check("Table 8 (Exp 6 calibrated)", f"sev{sev} TS rel change", f"{rel:.1f}",
          100 * (x.loc["Temperature Scaling", "ECE"] / x.loc["Baseline", "ECE"] - 1))
    if ct is not None:
        q = per[per.Severity == sev].pivot(index="Corruption", columns="Method", values="ECE")
        check("Table 8 (Exp 6 calibrated)", f"sev{sev} TS cells improved", ct, int((q["Temperature Scaling"] < q.Baseline).sum()), tol=0)
        check("Table 8 (Exp 6 calibrated)", f"sev{sev} Dir cells improved", cd, int((q.Dirichlet < q.Baseline).sum()), tol=0)
m15 = mean[mean.Severity.between(1, 5)].pivot(index="Severity", columns="Method")
ts_rm = pc(m15[("ECE", "Baseline")] - m15[("ECE", "Temperature Scaling")])
dr_rm = pc(m15[("ECE", "Baseline")] - m15[("ECE", "Dirichlet")])
check("Exp 6 text", "TS removes min", "0.53", ts_rm.min()); check("Exp 6 text", "TS removes max", "0.89", ts_rm.max())
check("Exp 6 text", "Dir removes min", "0.62", dr_rm.min()); check("Exp 6 text", "Dir removes max", "1.11", dr_rm.max())
r80d = pc((m15[("Risk@80", "Baseline")].to_numpy()[:, None] - m15["Risk@80"][["Temperature Scaling", "Dirichlet"]].to_numpy()))
check("Exp 6 text", "max |Risk@80 diff| <= 0.17", "0.17", np.abs(r80d).max(), tol=0.005)
check("Exp 6 text", "TS sev5 / clean ECE (16x)", 16, mean.query("Severity==5 and Method=='Temperature Scaling'").ECE.iloc[0] / pool0.loc["Temperature Scaling", "ECE"], tol=0.5)
check("Exp 6 text", "Dir sev5 / clean ECE (18x)", 18, mean.query("Severity==5 and Method=='Dirichlet'").ECE.iloc[0] / pool0.loc["Dirichlet", "ECE"], tol=0.5)

# Table 10 + Section 7 Gaussian-noise numbers
T10 = {"brightness": (87.44, 5.17, 4.53, 4.43), "motion_blur": (68.70, 17.16, 16.29, 15.46),
       "pixelate": (69.10, 17.91, 17.09, 16.85), "gaussian_noise": (36.62, 42.96, 41.88, 44.61),
       "contrast": (19.72, 63.74, 62.66, 60.03)}
q5 = per[per.Severity == 5]
for c, (acc, er, et, edr) in T10.items():
    x = q5[q5.Corruption == c].set_index("Method")
    check("Table 10 (Exp 7 per corruption)", f"{c} acc", f"{acc:.2f}", pc(x.loc["Baseline", "Accuracy"]))
    for m, e in zip(M, (er, et, edr)):
        check("Table 10 (Exp 7 per corruption)", f"{c} {m} ECE", f"{e:.2f}", pc(x.loc[m, "ECE"]))
g = per[per.Corruption == "gaussian_noise"].pivot(index="Severity", columns="Method", values="ECE")
for sev, d_, r_ in ((3, 30.85, 29.61), (4, 38.53, 36.50), (5, 44.61, 42.96)):
    check("Section 7 (Dirichlet fails)", f"gauss sev{sev} Dir", f"{d_:.2f}", pc(g.loc[sev, "Dirichlet"]))
    check("Section 7 (Dirichlet fails)", f"gauss sev{sev} raw", f"{r_:.2f}", pc(g.loc[sev, "Baseline"]))
others = per[per.Corruption != "gaussian_noise"].pivot_table(index=["Corruption", "Severity"], columns="Method", values="ECE")
check("Section 7 (Dirichlet fails)", "Dir improves all 20 other cells", 20, int((others.Dirichlet < others.Baseline).sum()), tol=0)
check("Section 7 (Dirichlet fails)", "gauss sev1-2 Dir improves", 2, int((g.loc[[1, 2], "Dirichlet"] < g.loc[[1, 2], "Baseline"]).sum()), tol=0)

show()
