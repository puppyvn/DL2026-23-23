"""Run calibration and distribution-shift experiments using cached logits."""
import argparse
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
import os
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))   # repository root, for shared/
from shared import data as shared_data                       # one download folder for every part
from utils import (
    K, BINS, LAMBDAS, CORRUPTIONS, SEVERITIES, COLORS, METHOD_NAMES, DEFAULT_ROOT,
    project_paths, display, evaluate, make_split, fit_temperature,
    select_regularization, fit_dirichlet, calibrated_predictions,
    draw_reliability, draw_risk, finish_figure,
    SEVERITY_COLORS, TEMPERATURE_COLORS, CORRUPTION_COLORS,
    REPORT_PALETTE, REFERENCE_COLOR, GUIDE_COLOR, style_boxplot,
)


def load_logits(cache, need_corruptions=False):
    clean = cache / "clean_logits.npz"
    if not clean.exists():
        raise FileNotFoundError("Clean logits not found. Run  python -m shared.prepare_data  from the repository root.")
    with np.load(clean, allow_pickle=False) as saved:
        Z, Y = saved["logits"].astype(np.float64), saved["labels"].astype(int)
        provenance = str(saved["provenance"])
    if Z.shape != (10000, K) or Y.shape != (10000,) or not np.isfinite(Z).all():
        raise ValueError("Invalid clean-logit cache.")
    if not np.array_equal(np.unique(Y), np.arange(K)) or not np.all(np.bincount(Y) == 1000):
        raise ValueError("Expected 1,000 test samples for each of 10 classes.")
    corruptions = {}
    if need_corruptions:
        for name in CORRUPTIONS:
            path = cache / f"corruption_{name}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Missing {path.name}. Run  python -m shared.prepare_data  from the repository root.")
            with np.load(path, allow_pickle=False) as saved:
                if str(saved["provenance"]) != provenance or not np.array_equal(saved["labels"], Y):
                    raise ValueError(f"Checkpoint, preprocessing, or label mismatch: {path}")
                corruptions[name] = saved["logits"]
            if corruptions[name].shape != (5, 10000, K) or not np.isfinite(corruptions[name]).all():
                raise ValueError(f"Invalid corruption-logit cache: {path}")
    return Z, Y, provenance, corruptions

def run_part4(Z, Y, RESULTS, FIGURES, PREPROCESS_ID, SEED=42, show=True):
    cal_idx, test_idx = make_split(Y, SEED)
    Z_cal, y_cal = Z[cal_idx], Y[cal_idx]
    Z_test, y_test = Z[test_idx], Y[test_idx]
    np.savez(RESULTS / f"split_seed{SEED}.npz", cal_idx=cal_idx, test_idx=test_idx)
    display(pd.DataFrame({"class": range(K),
        "calibration": np.bincount(y_cal, minlength=K),
        "final_test": np.bincount(y_test, minlength=K)}))

    temperature = fit_temperature(Z_cal, y_cal)
    print(f"T fitted on clean calibration only = {temperature:.6f}")

    best_lambda, cv_table = select_regularization(Z_cal, y_cal, SEED)
    cv_table.to_csv(RESULTS / f"dirichlet_cv_seed{SEED}.csv", index=False)
    display(cv_table)
    print("Selected lambda:", best_lambda, "— final test not used.")

    dir_W, dir_b = fit_dirichlet(Z_cal, y_cal, best_lambda)
    np.savez(RESULTS / f"calibrators_seed{SEED}.npz", temperature=temperature,
             W=dir_W, b=dir_b, regularization=best_lambda,
             cal_idx=cal_idx, test_idx=test_idx, provenance=PREPROCESS_ID)
    print("Saved T, W, b and split indices. Extension will reuse these mappings.")

    P_clean = calibrated_predictions(Z_test, temperature, dir_W, dir_b)
    assert np.array_equal(P_clean["Baseline"].argmax(1), P_clean["Temperature Scaling"].argmax(1))
    clean_table = pd.DataFrame([{"Method": name, **evaluate(p, y_test)}
                               for name, p in P_clean.items()])
    clean_table.to_csv(RESULTS / f"part4_clean_seed{SEED}.csv", index=False)
    np.savez_compressed(RESULTS / f"part4_predictions_seed{SEED}.npz",
                        labels=y_test, **{name.replace(" ", "_"): p for name, p in P_clean.items()})
    display(clean_table.round(5))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, name in zip(axes, METHOD_NAMES):
        draw_reliability(ax, P_clean[name], y_test, name, COLORS[name])
    fig.suptitle(f"Part 4 — Clean final test (n=5000, seed={SEED}); marker size = bin count")
    finish_figure(fig, "part4_clean_reliability", FIGURES, show)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, name in zip(axes, METHOD_NAMES):
        p = P_clean[name]
        ax.hist(p.max(1), bins=np.linspace(0, 1, BINS+1), color=COLORS[name], alpha=.75)
        ax.axvline((p.argmax(1) == y_test).mean(), color=REFERENCE_COLOR, ls="--", label="Accuracy")
        ax.axvline(p.max(1).mean(), color=REPORT_PALETTE["red"], ls=":", label="Mean confidence")
        ax.set(title=name, xlabel="Confidence", ylabel="Number of samples", xlim=(0, 1))
    axes[0].legend(fontsize=8)
    fig.suptitle("Part 4 — Confidence distribution on the same final test")
    finish_figure(fig, "part4_clean_confidence_histogram", FIGURES, show)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name in METHOD_NAMES:
        draw_risk(ax, P_clean[name], y_test, name, COLORS[name])
    for coverage in (.6, .8):
        ax.axvline(coverage, color=GUIDE_COLOR, ls=":", alpha=.6)
    ax.set_title("Part 4 — Clean final test Risk–Coverage")
    ax.legend()
    finish_figure(fig, "part4_clean_risk_coverage", FIGURES, show)

    return SimpleNamespace(P_clean=P_clean, y_test=y_test, test_idx=test_idx, temperature=temperature, dir_W=dir_W, dir_b=dir_b, clean_table=clean_table)


def run_sensitivity(Z, Y, result, RESULTS, FIGURES, PREPROCESS_ID, SEED=42, SPLIT_SEEDS=(42, 1, 2), show=True):
    P_clean, y_test = result.P_clean, result.y_test
    temperature, dir_W, dir_b = result.temperature, result.dir_W, result.dir_b
    test_idx, clean_table = result.test_idx, result.clean_table
    with np.load(RESULTS / f"calibrators_seed{SEED}.npz") as saved:
        best_lambda = float(saved["regularization"])
    sensitivity_table = None
    seed_predictions = {SEED: (P_clean, y_test)}
    sensitivity_rows = []
    for seed in SPLIT_SEEDS:
        if seed == SEED:
            probs, labels = P_clean, y_test
            fitted_T, selected_lambda = temperature, best_lambda
        else:
            ci, ti = make_split(Y, seed)
            fitted_T = fit_temperature(Z[ci], Y[ci])
            selected_lambda, cv = select_regularization(Z[ci], Y[ci], seed)
            W_seed, b_seed = fit_dirichlet(Z[ci], Y[ci], selected_lambda)
            probs = calibrated_predictions(Z[ti], fitted_T, W_seed, b_seed)
            labels = Y[ti]
            cv.to_csv(RESULTS / f"dirichlet_cv_seed{seed}.csv", index=False)
            np.savez(RESULTS / f"calibrators_seed{seed}.npz", temperature=fitted_T,
                     W=W_seed, b=b_seed, regularization=selected_lambda,
                     cal_idx=ci, test_idx=ti, provenance=PREPROCESS_ID)
            seed_predictions[seed] = (probs, labels)
        for name in METHOD_NAMES:
            sensitivity_rows.append({"Seed": seed, "Method": name,
                "T": fitted_T, "lambda": selected_lambda, **evaluate(probs[name], labels)})
        print("Completed seed:", seed)
    sensitivity_table = pd.DataFrame(sensitivity_rows)
    sensitivity_table.to_csv(RESULTS / "part4_split_sensitivity.csv", index=False)
    display(sensitivity_table.round(5))
    summary = sensitivity_table.groupby("Method")[["Accuracy", "ECE", "AdaptiveTopLabel", "ACE_classwise"]].agg(["mean", "std"])
    summary.to_csv(RESULTS / "part4_split_summary.csv")
    display(summary.round(5))

    fig, axes = plt.subplots(len(SPLIT_SEEDS), 3, figsize=(14, 4*len(SPLIT_SEEDS)), squeeze=False)
    for row, seed in enumerate(SPLIT_SEEDS):
        probs, labels = seed_predictions[seed]
        for ax, name in zip(axes[row], METHOD_NAMES):
            draw_reliability(ax, probs[name], labels, f"{name}, seed {seed}", COLORS[name])
    fig.suptitle("Part 4 — Sensitivity to calibration/final-test split")
    finish_figure(fig, "part4_reliability_all_seeds", FIGURES, show)


def run_shift(C, result, RESULTS, FIGURES, show=True):
    P_clean, y_test = result.P_clean, result.y_test
    temperature, dir_W, dir_b = result.temperature, result.dir_W, result.dir_b
    test_idx, clean_table = result.test_idx, result.clean_table

    shift_table = shift_mean = None
    P_shift = {}
    shift_rows = []
    for severity in SEVERITIES:
        P_shift[severity] = {}
        for corruption in CORRUPTIONS:
            logits = C[corruption][severity-1, test_idx].astype(np.float64)
            predictions = calibrated_predictions(logits, temperature, dir_W, dir_b)
            P_shift[severity][corruption] = predictions
            for name in METHOD_NAMES:
                shift_rows.append({"Method": name, "Corruption": corruption,
                                   "Severity": severity, **evaluate(predictions[name], y_test)})
    shift_table = pd.DataFrame(shift_rows)
    metric_columns = list(evaluate(P_clean["Baseline"], y_test))
    shift_mean = shift_table.groupby(["Method", "Severity"], as_index=False)[metric_columns].mean()
    shift_table.to_csv(RESULTS / "part4_shift_per_corruption.csv", index=False)
    shift_mean.to_csv(RESULTS / "part4_shift_mean.csv", index=False)
    display(shift_mean.round(5))

    pooled_shift = {}
    pooled_shift[0] = (P_clean, y_test)
    for severity in SEVERITIES:
        probs = {name: np.concatenate([P_shift[severity][c][name] for c in CORRUPTIONS])
                 for name in METHOD_NAMES}
        pooled_shift[severity] = (probs, np.tile(y_test, len(CORRUPTIONS)))
    pooled_rows = [{"Method": name, "Severity": severity, **evaluate(probs[name], labels)}
                  for severity, (probs, labels) in pooled_shift.items() for name in METHOD_NAMES]
    pd.DataFrame(pooled_rows).to_csv(RESULTS / "part4_shift_pooled_metrics.csv", index=False)
    print("Pooled metrics saved separately from mean-per-corruption metrics.")

    levels = (0,) + SEVERITIES
    fig, axes = plt.subplots(3, 6, figsize=(21, 10.5), sharex=True, sharey=True)
    for row, name in enumerate(METHOD_NAMES):
        for col, severity in enumerate(levels):
            probs, labels = pooled_shift[severity]
            title = "Clean" if severity == 0 else f"Severity {severity}"
            draw_reliability(axes[row, col], probs[name], labels, title, COLORS[name])
            axes[row, col].set_ylabel(name + "\nAccuracy" if col == 0 else "")
    fig.suptitle("Part 4 extension — Clean-fitted mappings; pooled corruptions; all nonempty bins")
    finish_figure(fig, "part4_shift_reliability", FIGURES, show)

    levels = (0,) + SEVERITIES
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, metric in zip(axes, ["Accuracy", "ECE", "AdaptiveTopLabel", "ACE_classwise"]):
        for name in METHOD_NAMES:
            clean_value = clean_table.set_index("Method").loc[name, metric]
            shifted = shift_mean[shift_mean.Method == name].set_index("Severity")
            values = [clean_value] + [shifted.loc[s, metric] for s in SEVERITIES]
            ax.plot(levels, values, "o-", color=COLORS[name], label=name)
        ax.set(title=metric, xlabel="Severity (0 = clean)", xticks=levels)
    axes[0].legend(fontsize=8)
    fig.suptitle("Part 4 extension — Mean of metrics computed separately for each corruption")
    finish_figure(fig, "part4_shift_metrics", FIGURES, show)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ax, severity in zip(axes.flat, (0,) + SEVERITIES):
        probs, labels = pooled_shift[severity]
        for name in METHOD_NAMES:
            draw_risk(ax, probs[name], labels, name, COLORS[name])
        ax.set_title("Clean" if severity == 0 else f"Severity {severity}")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Part 4 extension — Risk–Coverage, pooled corruptions; fixed clean mappings")
    finish_figure(fig, "part4_shift_risk_coverage", FIGURES, show)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["Risk@60", "Risk@80"]):
        for name in METHOD_NAMES:
            clean_value = clean_table.set_index("Method").loc[name, metric]
            shifted = shift_mean[shift_mean.Method == name].set_index("Severity")
            ax.plot((0,) + SEVERITIES, [clean_value] + [shifted.loc[s, metric] for s in SEVERITIES],
                    "o-", color=COLORS[name], label=name)
        ax.set(title=metric, xlabel="Severity (0 = clean)", ylabel="Risk")
    axes[0].legend(fontsize=8)
    fig.suptitle("Part 4 extension — Mean of per-corruption operating-point risks")
    finish_figure(fig, "part4_shift_operating_points", FIGURES, show)


def run_part1b(Z, Y, C, RESULTS, FIGURES, show=True):
    part1b = None
    raw_clean = softmax(Z, axis=1)
    rows_1b = []
    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            probs = softmax(C[corruption][severity-1].astype(np.float64), axis=1)
            rows_1b.append({"Corruption": corruption, "Severity": severity, **evaluate(probs, Y)})
    part1b = pd.DataFrame(rows_1b)
    part1b.to_csv(RESULTS / "part1b_per_corruption.csv", index=False)
    display(part1b.round(5))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    baseline_metrics = evaluate(raw_clean, Y)
    for ax, metric in zip(axes, ["Accuracy", "ECE", "AdaptiveTopLabel", "ACE_classwise"]):
        for corruption in CORRUPTIONS:
            values = part1b[part1b.Corruption == corruption].set_index("Severity")[metric]
            ax.plot((0,) + SEVERITIES, [baseline_metrics[metric]] + [values.loc[s] for s in SEVERITIES],
                    "o-", color=CORRUPTION_COLORS[corruption], alpha=.75, lw=1, label=corruption)
        means = part1b.groupby("Severity")[metric].mean()
        ax.plot((0,) + SEVERITIES, [baseline_metrics[metric]] + [means.loc[s] for s in SEVERITIES],
                "o-", color=REFERENCE_COLOR, lw=2.3, label="Mean across corruptions")
        ax.set(title=metric, xlabel="Severity (0 = clean)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Part 1B — Frozen classifier, raw confidence, all 10K indices")
    finish_figure(fig, "part1b_metrics", FIGURES, show)


def run_part2(Z, Y, RESULTS, FIGURES, show=True):
    P_stress = {}
    STRESS_TEMPERATURES = (.5, 1., 2.)
    P_stress = {t: softmax(Z / t, axis=1) for t in STRESS_TEMPERATURES}
    stress_table = pd.DataFrame([{"T": t, **evaluate(p, Y)} for t, p in P_stress.items()])
    stress_table.to_csv(RESULTS / "part2_temperature_stress.csv", index=False)
    display(stress_table.round(5))
    print("T<1 sharpens; T>1 softens. Over/underconfidence must be read from results.")
    print("TS preserves each sample's predicted class, not necessarily confidence ranking across samples.")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, t in zip(axes, STRESS_TEMPERATURES):
        draw_reliability(ax, P_stress[t], Y, f"Fixed T = {t}", TEMPERATURE_COLORS[t])
    fig.suptitle("Part 2 — Controlled stress on clean data; no fitting and no shift")
    finish_figure(fig, "part2_reliability", FIGURES, show)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for t, style in zip(STRESS_TEMPERATURES, ("-", "--", ":")):
        for ax in axes:
            draw_risk(ax, P_stress[t], Y, f"T={t}", color=TEMPERATURE_COLORS[t], linestyle=style)
    axes[0].set_title("Full coverage range")
    axes[1].set(xlim=(.5, 1), title="Zoom: coverage 50–100%")
    axes[0].legend()
    fig.suptitle("Part 2 — Confidence ranking under fixed temperature stress")
    finish_figure(fig, "part2_risk_coverage", FIGURES, show)


def run_part3(Z, Y, C, RESULTS, FIGURES, SEED=42, show=True):
    P_raw = {}
    confidence_groups, accuracy_groups = [], []
    P_raw[0] = (softmax(Z, axis=1), Y)
    for severity in SEVERITIES:
        P_raw[severity] = (
            np.concatenate([softmax(C[c][severity-1].astype(np.float64), axis=1) for c in CORRUPTIONS]),
            np.tile(Y, len(CORRUPTIONS)))
    # Use the same permutation and batches of 500 indices for every severity and corruption.
    fixed_batches = np.random.RandomState(SEED).permutation(len(Y)).reshape(-1, 500)
    for severity in (0,) + SEVERITIES:
        probs, labels = P_raw[severity]
        confidence_groups.append(probs.max(1))
        per_corruption = [probs] if severity == 0 else np.split(probs, len(CORRUPTIONS))
        accuracy_groups.append(np.concatenate([
            (p.argmax(1)[fixed_batches] == Y[fixed_batches]).mean(axis=1)
            for p in per_corruption]))
    raw_table = pd.DataFrame([{"Severity": s, **evaluate(p, y)} for s, (p, y) in P_raw.items()])
    raw_table.to_csv(RESULTS / "part3_pooled_metrics.csv", index=False)
    display(raw_table.round(5))
    print("Boxplots: clean has 20 batches; each shifted level has 20 batches × 5 corruptions.")
    print("These related batches are descriptive distributions, not independent replicates.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    severity_colors = [SEVERITY_COLORS[s] for s in (0,) + SEVERITIES]
    style_boxplot(axes[0].boxplot(confidence_groups, showfliers=False, patch_artist=True), severity_colors)
    style_boxplot(axes[1].boxplot(accuracy_groups, showfliers=False, patch_artist=True), severity_colors)
    labels = ["Clean"] + [f"Sev {s}" for s in SEVERITIES]
    for ax in axes:
        ax.set_xticks(range(1, 7), labels)
        ax.set(xlabel="Shift intensity", ylim=(0, 1.02))
    axes[0].set(title="Sample confidence", ylabel="Max softmax probability")
    axes[1].set(title="Fixed batches of 500, separately per corruption", ylabel="Batch accuracy")
    fig.suptitle("Part 3 — Raw confidence; outliers hidden for readability")
    finish_figure(fig, "part3_boxplots", FIGURES, show)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, severity in zip(axes.flat, (0,) + SEVERITIES):
        probs, labels = P_raw[severity]
        draw_reliability(ax, probs, labels, "Clean" if severity == 0 else f"Severity {severity}", SEVERITY_COLORS[severity])
    fig.suptitle("Part 3 — Raw-confidence reliability; pooled corruptions")
    finish_figure(fig, "part3_reliability", FIGURES, show)

    fig, ax = plt.subplots(figsize=(8, 5))
    for severity, (probs, labels) in P_raw.items():
        draw_risk(ax, probs, labels, "Clean" if severity == 0 else f"Severity {severity}", SEVERITY_COLORS[severity])
    ax.legend()
    ax.set_title("Part 3 — Raw-confidence Risk–Coverage; pooled corruptions")
    finish_figure(fig, "part3_risk_coverage", FIGURES, show)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("part4", "sensitivity", "shift", "part1b", "part2", "part3", "all"), default="part4")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Where results/ is written")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Shared download folder with logits/ (default: <repo>/downloads or $NNC_DATA_DIR)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2], help="Additional split seeds for sensitivity analysis")
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening interactive windows")
    parser.add_argument("--zip", action="store_true", help="Package this run's results")
    args = parser.parse_args()
    if any(seed < 0 or seed >= 2**32 for seed in [args.seed, *args.seeds]):
        parser.error("Seeds must be in [0, 2**32).")
    if args.no_show:
        plt.switch_backend("Agg")
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True, "grid.alpha": .22})
    if args.data_dir:
        os.environ["NNC_DATA_DIR"] = str(args.data_dir.expanduser().resolve())
    root, _, result_root, _ = project_paths(args.root)
    # A separate run directory prevents stale figures from entering the ZIP.
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    results = result_root / f"{args.part}_seed{args.seed}_{stamp}"
    figures = results / "figures"
    need_corruptions = args.part in ("shift", "part1b", "part3", "all")
    assert tuple(CORRUPTIONS) == tuple(shared_data.CORRUPTIONS)
    # Logits are computed once in the shared folder (downloading only what is missing) and reused by Part 3.
    cache = shared_data.ensure_logits(CORRUPTIONS if need_corruptions else ())
    Z, Y, provenance, C = load_logits(cache, need_corruptions)
    figures.mkdir(parents=True, exist_ok=True)
    show = not args.no_show
    if args.part in ("part4", "sensitivity", "shift", "all"):
        result = run_part4(Z, Y, results, figures, provenance, args.seed, show)
        if args.part in ("sensitivity", "all"):
            seeds = tuple(dict.fromkeys([args.seed, *args.seeds]))
            run_sensitivity(Z, Y, result, results, figures, provenance, args.seed, seeds, show)
        if args.part in ("shift", "all"):
            run_shift(C, result, results, figures, show)
    if args.part in ("part1b", "all"):
        run_part1b(Z, Y, C, results, figures, show)
    if args.part in ("part2", "all"):
        run_part2(Z, Y, results, figures, show)
    if args.part in ("part3", "all"):
        run_part3(Z, Y, C, results, figures, args.seed, show)
    import importlib.metadata
    manifest = {
        "part": args.part, "seed": args.seed, "split_seeds": list(dict.fromkeys([args.seed, *args.seeds])),
        "bins": BINS, "lambda_grid": LAMBDAS, "provenance": json.loads(provenance),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "matplotlib", "scikit-learn")},
        "aggregation": "Shift trends: mean across corruptions. Reliability/RC: pooled samples.",
    }
    (results / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.zip:
        zip_path = results.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(results.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(results))
        print("ZIP:", zip_path)
    print("Results:", results)


if __name__ == "__main__":
    main()
