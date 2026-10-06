"""Evaluation: the Part 2 experiment (steps 2-6 of the notebook) and the summary table.

Step 2  probabilities at T = 0.5, 1, 2; prediction and accuracy must be identical at every T
Step 3  Reliability Diagram per T, with ECE and ACE
Step 4  Risk-Coverage per T, with Risk@80% and Risk@60%
Step 5  direction shown by the diagram vs the direction created (below y = x: over, above: under)
Step 6  does Risk-Coverage keep (almost) the same order of predictions? Spearman and top-k overlap vs T = 1

Writes results/figures/*.png and results/tables/*.csv. Run data_preparation/prepare_data.py first.

    python evaluation/run_experiment.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from part2 import config                                                                   # noqa: E402
from part2.metrics import (ace_score, ece_score, predict, reliability_bins, risk_at,       # noqa: E402
                           signed_gap, spearman, topk_overlap)
from part2.plots import apply_style, reliability_figure, risk_coverage_figure             # noqa: E402

EXPECT = {0.5: "overconfident", 1.0: "near calibrated", 2.0: "underconfident"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=config.PROCESSED)
    ap.add_argument("--figures", type=Path, default=config.FIGURES)
    ap.add_argument("--tables", type=Path, default=config.TABLES)
    args = ap.parse_args()
    if not (args.data / "logits.npy").exists():
        sys.exit("data/processed/logits.npy not found: run  python data_preparation/prepare_data.py  first")
    args.figures.mkdir(parents=True, exist_ok=True); args.tables.mkdir(parents=True, exist_ok=True)
    np.random.seed(config.SEED)
    apply_style()

    logits, labels = np.load(args.data / "logits.npy"), np.load(args.data / "labels.npy")
    TS = config.TEMPERATURES

    # Step 2: T only rescales confidence; it cannot change the prediction
    REC = {T: predict(logits, labels, T) for T in TS}
    for T in TS:
        assert np.array_equal(REC[T]["pred"], REC[1.0]["pred"])
        assert np.array_equal(REC[T]["correct"], REC[1.0]["correct"])
    acc = REC[1.0]["correct"].mean()
    print(f"accuracy at every T: {100 * acc:.2f}%")

    # Step 3: reliability diagram, ECE, ACE
    ece = {T: ece_score(REC[T]["conf"], REC[T]["correct"]) for T in TS}
    ace = {T: ace_score(REC[T]["conf"], REC[T]["correct"]) for T in TS}
    reliability_figure(REC, acc, ece, ace, args.figures / "fig_p2_reliability.png")
    CAL = pd.DataFrame([{"T": T, "Acc %": 100 * acc, "mean conf %": 100 * REC[T]["conf"].mean(),
                         "ECE %": 100 * ece[T], "ACE %": 100 * ace[T]} for T in TS]).set_index("T")
    CAL.to_csv(args.tables / "p2_calibration.csv")

    # Step 4: risk-coverage, Risk@80%, Risk@60% (ranking by log-confidence)
    risk_coverage_figure(REC, acc, config.COVERAGES, args.figures / "fig_p2_risk_coverage.png")
    RC = pd.DataFrame([{"T": T, **{f"Risk@{int(100 * c)}% %": 100 * risk_at(REC[T]["score"], REC[T]["correct"], c)
                                   for c in config.COVERAGES}} for T in TS]).set_index("T")
    RC.to_csv(args.tables / "p2_risk_coverage.csv")

    # Step 5: direction on the diagram, weighted by the number of samples in each bin
    rows = []
    for T in TS:
        c, k = REC[T]["conf"], REC[T]["correct"]
        bc, ba, cnt = reliability_bins(c, k)
        below = cnt[(cnt > 0) & (bc > ba)].sum() / cnt.sum()
        above = cnt[(cnt > 0) & (ba > bc)].sum() / cnt.sum()
        rows.append({"T": T, "expected": EXPECT[T], "signed gap % (conf - acc)": 100 * signed_gap(c, k),
                     "samples below y=x (over)": round(100 * below, 1), "samples above y=x (under)": round(100 * above, 1),
                     "diagram says": "overconfident" if below > above else "underconfident"})
    DIR = pd.DataFrame(rows).set_index("T")
    DIR.to_csv(args.tables / "p2_direction.csv")

    # Step 6: is the order of predictions (almost) kept?
    rows = []
    for T in TS:
        if T == 1.0:
            continue
        row = {"T vs T=1": T}
        for c in config.COVERAGES:
            col = f"Risk@{int(100 * c)}% %"
            row[f"change in {col[:-2]} (pp)"] = RC.loc[T, col] - RC.loc[1.0, col]
        row["Spearman rho"] = spearman(REC[1.0]["score"], REC[T]["score"])
        for c in config.COVERAGES:
            row[f"overlap top-{int(100 * c)}%"] = topk_overlap(REC[1.0]["score"], REC[T]["score"], c)
        rows.append(row)
    RANK = pd.DataFrame(rows).set_index("T vs T=1")
    RANK.to_csv(args.tables / "p2_ranking.csv")

    # summary
    TABLE = pd.concat([CAL, RC], axis=1)
    TABLE["signed gap %"] = DIR["signed gap % (conf - acc)"]
    TABLE.round(2).to_csv(args.tables / "p2_table_main.csv")
    pd.set_option("display.width", 160)
    print("\nSummary (clean CIFAR-10 test, ResNet-18, 10,000 images)")
    print(TABLE.round(2).to_string())
    print("\n[1] Does the Reliability Diagram show the direction that was created?")
    for T in TS:
        print(f"    T={T}: expected {EXPECT[T]:<15} | gap {DIR.loc[T, 'signed gap % (conf - acc)']:+.2f}% | diagram: {DIR.loc[T, 'diagram says']}")
    print("[2] Does Risk-Coverage keep (almost) the same order when T changes?")
    print(RANK.round(4).to_string())
    print(f"\nfigures -> {args.figures}\ntables  -> {args.tables}")


if __name__ == "__main__":
    main()
