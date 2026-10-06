"""Paired bootstrap confidence intervals.

The test set is resampled 1,000 times; each draw uses the same images for T = 0.5, 1 and 2, so the noise the three
share cancels in their difference. An interval that excludes 0 means the difference is real. An interval that
contains 0 does NOT show "no change"; it only bounds how large the change can be.

AUROC (probability that a correct prediction is more confident than a wrong one) is added because, unlike
Risk@k and E-AURC, it does not depend on the error rate.

    python experiments/bootstrap_ci.py        -> results/tables/p2_bootstrap_ci.csv, p2_diff_vs_T1.csv
"""
import time

import pandas as pd

from common import TS, config, load
from part2.metrics import METRICS, ci, paired_bootstrap

LABEL = {"acc": "Accuracy", "gap": "Signed gap", "ece": "ECE", "ace": "ACE", "risk80": "Risk@80%",
         "risk60": "Risk@60%", "auroc": "AUROC"}


def main(B=1000):
    _, _, rec = load()
    t0 = time.time()
    boot = paired_bootstrap(rec, B=B, seed=config.SEED)
    print(f"paired bootstrap, {B} resamples: {time.time() - t0:.0f}s")

    rows = []
    for T in TS:
        for m, fn in METRICS.items():
            lo, hi = ci(boot[T][m])
            rows.append({"T": T, "metric": LABEL[m], "value": fn(rec[T]), "ci_low": lo, "ci_high": hi})
    values = pd.DataFrame(rows)
    values.to_csv(config.TABLES / "p2_bootstrap_ci.csv", index=False)

    rows = []
    for T in (0.5, 2.0):
        for m, fn in METRICS.items():
            if m == "acc":
                continue                                            # identical at every T by construction
            lo, hi = ci(boot[T][m] - boot[1.0][m])
            rows.append({"T vs T=1": T, "metric": LABEL[m], "difference": fn(rec[T]) - fn(rec[1.0]),
                         "ci_low": lo, "ci_high": hi, "ci_excludes_0": lo > 0 or hi < 0})
    diff = pd.DataFrame(rows)
    diff.to_csv(config.TABLES / "p2_diff_vs_T1.csv", index=False)

    show = diff.copy()
    for col in ("difference", "ci_low", "ci_high"):
        show[col] = show[col] * 100
    print("difference from T = 1 (x100: percentage points, or AUROC points), 95% paired CI")
    print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
