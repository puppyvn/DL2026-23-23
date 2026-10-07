"""Research question 1: does the raw confidence of the ResNet-18 (T = 1) reflect the probability of being correct?

Three views of the same clean-test predictions, used in the report's answer to RQ1:
  1. confidence ranges: mean confidence vs accuracy for predictions with confidence >= 0.9 and < 0.9;
  2. predicted classes: signed gap (mean confidence - accuracy) for each predicted class;
  3. noise floor: the ECE and ACE that a perfectly calibrated model with exactly these confidences would show on
     10,000 images (correctness drawn as Bernoulli(confidence), 1,000 simulations, seed 0). An observed ECE far
     above this floor is real miscalibration, not finite-sample noise.

    python experiments/calibration_profile.py   -> results/tables/p2_confidence_ranges.csv
                                                    results/tables/p2_class_gap.csv
                                                    results/tables/p2_ece_noise_floor.csv
"""
import numpy as np
import pandas as pd

from common import config, load
from part2.metrics import ace_score, ece_score

THRESHOLD = 0.9


def main(n_sim=1000):
    _, labels, rec = load()
    r = rec[1.0]
    conf, correct, pred = r["conf"], r["correct"], r["pred"]

    rows = []
    for name, mask in ((f"confidence >= {THRESHOLD}", conf >= THRESHOLD),
                       (f"confidence < {THRESHOLD}", conf < THRESHOLD),
                       ("all", np.ones_like(conf, dtype=bool))):
        rows.append({"range": name, "n": int(mask.sum()), "share": mask.mean(), "mean_conf": conf[mask].mean(),
                     "accuracy": correct[mask].mean(), "gap": conf[mask].mean() - correct[mask].mean()})
    ranges = pd.DataFrame(rows)
    ranges.to_csv(config.TABLES / "p2_confidence_ranges.csv", index=False)

    classes = pd.DataFrame([{"predicted_class": c, "n": int((pred == k).sum()),
                             "mean_conf": conf[pred == k].mean(), "accuracy": correct[pred == k].mean(),
                             "gap": conf[pred == k].mean() - correct[pred == k].mean()}
                            for k, c in enumerate(config.CLASSES)])
    classes.to_csv(config.TABLES / "p2_class_gap.csv", index=False)

    rng = np.random.default_rng(config.SEED)
    sims = np.array([[ece_score(conf, s), ace_score(conf, s)]
                     for s in ((rng.random(len(conf)) < conf).astype(float) for _ in range(n_sim))])
    floor = pd.DataFrame([{"metric": m, "observed": obs, "perfect_mean": sims[:, j].mean(),
                           "perfect_p95": np.percentile(sims[:, j], 95), "ratio_observed_to_mean": obs / sims[:, j].mean()}
                          for j, (m, obs) in enumerate((("ECE", ece_score(conf, correct)),
                                                       ("ACE", ace_score(conf, correct))))])
    floor.to_csv(config.TABLES / "p2_ece_noise_floor.csv", index=False)

    pct = lambda d, cols: d.assign(**{c: (100 * d[c]).round(2) for c in cols})
    print(pct(ranges, ["share", "mean_conf", "accuracy", "gap"]).to_string(index=False))
    print(pct(classes, ["mean_conf", "accuracy", "gap"]).sort_values("gap", ascending=False).to_string(index=False))
    print(pct(floor, ["observed", "perfect_mean", "perfect_p95"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
