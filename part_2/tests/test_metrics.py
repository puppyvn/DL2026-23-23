"""Checks of the metric code on synthetic data whose answer is known.  python tests/test_metrics.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from part2 import metrics as m      # noqa: E402

rng = np.random.default_rng(0)

# calibrated by construction -> ECE, ACE and gap close to 0
c = rng.uniform(.5, 1, 50000); k = (rng.uniform(size=50000) < c).astype(float)
assert m.ece_score(c, k) < .01 and m.ace_score(c, k) < .02 and abs(m.signed_gap(c, k)) < .01

# overconfident by 0.15 -> ECE, ACE and gap close to +0.15
k2 = (rng.uniform(size=50000) < c - .15).astype(float)
assert abs(m.ece_score(c, k2) - .15) < .02 and abs(m.ace_score(c, k2) - .15) < .02 and abs(m.signed_gap(c, k2) - .15) < .01

# temperature never changes the prediction; confidence strictly falls as T grows
z = rng.normal(0, 3, (2000, 10)); y = rng.integers(0, 10, 2000)
for T in (.5, 1, 2):
    p = m.softmax_T(z, T)
    assert np.allclose(p.sum(1), 1) and (p.argmax(1) == z.argmax(1)).all()
    assert np.allclose(np.log(p.max(1)), m.log_confidence(z, T))
assert (m.softmax_T(z, .5).max(1) > m.softmax_T(z, 1).max(1)).all() and (m.softmax_T(z, 1).max(1) > m.softmax_T(z, 2).max(1)).all()

# log-confidence keeps the order where float64 softmax rounds to exactly 1.0
big = np.zeros((2, 10)); big[:, 0] = [40, 45]
assert (m.softmax_T(big, .5).max(1) == 1.0).all() and m.log_confidence(big, .5)[1] > m.log_confidence(big, .5)[0]

# risk-coverage: full coverage = error rate; random ranking stays near the error rate
corr = (rng.uniform(size=10000) < .9).astype(float); s = rng.normal(size=10000)
cov, risk = m.risk_coverage(s, corr)
assert np.isclose(risk[-1], 1 - corr.mean()) and np.isclose(m.risk_at(s, corr, 1.0), 1 - corr.mean())
assert abs(m.risk_at(s, corr, .5) - (1 - corr.mean())) < .02
assert m.risk_at(corr, corr, .8) == 0.0                              # perfect ranking: no error in the top 80%

# ranking agreement
assert np.isclose(m.spearman(s, np.exp(s)), 1) and m.topk_overlap(s, s, .8) == 1 and np.isclose(m.topk_overlap(s, -s, .8), .75)

# AUROC: perfect ranking 1, reversed 0, all tied 0.5, random about 0.5
assert m.auroc(corr, corr) == 1.0 and m.auroc(-corr, corr) == 0.0 and m.auroc(np.ones(10000), corr) == 0.5
assert abs(m.auroc(s, corr) - .5) < .03

# E-AURC: 0 for the oracle; a random ranking gives about -(1 - e) ln(1 - e)
e = 1 - corr.mean()
assert np.isclose(m.e_aurc(m.oracle_score(corr), corr), 0)
assert abs(m.e_aurc(s, corr) - (-(1 - e) * np.log(1 - e))) < .01

# paired bootstrap: identical records give identical draws, and the interval covers the point value
rec = m.predict(z, y, 1.0)
bs = m.paired_bootstrap({"a": rec, "b": dict(rec)}, B=50, seed=1)
assert all(np.array_equal(bs["a"][k], bs["b"][k]) for k in bs["a"])
lo, hi = m.ci(bs["a"]["acc"]); assert lo <= rec["correct"].mean() <= hi

print("all metric tests passed")
