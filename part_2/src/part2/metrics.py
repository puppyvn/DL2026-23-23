"""Confidence, calibration and selective-prediction metrics used in Part 2.

Confidence = maximum softmax probability, prediction = argmax.
ECE uses 15 equal-width bins (lo, hi]; ACE uses 15 equal-mass bins on the top-label confidence.
Ranking uses log-confidence: same order as confidence, but it does not round to exactly 1.0 at T = 0.5.
"""
import numpy as np
from scipy.stats import spearmanr

N_BINS = 15


def softmax_T(logits, T=1.0):
    """softmax(z / T) in float64, with the row maximum subtracted first for numerical stability."""
    z = np.asarray(logits)
    z = (z - z.max(1, keepdims=True)).astype(np.float64) / T
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def log_confidence(logits, T=1.0):
    """log(max softmax(z / T)) = -log(1 + sum_{j != max} exp((z_j - z_max) / T)), computed with log1p."""
    z = np.asarray(logits)
    z = (z - z.max(1, keepdims=True)).astype(np.float64) / T
    e = np.exp(z)
    e[np.arange(len(e)), z.argmax(1)] = 0.0
    return -np.log1p(e.sum(1))


def predict(logits, labels, T=1.0):
    """Per-image record at temperature T: prediction, confidence, ranking score, correctness (0/1)."""
    p = softmax_T(logits, T)
    pred = p.argmax(1)
    return dict(pred=pred, conf=p.max(1), score=log_confidence(logits, T),
                correct=(pred == np.asarray(labels)).astype(float))


def reliability_bins(conf, correct, n_bins=N_BINS):
    """Mean confidence, accuracy and count of each equal-width bin (lo, hi]; NaN for empty bins."""
    idx = np.digitize(conf, np.linspace(0, 1, n_bins + 1)[1:-1], right=True)
    cnt = np.bincount(idx, minlength=n_bins).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        bc = np.bincount(idx, weights=conf, minlength=n_bins) / cnt
        ba = np.bincount(idx, weights=correct, minlength=n_bins) / cnt
    return bc, ba, cnt


def ece_score(conf, correct, n_bins=N_BINS):
    """Expected Calibration Error: sum over bins of (n_b / N) * |acc_b - conf_b|."""
    idx = np.digitize(conf, np.linspace(0, 1, n_bins + 1)[1:-1], right=True)
    sc = np.bincount(idx, weights=conf, minlength=n_bins)
    sa = np.bincount(idx, weights=correct, minlength=n_bins)
    return float(np.abs(sa - sc).sum() / len(conf))


def ace_score(conf, correct, n_bins=N_BINS):
    """Adaptive Calibration Error: mean |acc - conf| over equal-mass bins of the sorted confidence."""
    o = np.argsort(conf, kind="stable"); c = conf[o]; a = correct[o].astype(float)
    return float(np.mean([abs(a[p].mean() - c[p].mean()) for p in np.array_split(np.arange(len(c)), n_bins)]))


def signed_gap(conf, correct):
    """Mean confidence - accuracy. Positive = overconfident, negative = underconfident."""
    return float(np.mean(conf) - np.mean(correct))


def risk_coverage(score, correct):
    """Accept the k most confident predictions, k = 1..N. Returns (coverage, risk)."""
    err = 1.0 - np.asarray(correct, dtype=float)[np.argsort(-np.asarray(score), kind="stable")]
    k = np.arange(1, len(err) + 1)
    return k / len(err), np.cumsum(err) / k


def risk_at(score, correct, coverage):
    """Error rate on the round(coverage * N) most confident predictions."""
    k = int(round(coverage * len(score)))
    order = np.argsort(-np.asarray(score), kind="stable")[:k]
    return float(1.0 - np.asarray(correct, dtype=float)[order].mean())


def topk_overlap(score_a, score_b, frac):
    """Share of the top-frac accepted set that is the same under two rankings."""
    k = int(round(frac * len(score_a)))
    top = lambda s: np.argsort(-np.asarray(s), kind="stable")[:k]
    return len(np.intersect1d(top(score_a), top(score_b))) / k


def spearman(score_a, score_b):
    return float(spearmanr(score_a, score_b).statistic)


def oracle_score(correct):
    """Perfect ranking: every correct prediction above every wrong one (lowest possible risk at this accuracy)."""
    return np.asarray(correct, dtype=float)


def aurc(score, correct):
    """Area under the risk-coverage curve."""
    return float(risk_coverage(score, correct)[1].mean())


def e_aurc(score, correct):
    """Excess AURC: AURC minus the AURC of the oracle ranking at the same accuracy (Geifman et al., ICLR 2019).
    It still depends on the error rate e: a random ranking scores about -(1 - e) ln(1 - e)."""
    return aurc(score, correct) - aurc(oracle_score(correct), correct)


def auroc(score, correct):
    """P(a random correct prediction has a higher score than a random wrong one), ties count 1/2.
    0.5 = no information, 1 = perfect separation. Does not depend on the error rate."""
    from scipy.stats import rankdata
    c = np.asarray(correct).astype(bool); n1, n0 = c.sum(), (~c).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[c].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


METRICS = {
    "acc": lambda r: r["correct"].mean(),
    "gap": lambda r: signed_gap(r["conf"], r["correct"]),
    "ece": lambda r: ece_score(r["conf"], r["correct"]),
    "ace": lambda r: ace_score(r["conf"], r["correct"]),
    "risk80": lambda r: risk_at(r["score"], r["correct"], 0.8),
    "risk60": lambda r: risk_at(r["score"], r["correct"], 0.6),
    "auroc": lambda r: auroc(r["score"], r["correct"]),
}


def paired_bootstrap(records, metrics=None, B=1000, seed=0):
    """Resample the test set B times with the SAME indices for every record, so that differences between records
    (e.g. T = 0.5 minus T = 1) get valid paired intervals. Returns {record: {metric: array(B)}}."""
    metrics = metrics or METRICS
    rng = np.random.default_rng(seed)
    n = len(next(iter(records.values()))["correct"])
    out = {name: {m: np.empty(B) for m in metrics} for name in records}
    for b in range(B):
        idx = rng.integers(0, n, n)
        for name, r in records.items():
            sub = {k: v[idx] for k, v in r.items()}
            for m, fn in metrics.items():
                out[name][m][b] = fn(sub)
    return out


def ci(samples, level=95):
    lo, hi = np.percentile(samples, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(lo), float(hi)
