"""Shared loading code for the scripts in experiments/. Run data_preparation/prepare_data.py first."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from part2 import config                          # noqa: E402
from part2.metrics import predict                 # noqa: E402
from part2.plots import apply_style               # noqa: E402

TS = config.TEMPERATURES


def load():
    """Logits, labels and the per-image records at T = 0.5, 1, 2 (prediction, confidence, score, correctness)."""
    if not (config.PROCESSED / "logits.npy").exists():
        sys.exit("data/processed/logits.npy not found: run  python data_preparation/prepare_data.py  first")
    logits = np.load(config.PROCESSED / "logits.npy")
    labels = np.load(config.PROCESSED / "labels.npy")
    rec = {T: predict(logits, labels, T) for T in TS}
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    config.TABLES.mkdir(parents=True, exist_ok=True)
    apply_style()
    return logits, labels, rec
