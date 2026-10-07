"""Data preparation (step 1 of the notebook).

Reads the logits of the pretrained ResNet-18 on the 10,000 clean CIFAR-10 test images, checks them, and writes
the arrays the evaluation uses:
    data/processed/logits.npy   float32 (10000, 10)
    data/processed/labels.npy   int64   (10000,)
    data/processed/meta.json    source, checksum and baseline numbers at T = 1

The input is data/clean_logits.npz (keys: "ResNet_18", "labels", ...). It can be replaced by the output of
inference/extract_logits.py, which recomputes the same logits from the checkpoint.

    python data_preparation/prepare_data.py
    python data_preparation/prepare_data.py --source data/recomputed_logits.npz
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from part2 import config                                    # noqa: E402
from part2.metrics import ace_score, ece_score, predict      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=config.RAW_LOGITS, help="npz file with logits and labels")
    ap.add_argument("--model", default=config.MODEL_KEY, help="key of the model inside the npz file")
    ap.add_argument("--out", type=Path, default=config.PROCESSED)
    args = ap.parse_args()

    d = np.load(args.source)
    if args.model not in d.files or "labels" not in d.files:
        sys.exit(f"{args.source} must contain '{args.model}' and 'labels'; found {d.files}")
    logits = d[args.model].astype(np.float32)
    labels = d["labels"].astype(np.int64)

    # checks: official test set, one row of 10 logits per image, 1,000 images per class, no NaN/inf
    assert logits.shape == (10000, 10) and labels.shape == (10000,), (logits.shape, labels.shape)
    assert np.isfinite(logits).all(), "logits contain NaN or inf"
    assert (np.bincount(labels, minlength=10) == 1000).all(), "labels are not the balanced CIFAR-10 test set"

    r = predict(logits, labels, T=1.0)
    acc, ece, ace = r["correct"].mean(), ece_score(r["conf"], r["correct"]), ace_score(r["conf"], r["correct"])
    print(f"{args.model} on clean CIFAR-10 test, T = 1: accuracy {100 * acc:.2f}%  ECE {100 * ece:.2f}%  ACE {100 * ace:.2f}%")
    if abs(100 * acc - 93.07) > 0.5:
        print("warning: accuracy is far from the published 93.07% of the checkpoint; check the source of the logits")

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "logits.npy", logits)
    np.save(args.out / "labels.npy", labels)
    meta = dict(model=args.model, source=str(args.source.name),
                checkpoint="huyvnphan/PyTorch_CIFAR10 resnet18.pt, pretrained, no fine-tuning",
                data="CIFAR-10 official test set, clean, original order (10,000 images)",
                normalization=dict(mean=list(config.MEAN), std=list(config.STD)),
                accuracy=float(acc), ece=float(ece), ace=float(ace),
                sha256_logits=hashlib.sha256(logits.tobytes()).hexdigest())
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {args.out / 'logits.npy'}, labels.npy, meta.json")


if __name__ == "__main__":
    main()
