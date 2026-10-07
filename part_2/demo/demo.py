"""Demo: what happens to one test image when T changes.

Prints the prediction (identical at every T), the confidence at T = 0.5, 1 and 2, and whether the image is among
the 80% most confident predictions that a selective classifier would accept. The prediction never moves; the
confidence does; the accepted set barely moves.

    python demo/demo.py --index 0
    python demo/demo.py --index 0 --coverage 0.6
    python demo/demo.py --image path/to/32x32.png        # runs the checkpoint on your own image (needs PyTorch)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from part2 import config                                    # noqa: E402
from part2.metrics import predict, softmax_T                # noqa: E402


def from_test_set(index, coverage):
    logits = np.load(config.PROCESSED / "logits.npy"); labels = np.load(config.PROCESSED / "labels.npy")
    k = int(round(coverage * len(labels)))
    print(f"test image #{index}: true class = {config.CLASSES[labels[index]]}")
    for T in config.TEMPERATURES:
        r = predict(logits, labels, T)
        rank = int(np.where(np.argsort(-r["score"], kind="stable") == index)[0][0])      # 0 = most confident
        print(f"  T = {T:<3}  predicted {config.CLASSES[r['pred'][index]]:<10} confidence {r['conf'][index]:.4f}  "
              f"{'correct' if r['correct'][index] else 'wrong  '}  rank {rank + 1:>5} of {len(labels)}  "
              f"{'accepted' if rank < k else 'rejected'} at {int(100 * coverage)}% coverage")


def from_image(path):
    from PIL import Image
    import torch
    sys.path.insert(0, str(config.ROOT / "inference"))
    from extract_logits import load_model, normalise
    img = np.asarray(Image.open(path).convert("RGB").resize((32, 32)))[None]
    with torch.no_grad():
        z = load_model().forward(normalise(img)).numpy()
    print(f"image {path}")
    for T in config.TEMPERATURES:
        p = softmax_T(z, T)[0]
        print(f"  T = {T:<3}  predicted {config.CLASSES[p.argmax()]:<10} confidence {p.max():.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, default=0, help="index in the CIFAR-10 test set (0-9999)")
    ap.add_argument("--coverage", type=float, default=0.8)
    ap.add_argument("--image", type=Path, help="classify your own image instead of a test image")
    args = ap.parse_args()
    if args.image:
        from_image(args.image)
    elif not (config.PROCESSED / "logits.npy").exists():
        sys.exit("run  python data_preparation/prepare_data.py  first")
    else:
        from_test_set(args.index, args.coverage)


if __name__ == "__main__":
    main()
