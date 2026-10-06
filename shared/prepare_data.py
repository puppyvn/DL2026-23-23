"""Download / compute everything the four parts share, once, into downloads/ (or $NNC_DATA_DIR).

    python -m shared.prepare_data                  # everything: CIFAR-10, weights, CIFAR-10-C, logits
    python -m shared.prepare_data --only cifar10   # just CIFAR-10 (enough for Part 1)
    python -m shared.prepare_data --data-dir /content/drive/MyDrive/nnc_data   # keep it in Google Drive

What each part uses:
    Part 1   CIFAR-10 train + test (training and its own shift test sets)
    Part 2   committed logits (part_2/data/clean_logits.npz); weights only for the optional re-inference
    Part 3   clean + CIFAR-10-C logits of the frozen ResNet-18
    Part 4   the same logits as Part 3

Files already downloaded by the old per-part scripts (part_3/artifacts, part_4/src/artifacts,
part_2/external) are moved into the shared folder instead of being downloaded again.
"""
import argparse
import os
import shutil
from pathlib import Path


OLD_LOCATIONS = [  # (old path relative to repo root, new path relative to the data dir)
    ("part_4/src/artifacts/data/cifar-10-batches-py", "cifar10/cifar-10-batches-py"),
    ("part_3/artifacts/data/cifar-10-batches-py", "cifar10/cifar-10-batches-py"),
    ("part_3/cifar10c/artifacts/data/cifar-10-batches-py", "cifar10/cifar-10-batches-py"),
    ("part_2/external/cifar10/cifar-10-batches-py", "cifar10/cifar-10-batches-py"),
    ("part_1/data/cifar-10-batches-py", "cifar10/cifar-10-batches-py"),
    ("data/cifar-10-batches-py", "cifar10/cifar-10-batches-py"),
    ("part_4/src/artifacts/PyTorch_CIFAR10", "models/PyTorch_CIFAR10"),
    ("part_3/artifacts/PyTorch_CIFAR10", "models/PyTorch_CIFAR10"),
    ("part_3/cifar10c/artifacts/PyTorch_CIFAR10", "models/PyTorch_CIFAR10"),
    ("part_2/external/PyTorch_CIFAR10", "models/PyTorch_CIFAR10"),
    ("part_4/src/artifacts/cache/state_dicts.zip", "models/state_dicts.zip"),
    ("part_3/artifacts/cache/state_dicts.zip", "models/state_dicts.zip"),
    ("part_2/external/cifar10_weights.zip", "models/state_dicts.zip"),
    ("part_4/src/artifacts/cache/CIFAR-10-C.tar", "CIFAR-10-C/CIFAR-10-C.tar"),
    ("part_3/artifacts/cache/CIFAR-10-C.tar", "CIFAR-10-C/CIFAR-10-C.tar"),
]
OLD_FOLDERS = [  # (old folder, file pattern, new folder)
    ("part_4/src/artifacts/data/CIFAR-10-C", "*.npy", "CIFAR-10-C"),
    ("part_3/artifacts/data/CIFAR-10-C", "*.npy", "CIFAR-10-C"),
    ("part_3/cifar10c/artifacts/data/CIFAR-10-C", "*.npy", "CIFAR-10-C"),
    ("part_4/src/artifacts/cache", "*_logits.npz", "logits"),
    ("part_4/src/artifacts/cache", "corruption_*.npz", "logits"),
    ("part_3/artifacts/cache", "*_logits.npz", "logits"),
    ("part_3/artifacts/cache", "corruption_*.npz", "logits"),
]


def adopt_old_downloads(repo_root, data_dir):
    """Move files fetched by the old per-part scripts into the shared folder (never overwrites)."""
    moved = []
    pairs = [(repo_root / a, data_dir / b) for a, b in OLD_LOCATIONS]
    for old, pattern, new in OLD_FOLDERS:
        if (repo_root / old).is_dir():
            pairs += [(f, data_dir / new / f.name) for f in sorted((repo_root / old).glob(pattern))]
    for src, dst in pairs:
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved.append(f"{src.relative_to(repo_root)} -> {dst}")
    return moved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=("all", "cifar10", "weights", "cifar10c", "logits"), default="all")
    ap.add_argument("--data-dir", help="Where to keep the downloads (default: <repo>/downloads or $NNC_DATA_DIR)")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--delete-archive", action="store_true",
                    help="Delete CIFAR-10-C.tar (2.9 GB) after the five needed corruptions are extracted")
    ap.add_argument("--no-adopt", action="store_true", help="Do not move files from the old per-part folders")
    args = ap.parse_args()
    if args.data_dir:
        os.environ["NNC_DATA_DIR"] = str(Path(args.data_dir).expanduser().resolve())

    from shared import data as sd
    root = sd.data_dir()
    print("Shared data folder:", root)
    if not args.no_adopt:
        for line in adopt_old_downloads(sd.REPO_ROOT, root):
            print("  reused", line)

    if args.only in ("all", "cifar10"):
        sd.cifar10_dataset(train=True)
        sd.cifar10_dataset(train=False)
        print("CIFAR-10 ready:", sd.cifar10_dir())
    if args.only in ("all", "weights"):
        print("ResNet-18 weights ready:", sd.resnet18_weights()[1])
    if args.only in ("all", "cifar10c"):
        print("CIFAR-10-C ready:", sd.ensure_cifar10c(delete_archive=args.delete_archive))
    if args.only in ("all", "logits"):
        print("Logits ready:", sd.ensure_logits(device=args.device, batch_size=args.batch_size,
                                                delete_archive=args.delete_archive))
        _check_against_part2(sd)
    print("Done. Every part now reads from", root)


def _check_against_part2(sd):
    """The shared clean logits must reproduce the logits committed in Part 2 (same checkpoint)."""
    import numpy as np
    ref_path = sd.REPO_ROOT / "part_2" / "data" / "clean_logits.npz"
    if not ref_path.exists():
        return
    ref = np.load(ref_path)["ResNet_18"]
    with np.load(sd.logits_dir() / "clean_logits.npz") as saved:
        z = saved["logits"]
    agree = (z.argmax(1) == ref.argmax(1)).mean()
    print(f"Check vs Part 2 committed logits: prediction agreement {100 * agree:.2f}%, "
          f"max |difference| {np.abs(z - ref).max():.2e}")


if __name__ == "__main__":
    main()
