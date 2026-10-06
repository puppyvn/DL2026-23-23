"""Inference: recompute the ResNet-18 logits on the clean CIFAR-10 test set from the pretrained checkpoint.

This is the step that produced data/clean_logits.npz. Running it is optional: the evaluation uses the saved file.
It needs PyTorch (requirements-inference.txt), the model code of huyvnphan/PyTorch_CIFAR10 and its resnet18.pt
weights. Weights and CIFAR-10 come from the repository-wide downloads/ folder (shared/data.py), so they are
downloaded only once for all parts; run  python -m shared.prepare_data  to fetch them in advance.

    python inference/extract_logits.py                    # -> data/recomputed_logits.npz
    python data_preparation/prepare_data.py --source data/recomputed_logits.npz

On a GPU the logits can differ from the saved ones in the last digits (about 1e-6); accuracy does not change.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parents[2]))  # repository root, for shared/
from part2 import config                                    # noqa: E402
from shared import data as shared_data                      # noqa: E402


def load_model(model_source=None, device="cpu"):
    """Pretrained CIFAR-10 ResNet-18 in eval mode. By default the model code and weights come from the shared
    downloads/models folder (fetched once for every part); --model-source points to another clone instead."""
    if model_source is None:
        model_source, _ = shared_data.resnet18_weights()
    model_source = Path(model_source)
    sys.path.insert(0, str(model_source))
    from cifar10_models.resnet import resnet18
    return resnet18(pretrained=True).eval().to(device)


def normalise(images_uint8):
    """(N, 32, 32, 3) uint8 -> normalised (N, 3, 32, 32) float tensor, as used when the checkpoint was trained."""
    import torch
    x = torch.as_tensor(images_uint8).permute(0, 3, 1, 2).float() / 255.0
    return (x - torch.tensor(config.MEAN).view(1, 3, 1, 1)) / torch.tensor(config.STD).view(1, 3, 1, 1)


def main():
    import torch
    import torchvision
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-source", type=Path, default=None, help="default: shared downloads/models")
    ap.add_argument("--cifar-dir", type=Path, default=None, help="default: shared downloads/cifar10")
    ap.add_argument("--out", type=Path, default=config.ROOT / "data" / "recomputed_logits.npz")
    ap.add_argument("--batch", type=int, default=500)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model_source, device)
    test = (shared_data.cifar10_dataset(train=False) if args.cifar_dir is None
            else torchvision.datasets.CIFAR10(str(args.cifar_dir), train=False, download=True))
    x, y = normalise(test.data), np.array(test.targets, dtype=np.int64)
    with torch.no_grad():
        z = torch.cat([model(x[i:i + args.batch].to(device)).float().cpu() for i in range(0, len(x), args.batch)]).numpy()
    print(f"inference on {device}: {len(z)} images, accuracy {100 * (z.argmax(1) == y).mean():.2f}%")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **{config.MODEL_KEY: z.astype(np.float32), "labels": y})
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
