"""Download the frozen ResNet-18, CIFAR-10 and CIFAR-10-C, and cache the logits Part 3 needs.

Nothing is trained. The checkpoint is huyvnphan/PyTorch_CIFAR10 ResNet-18, the same one used in
Parts 2 and 4 (93.07% clean accuracy). Run once; later runs reuse the cache.

    python prepare_data.py                       # downloads everything into ./artifacts
    python prepare_data.py --root /path/to/dir   # e.g. a Google Drive folder

If the same files already exist from Part 4 (part_4/src/artifacts/cache), point --root at
part_4/src/artifacts: the cache format is identical and nothing is downloaded again.
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tarfile
import zipfile
import numpy as np
from common import K, CORRUPTIONS, MEAN, STD, DEFAULT_ROOT, project_paths

WEIGHTS_GDRIVE_ID = "17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq"
CIFAR10C_URL = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"


def load_model(root, cache, device):
    import torch
    import gdown
    repo = root / "PyTorch_CIFAR10"
    if not (repo / "cifar10_models" / "resnet.py").exists():
        subprocess.check_call(["git", "clone", "--depth", "1",
                               "https://github.com/huyvnphan/PyTorch_CIFAR10.git", str(repo)])
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    weight = repo / "cifar10_models" / "state_dicts" / "resnet18.pt"
    if not weight.exists():
        archive = cache / "state_dicts.zip"
        if not zipfile.is_zipfile(archive):
            tmp = cache / "state_dicts.downloading.zip"
            if gdown.download(id=WEIGHTS_GDRIVE_ID, output=str(tmp), quiet=False) is None or not zipfile.is_zipfile(tmp):
                raise RuntimeError("Weight download failed. Download state_dicts.zip from the PyTorch_CIFAR10 "
                                   f"README and put it at {archive}.")
            tmp.replace(archive)
        with zipfile.ZipFile(archive) as zf:
            name = [n for n in zf.namelist() if n.endswith("resnet18.pt")][0]
            weight.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, weight.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    sha = hashlib.sha256(weight.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("confidence_resnet", repo / "cifar10_models" / "resnet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.resnet18(pretrained=False)
    model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
    model = model.to(device).eval().requires_grad_(False)
    provenance = json.dumps({"checkpoint": sha, "commit": commit, "mean": MEAN, "std": STD})
    print("ResNet-18 checkpoint SHA-256:", sha)
    return model, provenance


def run_model(model, images_uint8, device, batch_size):
    """images: (N, 32, 32, 3) uint8 -> logits (N, 10) float64, same preprocessing as Parts 2 and 4."""
    import torch
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    out = []
    with torch.inference_mode():
        for start in range(0, len(images_uint8), batch_size):
            x = torch.from_numpy(np.array(images_uint8[start:start + batch_size], copy=True))
            x = x.permute(0, 3, 1, 2).float() / 255
            out.append(model(((x - mean) / std).to(device)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def cached(path, provenance, labels=None):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["provenance"]) != provenance:
            return None
        if labels is not None and not np.array_equal(saved["labels"], labels):
            return None
        return saved["logits"], saved["labels"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="Folder for downloads and the logit cache")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--batch-size", type=int, default=500)
    args = ap.parse_args()

    import torch
    import torchvision
    import requests
    from tqdm.auto import tqdm
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print("Device:", device)
    root, cache, _ = project_paths(args.root)
    model, provenance = load_model(root, cache, device)

    # Clean test set, official order.
    clean = torchvision.datasets.CIFAR10(root=str(root / "data"), train=False, download=True)
    Y = np.asarray(clean.targets, dtype=int)
    hit = cached(cache / "clean_logits.npz", provenance)
    if hit is None:
        Z = run_model(model, clean.data, device, args.batch_size)
        np.savez_compressed(cache / "clean_logits.npz", logits=Z, labels=Y, provenance=provenance)
    else:
        Z = hit[0]
    print(f"Clean accuracy: {(Z.argmax(1) == Y).mean():.4f}  (expected 0.9307)")

    # CIFAR-10-C: only the five planned corruptions are extracted from the archive.
    c_dir = root / "data" / "CIFAR-10-C"
    c_dir.mkdir(parents=True, exist_ok=True)
    needed = [f"{c}.npy" for c in CORRUPTIONS] + ["labels.npy"]
    todo = [c for c in CORRUPTIONS if cached(cache / f"corruption_{c}.npz", provenance, Y) is None]
    if todo and not all((c_dir / n).exists() for n in needed):
        archive = cache / "CIFAR-10-C.tar"
        if not archive.exists():
            tmp = cache / "CIFAR-10-C.downloading.tar"
            with requests.get(CIFAR10C_URL, stream=True, timeout=(30, 180)) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0)) or None
                with tmp.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="CIFAR-10-C") as bar:
                    for chunk in r.iter_content(4 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            tmp.replace(archive)
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                name = member.name.rsplit("/", 1)[-1]
                if member.isfile() and name in needed and not (c_dir / name).exists():
                    with tf.extractfile(member) as src, (c_dir / name).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    if todo:
        labels_c = np.load(c_dir / "labels.npy")
        assert np.array_equal(labels_c, np.tile(Y, len(labels_c) // len(Y))), "CIFAR-10-C labels do not match."
    for c in CORRUPTIONS:
        if c in todo:
            images = np.load(c_dir / f"{c}.npy", mmap_mode="r")
            assert images.shape == (50000, 32, 32, 3)
            logits = run_model(model, images, device, args.batch_size).reshape(5, 10000, K)
            np.savez_compressed(cache / f"corruption_{c}.npz", logits=logits, labels=Y, provenance=provenance)
            print(c, "computed")
        else:
            print(c, "cache reused")
    print("Logit cache ready in", cache)


if __name__ == "__main__":
    main()
