"""Single place that downloads and caches everything Parts 1-4 need, so each file is fetched once.

    downloads/                         (or $NNC_DATA_DIR, e.g. a Google Drive folder)
      cifar10/cifar-10-batches-py/     CIFAR-10 train + test (torchvision)           Parts 1, 2, 3, 4
      CIFAR-10-C/<corruption>.npy      the five planned corruptions + labels.npy     Parts 3, 4
      models/PyTorch_CIFAR10/          huyvnphan model code + resnet18.pt            Parts 2, 3, 4
      logits/clean_logits.npz          frozen ResNet-18 logits, clean test set       Parts 3, 4
      logits/corruption_<name>.npz     same model on CIFAR-10-C, shape (5, 10000, 10) Parts 3, 4

Every function checks what is already on disk and downloads or computes only what is missing.
Logit files carry a provenance string (checkpoint SHA-256, code commit, normalisation); a file made
with a different checkpoint is recomputed instead of being reused silently.

Prepare everything once from the repository root:
    python -m shared.prepare_data
"""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
K = 10
CORRUPTIONS = ("gaussian_noise", "motion_blur", "brightness", "contrast", "pixelate")
SEVERITIES = (1, 2, 3, 4, 5)
MEAN = (0.4914, 0.4822, 0.4465)      # normalisation of the huyvnphan checkpoints
STD = (0.2471, 0.2435, 0.2616)

MODEL_REPO = "https://github.com/huyvnphan/PyTorch_CIFAR10.git"
WEIGHTS_GDRIVE_ID = "17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq"     # archive with all huyvnphan checkpoints (~1 GB)
CIFAR10C_URL = os.environ.get("NNC_CIFAR10C_URL",
                              "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1")


# ----------------------------------------------------------------------------- locations

def data_dir():
    """Root of all downloads. Override with the NNC_DATA_DIR environment variable."""
    d = Path(os.environ.get("NNC_DATA_DIR", REPO_ROOT / "downloads")).expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def cifar10_dir():
    return _sub("cifar10")


def cifar10c_dir():
    return _sub("CIFAR-10-C")


def models_dir():
    return _sub("models")


def logits_dir():
    return _sub("logits")


def _sub(name):
    d = data_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------------------ CIFAR-10

def cifar10_dataset(train=False, transform=None):
    """torchvision CIFAR-10 from the shared folder; downloaded on first use only."""
    import torchvision
    return torchvision.datasets.CIFAR10(root=str(cifar10_dir()), train=train, transform=transform, download=True)


def cifar10_test_arrays():
    """Clean test set as (uint8 images NHWC, int labels), official order."""
    test = cifar10_dataset(train=False)
    return np.asarray(test.data), np.asarray(test.targets, dtype=int)


# ---------------------------------------------------------------------------- CIFAR-10-C

def ensure_cifar10c(corruptions=CORRUPTIONS, delete_archive=False):
    """Make <corruption>.npy and labels.npy available; downloads the 2.9 GB archive only if a file is missing.

    Files copied in by hand (e.g. extracted on your own computer and uploaded) are used as they are."""
    folder = cifar10c_dir()
    needed = [f"{c}.npy" for c in corruptions] + ["labels.npy"]
    missing = [n for n in needed if not (folder / n).exists()]
    if missing:
        archive = folder / "CIFAR-10-C.tar"
        if not _tar_ok(archive):
            _download(CIFAR10C_URL, archive)
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                name = member.name.rsplit("/", 1)[-1]
                if member.isfile() and name in missing:
                    partial = folder / (name + ".partial")
                    with tf.extractfile(member) as src, partial.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    partial.replace(folder / name)
        still = [n for n in needed if not (folder / n).exists()]
        if still:
            raise RuntimeError(f"{still} not found in {archive}")
        if delete_archive:
            archive.unlink()
    return folder


def cifar10c_images(corruption):
    """(50000, 32, 32, 3) uint8, memory-mapped: severities 1-5 stacked, 10,000 test images each."""
    images = np.load(ensure_cifar10c((corruption,)) / f"{corruption}.npy", mmap_mode="r")
    assert images.shape == (50000, 32, 32, 3), images.shape
    return images


def _tar_ok(path):
    if not path.exists():
        return False
    try:
        with tarfile.open(path) as tf:
            tf.getmembers()
        return True
    except (tarfile.TarError, EOFError):
        return False


def _download(url, target):
    import requests
    from tqdm.auto import tqdm
    tmp = target.with_suffix(target.suffix + ".downloading")
    with requests.get(url, stream=True, timeout=(30, 180)) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or None
        with tmp.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=target.name) as bar:
            for chunk in r.iter_content(4 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(target)


# --------------------------------------------------------------------------------- model

def resnet18_weights():
    """Path to huyvnphan resnet18.pt (and its model code); fetched once into downloads/models."""
    repo = models_dir() / "PyTorch_CIFAR10"
    if not (repo / "cifar10_models" / "resnet.py").exists():
        subprocess.check_call(["git", "clone", "--depth", "1", MODEL_REPO, str(repo)])
    weight = repo / "cifar10_models" / "state_dicts" / "resnet18.pt"
    if not weight.exists():
        archive = models_dir() / "state_dicts.zip"
        if not zipfile.is_zipfile(archive):
            import gdown
            tmp = models_dir() / "state_dicts.downloading.zip"
            if gdown.download(id=WEIGHTS_GDRIVE_ID, output=str(tmp), quiet=False) is None or not zipfile.is_zipfile(tmp):
                raise RuntimeError("Weight download failed (Google Drive quota?). Download state_dicts.zip from "
                                   f"{MODEL_REPO} and put it at {archive}, then run again.")
            tmp.replace(archive)
        with zipfile.ZipFile(archive) as zf:
            name = next(n for n in zf.namelist() if n.endswith("resnet18.pt"))
            weight.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, weight.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return repo, weight


def resnet18(device="cpu"):
    """Frozen pretrained CIFAR-10 ResNet-18 (eval mode) and its provenance string."""
    import torch
    repo, weight = resnet18_weights()
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    sha = hashlib.sha256(weight.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("huyvnphan_resnet", repo / "cifar10_models" / "resnet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.resnet18(pretrained=False)
    model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
    model = model.to(device).eval().requires_grad_(False)
    provenance = json.dumps({"checkpoint": sha, "commit": commit, "mean": MEAN, "std": STD})
    return model, provenance


def run_model(model, images_uint8, device="cpu", batch_size=500):
    """(N, 32, 32, 3) uint8 -> logits (N, 10) float64, with the checkpoint's normalisation."""
    import torch
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    out = []
    with torch.inference_mode():
        for start in range(0, len(images_uint8), batch_size):
            x = torch.from_numpy(np.array(images_uint8[start:start + batch_size], copy=True)).to(device)
            x = x.permute(0, 3, 1, 2).float() / 255
            out.append(model((x - mean) / std).float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


# -------------------------------------------------------------------------------- logits

def ensure_logits(corruptions=CORRUPTIONS, device="auto", batch_size=500, delete_archive=False):
    """Compute the clean and CIFAR-10-C logits of the frozen ResNet-18 once; later calls reuse them.

    Only downloads what a missing file needs: if every logit file exists with the right provenance,
    nothing (not even the model) is loaded."""
    folder = logits_dir()
    files = [folder / "clean_logits.npz"] + [folder / f"corruption_{c}.npz" for c in corruptions]
    stored = {f: _read_provenance(f) for f in files}
    if all(stored.values()) and len(set(stored.values())) == 1:
        return folder                                  # everything cached and consistent

    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, provenance = resnet18(device)
    images, labels = cifar10_test_arrays()
    if stored[files[0]] != provenance:
        z = run_model(model, images, device, batch_size)
        np.savez_compressed(files[0], logits=z, labels=labels, provenance=provenance)
        print(f"clean logits: accuracy {(z.argmax(1) == labels).mean():.4f} (expected 0.9307)")
    for c, f in zip(corruptions, files[1:]):
        if stored[f] == provenance:
            continue
        ensure_cifar10c(corruptions, delete_archive=False)
        c_labels = np.load(cifar10c_dir() / "labels.npy")
        assert np.array_equal(c_labels, np.tile(labels, len(c_labels) // len(labels))), "CIFAR-10-C label order"
        z = run_model(model, cifar10c_images(c), device, batch_size).reshape(len(SEVERITIES), len(labels), K)
        np.savez_compressed(f, logits=z, labels=labels, provenance=provenance)
        print(f"{c}: logits computed")
    if delete_archive and (cifar10c_dir() / "CIFAR-10-C.tar").exists():
        (cifar10c_dir() / "CIFAR-10-C.tar").unlink()
    return folder


def load_logits(corruptions=CORRUPTIONS, need_corruptions=True):
    """Return (clean logits, labels, provenance, {corruption: (5, 10000, 10)}), computing them if needed."""
    folder = ensure_logits(corruptions if need_corruptions else ())
    with np.load(folder / "clean_logits.npz", allow_pickle=False) as saved:
        Z, Y, provenance = saved["logits"].astype(np.float64), saved["labels"].astype(int), str(saved["provenance"])
    C = {}
    for c in (corruptions if need_corruptions else ()):
        with np.load(folder / f"corruption_{c}.npz", allow_pickle=False) as saved:
            if str(saved["provenance"]) != provenance or not np.array_equal(saved["labels"], Y):
                raise ValueError(f"corruption_{c}.npz does not match clean_logits.npz")
            C[c] = saved["logits"].astype(np.float64)
    return Z, Y, provenance, C


def _read_provenance(path):
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as saved:
            return str(saved["provenance"])
    except Exception:                                   # noqa: BLE001 - unreadable file is recomputed
        return None
