"""Download CIFAR data and cache frozen ResNet-18 logits. No model training."""
import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
import numpy as np
from utils import K, CORRUPTIONS, DEFAULT_ROOT, project_paths

def prepare(args):
    # Import inference-only dependencies after parsing command-line options.
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import requests
    import gdown
    from tqdm.auto import tqdm

    ROOT, CACHE, RESULTS, FIGURES = project_paths(args.root)
    REPO = ROOT / "PyTorch_CIFAR10"
    BATCH_SIZE = args.batch_size
    DEVICE = args.device
    if DEVICE == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu or --device auto.")
    print("Device:", DEVICE)
    if not (REPO / "cifar10_models" / "resnet.py").exists():
        subprocess.check_call(["git", "clone", "--depth", "1",
                               "https://github.com/huyvnphan/PyTorch_CIFAR10.git", str(REPO)])
    REPO_COMMIT = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    print("Architecture source commit:", REPO_COMMIT)

    WEIGHT = REPO / "cifar10_models" / "state_dicts" / "resnet18.pt"
    if not WEIGHT.exists():
        archive = CACHE / "state_dicts.zip"
        if not archive.exists() or not zipfile.is_zipfile(archive):
            # This Google Drive backup was used successfully in the original notebook.
            temporary = CACHE / "state_dicts.downloading.zip"
            downloaded = gdown.download(id="17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq",
                                        output=str(temporary), quiet=False)
            if downloaded is None or not zipfile.is_zipfile(temporary):
                raise RuntimeError("Failed to download the weights. Try running this cell again later.")
            temporary.replace(archive)
        with zipfile.ZipFile(archive) as zf:
            matches = [n for n in zf.namelist() if n.endswith("resnet18.pt")]
            if len(matches) != 1:
                raise RuntimeError(f"Could not identify a unique checkpoint: {matches}")
            WEIGHT.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(matches[0]) as source, WEIGHT.open("wb") as target:
                shutil.copyfileobj(source, target)
    WEIGHT_SHA256 = hashlib.sha256(WEIGHT.read_bytes()).hexdigest()
    print("Checkpoint:", WEIGHT)
    print("SHA256:", WEIGHT_SHA256)

    # Import the architecture file directly to avoid loading a repository from the previous notebook.
    spec = importlib.util.spec_from_file_location("confidence_resnet", REPO / "cifar10_models/resnet.py")
    resnet_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resnet_module)
    model = resnet_module.resnet18(pretrained=False)
    model.load_state_dict(torch.load(WEIGHT, map_location="cpu", weights_only=True))
    model = model.to(DEVICE).eval()
    model.requires_grad_(False)
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2471, 0.2435, 0.2616)
    PREPROCESS_ID = json.dumps({"checkpoint": WEIGHT_SHA256, "commit": REPO_COMMIT,
                                "mean": MEAN, "std": STD})
    print("Frozen ResNet-18 ready.")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    clean_dataset = torchvision.datasets.CIFAR10(
        root=str(ROOT / "data"), train=False, download=True, transform=transform)
    clean_loader = torch.utils.data.DataLoader(
        clean_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=args.workers)
    assert len(clean_dataset) == 10000
    print("10,000 test images; original index order is preserved.")

    clean_cache = CACHE / "clean_logits.npz"
    use_cache = False
    if clean_cache.exists():
        with np.load(clean_cache, allow_pickle=False) as saved:
            if str(saved["provenance"]) == PREPROCESS_ID:
                Z, Y = saved["logits"].astype(np.float64), saved["labels"].astype(int)
                use_cache = True
    if not use_cache:
        logits_batches, label_batches = [], []
        with torch.inference_mode():
            for x, y in tqdm(clean_loader, desc="Clean inference"):
                logits_batches.append(model(x.to(DEVICE)).cpu().numpy())
                label_batches.append(y.numpy())
        Z = np.concatenate(logits_batches).astype(np.float64)
        Y = np.concatenate(label_batches).astype(int)
        np.savez_compressed(clean_cache, logits=Z, labels=Y, provenance=PREPROCESS_ID)
    assert Z.shape == (10000, K) and Y.shape == (10000,)
    assert np.isfinite(Z).all()
    assert np.array_equal(Y, np.asarray(clean_dataset.targets))
    assert np.all(np.bincount(Y, minlength=K) == 1000)
    print("Cached" if use_cache else "Computed", Z.shape)
    print(f"Raw accuracy on all 10K: {(Z.argmax(1) == Y).mean():.4f}")

    NEED_CORRUPTIONS = args.dataset in ("corruptions", "all")
    C_DIR = ROOT / "data" / "CIFAR-10-C"
    if NEED_CORRUPTIONS:
        C_DIR.mkdir(parents=True, exist_ok=True)
        needed = [f"{c}.npy" for c in CORRUPTIONS] + ["labels.npy"]
        if not all((C_DIR / name).exists() for name in needed):
            archive = CACHE / "CIFAR-10-C.tar"
            if not archive.exists():
                temporary = CACHE / "CIFAR-10-C.downloading.tar"
                url = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
                with requests.get(url, stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("Content-Length", 0)) or None
                    with temporary.open("wb") as target, tqdm(total=total, unit="B", unit_scale=True, desc="CIFAR-10-C") as bar:
                        for chunk in response.iter_content(4 << 20):
                            if chunk:
                                target.write(chunk)
                                bar.update(len(chunk))
                # Rename the file only after the download completes and the archive is readable.
                with tarfile.open(temporary) as tf:
                    tf.getmembers()
                temporary.replace(archive)
            with tarfile.open(archive) as tf:
                for member in tf.getmembers():
                    name = Path(member.name).name
                    if member.isfile() and name in needed and not (C_DIR / name).exists():
                        with tf.extractfile(member) as source, (C_DIR / (name + ".partial")).open("wb") as target:
                            shutil.copyfileobj(source, target)
                        (C_DIR / (name + ".partial")).replace(C_DIR / name)
        assert all((C_DIR / name).exists() for name in needed)
        labels_corrupt = np.load(C_DIR / "labels.npy", allow_pickle=False)
        assert labels_corrupt.shape in ((10000,), (50000,))
        assert np.array_equal(labels_corrupt, np.tile(Y, len(labels_corrupt)//len(Y)))
        print("CIFAR-10-C ready; labels match original clean indices.")
    else:
        print("Skipped CIFAR-10-C.")

    C = {}
    if NEED_CORRUPTIONS:
        mean_tensor = torch.tensor(MEAN).view(1, 3, 1, 1)
        std_tensor = torch.tensor(STD).view(1, 3, 1, 1)
        for corruption in CORRUPTIONS:
            cache_file = CACHE / f"corruption_{corruption}.npz"
            cached = False
            if cache_file.exists():
                with np.load(cache_file, allow_pickle=False) as saved:
                    if str(saved["provenance"]) == PREPROCESS_ID and np.array_equal(saved["labels"], Y):
                        C[corruption] = saved["logits"]
                        cached = True
            if not cached:
                images = np.load(C_DIR / f"{corruption}.npy", mmap_mode="r")
                assert images.shape == (50000, 32, 32, 3)
                blocks = []
                with torch.inference_mode():
                    for start in tqdm(range(0, len(images), BATCH_SIZE), desc=corruption):
                        x = torch.from_numpy(np.array(images[start:start+BATCH_SIZE], copy=True))
                        x = x.permute(0, 3, 1, 2).float() / 255
                        blocks.append(model(((x-mean_tensor)/std_tensor).to(DEVICE)).cpu().numpy())
                C[corruption] = np.concatenate(blocks).reshape(5, 10000, K)
                np.savez_compressed(cache_file, logits=C[corruption], labels=Y, provenance=PREPROCESS_ID)
                del images, blocks
            assert C[corruption].shape == (5, 10000, K)
            assert np.isfinite(C[corruption]).all()
            print(corruption, "cache reused" if cached else "computed")
    else:
        print("Skipped corruption inference.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("clean", "corruptions", "all"), default="clean")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Artifact directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch-size must be positive and workers must be nonnegative")
    prepare(args)


if __name__ == "__main__":
    main()
