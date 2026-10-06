"""
Data loader module for clean CIFAR-10 test set and synthetic CIFAR-10-C corruptions.
Reads directly from pre-cached parquet or downloads if missing.
"""

import os
import io
import urllib.request
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from src.corruptions import corrupt_cifar_image

# Official CIFAR-10 normalization constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

PARQUET_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet"
PARQUET_FILE = "data/raw/cifar10_test.parquet"


def ensure_cifar10_test_parquet(save_path: str = PARQUET_FILE):
    """Ensure CIFAR-10 test set parquet exists, downloading if necessary."""
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"[*] Downloading CIFAR-10 test set (23MB) from Hugging Face...")
        urllib.request.urlretrieve(PARQUET_URL, save_path)
        print(f"[*] Downloaded successfully to: {save_path}")


class CIFAR10TestDataset(Dataset):
    """
    In-memory CIFAR-10 test dataset with on-the-fly corruption application.
    """
    def __init__(
        self,
        parquet_path: str = PARQUET_FILE,
        corruption_name: str = "clean",
        severity: int = 0
    ):
        ensure_cifar10_test_parquet(parquet_path)
        df = pd.read_parquet(parquet_path, engine="fastparquet")
        
        self.labels = df["label"].values.astype(np.int64)
        
        # Pre-decode bytes into PIL images or numpy arrays in memory for high speed
        print(f"[*] Decoding {len(df)} test images into memory...")
        self.images = []
        for img_bytes in df["img.bytes"].values:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            self.images.append(img)
            
        self.corruption_name = corruption_name
        self.severity = severity
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        target = self.labels[idx]
        
        if self.severity > 0 and self.corruption_name != "clean":
            img = corrupt_cifar_image(img, self.corruption_name, self.severity)
            
        tensor = self.transform(img)
        return tensor, target


# Global cache to avoid re-reading parquet multiple times across severities
_CACHED_BASE_IMAGES = None
_CACHED_BASE_LABELS = None


class FastCorruptedCIFAR10(Dataset):
    """
    Fast dataset reusing decoded base images from memory across different severity runs.
    """
    def __init__(self, images, labels, corruption_name="clean", severity=0):
        self.images = images
        self.labels = labels
        self.corruption_name = corruption_name
        self.severity = severity
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        target = self.labels[idx]
        
        if self.severity > 0 and self.corruption_name != "clean":
            img = corrupt_cifar_image(img, self.corruption_name, self.severity)
            
        tensor = self.transform(img)
        return tensor, target


def get_cifar10_test_loader(
    root: str = "data/raw",
    corruption_name: str = "clean",
    severity: int = 0,
    batch_size: int = 256,
    num_workers: int = 0
) -> DataLoader:
    """
    Get DataLoader for clean or corrupted CIFAR-10 test dataset.
    """
    global _CACHED_BASE_IMAGES, _CACHED_BASE_LABELS
    parquet_path = os.path.join(root, "cifar10_test.parquet")
    
    if _CACHED_BASE_IMAGES is None:
        ensure_cifar10_test_parquet(parquet_path)
        print("[*] Loading and caching CIFAR-10 test set in memory...")
        df = pd.read_parquet(parquet_path, engine="fastparquet")
        _CACHED_BASE_LABELS = df["label"].values.astype(np.int64)
        _CACHED_BASE_IMAGES = [
            Image.open(io.BytesIO(b)).convert("RGB")
            for b in df["img.bytes"].values
        ]
        print(f"[*] Cached {len(_CACHED_BASE_IMAGES)} test images successfully.")
        
    dataset = FastCorruptedCIFAR10(
        images=_CACHED_BASE_IMAGES,
        labels=_CACHED_BASE_LABELS,
        corruption_name=corruption_name,
        severity=severity
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return loader
