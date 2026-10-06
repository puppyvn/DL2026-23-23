"""
Model loading utility for CIFAR-10 pretrained checkpoints.
Primary anchor model: edadaltocg/resnet18_cifar10 from Hugging Face Hub.
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
from huggingface_hub import hf_hub_download


def build_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    """Standard CIFAR-10 adapted ResNet-18 (3x3 conv1, no maxpool)."""
    model = models.resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_cifar_resnet34(num_classes: int = 10) -> nn.Module:
    """Standard CIFAR-10 adapted ResNet-34 (3x3 conv1, no maxpool)."""
    model = models.resnet34(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_cifar_resnet50(num_classes: int = 10) -> nn.Module:
    """Standard CIFAR-10 adapted ResNet-50 (3x3 conv1, no maxpool)."""
    model = models.resnet50(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def load_cifar10_model(model_name: str = "edadaltocg/resnet18_cifar10", device: str = "cpu") -> nn.Module:
    """
    Load pretrained CIFAR-10 model from Hugging Face Hub.
    """
    print(f"[*] Loading model checkpoint: {model_name} on {device}...")
    
    # 1. Download weights from Hugging Face Hub
    weights_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    
    # 2. Build corresponding architecture
    if "resnet18" in model_name:
        model = build_cifar_resnet18(num_classes=10)
    elif "resnet34" in model_name:
        model = build_cifar_resnet34(num_classes=10)
    elif "resnet50" in model_name:
        model = build_cifar_resnet50(num_classes=10)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
        
    # 3. Load state dict
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    print(f"[*] Successfully loaded {model_name} (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
    return model
