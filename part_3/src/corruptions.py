"""
CIFAR-10 Corruption transformations matching Hendrycks & Dietterich (2019) CIFAR-10-C.
Implements the 5 core corruptions:
1. Gaussian Noise
2. Blur (Gaussian / Defocus Blur)
3. Brightness
4. Contrast
5. Pixelation
Across Severity levels 1 to 5.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def apply_gaussian_noise(image: Image.Image, severity: int) -> Image.Image:
    """Add Gaussian noise with variance increasing with severity."""
    # Scale sigmas for [0, 1] range
    c = [0.04, 0.08, 0.14, 0.22, 0.32][severity - 1]
    arr = np.array(image, dtype=np.float32) / 255.0
    noise = np.random.normal(loc=0.0, scale=c, size=arr.shape)
    arr = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def apply_blur(image: Image.Image, severity: int) -> Image.Image:
    """Apply Gaussian / Defocus blur with increasing kernel radius."""
    radius = [0.5, 1.0, 1.5, 2.0, 2.8][severity - 1]
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_brightness(image: Image.Image, severity: int) -> Image.Image:
    """Adjust brightness with factor increasing with severity."""
    factor = [1.2, 1.4, 1.6, 1.8, 2.1][severity - 1]
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def apply_contrast(image: Image.Image, severity: int) -> Image.Image:
    """Decrease contrast with factor decreasing with severity."""
    factor = [0.8, 0.6, 0.45, 0.3, 0.15][severity - 1]
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def apply_pixelation(image: Image.Image, severity: int) -> Image.Image:
    """Downsample image to low resolution and resize back via Nearest Neighbor."""
    resolutions = [28, 22, 16, 12, 8][severity - 1]
    w, h = image.size
    small = image.resize((resolutions, resolutions), resample=Image.Resampling.BOX)
    return small.resize((w, h), resample=Image.Resampling.NEAREST)


CORRUPTIONS = {
    "gaussian_noise": apply_gaussian_noise,
    "blur": apply_blur,
    "brightness": apply_brightness,
    "contrast": apply_contrast,
    "pixelation": apply_pixelation
}


def corrupt_cifar_image(image: Image.Image, corruption_name: str, severity: int) -> Image.Image:
    """
    Apply specified corruption and severity to a PIL Image.
    
    Args:
        image: PIL Image (32x32)
        corruption_name: one of ['gaussian_noise', 'blur', 'brightness', 'contrast', 'pixelation']
        severity: int from 1 to 5 (or 0 for clean uncorrupted)
        
    Returns:
        Corrupted PIL Image
    """
    if severity == 0 or corruption_name == "clean":
        return image
    if corruption_name not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption: {corruption_name}. Available: {list(CORRUPTIONS.keys())}")
    if not (1 <= severity <= 5):
        raise ValueError(f"Severity must be in [1, 5], got {severity}")
        
    return CORRUPTIONS[corruption_name](image, severity)
