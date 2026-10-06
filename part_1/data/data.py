import io
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Custom transforms (not available natively in torchvision)
# ---------------------------------------------------------------------------

class GaussianNoise:
    """Add Gaussian noise to a PIL image."""
    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img).astype(np.float32) / 255.0
        noise = np.random.randn(*arr.shape) * self.std
        arr = np.clip(arr + noise, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))


class JPEGCompression:
    """Simulate JPEG compression artefacts."""
    def __init__(self, quality: int = 20):
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return Image.open(buf).copy()


class Pixelate:
    """Pixelate an image by downscaling then upscaling."""
    def __init__(self, factor: int = 4):
        self.factor = factor

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        small = img.resize((w // self.factor, h // self.factor), Image.BOX)
        return small.resize((w, h), Image.NEAREST)


def _apply_smooth_filter(img: Image.Image) -> Image.Image:
    """Helper to apply smooth filter; defined at module level for picklability."""
    return img.filter(ImageFilter.SMOOTH)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Data_loader:
    def __init__(self, shift=None):
        self.shift = shift
        self.data     = None   # Training split (50 000 samples)
        self.val_data = None   # Test split     (10 000 samples) — used for evaluation
        self.shift1 = None   # Camera Shift
        self.shift2 = None   # Lighting & Environment Shift
        self.shift3 = None   # Image Acquisition / Quality Shift
        self.shift4 = None   # Appearance / Color Shift
        self.shift5 = None   # Mixed Realistic Shift

    # ------------------------------------------------------------------
    # load_data  (training split — 50 000 samples)
    # ------------------------------------------------------------------
    def load_data(self):
        self.data = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=False,
            transform=None,   # keep as PIL so augmentation works uniformly
        )

    # ------------------------------------------------------------------
    # load_validation_data  (test split — 10 000 samples)
    # ------------------------------------------------------------------
    def load_validation_data(self):
        """
        Load the official CIFAR-10 *test* split (10 000 samples) into
        self.val_data as PIL images (no transform applied).

        Call augmentation_validation() afterwards to build the 5 shifted
        versions of this test set for distribution-shift experiments.
        """
        self.val_data = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=False,
            transform=None,   # keep as PIL so shift pipelines apply uniformly
        )

    # ------------------------------------------------------------------
    # data_shift_visualize  (defined ABOVE augmentation per user request)
    # ------------------------------------------------------------------
    def data_shift_visualize(self, n_images: int = 5, save_path: str = "data/shift_data.jpg"):
        """
        Plot n_images samples from the original dataset and each of the 5
        shift distributions side-by-side and save the result to *save_path*.

        Layout: 6 rows × (1 label column + n_images image columns)
          row 0 – Original
          row 1 – Mild shift
          row 2 – Moderate shift
          row 3 – Noise / Blur shift
          row 4 – Severe corruption
          row 5 – Extreme shift
        """
        if self.data is None:
            raise RuntimeError("Call load_data() before data_shift_visualize().")
        if self.shift1 is None:
            raise RuntimeError("Call augmentation() before data_shift_visualize().")

        datasets = [
            (self.data,   "Original"),
            (self.shift1, "Camera Shift\n(white balance · color temp\nexposure · saturation\nsensor noise · sharpness)"),
            (self.shift2, "Lighting & Environment\n(brightness · gamma · contrast\nshadow · haze · saturation\ncolor temperature)"),
            (self.shift3, "Image Acquisition\n(downsample→upsample\nJPEG compression · defocus blur\nmotion blur · Gaussian noise)"),
            (self.shift4, "Appearance / Color Shift\n(hue · saturation · channel scaling\ncontrast · gamma · brightness)"),
            (self.shift5, "Mixed Realistic Shift\n(camera + lighting\n+ quality + color\nat moderate severity)"),
        ]

        n_rows     = len(datasets)
        # Accent colours per row (dark-on-white friendly palette)
        row_colors = ["#333333", "#1b7837", "#2166ac", "#d6600a", "#c0392b", "#6a3d9a"]

        # Grid: 1 narrow label column + n_images image columns
        col_widths = [2.8] + [3.2] * n_images
        fig, axes = plt.subplots(
            n_rows, n_images + 1,
            figsize=(sum(col_widths), n_rows * 3.4),
            gridspec_kw={"width_ratios": col_widths},
        )

        # ── White background everywhere ────────────────────────────────
        fig.patch.set_facecolor("white")
        for row in axes:
            for ax in row:
                ax.set_facecolor("white")

        for row_idx, (dataset, label) in enumerate(datasets):

            # ── Left label cell ────────────────────────────────────────
            ax_lbl = axes[row_idx][0]
            ax_lbl.set_xlim(0, 1)
            ax_lbl.set_ylim(0, 1)
            ax_lbl.axis("off")
            # Coloured vertical bar on the right edge of the label cell
            ax_lbl.axvline(x=0.92, ymin=0.05, ymax=0.95,
                           color=row_colors[row_idx], linewidth=4)
            ax_lbl.text(
                0.84, 0.50, label,
                ha="right", va="center",
                fontsize=9, fontweight="bold",
                color=row_colors[row_idx],
                transform=ax_lbl.transAxes,
                linespacing=1.6,
            )

            # ── Image cells ────────────────────────────────────────────
            for col_idx in range(n_images):
                ax = axes[row_idx][col_idx + 1]
                img_pil, _ = dataset[col_idx]

                # Datasets built with a transform pipeline return tensors;
                # convert back to numpy for display.
                if isinstance(img_pil, torch.Tensor):
                    img_np = img_pil.permute(1, 2, 0).numpy()
                    img_np = np.clip(img_np, 0, 1)
                else:
                    img_np = np.array(img_pil)

                ax.imshow(img_np, interpolation="nearest")
                # Thin coloured border matching the row colour
                for spine in ax.spines.values():
                    spine.set_edgecolor(row_colors[row_idx])
                    spine.set_linewidth(1.8)
                    spine.set_visible(True)
                ax.set_xticks([])
                ax.set_yticks([])

                # Column header on the top row only
                if row_idx == 0:
                    ax.set_title(
                        f"Sample #{col_idx + 1}",
                        color="black", fontsize=10, fontweight="bold", pad=7,
                    )

        fig.suptitle(
            "Distribution Shift Comparison — CIFAR-10",
            color="black", fontsize=14, fontweight="bold", y=1.005,
        )
        plt.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
        plt.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[data_shift_visualize] Saved → {save_path}")

    # ------------------------------------------------------------------
    # augmentation
    # ------------------------------------------------------------------
    def augmentation(self):
        """
        Build 5 shifted copies of self.data using distinct augmentation
        pipelines and store them in self.shift1 … self.shift5.

        shift1 – Camera Shift
            Simulates images from a different camera (sensor/brand/config):
            white balance, color temperature, slight exposure change,
            slight saturation/contrast change, mild sensor noise, mild
            sharpening/blur.

        shift2 – Lighting & Environment Shift
            Simulates different lighting conditions (day/night, indoor/outdoor,
            uneven or dim light): brightness, gamma, contrast, shadow, mild
            haze, saturation, color temperature.

        shift3 – Image Acquisition / Quality Shift
            Simulates lower-quality capture in production: downsampling →
            upsampling, JPEG compression, defocus blur, motion blur, mild
            Gaussian noise.

        shift4 – Appearance / Color Shift
            Simulates a different appearance domain (different white balance,
            environment colour, camera processing pipeline): hue shift,
            saturation shift, channel-wise colour scaling, contrast, gamma,
            slight brightness shift.

        shift5 – Mixed Realistic Shift
            Combines moderate versions of Camera Shift + Lighting Shift +
            Image Quality Shift + Color Shift to mimic real production
            environments that simultaneously face multiple subtle changes.
        """
        if self.data is None:
            raise RuntimeError("Call load_data() before augmentation().")

        # ── 1. Camera Shift ────────────────────────────────────────────
        # Simulate a different camera: altered white-balance/color-temp,
        # slight exposure & saturation/contrast change, mild sensor noise,
        # and mild sharpening or blur.
        camera_transform = transforms.Compose([
            # Slight exposure change (brightness) + contrast/saturation
            transforms.ColorJitter(
                brightness=0.15,   # ±15 % exposure
                contrast=0.10,     # mild contrast shift
                saturation=0.15,   # mild saturation shift (colour temperature)
                hue=0.05,          # tiny white-balance hue tilt
            ),
            # Mild sensor noise
            GaussianNoise(std=0.03),
            # Mild sharpening/blur to mimic a different lens/sensor sharpness
            transforms.RandomChoice([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 0.8)),
                transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0),
            ]),
            transforms.ToTensor(),
        ])

        # ── 2. Lighting & Environment Shift ───────────────────────────
        # Simulate different ambient conditions: brightness, gamma,
        # contrast, shadow, mild haze, saturation, colour temperature.
        lighting_transform = transforms.Compose([
            # Random brightness & gamma-like contrast swing
            transforms.ColorJitter(
                brightness=0.40,   # broad lighting range (dim ↔ bright)
                contrast=0.35,     # contrast varies with scene lighting
                saturation=0.30,   # saturation shifts with colour temperature
                hue=0.08,          # warm/cool colour temperature tilt
            ),
            # Mild haze: slight additive whitening via sharpness reduction
            transforms.RandomAdjustSharpness(sharpness_factor=0.4, p=0.5),
            transforms.ToTensor(),
        ])

        # ── 3. Image Acquisition / Quality Shift ──────────────────────
        # Simulate lower-quality capture: downsampling → upsampling,
        # JPEG compression, defocus blur, motion blur, mild Gaussian noise.
        quality_transform = transforms.Compose([
            # Downsampling → upsampling (lower resolution camera)
            Pixelate(factor=2),
            # Defocus / Gaussian blur
            transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)),
            # Motion blur via PIL ImageFilter
            transforms.RandomApply(
                [transforms.Lambda(_apply_smooth_filter)],
                p=0.5,
            ),
            # JPEG compression (network transmission / storage artefacts)
            JPEGCompression(quality=40),
            # Mild sensor / quantisation noise
            GaussianNoise(std=0.04),
            transforms.ToTensor(),
        ])

        # ── 4. Appearance / Color Shift ───────────────────────────────
        # Simulate a different colour domain: hue shift, saturation shift,
        # channel-wise colour scaling, contrast, gamma, slight brightness.
        color_transform = transforms.Compose([
            # Hue shift + saturation shift (dominant appearance change)
            transforms.ColorJitter(
                brightness=0.10,   # slight brightness shift
                contrast=0.20,     # moderate contrast / gamma
                saturation=0.50,   # strong saturation domain shift
                hue=0.20,          # noticeable hue / white-balance shift
            ),
            # Channel-wise colour scaling: randomly amplify one channel
            transforms.Lambda(self._random_channel_scale),
            transforms.ToTensor(),
        ])

        # ── 5. Mixed Realistic Shift ───────────────────────────────────
        # Moderate combination of Camera + Lighting + Quality + Color,
        # mimicking a real production environment with multiple covariate
        # shifts at once but each at a mild-to-moderate severity.
        mixed_transform = transforms.Compose([
            # --- Camera component (mild) ---
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.08,
                saturation=0.10,
                hue=0.04,
            ),
            # --- Lighting component (mild) ---
            transforms.ColorJitter(
                brightness=0.20,
                contrast=0.15,
                saturation=0.15,
            ),
            # --- Quality component (mild) ---
            transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 1.0)),
            JPEGCompression(quality=55),
            GaussianNoise(std=0.025),
            # --- Color component (mild) ---
            transforms.ColorJitter(
                saturation=0.25,
                hue=0.10,
            ),
            transforms.ToTensor(),
        ])

        def _apply(dataset, tfm):
            """Return a new CIFAR-10 dataset-like object sharing the same
            underlying samples but using *tfm* as the transform."""
            new_ds = torchvision.datasets.CIFAR10.__new__(
                torchvision.datasets.CIFAR10
            )
            # Copy all attributes from the original dataset
            new_ds.__dict__.update(dataset.__dict__)
            new_ds.transform = tfm
            return new_ds

        self.shift1 = _apply(self.data, camera_transform)
        self.shift2 = _apply(self.data, lighting_transform)
        self.shift3 = _apply(self.data, quality_transform)
        self.shift4 = _apply(self.data, color_transform)
        self.shift5 = _apply(self.data, mixed_transform)

        print("[augmentation] 5 distribution shifts created:")
        print("  self.shift1 → Camera Shift")
        print("  self.shift2 → Lighting & Environment Shift")
        print("  self.shift3 → Image Acquisition / Quality Shift")
        print("  self.shift4 → Appearance / Color Shift")
        print("  self.shift5 → Mixed Realistic Shift")

    # ------------------------------------------------------------------
    # augmentation_validation
    # ------------------------------------------------------------------
    def augmentation_validation(self):
        """
        Apply the exact same 5 distribution-shift pipelines as augmentation(),
        but to self.val_data (the official CIFAR-10 *test* split) instead of
        the training split.

        Results are stored in self.shift1 … self.shift5 (overwriting any
        values set by a previous augmentation() call).

        Call load_validation_data() before calling this method.

        This is the correct approach for experiments: models should be
        evaluated on held-out test data, not the data they were trained on.
        """
        if self.val_data is None:
            raise RuntimeError(
                "Call load_validation_data() before augmentation_validation()."
            )

        # ── 1. Camera Shift ────────────────────────────────────────────
        camera_transform = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.10,
                saturation=0.15,
                hue=0.05,
            ),
            GaussianNoise(std=0.03),
            transforms.RandomChoice([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 0.8)),
                transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0),
            ]),
            transforms.ToTensor(),
        ])

        # ── 2. Lighting & Environment Shift ───────────────────────────
        lighting_transform = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.40,
                contrast=0.35,
                saturation=0.30,
                hue=0.08,
            ),
            transforms.RandomAdjustSharpness(sharpness_factor=0.4, p=0.5),
            transforms.ToTensor(),
        ])

        # ── 3. Image Acquisition / Quality Shift ──────────────────────
        quality_transform = transforms.Compose([
            Pixelate(factor=2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)),
            transforms.RandomApply(
                [transforms.Lambda(_apply_smooth_filter)],
                p=0.5,
            ),
            JPEGCompression(quality=40),
            GaussianNoise(std=0.04),
            transforms.ToTensor(),
        ])

        # ── 4. Appearance / Color Shift ───────────────────────────────
        color_transform = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.20,
                saturation=0.50,
                hue=0.20,
            ),
            transforms.Lambda(self._random_channel_scale),
            transforms.ToTensor(),
        ])

        # ── 5. Mixed Realistic Shift ───────────────────────────────────
        mixed_transform = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.08,
                saturation=0.10,
                hue=0.04,
            ),
            transforms.ColorJitter(
                brightness=0.20,
                contrast=0.15,
                saturation=0.15,
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 1.0)),
            JPEGCompression(quality=55),
            GaussianNoise(std=0.025),
            transforms.ColorJitter(
                saturation=0.25,
                hue=0.10,
            ),
            transforms.ToTensor(),
        ])

        def _apply(dataset, tfm):
            """Return a new CIFAR-10 dataset-like object sharing the same
            underlying samples but using *tfm* as the transform."""
            new_ds = torchvision.datasets.CIFAR10.__new__(
                torchvision.datasets.CIFAR10
            )
            new_ds.__dict__.update(dataset.__dict__)
            new_ds.transform = tfm
            return new_ds

        self.shift1 = _apply(self.val_data, camera_transform)
        self.shift2 = _apply(self.val_data, lighting_transform)
        self.shift3 = _apply(self.val_data, quality_transform)
        self.shift4 = _apply(self.val_data, color_transform)
        self.shift5 = _apply(self.val_data, mixed_transform)

        print("[augmentation_validation] 5 distribution shifts created on TEST split:")
        print("  self.shift1 → Camera Shift")
        print("  self.shift2 → Lighting & Environment Shift")
        print("  self.shift3 → Image Acquisition / Quality Shift")
        print("  self.shift4 → Appearance / Color Shift")
        print("  self.shift5 → Mixed Realistic Shift")

    # ------------------------------------------------------------------
    # _random_channel_scale  (helper for shift4)
    # ------------------------------------------------------------------
    @staticmethod
    def _random_channel_scale(img: Image.Image) -> Image.Image:
        """Randomly scale individual RGB channels to simulate different
        camera processing pipelines / white-balance offsets."""
        arr = np.array(img).astype(np.float32)
        # Each channel scaled independently by a factor in [0.75, 1.25]
        scales = np.random.uniform(0.75, 1.25, size=(1, 1, 3))
        arr = np.clip(arr * scales, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

