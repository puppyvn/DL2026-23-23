# Data

This project uses two public datasets and one set of public pretrained models. Nothing is collected by hand. All
data is fetched by one script, `shared/data.py`, into one folder, `downloads/` at the repository root (not tracked by
Git), and every part reads from that folder.

| Dataset / asset | Official URL | Version used | Used by |
|---|---|---|---|
| CIFAR-10 | https://www.cs.toronto.edu/~kriz/cifar.html | Python version, `cifar-10-python.tar.gz` | Parts 1, 2, 3, 4 |
| CIFAR-10-C | https://zenodo.org/records/2535967 (DOI [10.5281/zenodo.2535967](https://doi.org/10.5281/zenodo.2535967)) | Zenodo record 2535967, **v1** (9 Jan 2019), `CIFAR-10-C.tar` | Parts 3, 4 |
| Pretrained CIFAR-10 models | https://github.com/huyvnphan/PyTorch_CIFAR10 | code commit `641cac2`, weight archive `state_dicts.zip` | Parts 2, 3, 4 |

## 1. CIFAR-10

- **Source:** Krizhevsky (2009), *Learning Multiple Layers of Features from Tiny Images*.
  https://www.cs.toronto.edu/~kriz/cifar.html
- **Version:** the Python version, `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`
  (about 163 MiB, MD5 `c58f30108f718f92721af3b95e74349a`). It is downloaded with `torchvision.datasets.CIFAR10`, which
  checks this MD5.
- **Content:** 60,000 RGB images of 32 × 32 pixels in 10 classes (airplane, automobile, bird, cat, deer, dog, frog,
  horse, ship, truck). There are 50,000 training images and 10,000 test images, with 1,000 test images per class.
- **Order:** every part uses the official test-set order (`test_batch`, index 0–9,999). The logits, CIFAR-10-C
  arrays and calibration splits all refer to these indices.
- **Location:** `downloads/cifar10/cifar-10-batches-py/`

## 2. CIFAR-10-C

- **Source:** Hendrycks & Dietterich (2019), *Benchmarking Neural Network Robustness to Common Corruptions and
  Perturbations*, ICLR. https://zenodo.org/records/2535967
- **Version:** v1 of Zenodo record 2535967 (published 9 January 2019), file `CIFAR-10-C.tar`: 2,918,471,680
  bytes, MD5 `56bf5dcef84df0e2308c6dcbcbbd8499` (as listed on Zenodo), SHA-256
  `c72763e101c723b7c507b96205f7e938912a5d587376173b825850cf3cb876a7` (as listed by TensorFlow Datasets). `shared/data.py` checks the size and SHA-256 after downloading and refuses any other file.
- **Content:** 19 corruptions of the 10,000 CIFAR-10 test images, each at severities 1–5. Each corruption is one
  file, `<name>.npy`, with shape `(50000, 32, 32, 3)` and dtype `uint8`. Rows `(s − 1)·10000 … s·10000 − 1` hold
  severity `s`, in the official test-set order. `labels.npy` has shape `(50000,)`.
- **Subset used:** five corruptions, chosen in the project plan to cover noise, blur, illumination, contrast and
  resolution: `gaussian_noise`, `motion_blur`, `brightness`, `contrast`, `pixelate`. All five severities are used,
  which gives 250,000 shifted images. Only these five files and `labels.npy` are extracted from the archive.
- **Location:** `downloads/CIFAR-10-C/`

## 3. Pretrained models

- **Source:** https://github.com/huyvnphan/PyTorch_CIFAR10. Model code at commit
  `641cac24371b17052b9bb6e56af1c83b5e97cd7f`. The weights come from the repository's `state_dicts.zip` (Google
  Drive id `17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq`, about 1 GB).
- **Checkpoint used in Parts 2–4:** `resnet18.pt`, SHA-256
  `72d30ca70e7d54e24a26628113604c40bc872eb8dc7a44f50ec58c3a1400f1b0` (93.07% clean test accuracy). It is used
  frozen, with no fine-tuning. The SHA-256 is checked when the model is loaded and recorded in every
  `run_manifest.json`.
- **Ten checkpoints used in Part 2 and in the depth/architecture comparison:** ResNet-18/34/50,
  VGG-11/13/16/19 (with batch norm), DenseNet-121/161/169. Their test-set logits are committed in
  `part_2/data/clean_logits.npz`, so no download is needed to reproduce those results.
- **Part 1** trains its own width-scaled ResNet-50 models from scratch (no pretrained weights; see Section 5).

## 4. Data splits

| Part | Training | Validation / calibration | Evaluation | How the split is made |
|---|---|---|---|---|
| 1 (width) | 45,000 CIFAR-10 train images | 5,000 CIFAR-10 train images (checkpoint selection) | 10,000 CIFAR-10 test images | `torch.utils.data.random_split`, seed 42, `train_split: 0.9` (`part_1/src/train.py`) |
| 1 (shift) | — | — | 10,000 test images under 5 synthetic shifts | `part_1/data/data.py` (Section 5) |
| 2 | — | — | 10,000 CIFAR-10 test images | none: the full test set |
| 3 | — | — | 10,000 test images + 250,000 CIFAR-10-C images | none: every test index at every severity |
| 4 | — | 5,000 test images (calibration) | other 5,000 test images (final test), clean and CIFAR-10-C | stratified, 500 per class in each half, seeds 42 (main), 1, 2 (`make_split` in `part_4/src/utils.py`) |

In Part 4 the same 5,000 final-test indices are used for clean CIFAR-10 and for every CIFAR-10-C corruption and
severity. Calibration maps (Temperature Scaling, Dirichlet with λ chosen by 5-fold cross-validation) are fitted on
the calibration half only. The split indices are saved as `split_seed<seed>.npz` with each run.

## 5. Preprocessing

**Parts 2, 3 and 4 (frozen pretrained ResNet-18):**

- Images stay at 32 × 32. There is no resizing, cropping or augmentation.
- `uint8` → float in [0, 1] (÷ 255), then per-channel normalisation with mean `(0.4914, 0.4822, 0.4465)` and std
  `(0.2471, 0.2435, 0.2616)`, the values the checkpoints were trained with.
- CIFAR-10-C images get exactly the same preprocessing as clean images.
- The model outputs logits. Confidence is the maximum softmax probability, and Part 2 divides the logits by T
  before the softmax. Logits are saved in float64 with a provenance string: checkpoint SHA-256, code commit and
  normalisation.

**Part 1 (width-scaled ResNet-50 trained from scratch, `part_1/config_train.yaml`):**

- **Training:** `Resize(224)` → `RandomCrop(224, padding=4)` → `RandomHorizontalFlip` → `ToTensor` →
  `Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))`.
- **Validation and test:** `Resize(224)` → `ToTensor` → the same `Normalize`.
- **Synthetic shift test sets** (`Data_loader.augmentation_validation` in `part_1/data/data.py`; parameters in
  `part_1/data/shift_data_attribute.md`), each applied to the 10,000 test images:

  | Shift | What it simulates |
  |---|---|
  | Camera | colour jitter, sensor noise σ = 0.03, mild blur or sharpening |
  | Lighting & Environment | strong colour jitter, haze |
  | Image Acquisition / Quality | pixelate ×2, Gaussian blur, smoothing, JPEG quality 40, noise σ = 0.04 |
  | Appearance / Colour | hue/saturation jitter, random per-channel scaling 0.75–1.25 |
  | Mixed | mild versions of all four |

  These transforms are random and drawn each time an image is loaded, without a fixed seed. Re-running gives
  statistically equivalent but not bit-identical shifted images.

## 6. Derived data (logits)

| File | Content | Produced by |
|---|---|---|
| `part_2/data/clean_logits.npz` (committed) | float32 logits `(10000, 10)` of the ten checkpoints + `labels` | `part_2/inference/extract_logits.py` |
| `part_2/data/processed/logits.npy`, `labels.npy`, `meta.json` | ResNet-18 logits checked and copied for Part 2, with SHA-256 | `part_2/data_preparation/prepare_data.py` |
| `downloads/logits/clean_logits.npz` | ResNet-18 logits on the clean test set, `(10000, 10)` | `python -m shared.prepare_data` |
| `downloads/logits/corruption_<name>.npz` | ResNet-18 logits on CIFAR-10-C, `(5, 10000, 10)` per corruption | `python -m shared.prepare_data` |

`shared.prepare_data` also checks that its clean logits reproduce the ResNet-18 logits committed in Part 2. Expect
100% prediction agreement; on a GPU the values can differ in the last digits, which doesn't change any reported
number.

## 7. Scripts to reproduce the data

From the repository root:

```bash
pip install -r requirements.txt

python -m shared.prepare_data                     # CIFAR-10, ResNet-18 weights, CIFAR-10-C (5 corruptions), all logits
python -m shared.prepare_data --only cifar10      # only CIFAR-10 (all Part 1 needs)
python -m shared.prepare_data --only cifar10c     # only CIFAR-10-C
python -m shared.prepare_data --delete-archive    # remove CIFAR-10-C.tar after extraction (saves 2.9 GB)
python part_2/inference/extract_logits.py         # optional: recompute Part 2's ResNet-18 logits from the checkpoint
```

| Step | Script / function | Output |
|---|---|---|
| Download + MD5 check of CIFAR-10 | `shared/data.py: cifar10_dataset` (torchvision) | `downloads/cifar10/` |
| Download + SHA-256 check of CIFAR-10-C, extract 5 corruptions | `shared/data.py: ensure_cifar10c` | `downloads/CIFAR-10-C/` |
| Model code + `resnet18.pt` + SHA-256 check | `shared/data.py: resnet18_weights`, `resnet18` | `downloads/models/` |
| Preprocessing + inference → logits | `shared/data.py: run_model`, `ensure_logits` | `downloads/logits/` |
| Part 4 calibration / final-test split | `part_4/src/utils.py: make_split` | `split_seed<seed>.npz` in each run folder |
| Part 1 train/validation split and transforms | `part_1/src/train.py: build_dataloaders` | in memory |
| Part 1 shift test sets | `part_1/data/data.py: Data_loader.augmentation_validation` | in memory |

Every step skips files that already exist. You can also run each part directly: it downloads only what it is
missing, into the same folder.

**Other locations.** Set `NNC_DATA_DIR=/path` (or pass `--data-dir /path`) to keep the downloads elsewhere, for
example `/content/drive/MyDrive/nnc_data` in Colab.

**Manual download.** You can download `CIFAR-10-C.tar` yourself from the Zenodo link above and put it in
`downloads/CIFAR-10-C/`; its checksum is verified before use. Alternatively, put the five `.npy` files and
`labels.npy` there directly; the archive is then not needed.

**Disk space:** about 0.2 GB for CIFAR-10, 1 GB for the weight archive, 2.9 GB for CIFAR-10-C (0.8 GB after
`--delete-archive`), and 0.1 GB for the logits.

## 8. Licences and citation

- CIFAR-10: A. Krizhevsky. *Learning Multiple Layers of Features from Tiny Images.* Technical report, University
  of Toronto, 2009.
- CIFAR-10-C: D. Hendrycks and T. Dietterich. *Benchmarking Neural Network Robustness to Common Corruptions and
  Perturbations.* ICLR 2019. Zenodo record 2535967, licence CC BY 4.0.
- Pretrained models: H. Phan, *PyTorch_CIFAR10*, https://github.com/huyvnphan/PyTorch_CIFAR10 (MIT licence).
