# Part 3 — Can we trust the model forever? (CIFAR-10-C)

Raw confidence (T = 1) of a frozen CIFAR-10 ResNet-18 as the input distribution shifts further from the training
data. Same checkpoint as Parts 2 and 4 (`huyvnphan/PyTorch_CIFAR10` ResNet-18, 93.07% clean accuracy), official
CIFAR-10-C (Hendrycks & Dietterich, 2019), severities 1–5, five corruptions: Gaussian noise, motion blur,
brightness, contrast, pixelate. Nothing is trained.

| Plan | Question | Output |
|---|---|---|
| 3.1 | Does confidence fall as fast as accuracy? | `part3_boxplots` (per-image confidence vs accuracy of fixed 500-image batches), `part3_summary.csv` |
| 3.2 | Does calibration drift away from y = x? | `part3_reliability` (ECE per severity), `part3_pooled_metrics.csv` |
| 3.3 | Is confidence still useful for selective prediction? | `part3_risk_coverage`, Risk@60/80 and AUROC in `part3_summary.csv` |
| — | Which corruptions hurt most? | `part3_per_corruption.csv` (every corruption × severity, all 10K images) |

## Results (`outputs/`)

Five corruptions pooled at each severity:

| Severity | Accuracy | Median confidence | Median batch accuracy | Gap (conf − acc) | ECE | Risk@80% | AUROC |
|---|---|---|---|---|---|---|---|
| Clean | 93.07% | 98.39% | 93.0% | +1.78 | 2.02% | 1.44% | 0.896 |
| 1 | 90.32% | 98.34% | 91.0% | +3.42 | 3.43% | 2.62% | 0.890 |
| 2 | 84.88% | 98.18% | 86.8% | +6.70 | 6.70% | 6.62% | 0.865 |
| 3 | 77.80% | 97.90% | 79.6% | +11.49 | 11.49% | 13.01% | 0.847 |
| 4 | 71.03% | 97.24% | 76.5% | +16.47 | 16.47% | 20.21% | 0.826 |
| 5 | 56.63% | 95.22% | 67.9% | +29.05 | 29.05% | 36.70% | 0.774 |

Accuracy drops by 36 points while median confidence drops by about 3, and the gap equals the ECE from severity 2
onward: the model becomes increasingly **overconfident**. Confidence also ranks predictions worse (AUROC 0.896 →
0.774), so it loses value for accepting or deferring predictions. Corruptions differ a lot: at severity 5,
brightness keeps 87.98% accuracy (ECE 4.86%), while contrast falls to 19.77% (ECE 63.54%).

These outputs come from the Colab run `all_seed42_20261006T195259727606Z`. `run_part3.py` reproduces them exactly
from the same logit cache.

## Reproduce

```bash
cd part_3/cifar10c
pip install -r requirements.txt
python prepare_data.py      # ResNet-18 weights + CIFAR-10 + CIFAR-10-C (2.9 GB), caches logits in ./artifacts/cache
python run_part3.py         # tables + figures in ./artifacts/results/part3_<time>/
```

Or open `run_part3_colab.ipynb` in Colab (T4 GPU) and run all cells. Most of the time goes into downloading
CIFAR-10-C. If you already ran Part 4, reuse its cache instead of downloading again:
`python run_part3.py --root ../../part_4/src/artifacts`.

If the weight download from Google Drive fails, download `state_dicts.zip` from the
[PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10) README and put it in `artifacts/cache/`.

## Method details

- Confidence = maximum softmax probability, T = 1, 15 bins.
- ECE uses equal-width bins. ACE (`AdaptiveTopLabel`) uses equal-mass bins. Gap = mean confidence − accuracy
  (positive = overconfident).
- Risk@c is the error rate among the c·N most confident predictions. AUROC is how well confidence separates
  correct from wrong predictions (0.5 = no information).
- The 500-image batches are the same 20 index sets for every corruption and severity (seed 42). Each shifted
  severity therefore has 100 batches (20 × 5 corruptions). They describe a distribution, not independent
  replicates.
- `common.py` holds the metrics, `prepare_data.py` the downloads and inference, `run_part3.py` the analysis.

## Relation to the older files in `part_3/`

`part_3/src`, `part_3/experiments` and `part_3/outputs` hold an earlier exploratory run that used a different
checkpoint (`edadaltocg/resnet18_cifar10`, 94.46% clean accuracy) and home-made Gaussian noise that is much
stronger than CIFAR-10-C (σ up to 0.32 against 0.10). The results in this folder use the same model and benchmark
as the rest of the project and are the ones reported.
