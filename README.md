# When Can We Trust Neural Network Confidence?

An empirical study of whether the softmax confidence of CIFAR-10 classifiers reflects the probability of being
correct, how to check it, whether it survives distribution shift, and whether post-hoc calibration (Temperature
Scaling, Dirichlet Calibration) fixes it under normal conditions and under shift. No new method is proposed. All
results come from the scripts in this repository.

| Part | Question | Folder | Report |
|---|---|---|---|
| 1 | Why not trust raw confidence? Width, depth, architecture, distribution shift | `part_1/`, `part_2/experiments/depth_architecture.py` | Experiments 1, 3, 4 |
| 2 | How do we check confidence? Controlled temperature stress test (T = 0.5 / 1 / 2) | `part_2/` | Experiment 2 |
| 3 | Can we trust the model forever? Raw confidence under CIFAR-10-C shift | `part_3/` | Experiment 6 (raw confidence) |
| 4 | What to do after miscalibration? Temperature Scaling and Dirichlet, clean and under shift | `part_4/` | Experiments 5, 6, 7 |

Dataset information (official URLs, versions, splits, preprocessing, scripts): **[DATA.md](DATA.md)**.

## Repository structure

```
shared/            downloads + inference shared by all parts (data.py, prepare_data.py)
part_1/            width-scaled ResNet-50: training (src/), evaluation (experiments.py), results/
part_2/            temperature stress test: data_preparation/, evaluation/, experiments/, tests/, results/
part_3/            raw confidence under CIFAR-10-C: run_part3.py, common.py, outputs/
part_4/            post-hoc calibration: src/experiments.py, src/utils.py, output/ (output/shift_run/ = under shift)
run_colab.ipynb    one-click Colab run of Parts 2-4
requirements.txt   all dependencies
DATA.md            dataset documentation
downloads/         created on first run, not tracked: CIFAR-10, CIFAR-10-C, weights, logits
```

## Installation

You need Python 3.10–3.12 and Git. A CUDA GPU is required to train Part 1 and recommended for computing the
CIFAR-10-C logits. Everything else runs on a CPU.

```bash
git clone https://github.com/puppyvn/DL2026-23-23.git
cd DL2026-23-23
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
# For a GPU, first install the CUDA build of PyTorch from https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

Check the installation (runs in seconds and needs no download):

```bash
cd part_2 && python tests/test_metrics.py && cd ..
```

## Data (downloaded once for all parts)

```bash
python -m shared.prepare_data          # CIFAR-10 + ResNet-18 weights + CIFAR-10-C (5 corruptions) + logits
```

This downloads about 4 GB into `downloads/`, verifies the official checksums, and computes the ResNet-18 logits
once for Parts 3 and 4. Running it again downloads nothing. Use `--only cifar10` if you only run Part 1,
`--delete-archive` to free 2.9 GB after extraction, and `--data-dir PATH` (or `NNC_DATA_DIR=PATH`) to store the
data elsewhere. Details are in [DATA.md](DATA.md).

## Reproducing the main results

Run every command from the repository root unless it starts with `cd`. Each step prints its key numbers and writes
tables and figures. The values below are those in the report and in the committed result folders.

### Step 1: Experiment 2, temperature stress test (Part 2). CPU, about 2 minutes, no download.

```bash
cd part_2
python data_preparation/prepare_data.py        # checks the committed logits -> data/processed/
python evaluation/run_experiment.py            # Table 6, Figures 3-4
python experiments/run_all.py                  # bootstrap CIs, controls, thresholds, 10 checkpoints, depth/architecture, RQ1 profile
cd ..
```

| Output | Expected (T = 0.5 / 1 / 2) |
|---|---|
| `part_2/results/tables/p2_table_main.csv` | accuracy 93.07% at every T; ECE 5.37 / 2.02 / 24.22%; Risk@80% 1.36 / 1.44 / 1.46% |
| `part_2/results/tables/p2_bootstrap_ci.csv` | AUROC 0.9014 / 0.8959 / 0.8914 |
| `part_2/results/figures/fig_p2_reliability.png`, `fig_p2_risk_coverage.png`, `fig_p2_positive_control.png` | report Figures 3–4 |
| `part_2/results/tables/p2_confidence_ranges.csv`, `p2_class_gap.csv`, `p2_ece_noise_floor.csv` | RQ1 answer: confidence ≥ 0.9: 98.11% vs 97.24% accuracy; < 0.9: 69.07% vs 60.11%; cat +4.93 points; ECE 2.02% vs 0.47% for a perfectly calibrated model |

### Step 2: Experiments 3–4, depth and architecture. Included in Step 1.

`part_2/experiments/depth_architecture.py` (run by `run_all.py`) evaluates the ten public checkpoints at T = 1.

| Output | Expected |
|---|---|
| `part_2/results/tables/p2_depth_architecture.csv` | ResNet-18/34/50 ECE 2.02 / 2.63 / 2.23%; VGG-13 accuracy 94.21%, ECE 1.11%, ACE 1.74%; DenseNet-121→169 ECE 2.02→2.37% |
| `part_2/results/tables/p2_depth_differences.csv` | ResNet-34 − ResNet-18: ECE +0.60 points, 95% CI excludes 0; accuracy CI includes 0 |
| `part_2/results/figures/fig_p2_depth_architecture.png` | accuracy vs ECE / ACE with 95% bootstrap intervals |

### Step 3: Experiment 1, model width (Part 1). GPU, long.

```bash
python -m shared.prepare_data --only cifar10
cd part_1
python -m src.train --config config_train.yaml       # trains 0.5x, 0.75x, 1x, 1.5x, 2x ResNet-50 -> checkpoints/
python experiments.py --config config_train.yaml     # Experiment A (width) and B (synthetic shifts)
cd ..
```

Training uses the settings in `part_1/config_train.yaml`: 50 epochs, SGD with Nesterov momentum, learning rate
0.1, cosine schedule with 5-epoch warm-up, seed 42. The 2.0× model alone took about 14.6 h on the team's GPU
(`part_1/results/train/summary.csv`).

| Output | Expected |
|---|---|
| `part_1/results/experiments/summary_exp_a.csv` | accuracy 92.36 / 92.71 / 92.35 / 92.92 / 91.51%; ECE 2.16 / 2.04 / 2.40 / 2.16 / 1.94% (0.5× … 2×) |
| `part_1/results/experiments/summary_exp_b.csv` | 1.0× model under five synthetic shifts, e.g. Quality: accuracy 33.44%, ECE 36.02% |
| `part_1/results/experiments/fig_exp_a_*.png`, `fig_exp_b_*.png` | report Figure 1 (`fig_exp_b_*`: synthetic shifts, not used in the report) |

Results vary slightly between training runs. The synthetic shifts are random on every load (see DATA.md §5).

### Step 4: Experiment 6 (first half), raw confidence under CIFAR-10-C (Part 3). GPU about 5 minutes, CPU about 1 hour (first run).

```bash
python part_3/run_part3.py                      # -> part_3/results/part3_<time>/
```

| Output | Expected (clean → severity 5, five corruptions pooled) |
|---|---|
| `part3_summary.csv` | accuracy 93.07 → 56.63%; median confidence 98.39 → 95.22%; ECE 2.02 → 29.05%; Risk@80% 1.44 → 36.70%; AUROC 0.896 → 0.774 |
| `part3_per_corruption.csv` | severity 5: brightness 87.98% accuracy (ECE 4.86%), contrast 19.77% (ECE 63.54%) |
| `figures/part3_boxplots.png`, `part3_reliability.png`, `part3_risk_coverage.png` | report Figures 5–6 (Experiment 6) |

The committed copy of these results is in `part_3/outputs/`.

### Step 5: Experiments 5, 6 and 7, calibration under normal conditions and shift (Part 4). Seconds once Step 4 has run.

```bash
cd part_4/src
python experiments.py --part all --no-show --zip   # -> part_4/src/artifacts/results/all_seed42_<time>/
cd ../..
```

| Output | Expected |
|---|---|
| `part4_clean_seed42.csv` | fitted T = 1.0369; ECE raw 2.30%, Temperature Scaling 1.78%, Dirichlet 1.60%; accuracy 92.78 / 92.78 / 92.76% |
| `part4_split_summary.csv` | mean ECE over split seeds 42, 1, 2: 2.03 / 1.64 / 1.52% |
| `part4_shift_mean.csv` | severity 5, mean over corruptions: ECE raw 29.39%, TS 28.49%, Dirichlet 28.28% |
| `part4_shift_per_corruption.csv`, `part1b_per_corruption.csv` | per corruption × severity, used for the shift tables |
| `figures/part4_clean_reliability.png`, `part4_shift_reliability.png`, `part4_shift_metrics.png` | report Figures 2 and 8 (Experiments 5 and 6) |

The committed copies are in `part_4/output/` (clean) and `part_4/output/shift_run/` (under shift).

### Everything at once on Colab

Open `run_colab.ipynb` in Google Colab, select a T4 GPU, and run all cells. It runs the data step once, then
Steps 1, 2, 4 and 5, and downloads the result folders. Set `USE_DRIVE = True` to keep the downloads in Google
Drive between sessions. Part 1 (training) is not included.

## Reproducibility notes

- **Model:** Parts 2–4 use one frozen checkpoint, ResNet-18 from huyvnphan/PyTorch_CIFAR10 (SHA-256 `72d30ca7…`).
  It is checked when loaded and recorded in each `run_manifest.json`.
- **Seeds:** split seeds 42 / 1 / 2 (Part 4); bootstrap seed 0 with 1,000 resamples (Parts 2 and 4); batch seed 42
  (Part 3); training seed 42 (Part 1).
- **GPU differences:** recomputing the logits on a GPU changes values only in the last digits. Two full Colab runs
  agreed with each other and with the committed results to all reported decimals.
- **Bins:** 15 bins throughout. ECE uses equal-width bins; ACE uses equal-mass bins.

## Team

Nguyen Tat Hoang Viet (23BA14320), Luu Huu Tinh (23BA14283), Hoang Gia Thanh (23BA14261), Phan Duy Hoang
(23BA14117), Le Dac Duy (23BA14084), Phan Minh Trang (23BA14290), Le Huu Duc (2411139).

## References

- C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. On Calibration of Modern Neural Networks. ICML 2017.
- J. Nixon et al. Measuring Calibration in Deep Learning. CVPR Workshops 2019.
- D. Hendrycks, T. Dietterich. Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. ICLR 2019.
- M. Kull et al. Beyond Temperature Scaling: Dirichlet Calibration. NeurIPS 2019.
- Y. Ovadia et al. Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift. NeurIPS 2019.
- Y. Geifman, R. El-Yaniv. Selective Classification for Deep Neural Networks. NeurIPS 2017.
