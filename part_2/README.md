# When Can We Trust Neural Network Confidence? — Part 2

Part 2 asks whether two tools can be trusted to read a model's confidence: the **Reliability Diagram** (does a confidence of 0.8 mean 80% correct?) and the **Risk–Coverage curve** (are the most confident predictions the correct ones?). We create a confidence fault on purpose and check what each tool reports.

The fault: the logits of a pretrained CIFAR-10 ResNet-18 are divided by a temperature T before the softmax, with T = 0.5 (sharper, pushes towards overconfidence), T = 1 (baseline) and T = 2 (softer, pushes towards underconfidence). Dividing by T never changes the predicted class, so accuracy is the same at every T and only the confidence moves. Everything runs on the 10,000 clean CIFAR-10 test images; no weight is changed and nothing is trained.

## Repository layout

| Folder | Role | Entry point |
| --- | --- | --- |
| `data/` | input logits and labels (`clean_logits.npz`) | [data/README.md](data/README.md) |
| `data_preparation/` | checks the logits and writes the arrays used by the evaluation | `prepare_data.py` |
| `training/` | not applicable to Part 2, explained in [training/README.md](training/README.md) | — |
| `evaluation/` | the experiment: temperatures, reliability, risk–coverage, ranking checks, summary | `run_experiment.py` |
| `experiments/` | supporting experiments: bootstrap intervals, positive and reverse controls, fixed thresholds, ten checkpoints | `run_all.py` |
| `inference/` | recomputes the logits from the pretrained checkpoint on the test set | `extract_logits.py` |
| `demo/` | one test image (or your own image) at T = 0.5, 1, 2 | `demo.py` |
| `src/part2/` | shared code: settings, metrics, figures | `config.py`, `metrics.py`, `plots.py` |
| `tests/` | metric checks on synthetic data with known answers | `test_metrics.py` |
| `results/` | figures and tables produced by the evaluation | — |

## Installation

Python 3.10 or newer. The main pipeline needs CPU only.

```bash
git clone <this repository>
cd part2-final-repo
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

Run from the repository root:

```bash
python tests/test_metrics.py                 # metric code behaves as expected
python data_preparation/prepare_data.py      # -> data/processed/
python evaluation/run_experiment.py          # -> results/figures/, results/tables/
```

The whole pipeline takes a few seconds. `prepare_data.py` should print accuracy 93.07%, ECE 2.02% and ACE 2.41% at T = 1. `run_experiment.py` writes two figures and five tables and prints this summary:

| T | Accuracy | Mean confidence | ECE | ACE | Signed gap | Risk@80% | Risk@60% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 93.07% | 98.44% | 5.37% | 5.37% | +5.37% | 1.36% | 0.70% |
| 1 | 93.07% | 94.85% | 2.02% | 2.41% | +1.78% | 1.44% | 0.82% |
| 2 | 93.07% | 68.85% | 24.22% | 24.22% | −24.22% | 1.46% | 0.88% |

Ranking against T = 1: Spearman ρ = 0.9937 (T = 0.5) and 0.9982 (T = 2); 99.45% and 99.56% of the top-80% accepted set are the same images.

| Output | Content |
| --- | --- |
| `results/figures/fig_p2_reliability.png` | Reliability Diagram per T. Blue: accuracy per bin; pink: overconfidence gap; teal: underconfidence gap |
| `results/figures/fig_p2_risk_coverage.png` | Risk–Coverage curves of the three temperatures |
| `results/tables/p2_calibration.csv` | accuracy, mean confidence, ECE, ACE per T |
| `results/tables/p2_risk_coverage.csv` | Risk@80% and Risk@60% per T |
| `results/tables/p2_direction.csv` | signed gap and share of images below / above y = x per T |
| `results/tables/p2_ranking.csv` | change in Risk@k, Spearman ρ and top-k overlap vs T = 1 |
| `results/tables/p2_table_main.csv` | summary table above |
| `results/figures/fig_p2_positive_control.png`, `fig_p2_reverse_control.png`, `fig_p2_thresholds.png`, `fig_p2_robustness_10_models.png` | figures of the supporting experiments |
| `results/tables/p2_bootstrap_ci.csv`, `p2_diff_vs_T1.csv`, `p2_positive_control.csv`, `p2_reverse_control.csv`, `p2_thresholds.csv`, `p2_robustness_10_models.csv` | tables of the supporting experiments (proportions, not percentages, unless the column says %) |

### Supporting experiments

```bash
python experiments/run_all.py                # about one minute; or run each script on its own
```

| Script | What it computes | Output |
| --- | --- | --- |
| `bootstrap_ci.py` | 95% paired bootstrap intervals (1,000 resamples, seed 0) of each metric at every T and of its difference from T = 1, including AUROC | `p2_bootstrap_ci.csv`, `p2_diff_vs_T1.csv` |
| `positive_control.py` | Risk@80%, Risk@60%, E-AURC, AUROC and Spearman ρ after adding Gaussian noise (σ = 0.003 to 0.3, 20 runs each) to the T = 1 confidence, with oracle and random rankings | `fig_p2_positive_control.png`, `p2_positive_control.csv` |
| `reverse_control.py` | the same metrics for two confidences built to be calibrated (constant = accuracy; precision of the predicted class), and 200 random tie-breaks of the constant one | `fig_p2_reverse_control.png`, `p2_reverse_control.csv` |
| `thresholds.py` | coverage and risk of the rule "accept if confidence ≥ threshold" at every T, errors rejected at 80% coverage | `fig_p2_thresholds.png`, `p2_thresholds.csv` |
| `robustness_10_models.py` | the stress test on the ten checkpoints in `data/clean_logits.npz`, with intervals and differences from T = 1 | `fig_p2_robustness_10_models.png`, `p2_robustness_10_models.csv` |

Expected values for checking a run: ECE change vs T = 1 of +3.35 and +22.19 pp; Risk@80% change of −0.075 and +0.025 pp; AUROC 0.9014 / 0.8959 / 0.8914 at T = 0.5 / 1 / 2; constant confidence gives ECE 0.00% and AUROC 0.500.

Definitions: confidence is the maximum softmax probability; ECE uses 15 equal-width bins (lo, hi]; ACE uses 15 equal-mass bins; signed gap = mean confidence − accuracy; Risk@c is the error rate among the c·N most confident predictions, ranked by log-confidence (same order as confidence, but no ties at 1.0 when T = 0.5).

## Demo

```bash
python demo/demo.py --index 0
```

```text
test image #0: true class = cat
  T = 0.5  predicted cat        confidence 1.0000  correct  rank  6454 of 10000  accepted at 80% coverage
  T = 1.0  predicted cat        confidence 0.9826  correct  rank  5775 of 10000  accepted at 80% coverage
  T = 2.0  predicted cat        confidence 0.7213  correct  rank  5451 of 10000  accepted at 80% coverage
```

Use `--index` for any test image (0–9999) and `--coverage` for another operating point.

## Inference from the checkpoint (optional)

`data/clean_logits.npz` was produced by running the pretrained network on the test set. To regenerate it:

```bash
pip install -r requirements-inference.txt
python inference/extract_logits.py                                   # -> data/recomputed_logits.npz
python data_preparation/prepare_data.py --source data/recomputed_logits.npz
python evaluation/run_experiment.py
```

The first run clones the model code of [huyvnphan/PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10) into `external/`, downloads its weight archive (about 1 GB) and the CIFAR-10 test set. On a GPU the logits can differ from the saved ones in the last digits; accuracy is unchanged. To classify your own image with the same model:

```bash
python demo/demo.py --image path/to/image.png
```
