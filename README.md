# When Can We Trust Neural Network Confidence?

| Part | Question | Folder |
|---|---|---|
| 1 | Why not trust raw confidence? Width, depth, architecture, distribution shift | `part_1/` |
| 2 | How do we check confidence? Controlled temperature stress test | `part_2/` |
| 3 | Can we trust the model forever? Raw confidence under CIFAR-10-C shift | `part_3/` |
| 4 | What to do after miscalibration? Temperature Scaling and Dirichlet, clean and under shift | `part_4/` |

## Data: downloaded once for all parts

Every download lives in one folder, `downloads/` at the repository root (not tracked by Git). The code that
fetches it is in `shared/data.py`, and every part reads from there:

| Item | Size | Used by | Location |
|---|---|---|---|
| CIFAR-10 (train + test) | 170 MB | 1, 2, 3, 4 | `downloads/cifar10/` |
| huyvnphan ResNet-18 weights + model code | ~1 GB archive | 2 (optional re-inference), 3, 4 | `downloads/models/` |
| CIFAR-10-C, five corruptions | 2.9 GB archive | 3, 4 | `downloads/CIFAR-10-C/` |
| ResNet-18 logits, clean + CIFAR-10-C | ~20 MB | 3, 4 | `downloads/logits/` |

Prepare everything once from the repository root:

```bash
pip install -r shared/requirements.txt
python -m shared.prepare_data                     # everything
python -m shared.prepare_data --only cifar10      # only what Part 1 needs
python -m shared.prepare_data --delete-archive    # also delete CIFAR-10-C.tar after extracting (saves 2.9 GB)
```

Each step skips what already exists, so running it again costs nothing. You don't have to run it first: each
part downloads only what it needs, into the same folder, on first use. Files that the old per-part scripts
already downloaded (`part_3/artifacts`, `part_4/src/artifacts`, `part_2/external`) are moved into
`downloads/` automatically instead of being downloaded again.

To keep the downloads somewhere else, such as Google Drive in Colab, set `NNC_DATA_DIR=/path/to/folder` or pass
`--data-dir /path/to/folder`. If you extracted CIFAR-10-C yourself, put the five `.npy` files and `labels.npy` in
`downloads/CIFAR-10-C/`; the archive is then never downloaded.

## Running the parts

```bash
# Part 1: trains the width-scaled ResNet-50 models (long); CIFAR-10 comes from downloads/cifar10
cd part_1 && python -m src.train --config config_train.yaml && python experiments.py --config config_train.yaml && cd ..
# Part 2: uses the logits committed in part_2/data; no download needed
cd part_2 && python data_preparation/prepare_data.py && python evaluation/run_experiment.py && python experiments/run_all.py && cd ..
# Part 3 and Part 4: share downloads/logits
python part_3/run_part3.py
cd part_4/src && python experiments.py --part all --no-show && cd ../..
```

`run_colab.ipynb` runs Parts 2–4 on Colab: it downloads the data once, optionally into Google Drive.
