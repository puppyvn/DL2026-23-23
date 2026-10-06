"""Settings shared by every Part 2 script. Change them here, not inside the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repository root

# data
RAW_LOGITS = ROOT / "data" / "clean_logits.npz"     # input: logits of the pretrained checkpoints on the CIFAR-10 test set
MODEL_KEY = "ResNet_18"                             # the checkpoint analysed in Part 2
PROCESSED = ROOT / "data" / "processed"             # written by data_preparation/prepare_data.py

# outputs
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"

# experiment
TEMPERATURES = [0.5, 1.0, 2.0]       # 0.5 = sharper (overconfidence stress), 1 = baseline, 2 = softer (underconfidence stress)
N_BINS = 15                          # equal-width bins for ECE / reliability, equal-mass bins for ACE
COVERAGES = (0.8, 0.6)               # operating points reported as Risk@80% and Risk@60%
SEED = 0

# CIFAR-10 normalisation of the pretrained checkpoint (huyvnphan/PyTorch_CIFAR10)
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2471, 0.2435, 0.2616)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
