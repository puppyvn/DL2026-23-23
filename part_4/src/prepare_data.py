"""Part 4 needs the clean and CIFAR-10-C logits of the frozen ResNet-18.

They are prepared once for the whole repository by shared/prepare_data.py and stored in downloads/logits
(Part 3 reads the same files). This wrapper only calls it; experiments.py also computes them on first use.

    python prepare_data.py            # same as: python -m shared.prepare_data   (from the repository root)
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from shared.prepare_data import main   # noqa: E402

if __name__ == "__main__":
    main()
