# Data

`clean_logits.npz` is the only input. It holds the logits of pretrained CIFAR-10 checkpoints from [huyvnphan/PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10) on the 10,000 images of the official, unmodified CIFAR-10 test set, in the original order.

| Key | Shape | Content |
| --- | --- | --- |
| `ResNet_18` | (10000, 10) float32 | logits of ResNet-18, the model used in Part 2 |
| `labels` | (10000,) int64 | true class of each test image, 1,000 per class |
| other keys (`ResNet_34`, `VGG_16`, ...) | (10000, 10) | logits of other checkpoints, not used here |

Images were normalised with mean (0.4914, 0.4822, 0.4465) and std (0.2471, 0.2435, 0.2616) before the forward pass. `inference/extract_logits.py` recomputes the `ResNet_18` array from the checkpoint.

`processed/` is created by `data_preparation/prepare_data.py` and is not tracked by Git.
