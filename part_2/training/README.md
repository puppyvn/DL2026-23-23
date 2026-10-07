# Training

Part 2 trains nothing, by design.

The question of Part 2 is whether two measurement tools, the Reliability Diagram and the Risk–Coverage curve, react correctly to a confidence fault whose direction we know in advance. To know the fault exactly, the model must stay fixed: we take a pretrained CIFAR-10 ResNet-18 ([huyvnphan/PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10)), change no weight, and only divide its logits by a temperature T chosen by hand (0.5, 1, 2). Dividing by T cannot change the predicted class, so any change the tools report comes from the confidence alone. Fitting a model, or fitting T, would remove that control.

The only computation that touches the network is inference on the 10,000 clean test images, in [`inference/extract_logits.py`](../inference/extract_logits.py).
