import copy
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HUB_SOURCE   = "pytorch/vision"
DEFAULT_MODEL = "resnet50"
DEFAULT_WEIGHTS = "IMAGENET1K_V2"

# Scale factors for the resize_model experiment
SCALE_FACTORS: List[float] = [0.5, 0.75, 1.0, 1.5, 2.0]

# ResNet layer config: (block_type, [layers], base_channels)
# base_channels are the output channels of the four stages at scale 1×.
_RESNET_CONFIGS = {
    "resnet18":  (BasicBlock,  [2, 2, 2, 2],  [64, 128, 256, 512]),
    "resnet34":  (BasicBlock,  [3, 4, 6, 3],  [64, 128, 256, 512]),
    "resnet50":  (Bottleneck,  [3, 4, 6, 3],  [64, 128, 256, 512]),
    "resnet101": (Bottleneck,  [3, 4, 23, 3], [64, 128, 256, 512]),
    "resnet152": (Bottleneck,  [3, 8, 36, 3], [64, 128, 256, 512]),
}


# ---------------------------------------------------------------------------
# Helper: build a width-scaled ResNet from scratch
# ---------------------------------------------------------------------------

def _build_scaled_resnet(
    arch: str,
    scale: float,
    num_classes: int = 1000,
) -> ResNet:
    """
    Construct a ResNet whose channel widths at every stage are multiplied
    by *scale* relative to the standard configuration.

    The stem conv (64 ch) is also scaled so the whole network is consistent.
    Channel counts are rounded to the nearest even integer (≥ 2).
    """
    if arch not in _RESNET_CONFIGS:
        raise ValueError(
            f"Unsupported arch '{arch}'. Choose from {list(_RESNET_CONFIGS)}."
        )
    block, layers, base_ch = _RESNET_CONFIGS[arch]

    # Scale each stage's width and ensure it is even & ≥ 2
    def _scale(ch: int) -> int:
        return max(2, round(ch * scale / 2) * 2)

    stem_ch = _scale(64)
    w1, w2, w3, w4 = [_scale(c) for c in base_ch]

    # torchvision ResNet accepts `width_per_group` only for resnext variants,
    # so we build the model normally then monkey-patch the stem + layer dims
    # via the standard constructor kwargs that torchvision exposes:
    #   ResNet(block, layers, num_classes, zero_init_residual,
    #          groups, width_per_group, replace_stride_with_dilation, norm_layer)
    # The cleanest approach is to subclass and override _make_layer widths
    # through the `planes` argument, which is what the constructor does
    # sequentially.  We replicate that here.

    model = _ScaledResNet(
        block       = block,
        layers      = layers,
        stem_width  = stem_ch,
        stage_widths= (w1, w2, w3, w4),
        num_classes = num_classes,
    )
    return model


class _ScaledResNet(ResNet):
    """
    ResNet with fully configurable per-stage channel widths.

    We override __init__ rather than calling super().__init__ with a width
    factor because torchvision's `width_per_group` only scales groups, not
    the overall stage width.
    """

    def __init__(
        self,
        block,
        layers,
        stem_width: int,
        stage_widths: Tuple[int, int, int, int],
        num_classes: int = 1000,
        norm_layer=None,
    ):
        # Initialise nn.Module directly — skip ResNet.__init__ so we can
        # set inplanes ourselves before calling _make_layer.
        nn.Module.__init__(self)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes    = stem_width
        self.dilation    = 1
        self.groups      = 1
        self.base_width  = 64  # kept at 64 for Bottleneck expansion logic

        w1, w2, w3, w4 = stage_widths

        # Stem
        self.conv1   = nn.Conv2d(3, stem_width, kernel_size=7, stride=2,
                                 padding=3, bias=False)
        self.bn1     = norm_layer(stem_width)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        self.layer1 = self._make_layer(block, w1, layers[0])
        self.layer2 = self._make_layer(block, w2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, w3, layers[2], stride=2)
        self.layer4 = self._make_layer(block, w4, layers[3], stride=2)

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        expansion    = block.expansion
        self.fc      = nn.Linear(w4 * expansion, num_classes)

        # Weight initialisation (same as torchvision)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Model:
    """
    Wrapper responsible for:
      • load_model()      – load a pretrained model from torch.hub
      • count_parameter() – count trainable parameters of self.model
      • resize_model()    – produce scaled variants (0.5× … 2×) of a model
    """

    def __init__(self, arch: str = DEFAULT_MODEL, device: str = "cpu"):
        """
        Parameters
        ----------
        arch   : torchvision hub model key, e.g. 'resnet50'
        device : 'cpu' or 'cuda'
        """
        self.arch   = arch
        self.device = torch.device(device)
        self.model  = None   # populated by load_model()

    # ------------------------------------------------------------------
    # load_model
    # ------------------------------------------------------------------
    def load_model(self) -> nn.Module:
        """
        Load *self.arch* from torch.hub with pretrained ImageNet weights,
        set to eval mode and move to *self.device*.

        Returns
        -------
        nn.Module  (also stored in self.model)
        """
        model = torch.hub.load(
            HUB_SOURCE,
            self.arch,
            weights=DEFAULT_WEIGHTS,
            verbose=False,
        )
        model.eval()
        model.to(self.device)
        self.model = model
        print(f"[✓] Loaded '{self.arch}' → {self.device}  "
              f"({self.count_parameter():,} trainable params)")
        return model

    # ------------------------------------------------------------------
    # count_parameter
    # ------------------------------------------------------------------
    def count_parameter(self, model: nn.Module = None) -> int:
        """
        Count the number of trainable (requires_grad) parameters.

        Parameters
        ----------
        model : nn.Module, optional
            If None, uses self.model.

        Returns
        -------
        int  – total trainable parameter count
        """
        target = model if model is not None else self.model
        if target is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        return sum(p.numel() for p in target.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # resize_model
    # ------------------------------------------------------------------
    def resize_model(
        self,
        model: nn.Module = None,
        scales: List[float] = None,
        num_classes: int = 1000,
    ) -> List[Tuple[float, nn.Module]]:
        """
        Build width-scaled variants of the given model architecture.

        Each variant is constructed **from scratch** (random init) with the
        same layer depth as the original but with every stage's channel count
        multiplied by *scale*.  This lets you study how model capacity (size)
        affects confidence calibration independently of pretrained weights.

        Parameters
        ----------
        model       : nn.Module, optional
            Reference model — used only to infer the architecture name.
            Falls back to self.model if None.
        scales      : list of float, optional
            Width multipliers to apply. Defaults to [0.5, 0.75, 1.0, 1.5, 2.0].
        num_classes : int
            Number of output classes for the scaled models.

        Returns
        -------
        List of (scale_factor, nn.Module) tuples, one per entry in *scales*.
        The modules are on the same device as self.device, in eval mode.

        Example
        -------
        >>> m = Model(arch='resnet50', device='cpu')
        >>> m.load_model()
        >>> variants = m.resize_model()
        >>> for scale, net in variants:
        ...     print(f"{scale}× → {m.count_parameter(net):,} params")
        """
        if scales is None:
            scales = SCALE_FACTORS

        # Resolve architecture name
        ref = model if model is not None else self.model
        arch = self.arch  # canonical name used at construction time

        if arch not in _RESNET_CONFIGS:
            raise ValueError(
                f"resize_model currently supports ResNet architectures "
                f"({list(_RESNET_CONFIGS)}). Got arch='{arch}'."
            )

        scaled_models: List[Tuple[float, nn.Module]] = []

        print(f"\n[resize_model] Building {len(scales)} scaled variants of "
              f"'{arch}' (num_classes={num_classes})")
        print(f"{'Scale':>8}  {'Params':>14}  {'Relative':>10}")
        print("─" * 40)

        base_params = None
        for scale in scales:
            net = _build_scaled_resnet(arch, scale, num_classes=num_classes)
            net.eval()
            net.to(self.device)

            n_params = self.count_parameter(net)
            if base_params is None and scale == 1.0:
                base_params = n_params
            relative = (f"{n_params / base_params:.3f}×"
                        if base_params else "—")
            print(f"{scale:>7.2f}×  {n_params:>14,}  {relative:>10}")

            scaled_models.append((scale, net))

        print("─" * 40)
        print(f"[✓] resize_model done — {len(scaled_models)} variants ready.\n")
        return scaled_models