"""Model definitions for corruption detection.

Two architectures share one interface, :class:`CorruptionNet`, a dual-head
classifier:

``binary_logits``
    2 logits — *is this frame corrupted at all?*  This is the question a driver
    or QA harness actually asks, and it stays accurate even for artifact types
    the model has not been trained on.
``class_logits``
    ``len(CLASS_NAMES)`` logits — *which artifact is it?*  Useful for triage:
    knowing whether a bad frame is tearing (scanout) or block corruption
    (memory) points at different parts of the stack.

Training both heads from one trunk is cheap and the binary head acts as an
auxiliary task that regularises the shared features.

Architectures
-------------
``small_cnn``
    ~1M parameter CNN written from scratch.  Four stride-2 blocks with
    BatchNorm and a global-average-pool head.  Fast enough to train on CPU and
    small enough to plausibly run per-frame alongside a game.
``resnet18``
    torchvision ResNet18, optionally initialised from ImageNet weights, with
    the classifier replaced by the dual head.  Supports freezing the trunk for
    linear probing.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from . import config


class CorruptionOutput(NamedTuple):
    """Forward-pass result: logits for both heads."""

    binary_logits: torch.Tensor   # (B, 2)
    class_logits: torch.Tensor    # (B, num_classes)


def _conv_block(in_ch: int, out_ch: int, dropout: float) -> nn.Sequential:
    """Conv-BN-ReLU pair ending in a stride-2 downsample.

    Two 3x3 convolutions per resolution level give the receptive field needed
    to see block edges and tear seams, while the stride-2 second conv halves
    the spatial size instead of a pooling layer.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Dropout2d(dropout),
    )


class SmallCNN(nn.Module):
    """Compact from-scratch trunk producing a ``feature_dim``-wide embedding.

    Args:
        width: Base channel count; levels use ``width * (1, 2, 4, 8)``.
        dropout: Spatial dropout probability inside each block.
        in_channels: Input channel count (3 for RGB frames).

    Attributes:
        feature_dim: Width of the embedding returned by :meth:`forward`.
    """

    def __init__(
        self,
        width: int = config.DEFAULT_WIDTH,
        dropout: float = config.DEFAULT_DROPOUT,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        channels = (width, width * 2, width * 4, width * 8)
        blocks = []
        previous = in_channels
        for out_ch in channels:
            blocks.append(_conv_block(previous, out_ch, dropout))
            previous = out_ch

        self.features = nn.Sequential(*blocks)
        # Global average pooling keeps the model resolution-agnostic: artifacts
        # are local patterns, so the head should not depend on input size.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, H, W)`` frames to ``(B, feature_dim)`` embeddings."""
        return torch.flatten(self.pool(self.features(x)), 1)


class CorruptionNet(nn.Module):
    """Trunk plus binary and multi-class heads.

    Args:
        arch: One of :data:`src.config.SUPPORTED_ARCHS`.
        num_classes: Multi-class head width.
        pretrained: ``resnet18`` only — load ImageNet weights.  Requires network
            access on first use; ignored by ``small_cnn``.
        freeze_backbone: ``resnet18`` only — train the heads only.
        width: ``small_cnn`` only — base channel count.
        dropout: Dropout applied inside the trunk and before the heads.
    """

    def __init__(
        self,
        arch: str = config.DEFAULT_ARCH,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        width: int = config.DEFAULT_WIDTH,
        dropout: float = config.DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        if arch not in config.SUPPORTED_ARCHS:
            raise ValueError(
                f"unsupported arch {arch!r}; expected one of {config.SUPPORTED_ARCHS}"
            )
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        self.arch = arch
        self.num_classes = num_classes

        if arch == "small_cnn":
            self.backbone: nn.Module = SmallCNN(width=width, dropout=dropout)
            feature_dim = self.backbone.feature_dim
        else:
            self.backbone, feature_dim = _build_resnet18(pretrained, freeze_backbone)

        self.dropout = nn.Dropout(dropout)
        self.binary_head = nn.Linear(feature_dim, len(config.BINARY_CLASS_NAMES))
        self.class_head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> CorruptionOutput:
        """Run the trunk once and both heads on the shared embedding."""
        if x.ndim != 4:
            raise ValueError(f"expected a 4D (B, C, H, W) batch, got shape {tuple(x.shape)}")
        features = self.dropout(self.backbone(x))
        return CorruptionOutput(
            binary_logits=self.binary_head(features),
            class_logits=self.class_head(features),
        )

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference helper returning ``(binary_pred, class_pred)`` indices."""
        self.eval()
        out = self(x)
        return out.binary_logits.argmax(dim=1), out.class_logits.argmax(dim=1)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters, by default only those that receive gradients."""
        return sum(
            p.numel() for p in self.parameters()
            if p.requires_grad or not trainable_only
        )


def _build_resnet18(
    pretrained: bool, freeze_backbone: bool
) -> tuple[nn.Module, int]:
    """Return a headless ResNet18 trunk and its feature width.

    The final ``fc`` is replaced with ``nn.Identity`` so the trunk emits raw
    512-dim embeddings that :class:`CorruptionNet` feeds to both heads.

    Raises:
        ImportError: If torchvision is not installed.
    """
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "the 'resnet18' architecture requires torchvision; "
            "install it with `pip install torchvision`"
        ) from exc

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    feature_dim = model.fc.in_features
    model.fc = nn.Identity()

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    return model, feature_dim


def build_model(
    arch: str = config.DEFAULT_ARCH,
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = False,
    freeze_backbone: bool = False,
    width: int = config.DEFAULT_WIDTH,
    dropout: float = config.DEFAULT_DROPOUT,
) -> CorruptionNet:
    """Factory mirroring the training CLI flags; see :class:`CorruptionNet`."""
    return CorruptionNet(
        arch=arch,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        width=width,
        dropout=dropout,
    )
