# Purpose: Implements a 1x1-3x3-1x1 bottleneck residual block with explicit expansion and projection rules for image-classification CNNs.
"""Bottleneck residual block (ResNet-50 style, post-activation).

    x -> 1x1 conv (in -> width) -> norm -> act
      -> 3x3 conv (width -> width, stride, groups) -> norm -> act
      -> 1x1 conv (width -> mid_channels * expansion) -> norm
      + shortcut(x) -> act

``width = mid_channels * (base_width / 64) * groups`` (ResNeXt convention; equals
``mid_channels`` with defaults). Output channels are ``mid_channels * expansion``.
The stride is applied on the 3x3 conv ("ResNet v1.5" placement).

Shortcut rule: identity iff ``stride == 1`` and ``in_channels == mid_channels * expansion``;
otherwise 1x1 conv with stride + norm.

Shapes: ``(B, in_channels, H, W)`` -> ``(B, mid_channels*expansion, ceil(H/stride), ceil(W/stride))``.

Requires: torch.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn


def _default_norm(num_channels: int) -> nn.Module:
    return nn.BatchNorm2d(num_channels)


def _default_activation() -> nn.Module:
    return nn.ReLU(inplace=True)


class BottleneckBlock(nn.Module):
    """Bottleneck block; ``out_channels == mid_channels * expansion``."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        stride: int = 1,
        expansion: int = 4,
        groups: int = 1,
        base_width: int = 64,
        norm_layer: Callable[[int], nn.Module] = _default_norm,
        activation_layer: Callable[[], nn.Module] = _default_activation,
        zero_init_last_norm: bool = False,
    ) -> None:
        super().__init__()
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if expansion < 1:
            raise ValueError(f"expansion must be >= 1, got {expansion}")
        width = int(mid_channels * (base_width / 64.0)) * groups
        if width < 1:
            raise ValueError("computed bottleneck width must be >= 1")
        out_channels = mid_channels * expansion
        self.expansion = expansion
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)
        self.norm1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride=stride, padding=1, groups=groups, bias=False)
        self.norm2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, out_channels, 1, bias=False)
        self.norm3 = norm_layer(out_channels)
        self.act1 = activation_layer()
        self.act2 = activation_layer()
        self.act3 = activation_layer()

        self.shortcut: Optional[nn.Module]
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                norm_layer(out_channels),
            )
        else:
            self.shortcut = None

        if zero_init_last_norm and getattr(self.norm3, "weight", None) is not None:
            nn.init.zeros_(self.norm3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.act2(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        return self.act3(out + identity)
