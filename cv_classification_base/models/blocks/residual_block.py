# Purpose: Implements a basic two-convolution residual block with a projection shortcut for image-classification CNNs.
"""Basic residual block (ResNet-18/34 style, post-activation).

    out = act(norm2(conv2(act(norm1(conv1(x))))) + shortcut(x))

Shapes: ``(B, in_channels, H, W)`` -> ``(B, out_channels, ceil(H/stride), ceil(W/stride))``.
The shortcut is identity when ``stride == 1`` and ``in_channels == out_channels``; otherwise it is a
1x1 conv (with the given stride) followed by a norm layer (projection shortcut, "option B").

``expansion = 1`` so ``block_factory(in_ch, out_ch, stride)`` produces exactly ``out_ch`` channels;
this is the factory signature expected by models/cnn/residual_cnn.py.

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


class BasicResidualBlock(nn.Module):
    """Two 3x3 convolutions plus identity/projection shortcut."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm_layer: Callable[[int], nn.Module] = _default_norm,
        activation_layer: Callable[[], nn.Module] = _default_activation,
        zero_init_last_norm: bool = False,
    ) -> None:
        """
        Args:
            zero_init_last_norm: If True and the second norm has a ``weight``, initialize it to 0
                so each block starts as identity (a common training-stability trick).
        """
        super().__init__()
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = norm_layer(out_channels)
        self.act1 = activation_layer()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.norm2 = norm_layer(out_channels)
        self.act2 = activation_layer()

        self.shortcut: Optional[nn.Module]
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                norm_layer(out_channels),
            )
        else:
            self.shortcut = None

        if zero_init_last_norm and getattr(self.norm2, "weight", None) is not None:
            nn.init.zeros_(self.norm2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act2(out + identity)
