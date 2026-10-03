# Purpose: Provides a configurable 2D conv-norm-activation block (or pre-activation norm-act-conv) for image-classification CNNs.
"""Conv / norm / activation block with explicit ordering and padding.

Shapes: input ``(B, in_channels, H, W)`` -> output ``(B, out_channels, H_out, W_out)`` with
``H_out = floor((H + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1``.
With the default ``padding=None`` and an odd kernel, padding is ``dilation*(kernel_size//2)``,
so stride 1 preserves spatial size and stride 2 gives ``ceil(H/2)``.

Orders:
    "conv-norm-act" (post-activation, classic ResNet/VGG-BN style)
    "norm-act-conv" (pre-activation, ResNet-v2 style; norm runs on ``in_channels``)

Norm/activation are injected as factories so the block has no sibling dependency:
``norm_layer(num_channels) -> nn.Module`` (default BatchNorm2d) and
``activation_layer() -> nn.Module`` (default ReLU). Pass ``None`` to disable either.
Conv bias defaults to ``True`` only when there is no norm layer directly after the conv.

Requires: torch.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn

ORDERS = ("conv-norm-act", "norm-act-conv")


def _default_norm(num_channels: int) -> nn.Module:
    return nn.BatchNorm2d(num_channels)


def _default_activation() -> nn.Module:
    return nn.ReLU(inplace=True)


class ConvNormAct(nn.Module):
    """2D convolution with optional normalization and activation in an explicit order."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: Optional[bool] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = _default_norm,
        activation_layer: Optional[Callable[[], nn.Module]] = _default_activation,
        order: str = "conv-norm-act",
    ) -> None:
        super().__init__()
        if order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be >= 1")
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError(f"groups={groups} must divide in_channels and out_channels")
        if padding is None:
            if kernel_size % 2 == 0:
                raise ValueError("padding=None requires an odd kernel_size; pass padding explicitly")
            padding = dilation * (kernel_size // 2)
        if bias is None:
            bias = norm_layer is None or order == "norm-act-conv"

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        norm_channels = out_channels if order == "conv-norm-act" else in_channels
        norm = norm_layer(norm_channels) if norm_layer is not None else nn.Identity()
        act = activation_layer() if activation_layer is not None else nn.Identity()

        layers = [conv, norm, act] if order == "conv-norm-act" else [norm, act, conv]
        self.order = order
        self.block = nn.Sequential(*layers)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
