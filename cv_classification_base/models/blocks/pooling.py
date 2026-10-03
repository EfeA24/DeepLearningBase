# Purpose: Provides global/adaptive average, max and avg+max concatenation pooling layers for image-classification feature maps.
"""Pooling layers for channels-first feature maps ``(B, C, H, W)``.

Output shapes:
    GlobalAvgPool2d(flatten=True)        -> (B, C)
    GlobalMaxPool2d(flatten=True)        -> (B, C)
    AdaptiveAvgMaxConcatPool2d(s)        -> (B, 2C, s, s), or (B, 2C) when s == 1 and flatten=True
    build_pooling("avg"/"max", s)        -> (B, C, s, s), or (B, C) when s == 1 and flatten=True

The concat variant doubles the channel count: size a following head with ``2 * C`` inputs
(``pooled_channels(kind, C)`` computes this).

Requires: torch.
"""
from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import nn

POOL_KINDS = ("avg", "max", "avgmax")
SizeLike = Union[int, Tuple[int, int]]


def _check_4d(x: torch.Tensor) -> None:
    if x.dim() != 4:
        raise ValueError(f"expected (B, C, H, W) input, got shape {tuple(x.shape)}")


class GlobalAvgPool2d(nn.Module):
    """Mean over H and W: ``(B, C, H, W) -> (B, C)`` (or ``(B, C, 1, 1)`` if not flatten)."""

    def __init__(self, flatten: bool = True) -> None:
        super().__init__()
        self.flatten = flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_4d(x)
        return x.mean(dim=(2, 3), keepdim=not self.flatten)


class GlobalMaxPool2d(nn.Module):
    """Max over H and W: ``(B, C, H, W) -> (B, C)`` (or ``(B, C, 1, 1)`` if not flatten)."""

    def __init__(self, flatten: bool = True) -> None:
        super().__init__()
        self.flatten = flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_4d(x)
        return x.amax(dim=(2, 3), keepdim=not self.flatten)


class AdaptiveAvgMaxConcatPool2d(nn.Module):
    """Concatenate adaptive avg and max pooling along channels (avg first, then max)."""

    def __init__(self, output_size: SizeLike = 1, flatten: bool = True) -> None:
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(output_size)
        self.max = nn.AdaptiveMaxPool2d(output_size)
        self.flatten = flatten and output_size in (1, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_4d(x)
        out = torch.cat([self.avg(x), self.max(x)], dim=1)
        return out.flatten(1) if self.flatten else out


class _AdaptivePool(nn.Module):
    def __init__(self, pool: nn.Module, flatten: bool) -> None:
        super().__init__()
        self.pool = pool
        self.flatten = flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_4d(x)
        out = self.pool(x)
        return out.flatten(1) if self.flatten else out


def build_pooling(kind: str, output_size: SizeLike = 1, flatten: bool = True) -> nn.Module:
    """Create a pooling layer. ``flatten`` only applies when ``output_size`` is 1."""
    flatten = flatten and output_size in (1, (1, 1))
    if kind == "avg":
        return _AdaptivePool(nn.AdaptiveAvgPool2d(output_size), flatten)
    if kind == "max":
        return _AdaptivePool(nn.AdaptiveMaxPool2d(output_size), flatten)
    if kind == "avgmax":
        return AdaptiveAvgMaxConcatPool2d(output_size, flatten=flatten)
    raise ValueError(f"Unknown pooling kind {kind!r}; expected one of {POOL_KINDS}")


def pooled_channels(kind: str, channels: int) -> int:
    """Channel count after pooling ``channels`` input channels with ``kind``."""
    if kind not in POOL_KINDS:
        raise ValueError(f"Unknown pooling kind {kind!r}; expected one of {POOL_KINDS}")
    return 2 * channels if kind == "avgmax" else channels
