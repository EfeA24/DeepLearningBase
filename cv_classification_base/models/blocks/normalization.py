# Purpose: Builds batch/group/instance/layer/identity normalization layers for image-classification CNN blocks with explicit spatial dimensionality.
"""Normalization layer factory for convolutional feature maps.

Input contract: channels-first feature maps ``(B, C, *spatial)`` where ``len(spatial)`` equals
``spatial_dims`` (1 -> (B, C, L), 2 -> (B, C, H, W), 3 -> (B, C, D, H, W)).

Kinds:
    "batch"    -> BatchNorm{1,2,3}d over (B, *spatial) per channel; depends on batch statistics.
    "group"    -> GroupNorm(num_groups, C); batch-size independent; C % num_groups must be 0.
    "instance" -> InstanceNorm{1,2,3}d per sample and channel (affine configurable).
    "layer"    -> ChannelLayerNorm: LayerNorm over the channel axis only at each spatial
                  position (ConvNeXt-style). Use GroupNorm(1, C) if you want normalization
                  over (C, *spatial) jointly instead.
    "identity" -> nn.Identity (no normalization).

Requires: torch.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn

NORM_KINDS = ("batch", "group", "instance", "layer", "identity")


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel axis of a channels-first tensor ``(B, C, *spatial)``."""

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 3:
            raise ValueError(f"ChannelLayerNorm expects (B, C, *spatial), got shape {tuple(x.shape)}")
        x = x.movedim(1, -1)
        x = self.norm(x)
        return x.movedim(-1, 1)


def build_norm(
    kind: str,
    num_channels: int,
    spatial_dims: int = 2,
    num_groups: int = 32,
    eps: float = 1e-5,
    affine: bool = True,
    momentum: float = 0.1,
) -> nn.Module:
    """Create a normalization layer for ``num_channels`` channels.

    Args:
        kind: One of ``NORM_KINDS``.
        num_channels: Channel count C of the input feature map.
        spatial_dims: 1, 2 or 3 (selects the BatchNorm/InstanceNorm variant).
        num_groups: Groups for "group" (must divide ``num_channels``).
        eps: Numerical stability term.
        affine: Learnable scale/shift. Note InstanceNorm defaults to ``affine=False`` in torch;
            here the caller's value is used for every kind.
        momentum: Running-stat momentum for "batch" (ignored elsewhere).
    """
    if num_channels < 1:
        raise ValueError(f"num_channels must be >= 1, got {num_channels}")
    if spatial_dims not in (1, 2, 3):
        raise ValueError(f"spatial_dims must be 1, 2 or 3, got {spatial_dims}")
    if kind == "batch":
        cls = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[spatial_dims]
        return cls(num_channels, eps=eps, momentum=momentum, affine=affine)
    if kind == "group":
        if num_groups < 1 or num_channels % num_groups != 0:
            raise ValueError(
                f"num_groups={num_groups} must be >= 1 and divide num_channels={num_channels}"
            )
        return nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)
    if kind == "instance":
        cls = {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}[spatial_dims]
        return cls(num_channels, eps=eps, affine=affine)
    if kind == "layer":
        return ChannelLayerNorm(num_channels, eps=eps, affine=affine)
    if kind == "identity":
        return nn.Identity()
    raise ValueError(f"Unknown norm kind {kind!r}; expected one of {NORM_KINDS}")


def norm_factory(kind: str, spatial_dims: int = 2, **kwargs) -> Callable[[int], nn.Module]:
    """Return ``f(num_channels) -> nn.Module`` for injection into blocks (``norm_layer=``)."""
    if kind not in NORM_KINDS:
        raise ValueError(f"Unknown norm kind {kind!r}; expected one of {NORM_KINDS}")

    def make(num_channels: int) -> nn.Module:
        return build_norm(kind, num_channels, spatial_dims=spatial_dims, **kwargs)

    return make
