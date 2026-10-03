# Purpose: Provides a reference Path Aggregation Network neck (top-down plus bottom-up fusion) with documented level order for multi-scale object detectors.
"""Path Aggregation Network (Liu et al., 2018) reference neck.

Input contract:
    ``features``: list of L maps ordered HIGH resolution -> LOW resolution,
    map i of shape (B, in_channels[i], H_i, W_i) (e.g. strides 8, 16, 32).

Output contract:
    list of L maps in the SAME order and spatial sizes as the inputs, each
    (B, out_channels, H_i, W_i). Output strides equal input strides.

Computation:
    1. Lateral 1x1 convs project every level to ``out_channels``.
    2. Top-down (FPN) path: P_i = lateral_i + upsample(P_{i+1}), then 3x3 conv.
    3. Bottom-up path: N_0 = P_0; N_{i+1} = conv3x3(P_{i+1} + down(N_i)), where
       ``down`` is a stride-2 3x3 conv; if odd sizes make shapes differ by one,
       the downsampled map is resized to P_{i+1}'s exact size.
    Conv blocks are Conv-BN-SiLU. Not a copy of any specific YOLO/PANet release.

Requires: torch only.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _conv_bn_act(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


class PAN(nn.Module):
    """Path Aggregation Network neck.

    Args:
        in_channels: Channels of each input level (high-res first); at least 2 levels.
        out_channels: Channels of every output level.
    """

    def __init__(self, in_channels: Sequence[int], out_channels: int) -> None:
        super().__init__()
        if len(in_channels) < 2 or any(c <= 0 for c in in_channels) or out_channels <= 0:
            raise ValueError("PAN needs >= 2 positive in_channels and out_channels > 0")
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        levels = len(in_channels)
        self.lateral = nn.ModuleList(_conv_bn_act(c, out_channels, 1) for c in in_channels)
        self.top_down = nn.ModuleList(_conv_bn_act(out_channels, out_channels, 3) for _ in range(levels - 1))
        self.downsample = nn.ModuleList(_conv_bn_act(out_channels, out_channels, 3, stride=2) for _ in range(levels - 1))
        self.bottom_up = nn.ModuleList(_conv_bn_act(out_channels, out_channels, 3) for _ in range(levels - 1))

    def forward(self, features: Sequence[Tensor]) -> List[Tensor]:
        if len(features) != len(self.in_channels):
            raise ValueError(f"expected {len(self.in_channels)} feature maps, got {len(features)}")
        for i, (feat, channels) in enumerate(zip(features, self.in_channels)):
            if feat.ndim != 4 or feat.shape[1] != channels:
                raise ValueError(f"feature {i} must be (B, {channels}, H, W), got {tuple(feat.shape)}")

        laterals = [layer(feat) for layer, feat in zip(self.lateral, features)]
        levels = len(laterals)

        top_down: List[Tensor] = [laterals[-1]]
        for i in range(levels - 2, -1, -1):
            upsampled = F.interpolate(top_down[0], size=laterals[i].shape[-2:], mode="nearest")
            top_down.insert(0, self.top_down[i](laterals[i] + upsampled))

        outputs: List[Tensor] = [top_down[0]]
        for i in range(levels - 1):
            down = self.downsample[i](outputs[-1])
            target = top_down[i + 1]
            if down.shape[-2:] != target.shape[-2:]:
                down = F.interpolate(down, size=target.shape[-2:], mode="nearest")
            outputs.append(self.bottom_up[i](target + down))
        return outputs
