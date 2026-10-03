# Purpose: Provides a reference Feature Pyramid Network neck with an explicit input-channel/stride contract for multi-scale object detectors.
"""Feature Pyramid Network (Lin et al., 2017) reference neck.

Input contract:
    ``features``: list of L maps ordered HIGH resolution -> LOW resolution,
    map i of shape (B, in_channels[i], H_i, W_i) with H_i >= H_{i+1}
    (typically strides 8, 16, 32).

Output contract:
    list of L (+ ``num_extra_levels``) maps, same order, each (B, out_channels, H_i, W_i).
    Extra levels are produced by stride-2 max-pooling of the coarsest output
    (torchvision "LastLevelMaxPool" style) and double the stride each time.

Top-down merge: lateral 1x1 conv per level; coarser map upsampled with nearest
interpolation to the exact lateral size (handles odd sizes), summed, then a 3x3
output conv. Not tied to any specific detector release.

Requires: torch only.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FPN(nn.Module):
    """Feature Pyramid Network.

    Args:
        in_channels: Channels of each input level (high-res first).
        out_channels: Channels of every output level.
        strides: Optional declared input strides (high-res first); exposed via ``out_strides``.
        num_extra_levels: Extra coarser levels appended by max-pooling.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        strides: Optional[Sequence[int]] = None,
        num_extra_levels: int = 0,
    ) -> None:
        super().__init__()
        if not in_channels or any(c <= 0 for c in in_channels) or out_channels <= 0:
            raise ValueError("in_channels must be non-empty positive ints and out_channels > 0")
        if strides is not None and len(strides) != len(in_channels):
            raise ValueError("strides must match in_channels in length")
        if num_extra_levels < 0:
            raise ValueError("num_extra_levels must be >= 0")
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_extra_levels = num_extra_levels
        self.lateral_convs = nn.ModuleList(nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels)
        self.output_convs = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels
        )
        self.out_strides: Optional[List[int]] = None
        if strides is not None:
            self.out_strides = list(strides) + [strides[-1] * 2 ** (i + 1) for i in range(num_extra_levels)]
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: Sequence[Tensor]) -> List[Tensor]:
        if len(features) != len(self.in_channels):
            raise ValueError(f"expected {len(self.in_channels)} feature maps, got {len(features)}")
        for i, (feat, channels) in enumerate(zip(features, self.in_channels)):
            if feat.ndim != 4 or feat.shape[1] != channels:
                raise ValueError(f"feature {i} must be (B, {channels}, H, W), got {tuple(feat.shape)}")

        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        merged: List[Tensor] = [laterals[-1]]
        for lateral in reversed(laterals[:-1]):
            upsampled = F.interpolate(merged[0], size=lateral.shape[-2:], mode="nearest")
            merged.insert(0, lateral + upsampled)
        outputs = [conv(m) for conv, m in zip(self.output_convs, merged)]
        for _ in range(self.num_extra_levels):
            outputs.append(F.max_pool2d(outputs[-1], kernel_size=1, stride=2))
        return outputs
