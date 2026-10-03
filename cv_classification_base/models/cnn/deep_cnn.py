# Purpose: Defines a configurable VGG-like sequential convolutional baseline for image classification.
"""Configurable sequential CNN baseline (no state-of-the-art claims).

Architecture: ``len(stage_channels)`` stages; each stage has ``convs_per_stage`` Conv3x3 ->
[BatchNorm] -> ReLU layers followed by MaxPool2d(2) (except optionally the last stage).
Then global average pooling -> Dropout -> Linear(stage_channels[-1], num_classes).

Input: ``(B, in_channels, H, W)`` with ``H, W >= 2 ** num_pooled_stages``.
Output: raw logits ``(B, num_classes)``.

``forward_features`` returns the final feature map ``(B, stage_channels[-1], H', W')`` for reuse
as a backbone.

Requires: torch.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class DeepCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        stage_channels: Sequence[int] = (32, 64, 128, 256),
        convs_per_stage: int = 2,
        use_batchnorm: bool = True,
        pool_last_stage: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes < 1 or in_channels < 1:
            raise ValueError("num_classes and in_channels must be >= 1")
        if len(stage_channels) == 0 or any(c < 1 for c in stage_channels):
            raise ValueError("stage_channels must be a non-empty sequence of positive ints")
        if convs_per_stage < 1:
            raise ValueError(f"convs_per_stage must be >= 1, got {convs_per_stage}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        layers = []
        prev = in_channels
        num_pools = 0
        for stage_idx, ch in enumerate(stage_channels):
            for _ in range(convs_per_stage):
                layers.append(nn.Conv2d(prev, ch, 3, padding=1, bias=not use_batchnorm))
                if use_batchnorm:
                    layers.append(nn.BatchNorm2d(ch))
                layers.append(nn.ReLU(inplace=True))
                prev = ch
            is_last = stage_idx == len(stage_channels) - 1
            if not is_last or pool_last_stage:
                layers.append(nn.MaxPool2d(2))
                num_pools += 1
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(prev, num_classes)
        self.out_channels = prev
        self.min_input_size = 2 ** num_pools

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected (B, C, H, W) input, got shape {tuple(x.shape)}")
        if min(x.shape[-2:]) < self.min_input_size:
            raise ValueError(f"spatial size must be >= {self.min_input_size}, got {tuple(x.shape[-2:])}")
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.forward_features(x)).flatten(1)
        return self.classifier(self.dropout(x))
