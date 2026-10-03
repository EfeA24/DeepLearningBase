# Purpose: Defines a small educational CNN image classifier with configurable channels and class count.
"""Small educational CNN classifier.

Architecture: for each entry in ``channels``: Conv3x3 -> BatchNorm -> ReLU -> MaxPool2d(2);
then global average pooling -> Dropout -> Linear(channels[-1], num_classes).

Input: float images ``(B, in_channels, H, W)``, typically normalized. Because of global average
pooling any ``H, W >= 2 ** len(channels)`` works (each stage halves the spatial size with floor).
Output: raw logits ``(B, num_classes)`` (no softmax).

Intended for learning/debugging/small datasets (MNIST/CIFAR-sized); not a competitive model.

Requires: torch.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes < 1 or in_channels < 1:
            raise ValueError("num_classes and in_channels must be >= 1")
        if len(channels) == 0 or any(c < 1 for c in channels):
            raise ValueError(f"channels must be a non-empty sequence of positive ints, got {channels}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        layers = []
        prev = in_channels
        for ch in channels:
            layers += [
                nn.Conv2d(prev, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            prev = ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(prev, num_classes)
        self.min_input_size = 2 ** len(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected (B, C, H, W) input, got shape {tuple(x.shape)}")
        if min(x.shape[-2:]) < self.min_input_size:
            raise ValueError(f"spatial size must be >= {self.min_input_size}, got {tuple(x.shape[-2:])}")
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(self.dropout(x))
