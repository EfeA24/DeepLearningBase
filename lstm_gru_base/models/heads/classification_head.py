# Purpose: Maps a pooled sequence representation to raw class logits for LSTM/GRU sequence classification.
"""Sequence classification head.

Contract:
    - Input: pooled features ``(B, H)`` (from models/pooling/* or ``h_n[-1]``); pooling is NOT done here.
    - Output: raw logits ``(B, C)`` - no softmax; use ``nn.CrossEntropyLoss`` (or BCEWithLogits for
      ``C = 1`` binary setups).
    - Optional hidden layer: ``Linear -> activation -> dropout -> Linear``; otherwise ``dropout -> Linear``.

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class SequenceClassificationHead(nn.Module):
    """Pooled features -> class logits."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: Optional[int] = None,
        dropout: float = 0.0,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1.")
        layers = []
        if hidden_size is not None:
            layers += [nn.Linear(input_size, hidden_size), activation if activation is not None else nn.Tanh(), nn.Dropout(dropout)]
            in_features = hidden_size
        else:
            layers.append(nn.Dropout(dropout))
            in_features = input_size
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)
        self.input_size = input_size

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2 or features.shape[-1] != self.input_size:
            raise ValueError(f"features must be (B, {self.input_size}), got {tuple(features.shape)}.")
        return self.net(features)
