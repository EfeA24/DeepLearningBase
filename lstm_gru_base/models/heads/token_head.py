# Purpose: Maps per-timestep recurrent outputs to per-token logits (tagging/frame labelling) for LSTM/GRU sequence labelling models.
"""Per-timestep (token) head.

Contract:
    - Input ``(B, T, H)``; output raw logits ``(B, T, C)``, positions preserved (padding positions included).
    - Loss: flatten to ``(B*T, C)`` and ``(B*T,)`` targets with padded labels set to ``ignore_index`` (-100).
      No softmax inside.

Requires: torch only.
"""
from __future__ import annotations

import torch
from torch import nn


class TokenClassificationHead(nn.Module):
    """Applies dropout + Linear to every time step."""

    def __init__(self, input_size: int, num_labels: int, dropout: float = 0.0) -> None:
        super().__init__()
        if num_labels < 1:
            raise ValueError("num_labels must be >= 1.")
        self.input_size = input_size
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(input_size, num_labels)

    def forward(self, outputs: torch.Tensor) -> torch.Tensor:
        if outputs.dim() != 3 or outputs.shape[-1] != self.input_size:
            raise ValueError(f"outputs must be (B, T, {self.input_size}), got {tuple(outputs.shape)}.")
        return self.proj(self.dropout(outputs))
