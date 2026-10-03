# Purpose: Produces per-time-step class logits from recurrent hidden states for token/step labelling tasks.
"""Token (per-step) classification head.

- Input: hidden states (B, T, H). Output: raw logits (B, T, num_labels), positions preserved
  (padded steps also get logits; exclude them in the loss with ignore_index -100 or a mask).
- No softmax is applied.

Requires: torch only. No sibling files.
"""
from __future__ import annotations

import torch
from torch import nn


class TokenClassificationHead(nn.Module):
    """Dropout + linear projection applied independently at every time step."""

    def __init__(self, input_size: int, num_labels: int, dropout: float = 0.0) -> None:
        super().__init__()
        if input_size <= 0 or num_labels <= 0:
            raise ValueError("input_size and num_labels must be positive.")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_size, num_labels)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() != 3:
            raise ValueError(f"hidden must have shape (B, T, H), got {tuple(hidden.shape)}.")
        return self.classifier(self.dropout(hidden))
