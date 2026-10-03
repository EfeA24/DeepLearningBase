# Purpose: Token classification head mapping every encoder position to raw label logits (NER, POS tagging).
"""Token classification head.

Input: hidden_states (B, T, D). Output: logits (B, T, num_labels), one row per position including
special and padding positions; the loss/inference code ignores them via labels == -100 or the
attention mask. No softmax is applied.

Requires: torch only.
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["TokenClassificationHead"]


class TokenClassificationHead(nn.Module):
    """Dropout -> Linear(D, num_labels) applied position-wise."""

    def __init__(self, d_model: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model <= 0 or num_labels <= 0:
            raise ValueError("d_model and num_labels must be positive")
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be (B, T, D), got shape {tuple(hidden_states.shape)}")
        return self.classifier(self.dropout(hidden_states))
