# Purpose: Sequence classification head that pools encoder hidden states ([CLS]/first token or masked mean, explicit) into raw class logits.
"""Sequence classification head.

Input: hidden_states (B, T, D); attention_mask (B, T) bool or 0/1, True/1 = valid token
    (required for pooling="mean"; ignored for "cls").
Output: logits (B, num_classes). No softmax/sigmoid is applied (use CE / BCE-with-logits losses).

Pooling:
    "cls":  hidden_states[:, 0] - assumes the first position is [CLS] (right padding).
    "mean": average over valid positions only.
Optional BERT-style pooler: Linear(D, D) + tanh before the classifier (``pooler=True``).

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

__all__ = ["ClassificationHead"]


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype.is_floating_point:
        raise TypeError("attention_mask must be bool or integer 0/1")
    weights = (mask != 0).unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


class ClassificationHead(nn.Module):
    """Pool -> (optional dense+tanh) -> dropout -> Linear -> logits."""

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        pooling: str = "cls",
        dropout: float = 0.1,
        pooler: bool = False,
    ) -> None:
        super().__init__()
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")
        if d_model <= 0 or num_classes <= 0:
            raise ValueError("d_model and num_classes must be positive")
        self.pooling = pooling
        self.pooler = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh()) if pooler else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def pool(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, T, D) -> (B, D) using the configured pooling."""
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be (B, T, D), got shape {tuple(hidden_states.shape)}")
        if self.pooling == "cls":
            return hidden_states[:, 0]
        if attention_mask is None:
            raise ValueError("pooling='mean' requires attention_mask")
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("attention_mask must be (B, T) matching hidden_states")
        return _masked_mean(hidden_states, attention_mask)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pooled = self.pooler(self.pool(hidden_states, attention_mask))
        return self.classifier(self.dropout(pooled))
