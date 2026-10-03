# Purpose: Maps recurrent hidden-state sequences to one class-logit vector per sequence with an explicit, length-aware pooling strategy.
"""Sequence classification head.

- Input: hidden states (B, T, H) (batch-first), plus optional ``lengths`` (B,) or ``mask``
  (B, T) bool with True = valid position.
- Output: raw logits (B, num_classes); no softmax is applied.
- Pooling (``pooling``):
  * "last": last valid step. Right padding (default): index ``length - 1``; left padding:
    the final position. Without lengths/mask, the final position is used.
  * "mean": average over valid steps.
  * "max": element-wise max over valid steps.
  Every sequence must have at least one valid step (ValueError otherwise).
- For bidirectional encoders "last" only covers the forward direction's final step; prefer
  merged final states (bidirectional_rnn.merge_final_state) or "mean"/"max".

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

_POOLINGS = ("last", "mean", "max")


def _resolve_mask(hidden: torch.Tensor, lengths: Optional[torch.Tensor], mask: Optional[torch.Tensor], padding_side: str) -> Optional[torch.Tensor]:
    if mask is not None:
        if mask.shape != hidden.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be a bool tensor of shape (B, T).")
        return mask
    if lengths is None:
        return None
    steps = hidden.shape[1]
    positions = torch.arange(steps, device=hidden.device).unsqueeze(0)
    lengths = lengths.to(hidden.device).unsqueeze(1)
    if padding_side == "right":
        return positions < lengths
    return positions >= steps - lengths


class SequenceClassificationHead(nn.Module):
    """Pool (B, T, H) -> (B, H), then dropout + linear -> (B, num_classes) logits."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        pooling: str = "last",
        dropout: float = 0.0,
        padding_side: str = "right",
    ) -> None:
        super().__init__()
        if pooling not in _POOLINGS:
            raise ValueError(f"pooling must be one of {_POOLINGS}, got {pooling!r}.")
        if padding_side not in ("right", "left"):
            raise ValueError(f"padding_side must be 'right' or 'left', got {padding_side!r}.")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        self.pooling = pooling
        self.padding_side = padding_side
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_size, num_classes)

    def pool(self, hidden: torch.Tensor, lengths: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hidden.dim() != 3:
            raise ValueError(f"hidden must have shape (B, T, H), got {tuple(hidden.shape)}.")
        valid = _resolve_mask(hidden, lengths, mask, self.padding_side)
        if valid is None:
            if self.pooling == "last":
                return hidden[:, -1]
            if self.pooling == "mean":
                return hidden.mean(dim=1)
            return hidden.max(dim=1).values

        counts = valid.sum(dim=1)
        if bool((counts == 0).any()):
            raise ValueError("Every sequence needs at least one valid step for pooling.")
        if self.pooling == "last":
            if self.padding_side == "left":
                return hidden[:, -1]
            batch_index = torch.arange(hidden.shape[0], device=hidden.device)
            return hidden[batch_index, counts - 1]
        expanded = valid.unsqueeze(-1)
        if self.pooling == "mean":
            summed = hidden.masked_fill(~expanded, 0.0).sum(dim=1)
            return summed / counts.unsqueeze(-1).to(hidden.dtype)
        return hidden.masked_fill(~expanded, torch.finfo(hidden.dtype).min).max(dim=1).values

    def forward(self, hidden: torch.Tensor, lengths: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.classifier(self.dropout(self.pool(hidden, lengths, mask)))
