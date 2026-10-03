# Purpose: Accumulates token-level accuracy for sequence labelling or language modelling, ignoring ignore_index labels and masked positions.
"""Token accuracy accumulator.

Inputs per ``update``:
    - ``logits`` (B, T, V) float, OR ``preds`` (B, T) int via ``update_from_preds``.
    - ``labels`` (B, T) int; positions equal to ``ignore_index`` (default -100) are skipped.
    - ``mask`` (optional, B, T) bool with **True = valid / counted** position.
A position counts only if label != ignore_index AND mask is True. Labels must already be aligned
with predictions (for causal LM, shift beforehand: logits[:, :-1] vs labels[:, 1:]).
Accuracy = correct valid tokens / valid tokens across all batches (token-weighted).

Requires: torch.
"""
from __future__ import annotations

from typing import Optional

import torch


class TokenAccuracy:
    """Token-weighted accuracy accumulator."""

    def __init__(self, ignore_index: int = -100) -> None:
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        if logits.ndim != labels.ndim + 1 or logits.shape[:-1] != labels.shape:
            raise ValueError(f"logits {tuple(logits.shape)} must be labels shape {tuple(labels.shape)} + (V,)")
        self.update_from_preds(logits.argmax(dim=-1), labels, mask)

    @torch.no_grad()
    def update_from_preds(self, preds: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        if preds.shape != labels.shape:
            raise ValueError(f"preds {tuple(preds.shape)} and labels {tuple(labels.shape)} must match")
        valid = labels != self.ignore_index
        if mask is not None:
            if mask.shape != labels.shape:
                raise ValueError(f"mask {tuple(mask.shape)} must match labels {tuple(labels.shape)}")
            if mask.dtype != torch.bool:
                raise TypeError("mask must be a bool tensor with True = valid position")
            valid = valid & mask
        self.correct += int((preds.eq(labels) & valid).sum().item())
        self.total += int(valid.sum().item())

    def compute(self) -> float:
        if self.total == 0:
            raise RuntimeError("No valid tokens accumulated")
        return self.correct / self.total
