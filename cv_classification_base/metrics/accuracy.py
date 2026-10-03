# Purpose: Computes and accumulates top-1/top-k accuracy from logits or predicted ids for single-label image classification.
"""Accuracy for single-label classification.

Input modes (explicit, chosen at construction):
    "logits": outputs ``(B, C)`` raw scores or probabilities; top-k computed via ``topk``.
    "preds":  outputs ``(B,)`` predicted class ids (only ``top_k=1`` is meaningful).
Targets: int ``(B,)`` class ids. Targets equal to ``ignore_index`` (if set) are excluded.

``AccuracyMeter`` is sample-weighted: it sums correct predictions and sample counts across
batches, so unequal batch sizes are handled correctly. ``compute()`` returns NaN when no samples
were seen (rather than inventing a value).

Requires: torch.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

INPUT_MODES = ("logits", "preds")


def count_correct(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    input_mode: str = "logits",
    top_k: int = 1,
    ignore_index: Optional[int] = None,
) -> Tuple[int, int]:
    """Return ``(num_correct, num_counted)`` for one batch."""
    if input_mode not in INPUT_MODES:
        raise ValueError(f"input_mode must be one of {INPUT_MODES}, got {input_mode!r}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    targets = targets.reshape(-1)
    if ignore_index is not None:
        keep = targets != ignore_index
    else:
        keep = torch.ones_like(targets, dtype=torch.bool)

    if input_mode == "logits":
        if outputs.dim() != 2 or outputs.shape[0] != targets.shape[0]:
            raise ValueError(f"logits must be (B, C) matching targets, got {tuple(outputs.shape)}")
        if top_k > outputs.shape[1]:
            raise ValueError(f"top_k={top_k} exceeds number of classes {outputs.shape[1]}")
        topk_ids = outputs.topk(top_k, dim=1).indices
        correct = (topk_ids == targets.unsqueeze(1)).any(dim=1)
    else:
        if top_k != 1:
            raise ValueError("input_mode='preds' only supports top_k=1")
        preds = outputs.reshape(-1)
        if preds.shape != targets.shape:
            raise ValueError("preds and targets must have the same number of samples")
        correct = preds == targets
    return int((correct & keep).sum()), int(keep.sum())


def accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    input_mode: str = "logits",
    top_k: int = 1,
    ignore_index: Optional[int] = None,
) -> float:
    """Accuracy of a single batch (NaN if nothing is counted)."""
    correct, total = count_correct(outputs, targets, input_mode, top_k, ignore_index)
    return correct / total if total > 0 else math.nan


class AccuracyMeter:
    """Sample-weighted accuracy accumulator: ``update`` per batch, ``compute`` at epoch end."""

    def __init__(
        self, input_mode: str = "logits", top_k: int = 1, ignore_index: Optional[int] = None
    ) -> None:
        if input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {INPUT_MODES}, got {input_mode!r}")
        self.input_mode = input_mode
        self.top_k = top_k
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    @torch.no_grad()
    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        c, t = count_correct(outputs.detach(), targets.detach(), self.input_mode, self.top_k, self.ignore_index)
        self.correct += c
        self.total += t

    def compute(self) -> float:
        return self.correct / self.total if self.total > 0 else math.nan
