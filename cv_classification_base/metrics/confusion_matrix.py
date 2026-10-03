# Purpose: Accumulates a fixed-size confusion matrix over batches for single-label image classification.
"""Confusion-matrix accumulator.

Layout: ``matrix[i, j]`` = number of samples with TRUE class ``i`` predicted as class ``j``
(rows = targets, columns = predictions), int64 ``(num_classes, num_classes)``. Class order is
``0..num_classes-1``; map ids to names with the caller's own class list.

``update`` accepts either logits ``(B, C)`` (``input_mode="logits"``, argmax taken) or predicted
ids ``(B,)`` (``input_mode="preds"``), plus int targets ``(B,)``. Targets equal to
``ignore_index`` are skipped. Out-of-range ids raise ValueError.

Derived metrics: see metrics/precision.py, recall.py, f1.py (they accept this matrix).

Requires: torch.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch

INPUT_MODES = ("logits", "preds")


def confusion_matrix_from_preds(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Return an int64 ``(C, C)`` confusion matrix for 1D prediction/target ids in ``[0, C)``."""
    if preds.shape != targets.shape or preds.dim() != 1:
        raise ValueError("preds and targets must be 1D tensors of equal length")
    preds = preds.long()
    targets = targets.long()
    if preds.numel() and (
        preds.min() < 0 or preds.max() >= num_classes or targets.min() < 0 or targets.max() >= num_classes
    ):
        raise ValueError(f"class ids must lie in [0, {num_classes})")
    flat = targets * num_classes + preds
    counts = torch.bincount(flat, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


class ConfusionMatrix:
    """Batch-wise accumulator: ``update`` -> ``compute`` -> ``reset``. Stored on CPU."""

    def __init__(
        self, num_classes: int, input_mode: str = "logits", ignore_index: Optional[int] = None
    ) -> None:
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {INPUT_MODES}, got {input_mode!r}")
        self.num_classes = num_classes
        self.input_mode = input_mode
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        if self.input_mode == "logits":
            if outputs.dim() != 2 or outputs.shape[1] != self.num_classes:
                raise ValueError(f"logits must be (B, {self.num_classes}), got {tuple(outputs.shape)}")
            preds = outputs.argmax(dim=1)
        else:
            preds = outputs
        preds = preds.detach().reshape(-1).cpu()
        targets = targets.detach().reshape(-1).cpu()
        if preds.shape != targets.shape:
            raise ValueError("predictions and targets must contain the same number of samples")
        if self.ignore_index is not None:
            keep = targets != self.ignore_index
            preds, targets = preds[keep], targets[keep]
        self.matrix += confusion_matrix_from_preds(preds, targets, self.num_classes)

    def compute(self) -> torch.Tensor:
        """Return a copy of the accumulated ``(C, C)`` matrix."""
        return self.matrix.clone()

    @property
    def total(self) -> int:
        return int(self.matrix.sum())


def format_confusion_matrix(matrix: torch.Tensor, class_names: Sequence[str]) -> str:
    """Render a matrix as a plain-text table (rows = true, cols = predicted). No printing."""
    if matrix.shape != (len(class_names), len(class_names)):
        raise ValueError("class_names length must match the matrix size")
    width = max(max(len(n) for n in class_names), len(str(int(matrix.max()) if matrix.numel() else 0)), 4)
    header = " " * (width + 1) + " ".join(n.rjust(width) for n in class_names)
    rows: List[str] = [header]
    for name, row in zip(class_names, matrix.tolist()):
        rows.append(name.rjust(width) + " " + " ".join(str(v).rjust(width) for v in row))
    return "\n".join(rows)
