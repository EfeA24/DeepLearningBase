# Purpose: Accumulates sample-weighted top-1/top-k classification accuracy from logits and integer labels across batches.
"""Top-k accuracy accumulator.

Inputs per ``update``:
    - ``logits``: float tensor (B, C), raw scores or probabilities (ranking is identical).
    - ``labels``: int tensor (B,) with class ids in [0, C).
    - ``sample_weight`` (optional): float tensor (B,) of non-negative per-sample weights.
Accuracy@k = sum(weight * [label in top-k]) / sum(weight). Batches of different sizes are
weighted by their samples (never by averaging batch means). Rows whose label equals
``ignore_index`` are excluded.

Distributed: ``state()`` returns tensors (correct_k..., total) that can be summed across ranks.

Requires: torch.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch


class TopKAccuracy:
    """Accumulates accuracy for each k in ``topk``.

    Args:
        topk: k values, each >= 1 (e.g. (1, 5)).
        ignore_index: Optional label value to skip.
    """

    def __init__(self, topk: Sequence[int] = (1,), ignore_index: Optional[int] = None) -> None:
        if not topk or any(int(k) < 1 for k in topk):
            raise ValueError(f"topk must contain integers >= 1, got {topk}")
        self.topk = tuple(sorted({int(k) for k in topk}))
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self._correct = torch.zeros(len(self.topk), dtype=torch.float64)
        self._total = torch.zeros((), dtype=torch.float64)

    @torch.no_grad()
    def update(
        self, logits: torch.Tensor, labels: torch.Tensor, sample_weight: Optional[torch.Tensor] = None
    ) -> None:
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape (B, C), got {tuple(logits.shape)}")
        if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
            raise ValueError(f"labels must have shape ({logits.shape[0]},), got {tuple(labels.shape)}")
        if labels.dtype.is_floating_point:
            raise TypeError("labels must be an integer tensor")
        num_classes = logits.shape[1]
        if self.topk[-1] > num_classes:
            raise ValueError(f"max k={self.topk[-1]} exceeds number of classes {num_classes}")
        weights = (
            torch.ones(labels.shape[0], dtype=torch.float64, device=labels.device)
            if sample_weight is None
            else sample_weight.detach().to(device=labels.device, dtype=torch.float64)
        )
        if weights.shape != labels.shape:
            raise ValueError("sample_weight must have shape (B,)")
        if (weights < 0).any():
            raise ValueError("sample_weight must be non-negative")
        if self.ignore_index is not None:
            keep = labels != self.ignore_index
            logits, labels, weights = logits[keep], labels[keep], weights[keep]
        if labels.numel() == 0:
            return
        if labels.min() < 0 or labels.max() >= num_classes:
            raise ValueError(f"labels must be in [0, {num_classes})")
        top = logits.topk(self.topk[-1], dim=1).indices
        hits = top.eq(labels.unsqueeze(1))
        for i, k in enumerate(self.topk):
            correct_k = hits[:, :k].any(dim=1).to(torch.float64)
            self._correct[i] += (correct_k * weights).sum().cpu()
        self._total += weights.sum().cpu()

    def compute(self) -> Dict[str, float]:
        """Return ``{"top{k}": accuracy}``; raises if no samples were accumulated."""
        if self._total.item() == 0:
            raise RuntimeError("No samples accumulated; call update() before compute()")
        return {f"top{k}": (self._correct[i] / self._total).item() for i, k in enumerate(self.topk)}

    def state(self) -> Dict[str, torch.Tensor]:
        return {"correct": self._correct.clone(), "total": self._total.clone()}
