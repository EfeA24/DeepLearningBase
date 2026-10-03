# Purpose: Implements focal loss for multi-class (softmax) and binary/multi-label (sigmoid) image classification from raw logits.
"""Focal loss (Lin et al., 2017): ``FL(p_t) = -alpha_t * (1 - p_t) ** gamma * log(p_t)``.

Modes:
    "multiclass": logits ``(B, C)``, targets int64 ``(B,)`` in ``[0, C)`` or ``ignore_index``.
        ``p_t = softmax(logits)[y]``. ``alpha`` is None, a float applied to every class, or a
        per-class ``(C,)`` weight tensor (``alpha_t = alpha[y]``).
    "binary": logits of any shape (e.g. ``(B,)`` or multi-label ``(B, L)``), targets of the same
        shape with values in [0, 1]. ``p_t = p*t + (1-p)*(1-t)`` with ``p = sigmoid(logits)``.
        ``alpha`` is None or a float in [0, 1] weighting positives
        (``alpha_t = alpha*t + (1-alpha)*(1-t)``; the paper uses 0.25).

``gamma = 0`` and ``alpha = None`` reduce to plain CE / BCE. Reductions: "mean" averages over
non-ignored elements (0 with grad if none), "sum", or "none" (per-element, ignored entries 0).

Requires: torch.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn

MODES = ("multiclass", "binary")
REDUCTIONS = ("mean", "sum", "none")


class FocalLoss(nn.Module):
    def __init__(
        self,
        mode: str = "multiclass",
        gamma: float = 2.0,
        alpha: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if reduction not in REDUCTIONS:
            raise ValueError(f"reduction must be one of {REDUCTIONS}, got {reduction!r}")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.mode = mode
        self.gamma = float(gamma)
        self.reduction = reduction
        self.ignore_index = ignore_index

        self.alpha_scalar: Optional[float] = None
        self.alpha_vector: Optional[torch.Tensor]
        if alpha is None:
            self.alpha_vector = None
        elif isinstance(alpha, (int, float)):
            if mode == "binary" and not 0.0 <= alpha <= 1.0:
                raise ValueError(f"binary alpha must be in [0, 1], got {alpha}")
            if alpha < 0:
                raise ValueError(f"alpha must be >= 0, got {alpha}")
            self.alpha_scalar = float(alpha)
            self.alpha_vector = None
        else:
            if mode != "multiclass":
                raise ValueError("per-class alpha tensors are only supported in multiclass mode")
            vec = torch.as_tensor(alpha, dtype=torch.float32)
            if vec.dim() != 1 or torch.any(vec < 0):
                raise ValueError("per-class alpha must be a non-negative 1D tensor of length C")
            self.register_buffer("alpha_vector", vec)

    def _reduce(self, loss: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        loss = loss * valid.to(loss.dtype)
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        count = valid.sum()
        if count == 0:
            return loss.sum() * 0.0
        return loss.sum() / count.to(loss.dtype)

    def _multiclass(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(f"multiclass logits must be (B, C), got shape {tuple(logits.shape)}")
        if targets.shape != logits.shape[:1] or targets.dtype != torch.long:
            raise ValueError("multiclass targets must be int64 with shape (B,)")
        valid = targets != self.ignore_index
        safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
        if torch.any(safe_targets < 0) or torch.any(safe_targets >= logits.shape[1]):
            raise ValueError("targets contain class ids outside [0, C)")
        log_p = F.log_softmax(logits.float(), dim=1)
        log_pt = log_p.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.alpha_vector is not None:
            if self.alpha_vector.numel() != logits.shape[1]:
                raise ValueError("per-class alpha length must equal the number of classes")
            loss = loss * self.alpha_vector.to(loss.device)[safe_targets]
        elif self.alpha_scalar is not None:
            loss = loss * self.alpha_scalar
        return self._reduce(loss, valid).to(logits.dtype)

    def _binary(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.shape != logits.shape:
            raise ValueError(
                f"binary targets must match logits shape {tuple(logits.shape)}, got {tuple(targets.shape)}"
            )
        targets = targets.to(torch.float32)
        if torch.any(targets < 0) or torch.any(targets > 1):
            raise ValueError("binary targets must be in [0, 1]")
        logits_f = logits.float()
        ce = F.binary_cross_entropy_with_logits(logits_f, targets, reduction="none")
        p = torch.sigmoid(logits_f)
        pt = p * targets + (1.0 - p) * (1.0 - targets)
        loss = ce * ((1.0 - pt) ** self.gamma)
        if self.alpha_scalar is not None:
            alpha_t = self.alpha_scalar * targets + (1.0 - self.alpha_scalar) * (1.0 - targets)
            loss = loss * alpha_t
        valid = torch.ones_like(loss, dtype=torch.bool)
        return self._reduce(loss, valid).to(logits.dtype)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.mode == "multiclass":
            return self._multiclass(logits, targets)
        return self._binary(logits, targets)
