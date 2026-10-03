# Purpose: Provides a LayerNorm wrapper (optional bias) over the last dimension for pre-norm or post-norm seq2seq transformer blocks.
"""LayerNorm over the last (d_model) dimension.

    y = (x - mean) / sqrt(var + eps) * weight (+ bias)      x: (..., d_model)

Pre/post-norm compatibility:
    post-norm (original Transformer): norm AFTER each residual add; no final stack norm.
    pre-norm (``norm_first=True``): norm BEFORE each sub-layer; the encoder/decoder stack
        then needs one FINAL LayerNorm, otherwise the residual stream is unnormalized.
Statistics are computed in float32 under autocast by ``F.layer_norm``.

Requires: torch only.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.Module):
    """LayerNorm with an explicit ``bias`` switch (bias-free variants are common in newer models)."""

    def __init__(self, d_model: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.d_model:
            raise ValueError(f"last dim must be {self.d_model}, got {x.size(-1)}")
        return F.layer_norm(x, (self.d_model,), self.weight, self.bias, self.eps)
