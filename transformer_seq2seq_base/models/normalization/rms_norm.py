# Purpose: Provides an RMSNorm variant (float32 statistics, cast back) as a drop-in norm for seq2seq transformer blocks.
"""Root-mean-square normalization.

    y = x / sqrt(mean(x**2, dim=-1) + eps) * weight      x: (..., d_model)
No mean subtraction and no bias. Statistics are computed in float32 and the result is cast
back to the input dtype (safe under fp16/bf16).

Compatibility notes: usable wherever LayerNorm is used in this template (pass
``norm_factory=lambda d: RMSNorm(d)`` to blocks). Pretrained checkpoints trained with
LayerNorm are NOT interchangeable with RMSNorm weights (T5 uses RMS-style norm; the
original Transformer/BART/Marian use LayerNorm).

Requires: torch only.
"""
from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm with a learned per-channel scale."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.d_model:
            raise ValueError(f"last dim must be {self.d_model}, got {x.size(-1)}")
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(dtype=x.dtype)
