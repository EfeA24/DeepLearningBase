# Purpose: RMSNorm layer (root-mean-square normalization, float32 compute then cast back) for transformer encoder variants.
"""RMSNorm.

y = x / sqrt(mean(x^2 over last dim) + eps) * weight      (no mean subtraction, no bias)

Dtype behavior: the normalization is computed in float32, cast back to the input dtype, and then
multiplied by ``weight`` (LLaMA convention), so the output dtype follows normal type promotion of
(input dtype, weight dtype).
Shape: (..., dim) -> (..., dim).

Requires: torch only.
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["RMSNorm"]


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization over the last dimension."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.dim = dim
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dim {self.dim}, got {x.shape[-1]}")
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed = normed.to(x.dtype)
        return normed * self.weight if self.weight is not None else normed

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}, affine={self.weight is not None}"
