# Purpose: Provides a reusable RMSNorm layer for pre-norm causal decoder (GPT/Llama-style) language models.
"""RMSNorm (Zhang & Sennrich, 2019).

    y = weight * (x / sqrt(mean(x^2, dim=-1) + eps))

Contract:
    input  (..., d_model), any floating dtype
    output (..., d_model), same dtype as the input

Dtype semantics: the statistic is always computed in float32 (``x.float()``) for
stability under fp16/bf16, the normalized value is cast back to the input dtype,
and only then multiplied by ``weight`` (the Llama convention). ``weight`` is cast to
the input dtype for the product, so the output dtype always equals the input dtype
(even when parameters are kept in float32 and activations are bf16).

No mean subtraction and no bias (unlike LayerNorm).

Requires: torch.
"""
from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the last dimension.

    Args:
        dim: Size of the normalized (last) dimension.
        eps: Added to the mean square before the reciprocal square root.
        elementwise_affine: If True, learn a per-feature scale initialised to 1.
    """

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
            raise ValueError(f"Expected last dim {self.dim}, got shape {tuple(x.shape)}")
        input_dtype = x.dtype
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed = normed.to(input_dtype)
        if self.weight is not None:
            normed = normed * self.weight.to(input_dtype)
        return normed

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}, elementwise_affine={self.weight is not None}"
