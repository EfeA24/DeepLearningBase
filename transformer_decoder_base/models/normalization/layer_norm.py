# Purpose: Provides a LayerNorm adapter with optional bias for GPT-style causal decoder blocks.
"""LayerNorm over the last dimension with an explicit ``bias`` switch.

    y = (x - mean) / sqrt(var + eps) * weight + bias      (biased variance)

Contract:
    input  (..., d_model)
    output (..., d_model), same dtype as the input

``bias=False`` reproduces the bias-free LayerNorm used by several modern decoders
(e.g. nanoGPT's option). Implemented with ``F.layer_norm`` so it behaves the same on
every torch >= 2.1 build. When the input is fp16/bf16 the statistic is computed in
float32 by default (``compute_in_float32=True``) and the result is cast back.

Requires: torch.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.Module):
    """LayerNorm with optional bias and optional float32 statistics.

    Args:
        dim: Size of the normalized (last) dimension.
        eps: Numerical stability constant.
        bias: Learn an additive bias (initialised to 0).
        elementwise_affine: Learn a multiplicative weight (initialised to 1).
        compute_in_float32: Upcast reduced-precision inputs before normalizing.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        bias: bool = True,
        elementwise_affine: bool = True,
        compute_in_float32: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.dim = dim
        self.eps = eps
        self.compute_in_float32 = compute_in_float32
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None
        self.bias = nn.Parameter(torch.zeros(dim)) if (bias and elementwise_affine) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got shape {tuple(x.shape)}")
        input_dtype = x.dtype
        if self.compute_in_float32 and input_dtype in (torch.float16, torch.bfloat16):
            weight = self.weight.float() if self.weight is not None else None
            bias = self.bias.float() if self.bias is not None else None
            out = F.layer_norm(x.float(), (self.dim,), weight, bias, self.eps)
            return out.to(input_dtype)
        return F.layer_norm(x, (self.dim,), self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}, bias={self.bias is not None}"
