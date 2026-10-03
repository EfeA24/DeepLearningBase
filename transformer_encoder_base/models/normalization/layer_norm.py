# Purpose: Reference LayerNorm (and a factory for torch.nn.LayerNorm) with explicit normalized dims for transformer encoders.
"""Layer normalization.

``LayerNorm`` is a readable reference: y = (x - mean) / sqrt(var + eps) * weight + bias, where mean
and (biased) variance are taken over the trailing ``normalized_shape`` dims. Statistics are computed
in float32 and the result is cast back to the input dtype (safe under fp16/bf16 autocast).
``build_layer_norm`` returns the optimized ``torch.nn.LayerNorm`` with the same arguments; prefer it
for production. Typical eps: 1e-12 (original BERT), 1e-5 (PyTorch default).

Shape: (..., *normalized_shape) -> same shape.

Requires: torch only (``bias`` argument of nn.LayerNorm requires torch >= 2.1).
"""
from __future__ import annotations

from typing import Sequence, Tuple, Union

import torch
from torch import nn

__all__ = ["LayerNorm", "build_layer_norm"]

ShapeSpec = Union[int, Sequence[int]]


def _as_shape(normalized_shape: ShapeSpec) -> Tuple[int, ...]:
    shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    if not shape or any(s <= 0 for s in shape):
        raise ValueError(f"normalized_shape must be positive, got {normalized_shape}")
    return shape


class LayerNorm(nn.Module):
    """Reference LayerNorm over the trailing ``normalized_shape`` dims."""

    def __init__(
        self,
        normalized_shape: ShapeSpec,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.normalized_shape = _as_shape(normalized_shape)
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape)) if bias else None
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[-len(self.normalized_shape):]) != self.normalized_shape:
            raise ValueError(f"trailing dims {tuple(x.shape)} do not match {self.normalized_shape}")
        dims = tuple(range(-len(self.normalized_shape), 0))
        x_float = x.float()
        mean = x_float.mean(dim=dims, keepdim=True)
        var = x_float.var(dim=dims, unbiased=False, keepdim=True)
        y = ((x_float - mean) * torch.rsqrt(var + self.eps)).to(x.dtype)
        if self.weight is not None:
            y = y * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, affine={self.weight is not None}"


def build_layer_norm(
    normalized_shape: ShapeSpec,
    eps: float = 1e-5,
    elementwise_affine: bool = True,
    bias: bool = True,
) -> nn.LayerNorm:
    """Construct the optimized torch.nn.LayerNorm with explicit arguments."""
    return nn.LayerNorm(_as_shape(normalized_shape), eps=eps, elementwise_affine=elementwise_affine, bias=bias)
