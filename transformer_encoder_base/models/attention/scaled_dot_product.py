# Purpose: Reference (educational) scaled dot-product attention for encoder models with explicit mask polarity, dropout and optional attention weights.
"""Reference scaled dot-product attention (SDPA).

This is a readable reference, not a replacement for the fused
``torch.nn.functional.scaled_dot_product_attention`` (use that in production; it is what
``ScaledDotProductAttention(use_fused=True)`` calls).

Shapes:
    query (B, H, Tq, Dh), key (B, H, Tk, Dh), value (B, H, Tk, Dv) -> output (B, H, Tq, Dv)
    weights (B, H, Tq, Tk), rows sum to 1 over valid keys.
Mask (``attn_mask``), broadcastable to (B, H, Tq, Tk):
    bool: True = attend, False = blocked (same polarity as F.scaled_dot_product_attention).
    floating: additive bias, 0 = attend, -inf = blocked.
Rows with no attendable key: this reference returns zeros for those rows (weights all 0),
whereas the fused kernel may return NaN.
Dropout: like the fused API, dropout is applied whenever ``dropout_p > 0``; pass 0.0 in eval.

Requires: torch only.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["scaled_dot_product_attention", "ScaledDotProductAttention"]


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    return_weights: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """softmax(Q K^T * scale + mask) V. Returns (output, weights or None).

    Softmax is computed in float32 and cast back to the query dtype.
    """
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key and value must be 4-D (B, H, T, D)")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError(f"query/key head dims differ: {query.shape[-1]} vs {key.shape[-1]}")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError(f"key/value lengths differ: {key.shape[-2]} vs {value.shape[-2]}")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}")

    scale = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
    scores = torch.matmul(query, key.transpose(-2, -1)).float() * scale

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        elif attn_mask.dtype.is_floating_point:
            scores = scores + attn_mask.float()
        else:
            raise TypeError(f"attn_mask must be bool or floating, got {attn_mask.dtype}")

    weights = torch.softmax(scores, dim=-1)
    # Fully blocked rows produce NaN (softmax of all -inf); define their output as zero.
    weights = torch.nan_to_num(weights, nan=0.0)
    weights = weights.to(query.dtype)
    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p, training=True)
    output = torch.matmul(weights, value)
    return output, (weights if return_weights else None)


class ScaledDotProductAttention(nn.Module):
    """Module wrapper choosing between the fused PyTorch kernel and the reference implementation.

    The fused path cannot return weights; ``need_weights=True`` always uses the reference path.
    Dropout is active only in training mode.
    """

    def __init__(self, dropout: float = 0.0, use_fused: bool = True) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.dropout = dropout
        self.use_fused = use_fused

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        dropout_p = self.dropout if self.training else 0.0
        if self.use_fused and not need_weights:
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p)
            return out, None
        return scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, return_weights=need_weights
        )
