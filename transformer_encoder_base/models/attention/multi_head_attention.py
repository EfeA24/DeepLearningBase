# Purpose: Self-contained multi-head attention (projections, head split/merge, mask handling, output projection) for bidirectional encoder blocks.
"""Multi-head attention for encoders.

Shapes (batch-first):
    query (B, Tq, D); key/value (B, Tk, D) (default: key = value = query -> self-attention).
    output (B, Tq, D); optional per-head weights (B, H, Tq, Tk).
Mask (``attention_mask``):
    bool or integer 0/1: True/1 = attend.
    floating: additive bias (0 = attend, -inf = blocked).
    Shapes: (B, Tk) key padding mask, (Tq, Tk), (B, Tq, Tk) or (B, 1|H, Tq, Tk).
    A 2-D mask matching both (B, Tk) and (Tq, Tk) is read as key padding.
Padded query positions still produce (meaningless) outputs; downstream code must ignore them via
the same padding mask. A sequence with zero valid tokens yields NaN on the fused path.

Implementation: uses F.scaled_dot_product_attention unless ``need_weights=True`` or
``use_fused=False``, in which case a local reference computation is used (kept self-contained so
this file can be copied alone).

Requires: torch only.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["MultiHeadAttention"]


def _normalize_mask(
    mask: Optional[torch.Tensor], batch_size: int, query_len: int, key_len: int
) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if not mask.dtype.is_floating_point and mask.dtype != torch.bool:
        mask = mask != 0
    if mask.dim() == 2:
        if mask.shape == (batch_size, key_len):
            mask = mask[:, None, None, :]
        elif mask.shape == (query_len, key_len):
            mask = mask[None, None, :, :]
        else:
            raise ValueError(f"2-D mask must be (B, Tk) or (Tq, Tk), got {tuple(mask.shape)}")
    elif mask.dim() == 3:
        mask = mask[:, None, :, :]
    elif mask.dim() != 4:
        raise ValueError(f"mask must have 2, 3 or 4 dims, got shape {tuple(mask.shape)}")
    if mask.shape[-1] != key_len:
        raise ValueError(f"mask key dim {mask.shape[-1]} != key length {key_len}")
    return mask


class MultiHeadAttention(nn.Module):
    """Standard MHA: Q/K/V linear projections, H parallel heads, output projection."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        use_fused: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.use_fused = use_fused
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (output (B, Tq, D), weights (B, H, Tq, Tk) or None)."""
        key = query if key is None else key
        value = key if value is None else value
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError("query, key and value must be (B, T, D)")
        if query.shape[-1] != self.d_model:
            raise ValueError(f"expected hidden size {self.d_model}, got {query.shape[-1]}")
        batch, query_len, _ = query.shape
        key_len = key.shape[1]

        q = self._split(self.q_proj(query))
        k = self._split(self.k_proj(key))
        v = self._split(self.v_proj(value))
        mask = _normalize_mask(attention_mask, batch, query_len, key_len)
        dropout_p = self.dropout if self.training else 0.0

        weights: Optional[torch.Tensor] = None
        if self.use_fused and not need_weights:
            context = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)).float() / math.sqrt(self.head_dim)
            if mask is not None:
                if mask.dtype == torch.bool:
                    scores = scores.masked_fill(~mask, float("-inf"))
                else:
                    scores = scores + mask.float()
            weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0).to(q.dtype)
            attn = F.dropout(weights, p=dropout_p, training=True) if dropout_p > 0 else weights
            context = torch.matmul(attn, v)

        context = context.transpose(1, 2).reshape(batch, query_len, self.d_model)
        return self.out_proj(context), (weights if need_weights else None)
