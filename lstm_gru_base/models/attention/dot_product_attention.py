# Purpose: Implements (scaled) dot-product and Luong "general" attention with explicit mask polarity for LSTM/GRU attention models; a readable reference, not a replacement for optimized SDPA.
"""Dot-product attention for recurrent models.

score(q, k_j) = q . k_j            (``mode="dot"``)
score(q, k_j) = q . (W k_j)        (``mode="general"``, Luong et al. 2015)
Optionally divided by sqrt(d) (``scaled=True``).

Shapes (batch-first):
    - ``query``: ``(B, D)`` or ``(B, Tq, D)``; ``keys``: ``(B, Tk, Dk)`` (``Dk == D`` for "dot");
      ``values``: ``(B, Tk, Dv)`` (defaults to keys).
    - ``mask``: bool, True = attend/valid; ``(B, Tk)`` or ``(B, Tq, Tk)``. (``F.scaled_dot_product_attention``
      also uses True = attend for boolean masks; ``nn.MultiheadAttention.key_padding_mask`` uses True = pad.)
Returns ``(context, weights)`` with ``(B, [Tq,] Dv)`` and ``(B, [Tq,] Tk)``; fully masked rows yield zeros.

Requires: torch only.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


def dot_product_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    scaled: bool = True,
    dropout_p: float = 0.0,
    training: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Functional (scaled) dot-product attention; see module docstring for shapes."""
    values = keys if values is None else values
    single_step = query.dim() == 2
    q = query.unsqueeze(1) if single_step else query
    if q.dim() != 3 or keys.dim() != 3 or q.shape[-1] != keys.shape[-1]:
        raise ValueError(f"Incompatible shapes query={tuple(query.shape)} keys={tuple(keys.shape)}.")
    scores = torch.bmm(q, keys.transpose(1, 2))
    if scaled:
        scores = scores / math.sqrt(q.shape[-1])
    valid = None
    if mask is not None:
        valid = mask.bool()
        if valid.dim() == 2:
            valid = valid.unsqueeze(1)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    if valid is not None:
        weights = weights * valid
    if dropout_p > 0:
        weights = nn.functional.dropout(weights, p=dropout_p, training=training)
    context = torch.bmm(weights, values)
    if single_step:
        return context.squeeze(1), weights.squeeze(1)
    return context, weights


class DotProductAttention(nn.Module):
    """Module form supporting ``mode="dot"`` or ``mode="general"`` (learned key projection)."""

    def __init__(
        self,
        query_dim: int,
        key_dim: Optional[int] = None,
        mode: str = "dot",
        scaled: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        key_dim = query_dim if key_dim is None else key_dim
        if mode not in ("dot", "general"):
            raise ValueError(f"mode must be 'dot' or 'general', got {mode!r}.")
        if mode == "dot" and key_dim != query_dim:
            raise ValueError("mode='dot' requires key_dim == query_dim; use mode='general' otherwise.")
        self.mode = mode
        self.scaled = scaled
        self.dropout = dropout
        self.key_proj = nn.Linear(key_dim, query_dim, bias=False) if mode == "general" else None

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        values = keys if values is None else values
        projected_keys = self.key_proj(keys) if self.key_proj is not None else keys
        return dot_product_attention(query, projected_keys, values, mask, self.scaled, self.dropout, self.training)
