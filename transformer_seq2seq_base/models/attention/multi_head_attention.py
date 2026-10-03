# Purpose: Implements the shared multi-head attention projection primitive (separate q/k/v inputs, boolean mask, cache-friendly projection split) for seq2seq transformers.
"""Multi-head attention primitive used by the self/causal/cross attention wrappers.

Shapes:
    query (B, Tq, d_model), key/value (B, Tk, d_model)
    projected heads q (B, H, Tq, Dh), k/v (B, H, Tk, Dh), Dh = d_model // H
    attn_mask: bool, broadcastable to (B, H, Tq, Tk), True = query MAY attend to key
               (same polarity as ``F.scaled_dot_product_attention`` boolean masks).
    output (B, Tq, d_model) after the output projection; optional weights (B, H, Tq, Tk).

Fully masked rows (no allowed key) are made finite by unmasking that row; such positions
carry no information and must be ignored downstream (e.g. labels = -100 at pads).

The split API (``project_query``, ``project_key_value``, ``attend``) lets wrappers cache
projected K/V for incremental decoding and cross-attention memory reuse.

Requires: torch only.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
    """Standard MHA with separate q/k/v projections.

    Args:
        d_model: model width (must be divisible by ``num_heads``).
        num_heads: number of heads H.
        dropout: attention-probability dropout (training only).
        bias: use biases in projections.
        use_sdpa: use ``F.scaled_dot_product_attention`` when weights are not requested.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, bias: bool = True, use_sdpa: bool = True) -> None:
        super().__init__()
        if d_model < 1 or num_heads < 1:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.use_sdpa = use_sdpa
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, H, T, Dh)."""
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, Dh) -> (B, T, d_model)."""
        batch, _, length, _ = x.shape
        return x.transpose(1, 2).reshape(batch, length, self.d_model)

    def project_query(self, query: torch.Tensor) -> torch.Tensor:
        return self.split_heads(self.q_proj(query))

    def project_key_value(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.split_heads(self.k_proj(key)), self.split_heads(self.v_proj(value))

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Attention over already projected heads; returns (output (B,Tq,D), weights or None)."""
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                raise TypeError(f"attn_mask must be bool (True = may attend), got {attn_mask.dtype}")
            attn_mask = attn_mask | ~attn_mask.any(dim=-1, keepdim=True)
        dropout_p = self.dropout if self.training else 0.0
        if self.use_sdpa and not need_weights:
            context = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
            weights = None
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            weights = scores.softmax(dim=-1)
            context = torch.matmul(F.dropout(weights, p=dropout_p, training=self.training), v)
        return self.out_proj(self.merge_heads(context)), weights

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q = self.project_query(query)
        k, v = self.project_key_value(key, value)
        return self.attend(q, k, v, attn_mask=attn_mask, need_weights=need_weights)
