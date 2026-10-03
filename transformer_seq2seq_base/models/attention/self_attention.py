# Purpose: Provides seq2seq encoder self-attention that applies a source key-padding mask over bidirectional attention.
"""Encoder self-attention wrapper.

    x (B, S, d_model), attention_mask (B, S) with 1/True = real token, 0/False = pad
    -> output (B, S, d_model), optional weights (B, H, S, S)
The padding mask is broadcast to (B, 1, 1, S) (True = may attend); no causal masking.

The attention primitive is injected (``attention=``) or, by default, built lazily from
models/attention/multi_head_attention.py. An injected module must implement
``forward(query, key, value, attn_mask=None, need_weights=False) -> (out, weights)``.

Requires: torch; sibling models/attention/multi_head_attention.py when ``attention`` is None
(imported inside __init__ with the template root on sys.path).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class EncoderSelfAttention(nn.Module):
    """Bidirectional self-attention with key-padding mask."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        attention: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if attention is None:
            from models.attention.multi_head_attention import MultiHeadAttention

            attention = MultiHeadAttention(d_model, num_heads, dropout=dropout, bias=bias)
        self.attention = attention

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mask = None
        if attention_mask is not None:
            if attention_mask.shape != x.shape[:2]:
                raise ValueError(
                    f"attention_mask must be (B, S)={tuple(x.shape[:2])}, got {tuple(attention_mask.shape)}"
                )
            mask = attention_mask.to(torch.bool)[:, None, None, :]
        return self.attention(x, x, x, attn_mask=mask, need_weights=need_weights)
