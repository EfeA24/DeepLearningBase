# Purpose: Provides seq2seq decoder causal self-attention with target padding mask and an optional incremental key/value cache for generation.
"""Decoder causal self-attention wrapper.

Inputs:
    x                      (B, T, d_model) new decoder positions (T = 1 during cached decoding)
    decoder_attention_mask optional (B, past + T), 1/True = real target token, 0/False = pad
    past_key_value         optional (k, v), each (B, H, past, Dh) from the previous call
Output: (output (B, T, d_model), present_key_value (k, v) with length past + T, or None
        when ``use_cache=False``).
Mask polarity: True = may attend. Query i (absolute position past + i) sees keys <= past + i.

The attention primitive is injected or lazily built from models/attention/multi_head_attention.py
and must expose ``project_query``, ``project_key_value`` and ``attend`` (see that file).

Requires: torch; sibling models/attention/multi_head_attention.py when ``attention`` is None.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

KeyValue = Tuple[torch.Tensor, torch.Tensor]


class CausalSelfAttention(nn.Module):
    """Masked (causal) self-attention with optional KV cache."""

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
        decoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[KeyValue] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KeyValue]]:
        q = self.attention.project_query(x)
        k, v = self.attention.project_key_value(x, x)
        past_length = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_length = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_length, total_length = x.size(1), k.size(2)
        mask = torch.ones((new_length, total_length), dtype=torch.bool, device=x.device)
        mask = torch.tril(mask, diagonal=past_length)[None, None]
        if decoder_attention_mask is not None:
            if decoder_attention_mask.shape != (x.size(0), total_length):
                raise ValueError(
                    f"decoder_attention_mask must be (B, past+T)={(x.size(0), total_length)}, "
                    f"got {tuple(decoder_attention_mask.shape)}"
                )
            mask = mask & decoder_attention_mask.to(torch.bool)[:, None, None, :]
        output, _ = self.attention.attend(q, k, v, attn_mask=mask)
        return output, ((k, v) if use_cache else None)
