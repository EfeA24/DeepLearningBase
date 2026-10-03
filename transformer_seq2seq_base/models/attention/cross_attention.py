# Purpose: Provides seq2seq decoder-to-encoder cross-attention with a memory key-padding mask and optional cached memory keys/values.
"""Cross-attention wrapper (decoder queries over encoder memory).

Inputs:
    x                     (B, T, d_model) decoder states (queries)
    memory                (B, S, d_model) encoder output; may be None if ``memory_key_value`` given
    memory_attention_mask optional (B, S), 1/True = real source token, 0/False = pad
    memory_key_value      optional cached projected (k, v), each (B, H, S, Dh)
Output: (output (B, T, d_model), memory_key_value or None when ``use_cache=False``).
Mask is broadcast to (B, 1, 1, S), True = may attend; no causal structure.
Memory K/V are identical at every decoding step, so caching them avoids re-projection.

Requires: torch; sibling models/attention/multi_head_attention.py when ``attention`` is None.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

KeyValue = Tuple[torch.Tensor, torch.Tensor]


class CrossAttention(nn.Module):
    """Decoder-encoder attention with memory padding mask."""

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
        memory: Optional[torch.Tensor] = None,
        memory_attention_mask: Optional[torch.Tensor] = None,
        memory_key_value: Optional[KeyValue] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KeyValue]]:
        q = self.attention.project_query(x)
        if memory_key_value is not None:
            k, v = memory_key_value
        else:
            if memory is None:
                raise ValueError("either memory or memory_key_value must be provided")
            k, v = self.attention.project_key_value(memory, memory)
        mask = None
        if memory_attention_mask is not None:
            if memory_attention_mask.shape != (x.size(0), k.size(2)):
                raise ValueError(
                    f"memory_attention_mask must be (B, S)={(x.size(0), k.size(2))}, "
                    f"got {tuple(memory_attention_mask.shape)}"
                )
            mask = memory_attention_mask.to(torch.bool)[:, None, None, :]
        output, _ = self.attention.attend(q, k, v, attn_mask=mask)
        return output, ((k, v) if use_cache else None)
