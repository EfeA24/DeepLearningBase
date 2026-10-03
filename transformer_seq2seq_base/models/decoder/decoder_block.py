# Purpose: Implements one seq2seq transformer decoder layer (masked self-attention, cross-attention, feed-forward) with documented residual/norm order and cache passthrough.
"""Single decoder layer.

Inputs:
    x                      (B, T, d_model) target states (T = 1 when decoding with cache)
    memory                 (B, S, d_model) encoder output (may be None if cross K/V cached)
    decoder_attention_mask optional (B, past + T), 1/True = real target token
    memory_attention_mask  optional (B, S), 1/True = real source token
    past_key_value         optional dict {"self": (k, v), "cross": (k, v)} from the last step
Output: (x (B, T, d_model), present dict with the same keys, or None if ``use_cache=False``).

Residual/norm order:
    norm_first=True:  x += Drop(SelfAttn(N1(x))); x += Drop(CrossAttn(N2(x), memory)); x += Drop(FFN(N3(x)))
    norm_first=False: x = N1(x + Drop(SelfAttn(x))); x = N2(x + Drop(CrossAttn(x))); x = N3(x + Drop(FFN(x)))

Injected sub-module contracts:
    self_attention(x, decoder_attention_mask, past_key_value=, use_cache=) -> (out, present)
    cross_attention(x, memory, memory_attention_mask, memory_key_value=, use_cache=) -> (out, kv)
    feed_forward(x) -> x;  norm_factory(d_model) -> nn.Module (default nn.LayerNorm)

Requires: torch; when not injected, siblings models/attention/causal_attention.py,
models/attention/cross_attention.py (+ multi_head_attention.py) and
models/feedforward/feed_forward.py, imported lazily in __init__.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Dict, Optional, Tuple

import torch
from torch import nn

LayerCache = Dict[str, Tuple[torch.Tensor, torch.Tensor]]


class DecoderBlock(nn.Module):
    """Masked self-attention -> cross-attention -> FFN decoder layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_first: bool = True,
        layer_norm_eps: float = 1e-5,
        self_attention: Optional[nn.Module] = None,
        cross_attention: Optional[nn.Module] = None,
        feed_forward: Optional[nn.Module] = None,
        norm_factory: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        if self_attention is None:
            from models.attention.causal_attention import CausalSelfAttention

            self_attention = CausalSelfAttention(d_model, num_heads, dropout=dropout)
        if cross_attention is None:
            from models.attention.cross_attention import CrossAttention

            cross_attention = CrossAttention(d_model, num_heads, dropout=dropout)
        if feed_forward is None:
            from models.feedforward.feed_forward import PositionwiseFeedForward

            feed_forward = PositionwiseFeedForward(d_model, d_ff, activation=activation, dropout=dropout)
        if norm_factory is None:
            norm_factory = lambda dim: nn.LayerNorm(dim, eps=layer_norm_eps)  # noqa: E731
        self.norm_first = norm_first
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.norm1 = norm_factory(d_model)
        self.norm2 = norm_factory(d_model)
        self.norm3 = norm_factory(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor],
        decoder_attention_mask: Optional[torch.Tensor] = None,
        memory_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[LayerCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[LayerCache]]:
        self_past = past_key_value.get("self") if past_key_value else None
        cross_past = past_key_value.get("cross") if past_key_value else None

        h = self.norm1(x) if self.norm_first else x
        h, self_present = self.self_attention(h, decoder_attention_mask, past_key_value=self_past, use_cache=use_cache)
        x = x + self.dropout(h) if self.norm_first else self.norm1(x + self.dropout(h))

        h = self.norm2(x) if self.norm_first else x
        h, cross_present = self.cross_attention(
            h, memory, memory_attention_mask, memory_key_value=cross_past, use_cache=use_cache
        )
        x = x + self.dropout(h) if self.norm_first else self.norm2(x + self.dropout(h))

        h = self.norm3(x) if self.norm_first else x
        h = self.feed_forward(h)
        x = x + self.dropout(h) if self.norm_first else self.norm3(x + self.dropout(h))

        present = {"self": self_present, "cross": cross_present} if use_cache else None
        return x, present
