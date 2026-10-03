# Purpose: Implements one seq2seq transformer encoder layer (self-attention + feed-forward) with an explicit norm_first residual/norm order.
"""Single encoder layer.

    x (B, S, d_model), attention_mask (B, S) 1/True = real token -> (B, S, d_model)

Residual/norm order:
    norm_first=True  (pre-norm):  x = x + Drop(SelfAttn(Norm1(x)));  x = x + Drop(FFN(Norm2(x)))
                                  -> the encoder stack must apply a final norm.
    norm_first=False (post-norm): x = Norm1(x + Drop(SelfAttn(x)));  x = Norm2(x + Drop(FFN(x)))

Sub-modules are injectable: ``self_attention`` must follow
``(x, attention_mask) -> (out, weights)``; ``feed_forward`` maps (B, S, D) -> (B, S, D);
``norm_factory(d_model) -> nn.Module`` (default ``nn.LayerNorm``).

Requires: torch; when not injected, siblings models/attention/self_attention.py (+ its
multi_head_attention.py) and models/feedforward/feed_forward.py, imported lazily in __init__.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import torch
from torch import nn


class EncoderBlock(nn.Module):
    """Self-attention + FFN encoder layer."""

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
        feed_forward: Optional[nn.Module] = None,
        norm_factory: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        if self_attention is None:
            from models.attention.self_attention import EncoderSelfAttention

            self_attention = EncoderSelfAttention(d_model, num_heads, dropout=dropout)
        if feed_forward is None:
            from models.feedforward.feed_forward import PositionwiseFeedForward

            feed_forward = PositionwiseFeedForward(d_model, d_ff, activation=activation, dropout=dropout)
        if norm_factory is None:
            norm_factory = lambda dim: nn.LayerNorm(dim, eps=layer_norm_eps)  # noqa: E731
        self.norm_first = norm_first
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.norm1 = norm_factory(d_model)
        self.norm2 = norm_factory(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.norm_first:
            attn_out, _ = self.self_attention(self.norm1(x), attention_mask)
            x = x + self.dropout(attn_out)
            return x + self.dropout(self.feed_forward(self.norm2(x)))
        attn_out, _ = self.self_attention(x, attention_mask)
        x = self.norm1(x + self.dropout(attn_out))
        return self.norm2(x + self.dropout(self.feed_forward(x)))
