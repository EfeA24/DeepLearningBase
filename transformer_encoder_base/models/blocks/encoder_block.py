# Purpose: One transformer encoder layer (self-attention + FFN with residuals) with an explicit norm_first flag selecting pre-norm or post-norm order.
"""Transformer encoder block.

Residual/norm order:
    norm_first=False (post-norm, original Transformer/BERT):
        x = norm1(x + dropout(attn(x)))
        x = norm2(x + ffn(x))
    norm_first=True (pre-norm, more stable for deep stacks; add a final norm after the stack):
        x = x + dropout(attn(norm1(x)))
        x = x + ffn(norm2(x))
``ffn`` is expected to apply its own output dropout (as models/feedforward/* do).

Shapes: x (B, T, D) -> (B, T, D).
Mask: ``attention_mask`` (B, T) bool or 0/1 with True/1 = valid token (or any 3-D/4-D mask accepted
by the attention module, True = attend; floating = additive 0/-inf).

Injected components (dependency injection; preferred):
    attention: module called as ``attention(x, attention_mask=mask, need_weights=bool)`` returning
        ``(output (B, T, D), weights or None)``.
    feed_forward: module mapping (B, T, D) -> (B, T, D).
If omitted they are built lazily from sibling files (template root must be on sys.path).

Requires: torch; when not injected, models/attention/multi_head_attention.py (MultiHeadAttention)
and models/feedforward/feed_forward.py (FeedForward).
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import nn

__all__ = ["TransformerEncoderBlock"]


class TransformerEncoderBlock(nn.Module):
    """Single encoder layer with configurable residual/normalization order."""

    def __init__(
        self,
        d_model: int,
        num_heads: Optional[int] = None,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = False,
        layer_norm_eps: float = 1e-5,
        attention: Optional[nn.Module] = None,
        feed_forward: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if attention is None:
            if num_heads is None:
                raise ValueError("num_heads is required when no attention module is injected")
            from models.attention.multi_head_attention import MultiHeadAttention

            attention = MultiHeadAttention(d_model, num_heads, dropout=attention_dropout)
        if feed_forward is None:
            if d_ff is None:
                raise ValueError("d_ff is required when no feed_forward module is injected")
            from models.feedforward.feed_forward import FeedForward

            feed_forward = FeedForward(d_model, d_ff, activation=activation, dropout=dropout)
        self.norm_first = norm_first
        self.attention = attention
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attention_dropout = nn.Dropout(dropout)

    def _attend(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor], need_weights: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out, weights = self.attention(x, attention_mask=attention_mask, need_weights=need_weights)
        return self.attention_dropout(out), weights

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Return hidden states (B, T, D), or (hidden, attention weights) if ``return_attention``."""
        if x.dim() != 3:
            raise ValueError(f"x must be (B, T, D), got shape {tuple(x.shape)}")
        if self.norm_first:
            attn_out, weights = self._attend(self.norm1(x), attention_mask, return_attention)
            x = x + attn_out
            x = x + self.feed_forward(self.norm2(x))
        else:
            attn_out, weights = self._attend(x, attention_mask, return_attention)
            x = self.norm1(x + attn_out)
            x = self.norm2(x + self.feed_forward(x))
        return (x, weights) if return_attention else x
