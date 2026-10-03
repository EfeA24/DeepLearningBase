# Purpose: Pre-norm transformer encoder block (normalize before attention/FFN, residual after) as a single explicit layer variant.
"""Pre-norm encoder block (single layer).

    h = x + dropout(attn(norm_attn(x)))
    y = h + ffn(norm_ffn(h))

The residual stream is never normalized inside the block, so a stack of pre-norm blocks needs a
final LayerNorm (see models/encoder/transformer_encoder.py ``final_norm``).
``ffn`` applies its own output dropout.

Shapes: (B, T, D) -> (B, T, D).
Mask: ``attention_mask`` (B, T) bool or 0/1, True/1 = valid token; forwarded to the attention module.

Components are injected: ``attention`` is called as
``attention(x, attention_mask=mask, need_weights=bool) -> (out, weights|None)``; ``feed_forward``
maps (B, T, D) -> (B, T, D). ``norm_factory(d_model)`` builds the norms (default nn.LayerNorm),
e.g. pass an RMSNorm constructor. ``PreNormEncoderBlock.build`` constructs default components.

Requires: torch; ``build`` lazily imports models/attention/multi_head_attention.py and
models/feedforward/feed_forward.py (template root on sys.path).
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

__all__ = ["PreNormEncoderBlock"]


class PreNormEncoderBlock(nn.Module):
    """x + Attn(Norm(x)), then + FFN(Norm(.))."""

    def __init__(
        self,
        d_model: int,
        attention: nn.Module,
        feed_forward: nn.Module,
        dropout: float = 0.1,
        norm_factory: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        make_norm = norm_factory or (lambda dim: nn.LayerNorm(dim))
        self.norm_attn = make_norm(d_model)
        self.norm_ffn = make_norm(d_model)
        self.attention = attention
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)

    @classmethod
    def build(
        cls,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = "gelu",
        norm_factory: Optional[Callable[[int], nn.Module]] = None,
    ) -> "PreNormEncoderBlock":
        """Construct with default MultiHeadAttention and FeedForward components."""
        from models.attention.multi_head_attention import MultiHeadAttention
        from models.feedforward.feed_forward import FeedForward

        return cls(
            d_model,
            attention=MultiHeadAttention(d_model, num_heads, dropout=attention_dropout),
            feed_forward=FeedForward(d_model, d_ff, activation=activation, dropout=dropout),
            dropout=dropout,
            norm_factory=norm_factory,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        attn_out, weights = self.attention(
            self.norm_attn(x), attention_mask=attention_mask, need_weights=return_attention
        )
        x = x + self.dropout(attn_out)
        x = x + self.feed_forward(self.norm_ffn(x))
        return (x, weights) if return_attention else x
