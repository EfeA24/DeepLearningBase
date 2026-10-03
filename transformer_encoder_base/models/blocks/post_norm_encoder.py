# Purpose: Post-norm transformer encoder block (residual add then normalize, original BERT order) as a single explicit layer variant.
"""Post-norm encoder block (single layer).

    h = norm_attn(x + dropout(attn(x)))
    y = norm_ffn(h + ffn(h))

This is the original Transformer / BERT ordering; outputs are already normalized so no final
stack-level norm is required. Deep post-norm stacks usually need learning-rate warmup.
``ffn`` applies its own output dropout.

Shapes: (B, T, D) -> (B, T, D).
Mask: ``attention_mask`` (B, T) bool or 0/1, True/1 = valid token; forwarded to the attention module.

Components are injected: ``attention`` is called as
``attention(x, attention_mask=mask, need_weights=bool) -> (out, weights|None)``; ``feed_forward``
maps (B, T, D) -> (B, T, D). ``norm_factory(d_model)`` builds the norms (default
nn.LayerNorm(d_model, eps=1e-12) as in BERT). ``PostNormEncoderBlock.build`` constructs defaults.

Requires: torch; ``build`` lazily imports models/attention/multi_head_attention.py and
models/feedforward/feed_forward.py (template root on sys.path).
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

__all__ = ["PostNormEncoderBlock"]


class PostNormEncoderBlock(nn.Module):
    """Norm(x + Attn(x)), then Norm(. + FFN(.))."""

    def __init__(
        self,
        d_model: int,
        attention: nn.Module,
        feed_forward: nn.Module,
        dropout: float = 0.1,
        norm_factory: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        make_norm = norm_factory or (lambda dim: nn.LayerNorm(dim, eps=1e-12))
        self.attention = attention
        self.feed_forward = feed_forward
        self.norm_attn = make_norm(d_model)
        self.norm_ffn = make_norm(d_model)
        self.dropout = nn.Dropout(dropout)

    @classmethod
    def build(
        cls,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        norm_factory: Optional[Callable[[int], nn.Module]] = None,
    ) -> "PostNormEncoderBlock":
        """Construct with default MultiHeadAttention and FeedForward components (BERT-like dropouts)."""
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
        attn_out, weights = self.attention(x, attention_mask=attention_mask, need_weights=return_attention)
        x = self.norm_attn(x + self.dropout(attn_out))
        x = self.norm_ffn(x + self.feed_forward(x))
        return (x, weights) if return_attention else x
