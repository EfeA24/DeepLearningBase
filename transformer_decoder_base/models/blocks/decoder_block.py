# Purpose: Composes one pre-norm causal decoder block (attention + feed-forward with residuals) with explicit KV-cache passthrough.
"""Pre-norm decoder block (GPT-2 / Llama ordering):

    a, present = attention(attn_norm(x), attention_mask, position_ids, past_kv, use_cache)
    h   = x + dropout(a)
    out = h + dropout(feed_forward(ffn_norm(h)))

Block contract (shared with ``models/decoder/transformer_decoder.py`` and the gradient
checkpointing wrapper):
    forward(hidden_states (B, T, D), attention_mask=None, position_ids=None,
            past_kv=None, use_cache=False) -> (hidden_states (B, T, D), present_kv)

Injected components:
    attention:    module with the signature above returning ``(output, present_kv)``
                  (e.g. ``GroupedQueryAttention`` / ``CausalSelfAttention``); mask polarity,
                  cache layout (B, H_kv, T, D) and RoPE handling are the attention's contract.
    feed_forward: module (B, T, D) -> (B, T, D) (e.g. ``SwiGLU``, ``DecoderMLP``).
    attn_norm, ffn_norm: modules (B, T, D) -> (B, T, D) (e.g. ``RMSNorm``, ``LayerNorm``).

The block itself never creates or mutates caches; it only passes ``past_kv`` to the attention
and returns its ``present_kv``.

``build_default_decoder_block`` is an optional convenience factory that lazily imports the
sibling components listed below (absolute imports from the template root, so the template
root must be on ``sys.path`` when calling it).

Requires: torch. ``build_default_decoder_block`` additionally requires
models/attention/grouped_query_attention.py, models/feedforward/swiglu.py,
models/normalization/rms_norm.py.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch
from torch import nn


class DecoderBlock(nn.Module):
    """Pre-norm residual decoder block built from injected components.

    Args:
        attention: Causal self-attention module (see module docstring for signature).
        feed_forward: Position-wise feed-forward module.
        attn_norm: Normalization applied before attention.
        ffn_norm: Normalization applied before the feed-forward module.
        residual_dropout: Dropout on each residual branch output (training only).
    """

    def __init__(
        self,
        attention: nn.Module,
        feed_forward: nn.Module,
        attn_norm: nn.Module,
        ffn_norm: nn.Module,
        residual_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= residual_dropout < 1.0:
            raise ValueError(f"residual_dropout must be in [0, 1), got {residual_dropout}")
        self.attention = attention
        self.feed_forward = feed_forward
        self.attn_norm = attn_norm
        self.ffn_norm = ffn_norm
        self.dropout = nn.Dropout(residual_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_kv: Any = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Any]:
        attn_out, present = self.attention(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + self.dropout(attn_out)
        hidden_states = hidden_states + self.dropout(self.feed_forward(self.ffn_norm(hidden_states)))
        return hidden_states, present


def build_default_decoder_block(
    d_model: int,
    n_heads: int,
    n_kv_heads: Optional[int] = None,
    ffn_hidden_dim: Optional[int] = None,
    multiple_of: int = 256,
    dropout: float = 0.0,
    norm_eps: float = 1e-6,
    rope: Optional[Callable[..., Any]] = None,
) -> DecoderBlock:
    """Build a Llama-style block: RMSNorm + GQA (optional RoPE) + SwiGLU.

    ``rope`` is passed to the attention unchanged (e.g. ``RoPE(RotaryEmbedding(head_dim))``).
    """
    from models.attention.grouped_query_attention import GroupedQueryAttention
    from models.feedforward.swiglu import SwiGLU
    from models.normalization.rms_norm import RMSNorm

    attention = GroupedQueryAttention(d_model, n_heads, n_kv_heads=n_kv_heads, dropout=dropout, rope=rope)
    feed_forward = SwiGLU(d_model, hidden_dim=ffn_hidden_dim, multiple_of=multiple_of)
    return DecoderBlock(
        attention=attention,
        feed_forward=feed_forward,
        attn_norm=RMSNorm(d_model, eps=norm_eps),
        ffn_norm=RMSNorm(d_model, eps=norm_eps),
        residual_dropout=dropout,
    )
