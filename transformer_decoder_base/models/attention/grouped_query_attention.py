# Purpose: Implements grouped-query causal self-attention (GQA/MQA/MHA) with RoPE injection and KV caching for decoder LMs.
"""Grouped-query attention (Ainslie et al., 2023) for causal decoders.

n_kv_heads == n_heads -> standard multi-head attention
n_kv_heads == 1       -> multi-query attention
otherwise             -> GQA; n_heads must be divisible by n_kv_heads. Query head ``h``
                         uses key/value head ``h // (n_heads // n_kv_heads)`` (``repeat_kv``).

Shapes:
    x:            (B, T, d_model)               new tokens only when a cache is used
    q:            (B, n_heads, T, head_dim)
    k, v (cache): (B, n_kv_heads, T_past + T, head_dim)   cached BEFORE repeat_kv, AFTER RoPE
    output:       (B, T, d_model)

attention_mask (polarity: True/1 = may attend / valid):
    None                      -> pure causal; the query block is aligned to the END of the
                                 keys (query i sees keys j <= past_len + i).
    (B, T_past + T) bool/int  -> key padding mask (1 = real token) covering cached + new
                                 tokens; combined with the causal rule.
    (B, 1|H, T, T_past + T)   -> final mask used as-is (bool True = allowed, or additive
                                 float 0/-inf); the caller must already include causality
                                 (e.g. packed-sequence block-diagonal masks).
Query rows that would see no key at all (left-padding positions) are allowed to attend
everywhere to avoid NaNs; their outputs are meaningless and must be ignored downstream.

position_ids: (B, T) absolute positions for RoPE; default ``past_len + arange(T)``.
past_kv: None, a tuple ``(k, v)``, or a cache object exposing ``seq_len``/``update``
(see ``kv_cache.py``). Returns ``(output, present_kv)``; ``present_kv`` is None when
``use_cache`` is False and no cache object was passed.

``rope`` is an injected callable ``rope(q, k, position_ids) -> (q, k)`` (e.g. ``RoPE`` from
``models/embeddings/rotary/rope.py``); None disables rotary embeddings.

Requires: torch.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

RopeFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand (B, H_kv, T, D) to (B, H_kv * n_rep, T, D); head h maps to kv head h // n_rep."""
    if n_rep == 1:
        return x
    b, h_kv, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h_kv, n_rep, t, d).reshape(b, h_kv * n_rep, t, d)


def past_length(past_kv: Any) -> int:
    """Number of cached positions in a tuple/object cache (0 for None)."""
    if past_kv is None:
        return 0
    if hasattr(past_kv, "update"):
        return int(past_kv.seq_len)
    return int(past_kv[0].shape[2])


def update_kv(
    past_kv: Any, k: torch.Tensor, v: torch.Tensor, use_cache: bool
) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    """Merge new k/v with ``past_kv``; returns (k_all, v_all, present)."""
    if past_kv is None:
        return k, v, ((k, v) if use_cache else None)
    if hasattr(past_kv, "update"):
        k_all, v_all = past_kv.update(k, v)
        return k_all, v_all, past_kv
    k_past, v_past = past_kv
    k_all = torch.cat((k_past.to(k.dtype), k), dim=2)
    v_all = torch.cat((v_past.to(v.dtype), v), dim=2)
    return k_all, v_all, ((k_all, v_all) if use_cache else None)


def prepare_attention_mask(
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    q_len: int,
    kv_len: int,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], bool]:
    """Convert the documented mask forms into ``(attn_mask, is_causal)`` for SDPA."""
    past = kv_len - q_len
    if past < 0:
        raise ValueError(f"kv_len ({kv_len}) must be >= q_len ({q_len})")
    if attention_mask is None:
        if q_len == 1:
            return None, False
        if past == 0:
            return None, True
        causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=device).tril(diagonal=past)
        return causal, False
    if attention_mask.dim() == 2:
        if tuple(attention_mask.shape) != (batch_size, kv_len):
            raise ValueError(
                f"2D attention_mask must be (B, T_past + T) = {(batch_size, kv_len)}, "
                f"got {tuple(attention_mask.shape)}"
            )
        valid = attention_mask.to(device=device, dtype=torch.bool)
        causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=device).tril(diagonal=past)
        mask = causal[None, None, :, :] & valid[:, None, None, :]
        return mask | ~mask.any(dim=-1, keepdim=True), False
    if attention_mask.dim() == 4:
        if tuple(attention_mask.shape[-2:]) != (q_len, kv_len):
            raise ValueError(
                f"4D attention_mask must end with (T, T_past + T) = {(q_len, kv_len)}, "
                f"got {tuple(attention_mask.shape)}"
            )
        mask = attention_mask.to(device)
        if mask.dtype == torch.bool:
            return mask | ~mask.any(dim=-1, keepdim=True), False
        if mask.is_floating_point():
            return mask, False
        raise TypeError("4D attention_mask must be bool (True = allowed) or additive float")
    raise ValueError(f"attention_mask must be None, 2D or 4D, got {attention_mask.dim()}D")


class GroupedQueryAttention(nn.Module):
    """Causal GQA self-attention with optional RoPE and KV cache.

    Args:
        d_model: Model width.
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads (default ``n_heads``); must divide ``n_heads``.
        head_dim: Per-head size (default ``d_model // n_heads``).
        dropout: Attention-probability dropout (training only).
        bias: Use biases in the q/k/v/o projections.
        rope: Optional injected rotary callable ``(q, k, position_ids) -> (q, k)``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        rope: Optional[RopeFn] = None,
    ) -> None:
        super().__init__()
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if n_heads <= 0 or n_kv_heads <= 0:
            raise ValueError("n_heads and n_kv_heads must be positive")
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
        if head_dim is None:
            if d_model % n_heads != 0:
                raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
            head_dim = d_model // n_heads
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=bias)
        self.rope = rope

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_kv: Any = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Any]:
        if x.dim() != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must be (B, T, {self.d_model}), got {tuple(x.shape)}")
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        past_len = past_length(past_kv)
        if self.rope is not None:
            if position_ids is None:
                position_ids = torch.arange(past_len, past_len + t, device=x.device).unsqueeze(0).expand(b, t)
            q, k = self.rope(q, k, position_ids)

        k_all, v_all, present = update_kv(past_kv, k, v, use_cache)
        attn_mask, is_causal = prepare_attention_mask(attention_mask, b, t, k_all.shape[2], x.device)
        if attn_mask is not None and attn_mask.is_floating_point():
            attn_mask = attn_mask.to(q.dtype)

        out = F.scaled_dot_product_attention(
            q,
            repeat_kv(k_all, self.n_rep),
            repeat_kv(v_all, self.n_rep),
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim)
        return self.o_proj(out), present
