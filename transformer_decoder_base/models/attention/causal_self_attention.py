# Purpose: Implements GPT-style causal multi-head self-attention with fused QKV, optional RoPE and KV-cache passthrough.
"""Causal multi-head self-attention (all heads have their own K/V; see
``grouped_query_attention.py`` for GQA/MQA).

Shapes:
    x:           (B, T, d_model)       new tokens only when a cache is used
    q, k, v:     (B, n_heads, T, head_dim)
    cache k/v:   (B, n_heads, T_past + T, head_dim)   keys cached AFTER RoPE
    output:      (B, T, d_model)

attention_mask (True/1 = may attend / valid):
    None                       -> causal; queries aligned to the end of the keys
                                  (query i sees keys j <= T_past + i).
    (B, T_past + T) bool/int   -> key padding mask over cached + new tokens, combined
                                  with the causal rule.
    (B, 1|H, T, T_past + T)    -> final mask used as-is (bool True = allowed or additive
                                  float); caller must include causality.
Query rows with no visible key (left-padding positions) attend everywhere to avoid NaNs;
ignore their outputs.

past_kv / present_kv: None, tuple ``(k, v)`` (functional: a new tuple is returned when
``use_cache``), or a cache object with ``seq_len``/``update`` (updated in place and
returned). ``rope``: optional injected ``rope(q, k, position_ids) -> (q, k)``; when given
and ``position_ids`` is None, positions default to ``T_past + arange(T)``.

Self-contained (helpers duplicated from grouped_query_attention.py on purpose so this
file can be copied alone).

Requires: torch.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

RopeFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


def _past_length(past_kv: Any) -> int:
    if past_kv is None:
        return 0
    if hasattr(past_kv, "update"):
        return int(past_kv.seq_len)
    return int(past_kv[0].shape[2])


def _build_mask(
    attention_mask: Optional[torch.Tensor], batch_size: int, q_len: int, kv_len: int, device: torch.device
) -> Tuple[Optional[torch.Tensor], bool]:
    past = kv_len - q_len
    if attention_mask is None:
        if q_len == 1:
            return None, False
        if past == 0:
            return None, True
        return torch.ones(q_len, kv_len, dtype=torch.bool, device=device).tril(diagonal=past), False
    if attention_mask.dim() == 2:
        if tuple(attention_mask.shape) != (batch_size, kv_len):
            raise ValueError(
                f"2D attention_mask must be {(batch_size, kv_len)}, got {tuple(attention_mask.shape)}"
            )
        causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=device).tril(diagonal=past)
        mask = causal[None, None] & attention_mask.to(device=device, dtype=torch.bool)[:, None, None, :]
        return mask | ~mask.any(dim=-1, keepdim=True), False
    if attention_mask.dim() == 4:
        if tuple(attention_mask.shape[-2:]) != (q_len, kv_len):
            raise ValueError(
                f"4D attention_mask must end with {(q_len, kv_len)}, got {tuple(attention_mask.shape)}"
            )
        mask = attention_mask.to(device)
        if mask.dtype == torch.bool:
            return mask | ~mask.any(dim=-1, keepdim=True), False
        if mask.is_floating_point():
            return mask, False
        raise TypeError("4D attention_mask must be bool (True = allowed) or additive float")
    raise ValueError(f"attention_mask must be None, 2D or 4D, got {attention_mask.dim()}D")


class CausalSelfAttention(nn.Module):
    """Fused-QKV causal self-attention (GPT-2 layout) on top of ``F.scaled_dot_product_attention``.

    Args:
        d_model: Model width; must be divisible by ``n_heads``.
        n_heads: Number of attention heads.
        dropout: Attention-probability dropout (training only).
        bias: Biases in the QKV and output projections (GPT-2 uses True).
        rope: Optional injected rotary callable.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        rope: Optional[RopeFn] = None,
    ) -> None:
        super().__init__()
        if n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by a positive n_heads ({n_heads})")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
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
        q, k, v = self.qkv_proj(x).split(self.d_model, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        past_len = _past_length(past_kv)
        if self.rope is not None:
            if position_ids is None:
                position_ids = torch.arange(past_len, past_len + t, device=x.device).unsqueeze(0).expand(b, t)
            q, k = self.rope(q, k, position_ids)

        if past_kv is None:
            k_all, v_all = k, v
            present = (k, v) if use_cache else None
        elif hasattr(past_kv, "update"):
            k_all, v_all = past_kv.update(k, v)
            present = past_kv
        else:
            k_all = torch.cat((past_kv[0].to(k.dtype), k), dim=2)
            v_all = torch.cat((past_kv[1].to(v.dtype), v), dim=2)
            present = (k_all, v_all) if use_cache else None

        attn_mask, is_causal = _build_mask(attention_mask, b, t, k_all.shape[2], x.device)
        if attn_mask is not None and attn_mask.is_floating_point():
            attn_mask = attn_mask.to(q.dtype)
        out = F.scaled_dot_product_attention(
            q,
            k_all,
            v_all,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).reshape(b, t, self.d_model)
        return self.out_proj(out), present
