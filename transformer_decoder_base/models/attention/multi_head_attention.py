# Purpose: Provides a reference multi-head attention module with separate query/key/value inputs for decoder-side use.
"""Standard multi-head attention with explicit q/k/v inputs (reference implementation).

Shapes:
    query: (B, T_q, d_model)    key, value: (B, T_k, d_model)
    output: (B, T_q, d_model)
    weights (optional): (B, n_heads, T_q, T_k), softmax probabilities after masking

attn_mask (polarity: True = may attend):
    bool  (T_q, T_k) | (B, T_q, T_k) | (B, 1|H, T_q, T_k)
    float of the same shapes = additive mask (0 keep, -inf block)
key_padding_mask: bool (B, T_k), True = real token (NOT the torch.nn.MultiheadAttention
    convention, where True = pad).
is_causal: adds the causal rule with queries aligned to the END of the keys
    (query i sees keys j <= (T_k - T_q) + i), i.e. correct for KV-cached decoding.

Fully-masked query rows produce zeros (instead of NaN) in the explicit path; with the SDPA
path they attend everywhere. Either way their outputs should be ignored.

``need_weights=False`` uses ``F.scaled_dot_product_attention``; ``need_weights=True`` uses the
explicit softmax(QK^T / sqrt(d)) V path so the probabilities can be returned. This module
has no cache logic; see ``causal_self_attention.py`` / ``grouped_query_attention.py``.

Requires: torch.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _to_4d(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 2:
        return mask[None, None]
    if mask.dim() == 3:
        return mask[:, None]
    if mask.dim() == 4:
        return mask
    raise ValueError(f"attn_mask must be 2D, 3D or 4D, got {mask.dim()}D")


class MultiHeadAttention(nn.Module):
    """Multi-head attention with separate q/k/v projections.

    Args:
        d_model: Model width; divisible by ``n_heads``.
        n_heads: Number of heads.
        dropout: Attention-probability dropout (training only).
        bias: Biases in the projections.
        kdim, vdim: Input widths of key/value (default ``d_model``).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
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
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(kdim or d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(vdim or d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def _combined_mask(
        self,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool,
        batch_size: int,
        q_len: int,
        k_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Merge all mask sources into one 4D tensor: bool (True = allowed) or additive float."""
        allowed: Optional[torch.Tensor] = None
        additive: Optional[torch.Tensor] = None
        if is_causal:
            if k_len < q_len:
                raise ValueError("is_causal requires T_k >= T_q")
            allowed = torch.ones(q_len, k_len, dtype=torch.bool, device=device).tril(k_len - q_len)[None, None]
        if key_padding_mask is not None:
            if tuple(key_padding_mask.shape) != (batch_size, k_len):
                raise ValueError(
                    f"key_padding_mask must be (B, T_k) = {(batch_size, k_len)}, got {tuple(key_padding_mask.shape)}"
                )
            kpm = key_padding_mask.to(device=device, dtype=torch.bool)[:, None, None, :]
            allowed = kpm if allowed is None else allowed & kpm
        if attn_mask is not None:
            m = _to_4d(attn_mask.to(device))
            if m.dtype == torch.bool:
                allowed = m if allowed is None else allowed & m
            elif m.is_floating_point():
                additive = m.to(dtype)
            else:
                raise TypeError("attn_mask must be bool (True = allowed) or additive float")
        if additive is None:
            return allowed
        if allowed is not None:
            additive = additive.masked_fill(~allowed, float("-inf"))
        return additive

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError("query, key and value must be (B, T, d)")
        if key.shape[:2] != value.shape[:2]:
            raise ValueError("key and value must share (B, T_k)")
        b, t_q, _ = query.shape
        t_k = key.shape[1]
        q = self._split(self.q_proj(query))
        k = self._split(self.k_proj(key))
        v = self._split(self.v_proj(value))
        mask = self._combined_mask(attn_mask, key_padding_mask, is_causal, b, t_q, t_k, query.device, q.dtype)
        p = self.dropout if self.training else 0.0

        if not need_weights:
            if mask is not None and mask.dtype == torch.bool:
                mask = mask | ~mask.any(dim=-1, keepdim=True)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
            weights = None
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
            weights = torch.softmax(scores.float(), dim=-1).nan_to_num(0.0).to(q.dtype)
            out = F.dropout(weights, p=p) @ v
        out = out.transpose(1, 2).reshape(b, t_q, self.d_model)
        return self.out_proj(out), weights
