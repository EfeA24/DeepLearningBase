# Purpose: Builds boolean and additive causal attention masks (with KV-cache offsets) for decoder-only language models.
"""Causal masks.

Polarity: boolean masks use True = query may attend to key (same as the ``attn_mask`` bool
convention of ``F.scaled_dot_product_attention``). Additive masks hold 0.0 where attention is
allowed and ``fill_value`` (default -inf) where it is blocked.

Shape: (q_len, kv_len). With a KV cache the queries are the LAST ``q_len`` positions of the
``kv_len`` keys, so query i (0-based within the new block) may attend keys
``j <= (kv_len - q_len) + i``. Without a cache ``kv_len == q_len`` and this is the usual
lower-triangular mask.

Note: ``F.scaled_dot_product_attention(is_causal=True)`` aligns the mask to the TOP-LEFT,
which is only correct when q_len == kv_len; use these explicit masks during cached decoding
with q_len > 1.

Requires: torch.
"""
from __future__ import annotations

from typing import Optional

import torch


def causal_mask(
    q_len: int, kv_len: Optional[int] = None, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Return a bool (q_len, kv_len) mask, True = allowed, aligned to the end of the keys."""
    kv_len = q_len if kv_len is None else kv_len
    if q_len <= 0 or kv_len <= 0:
        raise ValueError(f"q_len and kv_len must be positive, got {q_len}, {kv_len}")
    if kv_len < q_len:
        raise ValueError(f"kv_len ({kv_len}) must be >= q_len ({q_len})")
    return torch.ones(q_len, kv_len, dtype=torch.bool, device=device).tril(diagonal=kv_len - q_len)


def bool_to_additive(
    mask: torch.Tensor, dtype: torch.dtype = torch.float32, fill_value: Optional[float] = None
) -> torch.Tensor:
    """Convert a bool mask (True = allowed) to an additive mask (0 allowed, ``fill_value`` blocked).

    ``fill_value`` defaults to -inf. Use ``torch.finfo(dtype).min`` to avoid NaNs on rows that are
    fully blocked (the softmax then becomes uniform instead of NaN).
    """
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be bool, got {mask.dtype}")
    fill = float("-inf") if fill_value is None else fill_value
    out = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    return out.masked_fill(~mask, fill)


def causal_additive_mask(
    q_len: int,
    kv_len: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    fill_value: Optional[float] = None,
) -> torch.Tensor:
    """Additive (q_len, kv_len) causal mask: 0 where allowed, ``fill_value`` (default -inf) elsewhere."""
    return bool_to_additive(causal_mask(q_len, kv_len, device), dtype=dtype, fill_value=fill_value)
