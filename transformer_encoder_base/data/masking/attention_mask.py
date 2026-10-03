# Purpose: Converts encoder padding masks between boolean (True=attend) and additive (0/-inf) attention-mask forms broadcastable to (B, H, Tq, Tk).
"""Encoder attention-mask conversions.

Conventions:
    boolean mask: True = query may attend to this key, False = blocked. This is the same
        polarity as ``torch.nn.functional.scaled_dot_product_attention(attn_mask=bool)``.
    additive mask: floating tensor added to attention scores; 0 = attend, -inf (or a very
        large negative number) = blocked.
    key padding mask input: (B, Tk) bool or integer 0/1, True/1 = valid token.

Broadcastable output shapes:
    (B, Tk)       -> (B, 1, 1, Tk)   key padding, same for every head and query
    (B, Tq, Tk)   -> (B, 1, Tq, Tk)  per-query mask
    (B, H, Tq, Tk) passes through.

Note: a query row whose keys are ALL blocked yields NaN in softmax with -inf masks. In an
encoder this only happens for an all-padding sequence; avoid such rows or use a finite fill.

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "to_bool_mask",
    "expand_key_padding_mask",
    "build_encoder_attention_mask",
    "bool_to_additive",
    "additive_to_bool",
]


def to_bool_mask(mask: torch.Tensor) -> torch.Tensor:
    """Bool or integer 0/1 mask -> bool (True = attend). Floating input is rejected (ambiguous)."""
    if mask.dtype == torch.bool:
        return mask
    if mask.dtype.is_floating_point:
        raise TypeError("Expected a bool or integer 0/1 mask; use additive_to_bool for additive masks.")
    return mask != 0


def expand_key_padding_mask(mask: torch.Tensor) -> torch.Tensor:
    """(B, Tk) valid mask -> (B, 1, 1, Tk) bool, broadcastable over heads and queries."""
    if mask.dim() != 2:
        raise ValueError(f"key padding mask must be (B, Tk), got shape {tuple(mask.shape)}")
    return to_bool_mask(mask)[:, None, None, :]


def build_encoder_attention_mask(
    mask: torch.Tensor,
    additive: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Normalize a (B,Tk) / (B,Tq,Tk) / (B,H,Tq,Tk) boolean-or-0/1 mask to a 4-D broadcastable mask.

    Args:
        mask: True/1 = attend.
        additive: return a floating 0/-inf mask instead of bool.
        dtype: floating dtype for the additive mask (default torch.float32).
    """
    if mask.dim() == 2:
        out = expand_key_padding_mask(mask)
    elif mask.dim() == 3:
        out = to_bool_mask(mask)[:, None, :, :]
    elif mask.dim() == 4:
        out = to_bool_mask(mask)
    else:
        raise ValueError(f"mask must have 2, 3 or 4 dims, got shape {tuple(mask.shape)}")
    if additive:
        return bool_to_additive(out, dtype=dtype or torch.float32)
    return out


def bool_to_additive(
    mask: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    blocked_value: Optional[float] = None,
) -> torch.Tensor:
    """Bool mask (True = attend) -> additive mask (0 where True, ``blocked_value`` where False).

    ``blocked_value`` defaults to -inf. Pass ``torch.finfo(dtype).min`` to avoid NaN for rows
    with no attendable key (they then attend uniformly to blocked keys instead).
    """
    if not dtype.is_floating_point:
        raise TypeError(f"additive masks need a floating dtype, got {dtype}")
    fill = float("-inf") if blocked_value is None else blocked_value
    out = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    return out.masked_fill(~to_bool_mask(mask), fill)


def additive_to_bool(mask: torch.Tensor) -> torch.Tensor:
    """Additive mask -> bool (True = attend). Values <= half the dtype minimum (incl. -inf) count as blocked."""
    if not mask.dtype.is_floating_point:
        raise TypeError(f"additive masks are floating tensors, got {mask.dtype}")
    threshold = torch.finfo(mask.dtype).min / 2
    return mask > threshold
