# Purpose: Provides head split/merge and attention-mask normalization helpers for multi-head encoder attention.
"""Shape and mask helpers for multi-head attention.

Shapes: hidden (B, T, D) with D = H * Dh; per-head tensors (B, H, T, Dh).

Mask normalization contract (``normalize_attention_mask``):
    bool or integer 0/1 masks: True/1 = attend (same polarity as F.scaled_dot_product_attention).
    floating masks: additive, 0 = attend, -inf = blocked; passed through unchanged.
    Accepted shapes: (B, Tk) key padding, (Tq, Tk), (B, Tq, Tk), (B, 1|H, Tq, Tk).
    Output is 4-D and broadcastable to (B, H, Tq, Tk).

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional

import torch

__all__ = ["split_heads", "merge_heads", "normalize_attention_mask", "bool_mask_to_additive"]


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """(B, T, D) -> (B, H, T, D // H)."""
    if x.dim() != 3:
        raise ValueError(f"expected (B, T, D), got shape {tuple(x.shape)}")
    batch, length, dim = x.shape
    if dim % num_heads != 0:
        raise ValueError(f"hidden size {dim} is not divisible by num_heads={num_heads}")
    return x.view(batch, length, num_heads, dim // num_heads).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, H, T, Dh) -> (B, T, H * Dh)."""
    if x.dim() != 4:
        raise ValueError(f"expected (B, H, T, Dh), got shape {tuple(x.shape)}")
    batch, heads, length, head_dim = x.shape
    return x.transpose(1, 2).reshape(batch, length, heads * head_dim)


def normalize_attention_mask(
    mask: Optional[torch.Tensor],
    batch_size: int,
    query_len: int,
    key_len: int,
) -> Optional[torch.Tensor]:
    """Return a 4-D mask broadcastable to (B, H, Tq, Tk); see module docstring for accepted inputs.

    A 2-D mask is interpreted as (B, Tk) key padding when its shape is (batch_size, key_len),
    otherwise as (Tq, Tk). If both interpretations match (B == Tq), key padding wins.
    """
    if mask is None:
        return None
    if not mask.dtype.is_floating_point and mask.dtype != torch.bool:
        mask = mask != 0
    if mask.dim() == 2:
        if mask.shape == (batch_size, key_len):
            mask = mask[:, None, None, :]
        elif mask.shape == (query_len, key_len):
            mask = mask[None, None, :, :]
        else:
            raise ValueError(
                f"2-D mask must be (B={batch_size}, Tk={key_len}) or (Tq={query_len}, Tk={key_len}), "
                f"got {tuple(mask.shape)}"
            )
    elif mask.dim() == 3:
        mask = mask[:, None, :, :]
    elif mask.dim() != 4:
        raise ValueError(f"mask must have 2, 3 or 4 dims, got shape {tuple(mask.shape)}")
    if mask.shape[-1] != key_len or mask.shape[-2] not in (1, query_len):
        raise ValueError(f"mask shape {tuple(mask.shape)} incompatible with Tq={query_len}, Tk={key_len}")
    if mask.shape[0] not in (1, batch_size):
        raise ValueError(f"mask batch dim {mask.shape[0]} incompatible with batch size {batch_size}")
    return mask


def bool_mask_to_additive(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Bool mask (True = attend) -> additive mask (0 / -inf) of floating ``dtype``."""
    if mask.dtype != torch.bool:
        raise TypeError(f"expected a bool mask, got {mask.dtype}")
    return torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(~mask, float("-inf"))
