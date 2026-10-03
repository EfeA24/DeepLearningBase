# Purpose: Builds valid/pad token masks from lengths or pad ids for bidirectional encoder (BERT-style) batches.
"""Padding-mask helpers for batch-first token batches.

Mask polarity (used throughout transformer_encoder_base):
    valid mask, dtype torch.bool, shape (B, T): True = real token (attend), False = padding.
    ``invert_mask`` converts it to the "True = pad" polarity expected by e.g.
    ``torch.nn.MultiheadAttention(key_padding_mask=...)``.

Hugging Face style ``attention_mask`` tensors (long, 1 = valid, 0 = pad) have the same
polarity; ``as_bool_mask`` converts them.

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "lengths_to_mask",
    "mask_from_pad_id",
    "mask_to_lengths",
    "invert_mask",
    "as_bool_mask",
]


def lengths_to_mask(
    lengths: torch.Tensor,
    max_len: Optional[int] = None,
    padding_side: str = "right",
) -> torch.Tensor:
    """Convert per-sequence lengths (B,) into a valid mask (B, T) with True = valid token.

    Args:
        lengths: integer tensor (B,) with values in [0, max_len].
        max_len: T; defaults to ``lengths.max()``.
        padding_side: "right" (tokens first, pads after) or "left" (pads first).
    """
    if lengths.dim() != 1:
        raise ValueError(f"lengths must be 1-D (B,), got shape {tuple(lengths.shape)}")
    if lengths.dtype.is_floating_point or lengths.dtype == torch.bool:
        raise TypeError(f"lengths must be an integer tensor, got {lengths.dtype}")
    if padding_side not in ("right", "left"):
        raise ValueError(f"padding_side must be 'right' or 'left', got {padding_side!r}")
    if lengths.numel() > 0 and int(lengths.min()) < 0:
        raise ValueError("lengths must be non-negative")
    longest = int(lengths.max()) if lengths.numel() > 0 else 0
    if max_len is None:
        max_len = longest
    elif max_len < longest:
        raise ValueError(f"max_len={max_len} is smaller than the longest length {longest}")
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    if padding_side == "right":
        return positions < lengths.unsqueeze(1)
    return positions >= (max_len - lengths).unsqueeze(1)


def mask_from_pad_id(input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Return a valid mask (same shape as ``input_ids``) with True where ``input_ids != pad_id``.

    Only correct when ``pad_id`` never occurs as a real token inside sequences.
    """
    if input_ids.dtype.is_floating_point:
        raise TypeError(f"input_ids must be an integer tensor, got {input_ids.dtype}")
    return input_ids != pad_id


def mask_to_lengths(mask: torch.Tensor) -> torch.Tensor:
    """Count valid positions per row of a (B, T) mask (True / non-zero = valid). Returns long (B,)."""
    if mask.dim() != 2:
        raise ValueError(f"mask must be (B, T), got shape {tuple(mask.shape)}")
    return as_bool_mask(mask).sum(dim=1, dtype=torch.long)


def invert_mask(mask: torch.Tensor) -> torch.Tensor:
    """Flip polarity (True = valid  <->  True = pad)."""
    return ~as_bool_mask(mask)


def as_bool_mask(mask: torch.Tensor) -> torch.Tensor:
    """Convert a bool or integer 0/1 mask to bool (non-zero = True). Floating masks are rejected

    because a float mask is ambiguous (0/1 indicator vs additive 0/-inf bias).
    """
    if mask.dtype == torch.bool:
        return mask
    if mask.dtype.is_floating_point:
        raise TypeError(
            "Floating masks are ambiguous here; pass a bool or integer 0/1 mask "
            "(use data/masking/attention_mask.py to convert additive masks)."
        )
    return mask != 0
