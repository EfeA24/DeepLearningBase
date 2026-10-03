# Purpose: Builds seq2seq encoder padding masks and broadcastable (B,1,1,S) attention masks with documented boolean/additive polarity.
"""Encoder-side masks.

Polarity used throughout transformer_seq2seq_base:
    padding mask  (B, S) bool: True = real token, False = pad.
    attention mask (B, 1, 1, S) bool: True = query MAY attend to this key
        (same polarity as ``F.scaled_dot_product_attention`` boolean masks).
    additive mask: 0.0 where allowed, a large negative value where blocked.

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional

import torch


def padding_mask_from_ids(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """(B, S) ids -> (B, S) bool mask, True = real token."""
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be (B, S), got {tuple(input_ids.shape)}")
    return input_ids != pad_token_id


def padding_mask_from_lengths(
    lengths: torch.Tensor, max_length: Optional[int] = None
) -> torch.Tensor:
    """(B,) lengths -> (B, S) bool mask for right-padded batches, True = real token."""
    if lengths.dim() != 1:
        raise ValueError(f"lengths must be (B,), got {tuple(lengths.shape)}")
    if lengths.numel() and int(lengths.min()) < 0:
        raise ValueError("lengths must be non-negative")
    longest = int(lengths.max()) if lengths.numel() else 0
    max_len = longest if max_length is None else int(max_length)
    if longest > max_len:
        raise ValueError(f"max_length={max_len} is smaller than the longest length")
    positions = torch.arange(max_len, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def make_encoder_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """(B, S) padding mask (bool or 0/1, nonzero = real) -> (B, 1, 1, S) bool, True = may attend.

    Broadcasts over heads and query positions in (B, H, S, S) self-attention scores.
    """
    if attention_mask.dim() != 2:
        raise ValueError(f"attention_mask must be (B, S), got {tuple(attention_mask.shape)}")
    return attention_mask.to(torch.bool)[:, None, None, :]


def to_additive_mask(mask: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Bool mask (True = allowed) -> additive float mask (0 allowed, finfo(dtype).min blocked).

    ``finfo.min`` is used instead of -inf so fully masked rows stay finite after softmax.
    """
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be bool (True = allowed), got {mask.dtype}")
    if not dtype.is_floating_point:
        raise TypeError(f"dtype must be floating point, got {dtype}")
    additive = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    return additive.masked_fill(~mask, torch.finfo(dtype).min)
