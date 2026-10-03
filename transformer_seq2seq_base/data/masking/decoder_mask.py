# Purpose: Combines seq2seq decoder causal and target padding masks into (B,1,T,K) boolean masks (True = may attend) with KV-cache offset support.
"""Decoder self-attention masks.

Polarity: True = query position may attend to key position (SDPA boolean convention).
    causal mask           (T, past + T): query i (absolute position past + i) sees keys <= past + i.
    decoder padding mask  (B, past + T): True = real target token, False = pad.
    combined mask         (B, 1, T, past + T) = causal AND key-is-real.

With right padding a pad query still sees earlier real keys, so no row is fully masked.
With left padding rows can be fully masked; attention code in this template makes such rows
finite, and their outputs must be ignored via labels = -100.

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional, Union

import torch


def make_causal_mask(
    target_length: int,
    *,
    past_length: int = 0,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """(T, past_length + T) bool lower-triangular mask, True = allowed."""
    if target_length < 1:
        raise ValueError(f"target_length must be >= 1, got {target_length}")
    if past_length < 0:
        raise ValueError(f"past_length must be >= 0, got {past_length}")
    total = past_length + target_length
    ones = torch.ones((target_length, total), dtype=torch.bool, device=device)
    return torch.tril(ones, diagonal=past_length)


def make_decoder_attention_mask(
    target_length: int,
    decoder_attention_mask: Optional[torch.Tensor] = None,
    *,
    past_length: int = 0,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Combine causal and padding masks.

    Args:
        target_length: number of NEW query positions T.
        decoder_attention_mask: optional (B, past_length + T) padding mask (nonzero = real).
        past_length: number of cached key positions.
        device: device used when no padding mask is given.

    Returns:
        (B, 1, T, past+T) bool if a padding mask is given, else (1, 1, T, past+T).
    """
    if decoder_attention_mask is not None:
        device = decoder_attention_mask.device
    causal = make_causal_mask(target_length, past_length=past_length, device=device)[None, None]
    if decoder_attention_mask is None:
        return causal
    expected = past_length + target_length
    if decoder_attention_mask.dim() != 2 or decoder_attention_mask.size(1) != expected:
        raise ValueError(
            f"decoder_attention_mask must be (B, {expected}), got {tuple(decoder_attention_mask.shape)}"
        )
    key_valid = decoder_attention_mask.to(torch.bool)[:, None, None, :]
    return causal & key_valid
