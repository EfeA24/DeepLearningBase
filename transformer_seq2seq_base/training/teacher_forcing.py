# Purpose: Builds training-time seq2seq decoder inputs, decoder masks and -100 labels from padded target tensors for teacher forcing.
"""Batched teacher-forcing helper (tensor level).

Input: target_ids (B, T) int64 right-padded with ``pad_token_id``, ending with EOS, without
the decoder start token; optional target_attention_mask (B, T) 1 = real token (required
when pad shares an id with EOS).
Output dict:
    labels                 (B, T) target ids with padding = ``ignore_index`` (-100)
    decoder_input_ids      (B, T) shift_right(labels): [start, y1, ...], -100 -> pad
    decoder_attention_mask (B, T) int64, 1 = real position, 0 = pad

Scheduled sampling (feeding model predictions instead of gold tokens) is intentionally NOT
implemented; if needed, replace selected ``decoder_input_ids`` positions with argmax
predictions from a no-grad forward pass under an explicit schedule chosen by the caller.

Requires: torch only (self-contained shift; same semantics as data/sequences/target_shift.py).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch


def build_teacher_forcing_batch(
    target_ids: torch.Tensor,
    *,
    pad_token_id: int,
    decoder_start_token_id: int,
    ignore_index: int = -100,
    target_attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Create labels, decoder inputs and decoder mask from padded targets."""
    if target_ids.dim() != 2 or target_ids.size(1) < 1:
        raise ValueError(f"target_ids must be (B, T) with T >= 1, got {tuple(target_ids.shape)}")
    if target_attention_mask is None:
        valid = target_ids != pad_token_id
    else:
        if target_attention_mask.shape != target_ids.shape:
            raise ValueError("target_attention_mask must match target_ids shape")
        valid = target_attention_mask.to(torch.bool)
    labels = target_ids.masked_fill(~valid, ignore_index)
    decoder_input_ids = labels_to_decoder_inputs(
        labels, pad_token_id=pad_token_id, decoder_start_token_id=decoder_start_token_id, ignore_index=ignore_index
    )
    return {
        "labels": labels,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": valid.long(),
    }


def labels_to_decoder_inputs(
    labels: torch.Tensor,
    *,
    pad_token_id: int,
    decoder_start_token_id: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """shift_right: (B, T) labels -> (B, T) decoder inputs with -100 replaced by pad."""
    if labels.dim() != 2 or labels.size(1) < 1:
        raise ValueError(f"labels must be (B, T) with T >= 1, got {tuple(labels.shape)}")
    shifted = labels.new_empty(labels.shape)
    shifted[:, 1:] = labels[:, :-1]
    shifted[:, 0] = decoder_start_token_id
    return shifted.masked_fill(shifted == ignore_index, pad_token_id)
