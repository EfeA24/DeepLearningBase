# Purpose: Provides shift_right/shift_left helpers that build seq2seq decoder inputs and shifted targets with explicit pad/start-token semantics.
"""Target shifting helpers for encoder-decoder (seq2seq) training.

Contract (batch-first, right padding, int64 ids):
    labels:            (B, T); padded positions hold ``ignore_index`` (default -100).
                       Labels normally END with EOS and do NOT start with the
                       decoder start token.
    decoder_input_ids: (B, T) = shift_right(labels): position 0 holds
                       ``decoder_start_token_id``; position t holds labels[:, t-1];
                       every ``ignore_index`` is replaced by ``pad_token_id`` so the
                       embedding layer never receives -100.

Example (start=0, pad=0, eos=2)::

    labels            = [[15, 16, 2, -100]]
    decoder_input_ids = [[ 0, 15, 16,    2]]   # position 3 is a pad-label query (ignored by loss)

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional

import torch


def _check_2d_int(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dim() != 2:
        raise ValueError(f"{name} must have shape (B, T), got {tuple(tensor.shape)}")
    if tensor.dtype.is_floating_point or tensor.dtype == torch.bool:
        raise TypeError(f"{name} must be an integer tensor, got {tensor.dtype}")
    if tensor.size(1) < 1:
        raise ValueError(f"{name} must have at least one time step")


def shift_right(
    labels: torch.Tensor,
    decoder_start_token_id: int,
    pad_token_id: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Build teacher-forcing decoder inputs from labels.

    Args:
        labels: (B, T) int target ids, padding = ``ignore_index`` (or already ``pad_token_id``).
        decoder_start_token_id: id placed at position 0 (BOS, or pad-as-start for T5/Marian style).
        pad_token_id: id written wherever the shifted labels contain ``ignore_index``.
        ignore_index: label padding value to replace.

    Returns:
        (B, T) int64 decoder input ids; the last label column is dropped.
    """
    _check_2d_int(labels, "labels")
    shifted = labels.new_empty(labels.shape)
    shifted[:, 1:] = labels[:, :-1]
    shifted[:, 0] = decoder_start_token_id
    return shifted.masked_fill(shifted == ignore_index, pad_token_id)


def shift_left(sequence_ids: torch.Tensor, fill_value: int = -100) -> torch.Tensor:
    """Shift ids one step to the left and append ``fill_value`` at the end.

    Useful to derive next-token labels from a decoder sequence that already starts
    with the decoder start token: ``labels = shift_left(decoder_input_ids, -100)``.
    The final column has no successor, so it receives ``fill_value`` (ignored by default).

    Args:
        sequence_ids: (B, T) int ids.
        fill_value: value written at the last position.

    Returns:
        (B, T) int ids.
    """
    _check_2d_int(sequence_ids, "sequence_ids")
    shifted = sequence_ids.new_empty(sequence_ids.shape)
    shifted[:, :-1] = sequence_ids[:, 1:]
    shifted[:, -1] = fill_value
    return shifted


def labels_from_padded_targets(
    target_ids: torch.Tensor,
    pad_token_id: int,
    ignore_index: int = -100,
    target_attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Replace target padding with ``ignore_index``.

    If ``target_attention_mask`` (B, T; 1/True = real token, 0/False = pad) is given it
    decides which positions are padding. Otherwise every ``pad_token_id`` is treated as
    padding, which is WRONG when pad shares an id with EOS; pass the mask in that case.
    """
    _check_2d_int(target_ids, "target_ids")
    if target_attention_mask is not None:
        if target_attention_mask.shape != target_ids.shape:
            raise ValueError(
                f"target_attention_mask shape {tuple(target_attention_mask.shape)} "
                f"!= target_ids shape {tuple(target_ids.shape)}"
            )
        is_pad = ~target_attention_mask.to(torch.bool)
    else:
        is_pad = target_ids == pad_token_id
    return target_ids.masked_fill(is_pad, ignore_index)
