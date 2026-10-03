# Purpose: Wraps pack_padded_sequence/pad_packed_sequence with explicit sorting and length handling for LSTM/GRU sequence models.
"""Packing helpers for running nn.LSTM / nn.GRU on right-padded, batch-first batches.

Contract:
    - Inputs are batch-first ``(B, T, F)`` and RIGHT-padded (valid prefix); left-padded batches
      must be converted first, otherwise packing silently reads padding as data.
    - ``lengths`` is ``(B,)`` int64; every length must be >= 1 (PyTorch rejects zero-length rows).
      PyTorch requires lengths on CPU, so this module moves them to CPU explicitly.
    - ``enforce_sorted=False`` (default) lets PyTorch sort/unsort internally; outputs keep the
      original batch order. ``enforce_sorted=True`` requires lengths sorted descending and raises otherwise.
    - Final states returned by an RNN on a PackedSequence are already in the ORIGINAL batch order
      and correspond to each sequence's last valid step.

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence


def _validate_lengths(lengths: torch.Tensor, batch_size: int, max_time: int) -> torch.Tensor:
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.as_tensor(lengths)
    if lengths.dim() != 1 or lengths.shape[0] != batch_size:
        raise ValueError(f"lengths must have shape ({batch_size},), got {tuple(lengths.shape)}.")
    lengths = lengths.to(device="cpu", dtype=torch.long)
    if lengths.numel() and int(lengths.min()) < 1:
        raise ValueError("All lengths must be >= 1 for packing; drop or handle empty sequences beforehand.")
    if lengths.numel() and int(lengths.max()) > max_time:
        raise ValueError(f"A length exceeds the padded time dimension ({max_time}).")
    return lengths


def pack_batch(
    padded: torch.Tensor,
    lengths: torch.Tensor,
    enforce_sorted: bool = False,
) -> PackedSequence:
    """Pack a right-padded batch-first tensor ``(B, T, ...)`` using ``lengths`` ``(B,)``."""
    if padded.dim() < 2:
        raise ValueError(f"padded must be at least (B, T), got shape {tuple(padded.shape)}.")
    cpu_lengths = _validate_lengths(lengths, padded.shape[0], padded.shape[1])
    if enforce_sorted and cpu_lengths.numel() > 1 and bool((cpu_lengths[:-1] < cpu_lengths[1:]).any()):
        raise ValueError("enforce_sorted=True requires lengths sorted in descending order.")
    return pack_padded_sequence(padded, cpu_lengths, batch_first=True, enforce_sorted=enforce_sorted)


def unpack_batch(
    packed: PackedSequence,
    padding_value: float = 0.0,
    total_length: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unpack to ``(B, T, ...)`` batch-first plus lengths ``(B,)`` (CPU int64).

    Pass ``total_length`` to restore the original padded width (needed when the longest
    sequence is shorter than the input ``T``, e.g. under DataParallel or for loss alignment).
    """
    return pad_packed_sequence(packed, batch_first=True, padding_value=padding_value, total_length=total_length)


def run_packed(
    rnn: torch.nn.Module,
    padded: torch.Tensor,
    lengths: torch.Tensor,
    initial_state=None,
    enforce_sorted: bool = False,
    padding_value: float = 0.0,
):
    """Run a batch-first nn.RNN/nn.LSTM/nn.GRU over a padded batch via packing.

    Returns ``(outputs, final_state)`` where outputs is ``(B, T, D*H)`` padded with
    ``padding_value`` (same ``T`` as the input) and ``final_state`` follows the module's state layout.
    """
    if not getattr(rnn, "batch_first", False):
        raise ValueError("run_packed expects an RNN module constructed with batch_first=True.")
    packed = pack_batch(padded, lengths, enforce_sorted=enforce_sorted)
    packed_out, final_state = rnn(packed, initial_state)
    outputs, _ = unpack_batch(packed_out, padding_value=padding_value, total_length=padded.shape[1])
    return outputs, final_state
