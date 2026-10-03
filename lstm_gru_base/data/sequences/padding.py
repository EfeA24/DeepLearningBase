# Purpose: Pads variable-length token or feature sequences into a batch-first tensor with lengths for LSTM/GRU sequence projects.
"""Padding helpers for gated-RNN batches.

Contract:
    - Input: a list of 1-D tensors/lists ``(T_i,)`` (token ids) or 2-D tensors ``(T_i, F)`` (features).
    - Output: padded tensor ``(B, T_max)`` or ``(B, T_max, F)``, batch-first, plus lengths ``(B,)`` int64.
    - ``side="right"`` puts real values at the start (the layout expected by ``pack_padded_sequence``);
      ``side="left"`` puts them at the end (useful for generation). Mask polarity: True = valid position.

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch

SequenceLike = Union[torch.Tensor, Sequence[int], Sequence[float]]


def _to_tensor(seq: SequenceLike, dtype: Optional[torch.dtype]) -> torch.Tensor:
    tensor = seq if isinstance(seq, torch.Tensor) else torch.as_tensor(seq)
    if dtype is not None:
        tensor = tensor.to(dtype)
    if tensor.dim() not in (1, 2):
        raise ValueError(f"Each sequence must be 1-D (T,) or 2-D (T, F); got shape {tuple(tensor.shape)}.")
    return tensor


def pad_sequences(
    sequences: Sequence[SequenceLike],
    padding_value: float = 0,
    side: str = "right",
    max_length: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad sequences to a common length.

    Args:
        sequences: Non-empty list of sequences, each ``(T_i,)`` or ``(T_i, F)`` with equal ``F``.
        padding_value: Fill value for padded positions (e.g. the PAD id, or -100 for labels).
        side: ``"right"`` or ``"left"`` padding.
        max_length: Target length; defaults to the longest sequence. Sequences longer than this
            raise ``ValueError`` (truncate explicitly beforehand).
        dtype: Optional dtype cast applied to every sequence.

    Returns:
        ``(padded, lengths)`` where ``padded`` is ``(B, T)`` / ``(B, T, F)`` and ``lengths`` is ``(B,)`` int64.
    """
    if side not in ("right", "left"):
        raise ValueError(f"side must be 'right' or 'left', got {side!r}.")
    if len(sequences) == 0:
        raise ValueError("pad_sequences requires at least one sequence.")
    tensors = [_to_tensor(s, dtype) for s in sequences]
    trailing = tensors[0].shape[1:]
    for t in tensors:
        if t.shape[1:] != trailing:
            raise ValueError(f"All sequences need the same trailing shape; got {tuple(trailing)} and {tuple(t.shape[1:])}.")
    lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long)
    target_len = int(lengths.max().item()) if max_length is None else int(max_length)
    if target_len < int(lengths.max().item()):
        raise ValueError(f"max_length={target_len} is shorter than the longest sequence ({int(lengths.max())}).")

    padded = tensors[0].new_full((len(tensors), target_len, *trailing), padding_value)
    for i, t in enumerate(tensors):
        n = t.shape[0]
        if n == 0:
            continue
        if side == "right":
            padded[i, :n] = t
        else:
            padded[i, target_len - n:] = t
    return padded, lengths


def pad_to_length(sequence: torch.Tensor, length: int, padding_value: float = 0, side: str = "right") -> torch.Tensor:
    """Pad a single ``(T,)`` or ``(T, F)`` tensor along time to exactly ``length`` steps."""
    padded, _ = pad_sequences([sequence], padding_value=padding_value, side=side, max_length=length)
    return padded[0]
