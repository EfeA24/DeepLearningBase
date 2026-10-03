# Purpose: Pads batches of variable-length token or feature sequences (left or right) for recurrent sequence models.
"""Batch padding utilities for variable-length sequences (standalone, especially reusable).

Contract:
- Input: a non-empty sequence of examples. Each example is a 1-D Python list/tuple of numbers
  or a tensor of shape (T_i, *feature_dims); time is always dim 0 of an example.
- Output tensor is batch-first: (B, T_pad, *feature_dims), where T_pad is the longest T_i
  or ``max_length`` when given.
- ``lengths``: int64 tensor (B,) of original lengths.
- ``mask``: bool tensor (B, T_pad) with True = valid (real) position, False = padding.
- ``padding_side="right"`` puts real items at positions [0, T_i); ``"left"`` puts them at
  [T_pad - T_i, T_pad) (useful when prompt ends must be aligned for generation).
- Sequences longer than ``max_length`` raise ValueError; truncate first (see truncation.py).
- For LM targets pad with ``padding_value=-100`` (the default ``ignore_index`` of cross entropy).

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch

SequenceLike = Union[Sequence[int], Sequence[float], torch.Tensor]


@dataclass
class PaddedBatch:
    """Padded batch: ``padded`` (B, T_pad, *F), ``lengths`` (B,) int64, ``mask`` (B, T_pad) bool True=valid."""

    padded: torch.Tensor
    lengths: torch.Tensor
    mask: torch.Tensor


def _to_tensor(seq: SequenceLike, dtype: Optional[torch.dtype]) -> torch.Tensor:
    if isinstance(seq, torch.Tensor):
        tensor = seq if dtype is None else seq.to(dtype)
    else:
        tensor = torch.as_tensor(list(seq), dtype=dtype)
    if tensor.dim() == 0:
        raise ValueError("Each sequence must have a time dimension (dim 0); got a scalar.")
    return tensor


def pad_batch(
    sequences: Sequence[SequenceLike],
    padding_value: Union[int, float] = 0,
    padding_side: str = "right",
    max_length: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> PaddedBatch:
    """Pad ``sequences`` into a batch-first tensor and return it with lengths and a validity mask.

    Args:
        sequences: Examples of shape (T_i, *F); feature dims must match across examples.
        padding_value: Fill value for padded positions (e.g. pad_id for inputs, -100 for LM targets).
        padding_side: "right" or "left".
        max_length: Optional fixed padded length; must be >= every T_i.
        dtype: Optional output dtype. Defaults to the dtype of the first non-empty example
            (int64 if every example is empty).
    """
    if len(sequences) == 0:
        raise ValueError("pad_batch requires at least one sequence.")
    if padding_side not in ("right", "left"):
        raise ValueError(f"padding_side must be 'right' or 'left', got {padding_side!r}.")
    if max_length is not None and max_length < 0:
        raise ValueError(f"max_length must be non-negative, got {max_length}.")

    tensors = [_to_tensor(seq, dtype) for seq in sequences]
    if dtype is None:
        dtype = next((t.dtype for t in tensors if t.numel() > 0), torch.long)
    device = tensors[0].device
    tensors = [t.to(device=device, dtype=dtype) for t in tensors]

    feature_shape = tensors[0].shape[1:]
    for index, tensor in enumerate(tensors):
        if tensor.shape[1:] != feature_shape:
            raise ValueError(
                f"Sequence {index} has feature shape {tuple(tensor.shape[1:])}, "
                f"expected {tuple(feature_shape)}."
            )

    lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long, device=device)
    longest = int(lengths.max().item())
    if max_length is not None and longest > max_length:
        raise ValueError(
            f"Longest sequence has length {longest} > max_length={max_length}; truncate before padding."
        )
    padded_length = longest if max_length is None else max_length

    batch_size = len(tensors)
    padded = torch.full(
        (batch_size, padded_length, *feature_shape), padding_value, dtype=dtype, device=device
    )
    mask = torch.zeros((batch_size, padded_length), dtype=torch.bool, device=device)
    for index, tensor in enumerate(tensors):
        length = tensor.shape[0]
        if length == 0:
            continue
        if padding_side == "right":
            padded[index, :length] = tensor
            mask[index, :length] = True
        else:
            padded[index, padded_length - length :] = tensor
            mask[index, padded_length - length :] = True
    return PaddedBatch(padded=padded, lengths=lengths, mask=mask)


def pad_sequences(
    sequences: Sequence[SequenceLike],
    padding_value: Union[int, float] = 0,
    padding_side: str = "right",
    max_length: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Return only the padded (B, T_pad, *F) tensor; see :func:`pad_batch` for arguments."""
    return pad_batch(sequences, padding_value, padding_side, max_length, dtype).padded


def unpad_sequences(
    padded: torch.Tensor, lengths: torch.Tensor, padding_side: str = "right"
) -> List[torch.Tensor]:
    """Invert padding: split (B, T_pad, *F) into a list of (T_i, *F) views using ``lengths`` (B,)."""
    if padding_side not in ("right", "left"):
        raise ValueError(f"padding_side must be 'right' or 'left', got {padding_side!r}.")
    if padded.dim() < 2:
        raise ValueError(f"padded must have shape (B, T, ...), got {tuple(padded.shape)}.")
    if lengths.dim() != 1 or lengths.shape[0] != padded.shape[0]:
        raise ValueError("lengths must have shape (B,) matching padded.shape[0].")
    padded_length = padded.shape[1]
    result = []
    for index, length in enumerate(lengths.tolist()):
        if not 0 <= length <= padded_length:
            raise ValueError(f"length {length} at index {index} is outside [0, {padded_length}].")
        if padding_side == "right":
            result.append(padded[index, :length])
        else:
            result.append(padded[index, padded_length - length :])
    return result
