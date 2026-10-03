# Purpose: Computes, validates and sorts sequence lengths and converts them to validity masks for padded recurrent batches.
"""Sequence length helpers.

- Lengths are int64 tensors of shape (B,), counting valid (non-pad) steps.
- Masks use True = valid position, shape (B, T).
- ``lengths_from_ids`` counts tokens != pad_id; it assumes pad ids appear only in padding
  (if the pad id can occur inside real content, track lengths explicitly instead).
- ``sort_by_length`` returns indices to sort and to restore order (e.g. for manual packing;
  ``pack_padded_sequence(enforce_sorted=False)`` already does this internally).

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def lengths_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """(B, T) bool mask (True = valid) -> (B,) int64 lengths."""
    if mask.dtype != torch.bool or mask.dim() != 2:
        raise ValueError("mask must be a (B, T) bool tensor.")
    return mask.sum(dim=1, dtype=torch.long)


def lengths_from_ids(token_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """(B, T) ids -> (B,) int64 count of tokens different from ``pad_id``."""
    if token_ids.dim() != 2:
        raise ValueError(f"token_ids must be (B, T), got {tuple(token_ids.shape)}.")
    return (token_ids != pad_id).sum(dim=1, dtype=torch.long)


def validate_lengths(
    lengths: torch.Tensor,
    batch_size: Optional[int] = None,
    max_length: Optional[int] = None,
    allow_zero: bool = False,
) -> None:
    """Raise ValueError/TypeError if lengths are not a valid (B,) integer tensor within bounds."""
    if lengths.dim() != 1:
        raise ValueError(f"lengths must be 1-D, got shape {tuple(lengths.shape)}.")
    if lengths.dtype.is_floating_point or lengths.dtype == torch.bool:
        raise TypeError(f"lengths must be an integer tensor, got {lengths.dtype}.")
    if batch_size is not None and lengths.shape[0] != batch_size:
        raise ValueError(f"lengths has {lengths.shape[0]} entries, expected {batch_size}.")
    if lengths.numel() == 0:
        return
    minimum = 0 if allow_zero else 1
    if int(lengths.min()) < minimum:
        raise ValueError(f"lengths must be >= {minimum}.")
    if max_length is not None and int(lengths.max()) > max_length:
        raise ValueError(f"lengths must be <= {max_length}.")


def lengths_to_mask(lengths: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    """(B,) lengths -> (B, T) bool mask with True = valid, assuming right padding."""
    if lengths.dim() != 1:
        raise ValueError(f"lengths must be 1-D, got shape {tuple(lengths.shape)}.")
    if max_length is None:
        max_length = int(lengths.max().item()) if lengths.numel() else 0
    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def sort_by_length(
    lengths: torch.Tensor, descending: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(sorted_lengths, sort_indices, restore_indices)``; ``x[sort][restore] == x``."""
    sorted_lengths, sort_indices = torch.sort(lengths, descending=descending, stable=True)
    restore_indices = torch.empty_like(sort_indices)
    restore_indices[sort_indices] = torch.arange(sort_indices.numel(), device=sort_indices.device)
    return sorted_lengths, sort_indices, restore_indices
