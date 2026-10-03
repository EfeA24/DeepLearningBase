# Purpose: Truncates token or feature sequences to a maximum length with explicit left/right policy and special-token preservation.
"""Sequence truncation helpers for recurrent sequence pipelines.

Contract:
- A sequence is a Python list/tuple (returned as a list) or a tensor whose dim 0 is time
  (returned as a tensor).
- ``side="right"`` drops items from the end of the content (keeps the beginning);
  ``side="left"`` drops items from the beginning (keeps the most recent items).
- ``keep_start`` / ``keep_end`` items (e.g. BOS / EOS) are always preserved and are counted
  inside ``max_length``. Content between them is truncated according to ``side``.
- Sequences already within ``max_length`` are returned unchanged (lists are copied).

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from typing import Any, List, Sequence, Union

import torch

SequenceLike = Union[Sequence[Any], torch.Tensor]


def truncate_sequence(
    sequence: SequenceLike,
    max_length: int,
    side: str = "right",
    keep_start: int = 0,
    keep_end: int = 0,
) -> SequenceLike:
    """Truncate one sequence to at most ``max_length`` items.

    Example: ``truncate_sequence([BOS, a, b, c, EOS], 4, keep_start=1, keep_end=1)``
    returns ``[BOS, a, b, EOS]`` (right side) or ``[BOS, b, c, EOS]`` (left side).
    """
    if side not in ("right", "left"):
        raise ValueError(f"side must be 'right' or 'left', got {side!r}.")
    if max_length < 0:
        raise ValueError(f"max_length must be non-negative, got {max_length}.")
    if keep_start < 0 or keep_end < 0:
        raise ValueError("keep_start and keep_end must be non-negative.")
    if keep_start + keep_end > max_length:
        raise ValueError(
            f"keep_start + keep_end ({keep_start + keep_end}) exceeds max_length ({max_length})."
        )

    is_tensor = isinstance(sequence, torch.Tensor)
    length = sequence.shape[0] if is_tensor else len(sequence)
    if length <= max_length:
        return sequence if is_tensor else list(sequence)
    if keep_start + keep_end > length:
        raise ValueError("keep_start + keep_end exceeds the sequence length.")

    head = sequence[:keep_start]
    tail = sequence[length - keep_end :]
    body = sequence[keep_start : length - keep_end]
    budget = max_length - keep_start - keep_end
    body_length = body.shape[0] if is_tensor else len(body)
    body = body[:budget] if side == "right" else body[body_length - budget :]

    if is_tensor:
        return torch.cat([head, body, tail], dim=0)
    return list(head) + list(body) + list(tail)


def truncate_batch(
    sequences: Sequence[SequenceLike],
    max_length: int,
    side: str = "right",
    keep_start: int = 0,
    keep_end: int = 0,
) -> List[SequenceLike]:
    """Apply :func:`truncate_sequence` to every sequence with the same policy."""
    return [truncate_sequence(seq, max_length, side, keep_start, keep_end) for seq in sequences]
