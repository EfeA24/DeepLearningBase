# Purpose: Splits causal-LM token streams into fixed-length chunks with an explicit remainder policy (drop/pad/keep).
"""Fixed-length, non-overlapping chunking of a token stream.

    chunk_tokens([0..9], chunk_size=4, remainder="drop") -> [[0,1,2,3], [4,5,6,7]]
    remainder="pad"  -> last chunk [8, 9, pad, pad]  (requires pad_id)
    remainder="keep" -> last chunk [8, 9]            (variable length; collator must pad)

``chunk_tensor`` returns a (N, chunk_size) LongTensor plus a (N, chunk_size) bool valid mask
(True = real token) for the "drop"/"pad" policies. Build labels from the mask
(pad -> -100) when training; see data/sequences/causal_shift.py.

Note: non-overlapping chunks lose the first-token prediction of every chunk under the
aligned-label shift (position 0 has no context); use sliding_window.py for evaluation that
scores every token with context.

Requires: torch.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

_POLICIES = ("drop", "pad", "keep")


def chunk_tokens(
    token_ids: Sequence[int],
    chunk_size: int,
    remainder: str = "drop",
    pad_id: Optional[int] = None,
) -> List[List[int]]:
    """Split ``token_ids`` into consecutive chunks of ``chunk_size`` tokens."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if remainder not in _POLICIES:
        raise ValueError(f"remainder must be one of {_POLICIES}, got {remainder!r}")
    if remainder == "pad" and pad_id is None:
        raise ValueError("remainder='pad' requires pad_id")
    ids = list(token_ids)
    chunks = [ids[i : i + chunk_size] for i in range(0, len(ids), chunk_size)]
    if chunks and len(chunks[-1]) < chunk_size:
        if remainder == "drop":
            chunks.pop()
        elif remainder == "pad":
            chunks[-1] = chunks[-1] + [pad_id] * (chunk_size - len(chunks[-1]))
    return chunks


def chunk_tensor(
    token_ids: torch.Tensor,
    chunk_size: int,
    remainder: str = "drop",
    pad_id: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunk a 1D tensor into (N, chunk_size) plus a bool valid mask; supports "drop" and "pad"."""
    if token_ids.dim() != 1:
        raise ValueError(f"token_ids must be 1D, got {tuple(token_ids.shape)}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if remainder not in ("drop", "pad"):
        raise ValueError("chunk_tensor supports remainder='drop' or 'pad' (use chunk_tokens for 'keep')")
    n = token_ids.numel()
    if remainder == "drop":
        usable = (n // chunk_size) * chunk_size
        chunks = token_ids[:usable].view(-1, chunk_size)
        return chunks, torch.ones_like(chunks, dtype=torch.bool)
    if pad_id is None:
        raise ValueError("remainder='pad' requires pad_id")
    n_chunks = (n + chunk_size - 1) // chunk_size
    total = n_chunks * chunk_size
    padded = torch.full((total,), pad_id, dtype=token_ids.dtype, device=token_ids.device)
    padded[:n] = token_ids
    valid = torch.zeros(total, dtype=torch.bool, device=token_ids.device)
    valid[:n] = True
    return padded.view(n_chunks, chunk_size), valid.view(n_chunks, chunk_size)
