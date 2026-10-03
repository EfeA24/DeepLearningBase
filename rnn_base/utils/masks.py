# Purpose: Builds boolean and additive sequence masks with documented shape and polarity for padded recurrent-model batches.
"""Sequence masks.

Polarity convention in this template: boolean masks use True = valid (real token),
False = padding, unless a function name says otherwise (``padding_mask_*`` returns True = pad).
Shapes: (B, T) for per-position masks; ``causal_mask`` is (T, T) with True = allowed
(query i may attend to key j <= i), useful if attention is added on top of an RNN.
Additive masks contain 0.0 at valid positions and the dtype's most negative finite value at
padding (finite instead of -inf so fully masked rows do not produce NaN after softmax).

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from typing import Optional, Union

import torch


def lengths_to_mask(
    lengths: torch.Tensor, max_length: Optional[int] = None, padding_side: str = "right"
) -> torch.Tensor:
    """(B,) lengths -> (B, T) bool mask, True = valid. T = max(lengths) if ``max_length`` is None."""
    if lengths.dim() != 1:
        raise ValueError(f"lengths must be 1-D, got shape {tuple(lengths.shape)}.")
    if padding_side not in ("right", "left"):
        raise ValueError(f"padding_side must be 'right' or 'left', got {padding_side!r}.")
    steps = int(lengths.max().item()) if max_length is None and lengths.numel() else (max_length or 0)
    positions = torch.arange(steps, device=lengths.device).unsqueeze(0)
    lengths = lengths.unsqueeze(1)
    if padding_side == "right":
        return positions < lengths
    return positions >= steps - lengths


def valid_mask_from_ids(token_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """(B, T) ids -> bool mask, True = token != pad_id."""
    return token_ids != pad_id


def padding_mask_from_ids(token_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """(B, T) ids -> bool mask, True = padding (inverse polarity, e.g. for key_padding_mask)."""
    return token_ids == pad_id


def invert_mask(mask: torch.Tensor) -> torch.Tensor:
    """Flip polarity of a bool mask (valid <-> padding)."""
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be bool, got {mask.dtype}.")
    return ~mask


def to_additive_mask(valid_mask: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Bool mask (True = valid) -> float mask: 0.0 valid, finfo(dtype).min at padding."""
    if valid_mask.dtype != torch.bool:
        raise TypeError(f"valid_mask must be bool, got {valid_mask.dtype}.")
    additive = torch.zeros(valid_mask.shape, dtype=dtype, device=valid_mask.device)
    return additive.masked_fill(~valid_mask, torch.finfo(dtype).min)


def causal_mask(steps: int, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    """(T, T) bool lower-triangular mask, True = allowed."""
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}.")
    return torch.ones(steps, steps, dtype=torch.bool, device=device).tril()


def expand_mask_like(mask: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    """(B, T) mask -> (B, T, 1, ...) broadcastable against ``tensor`` (B, T, *F)."""
    if tensor.shape[: mask.dim()] != mask.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} does not prefix tensor shape {tuple(tensor.shape)}.")
    return mask.view(*mask.shape, *([1] * (tensor.dim() - mask.dim())))
