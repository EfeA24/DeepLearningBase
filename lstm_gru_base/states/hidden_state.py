# Purpose: Validates and manipulates recurrent hidden-state tensors in PyTorch's (layers*directions, B, H) layout for LSTM/GRU models.
"""Hidden-state helpers.

Layout (PyTorch nn.RNN/nn.LSTM/nn.GRU, independent of ``batch_first``):
    ``h`` is ``(num_layers * num_directions, B, H)`` ordered
    ``[layer0_fwd, layer0_bwd, layer1_fwd, layer1_bwd, ...]``. The batch axis is ALWAYS dim 1.

Ops: zeros, validate, detach, select/reorder batch rows (e.g. beam search or dropping finished streams),
split into ``(L, D, B, H)``, and extract the top layer as ``(B, D*H)`` features.

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch


def zeros_hidden(
    num_layers: int,
    batch_size: int,
    hidden_size: int,
    num_directions: int = 1,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Zero hidden state ``(L*D, B, H)`` on the caller-specified device/dtype."""
    if num_directions not in (1, 2):
        raise ValueError("num_directions must be 1 or 2.")
    return torch.zeros(num_layers * num_directions, batch_size, hidden_size, device=device, dtype=dtype)


def validate_hidden(
    h: torch.Tensor,
    num_layers: int,
    batch_size: int,
    hidden_size: int,
    num_directions: int = 1,
) -> None:
    """Raise ``ValueError`` if ``h`` does not have shape ``(L*D, B, H)``."""
    expected = (num_layers * num_directions, batch_size, hidden_size)
    if not isinstance(h, torch.Tensor):
        raise TypeError(f"Hidden state must be a tensor, got {type(h).__name__}.")
    if tuple(h.shape) != expected:
        raise ValueError(f"Hidden state shape {tuple(h.shape)} != expected (L*D, B, H) = {expected}.")


def detach_hidden(h: torch.Tensor) -> torch.Tensor:
    """Cut the autograd graph (TBPTT boundary) while keeping values."""
    return h.detach()


def select_batch(h: torch.Tensor, indices: Union[torch.Tensor, Sequence[int]]) -> torch.Tensor:
    """Select/reorder batch rows along dim 1 (e.g. beam reordering, keeping unfinished sequences)."""
    index = torch.as_tensor(indices, dtype=torch.long, device=h.device)
    if index.dim() != 1:
        raise ValueError("indices must be 1-D.")
    return h.index_select(1, index)


def split_directions(h: torch.Tensor, num_layers: int, num_directions: int) -> torch.Tensor:
    """View ``(L*D, B, H)`` as ``(L, D, B, H)``."""
    if h.shape[0] != num_layers * num_directions:
        raise ValueError(f"First dim {h.shape[0]} != num_layers*num_directions = {num_layers * num_directions}.")
    return h.view(num_layers, num_directions, *h.shape[1:])


def top_layer_features(h: torch.Tensor, num_layers: int, num_directions: int = 1) -> torch.Tensor:
    """Return the last layer's state as ``(B, D*H)`` (directions concatenated forward first)."""
    top = split_directions(h, num_layers, num_directions)[-1]
    return torch.cat(list(top), dim=-1)
