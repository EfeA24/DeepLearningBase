# Purpose: Validates, reshapes, direction-merges and batch-selects recurrent hidden states without imposing any training policy.
"""Hidden-state shape helpers.

Layout (PyTorch): h_n has shape (num_layers * num_directions, B, H), layer-major:
[layer0_fwd, layer0_bwd, layer1_fwd, ...].
- ``split_layers_directions``: -> (num_layers, num_directions, B, H)
- ``combine_directions``: -> (num_layers, B, 2H) for "concat", (num_layers, B, H) for "sum"/"mean"
- ``last_layer``: -> (num_directions, B, H) of the top layer
- ``select_batch``: index_select along the batch dim (dim 1) for tensors or nested tuples/lists
  (e.g. dropping finished streams or reordering beams).

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch


def validate_hidden(
    hidden: torch.Tensor,
    num_layers: int,
    num_directions: int,
    batch_size: Optional[int] = None,
    hidden_size: Optional[int] = None,
) -> None:
    """Raise ValueError if ``hidden`` is not (num_layers * num_directions, B, H) with the given sizes."""
    if hidden.dim() != 3:
        raise ValueError(f"hidden must be 3-D (layers*dirs, B, H), got shape {tuple(hidden.shape)}.")
    if hidden.shape[0] != num_layers * num_directions:
        raise ValueError(f"hidden dim 0 is {hidden.shape[0]}, expected {num_layers * num_directions}.")
    if batch_size is not None and hidden.shape[1] != batch_size:
        raise ValueError(f"hidden batch size is {hidden.shape[1]}, expected {batch_size}.")
    if hidden_size is not None and hidden.shape[2] != hidden_size:
        raise ValueError(f"hidden size is {hidden.shape[2]}, expected {hidden_size}.")


def split_layers_directions(hidden: torch.Tensor, num_layers: int, num_directions: int) -> torch.Tensor:
    """(num_layers * num_directions, B, H) -> (num_layers, num_directions, B, H) (a view when contiguous)."""
    validate_hidden(hidden, num_layers, num_directions)
    return hidden.reshape(num_layers, num_directions, hidden.shape[1], hidden.shape[2])


def combine_directions(hidden: torch.Tensor, num_layers: int, mode: str = "concat") -> torch.Tensor:
    """Merge forward/backward states of a bidirectional h_n per layer."""
    if mode not in ("concat", "sum", "mean"):
        raise ValueError(f"mode must be 'concat', 'sum' or 'mean', got {mode!r}.")
    per_direction = split_layers_directions(hidden, num_layers, 2)
    forward, backward = per_direction[:, 0], per_direction[:, 1]
    if mode == "concat":
        return torch.cat([forward, backward], dim=-1)
    if mode == "sum":
        return forward + backward
    return (forward + backward) / 2.0


def last_layer(hidden: torch.Tensor, num_layers: int, num_directions: int = 1) -> torch.Tensor:
    """Top-layer states (num_directions, B, H)."""
    return split_layers_directions(hidden, num_layers, num_directions)[-1]


def select_batch(state: Any, indices: Union[torch.Tensor, Sequence[int]], batch_dim: int = 1) -> Any:
    """Select/reorder batch entries of a tensor or nested tuple/list state."""
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        index = torch.as_tensor(indices, dtype=torch.long, device=state.device)
        return state.index_select(batch_dim, index)
    if isinstance(state, (tuple, list)):
        return type(state)(select_batch(item, indices, batch_dim) for item in state)
    raise TypeError(f"Unsupported hidden-state element type: {type(state).__name__}.")
