# Purpose: Provides explicit reset/detach/carry helpers for nested recurrent states across batches in stateful LSTM/GRU training and streaming inference.
"""Recurrent state lifecycle helpers.

Contract:
    - A state is ``None``, a tensor ``(L*D, B, H)`` (GRU/RNN), or a tuple/list of such tensors (LSTM ``(h, c)``).
      The batch axis is dim 1.
    - The caller decides stream identity: resets are driven by an explicit ``reset_mask`` ``(B,)`` bool
      (True = reset that stream, e.g. a new document started). Nothing here guesses from data.
    - ``detach`` cuts the autograd graph between TBPTT chunks; values are carried unchanged.

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import torch

State = Any


def map_state(state: State, fn: Callable[[torch.Tensor], torch.Tensor]) -> State:
    """Apply ``fn`` to every tensor in a nested state, preserving structure."""
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return fn(state)
    if isinstance(state, tuple):
        return tuple(map_state(s, fn) for s in state)
    if isinstance(state, list):
        return [map_state(s, fn) for s in state]
    raise TypeError(f"Unsupported state type {type(state).__name__}.")


def detach_state(state: State) -> State:
    """Detach all tensors (TBPTT boundary)."""
    return map_state(state, lambda t: t.detach())


def reset_state(state: State, reset_mask: torch.Tensor) -> State:
    """Zero the batch rows where ``reset_mask`` ``(B,)`` is True; other rows are carried."""
    if reset_mask.dim() != 1:
        raise ValueError(f"reset_mask must be (B,), got {tuple(reset_mask.shape)}.")

    def _reset(t: torch.Tensor) -> torch.Tensor:
        if t.shape[1] != reset_mask.shape[0]:
            raise ValueError(f"State batch dim {t.shape[1]} != reset_mask length {reset_mask.shape[0]}.")
        keep = (~reset_mask.bool()).to(device=t.device, dtype=t.dtype).view(1, -1, *([1] * (t.dim() - 2)))
        return t * keep

    return map_state(state, _reset)


class RecurrentStateManager:
    """Holds one carried state with explicit caller-driven transitions.

    Typical TBPTT loop::

        manager = RecurrentStateManager()
        for chunk in chunks:
            outputs, new_state = model(chunk, manager.state)
            loss.backward(); optimizer.step()
            manager.update(new_state, detach=True)
    """

    def __init__(self, initial_state: State = None) -> None:
        self.state = initial_state

    def update(self, new_state: State, detach: bool = True) -> None:
        """Store the model's returned state (detached by default so the next chunk starts a new graph)."""
        self.state = detach_state(new_state) if detach else new_state

    def detach(self) -> None:
        self.state = detach_state(self.state)

    def reset(self) -> None:
        """Forget all state (model will use its own zero initial state)."""
        self.state = None

    def reset_where(self, reset_mask: torch.Tensor) -> None:
        """Zero specific streams (True in ``reset_mask``); no-op when no state is held."""
        if self.state is not None:
            self.state = reset_state(self.state, reset_mask)

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        self.state = map_state(self.state, lambda t: t.to(device=device, dtype=dtype))
