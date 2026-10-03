# Purpose: Manages recurrent hidden-state lifecycle across batches/streams with caller-controlled reset, masked reset and detach.
"""Hidden-state lifecycle manager (standalone, especially reusable).

The manager stores one recurrent state: None, a tensor, or a nested tuple/list of tensors
(e.g. LSTM ``(h, c)``). Every tensor has the batch on ``batch_dim`` (default 1, PyTorch's
(layers * dirs, B, H) layout). It never guesses stream identity: the caller decides when to

- ``reset()``                 forget everything (next ``get`` returns None or a fresh init);
- ``reset_where(mask)``       zero the state of streams where ``mask`` (B,) bool is True
                              (True = reset, e.g. a document ended in that batch row);
- ``detach()``                cut autograd history (between TBPTT chunks / batches);
- ``select(indices)``         keep/reorder batch rows (e.g. streams dropped from the batch).

Typical loop::

    manager = HiddenStateManager()
    for inputs, targets, doc_start in loader:          # doc_start: (B,) bool
        manager.reset_where(doc_start)
        output, new_state = model(inputs, manager.get())
        loss = loss_fn(output, targets); loss.backward(); optimizer.step()
        manager.set(new_state)
        manager.detach()

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Union

import torch


def _map_state(state: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return fn(state)
    if isinstance(state, (tuple, list)):
        return type(state)(_map_state(item, fn) for item in state)
    raise TypeError(f"Unsupported hidden-state element type: {type(state).__name__}.")


def _first_tensor(state: Any) -> Optional[torch.Tensor]:
    if isinstance(state, torch.Tensor):
        return state
    if isinstance(state, (tuple, list)):
        for item in state:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


class HiddenStateManager:
    """Holds a recurrent state and applies explicit reset/detach/select operations."""

    def __init__(self, batch_dim: int = 1) -> None:
        self.batch_dim = batch_dim
        self._state: Any = None

    @property
    def has_state(self) -> bool:
        return self._state is not None

    @property
    def batch_size(self) -> Optional[int]:
        tensor = _first_tensor(self._state)
        return None if tensor is None else tensor.shape[self.batch_dim]

    def get(self, init_fn: Optional[Callable[[], Any]] = None) -> Any:
        """Current state; if empty and ``init_fn`` is given, store and return ``init_fn()``."""
        if self._state is None and init_fn is not None:
            self.set(init_fn())
        return self._state

    def set(self, state: Any) -> None:
        """Store ``state`` (validated to be None or a tensor tree)."""
        _map_state(state, lambda t: t)
        self._state = state

    def reset(self) -> None:
        self._state = None

    def detach(self) -> Any:
        """Detach the stored state in place of the old one and return it."""
        self._state = _map_state(self._state, lambda t: t.detach())
        return self._state

    def reset_where(self, reset_mask: torch.Tensor) -> Any:
        """Zero batch rows where ``reset_mask`` (B,) bool is True; out-of-place (autograd-safe)."""
        if self._state is None:
            return None
        if reset_mask.dtype != torch.bool or reset_mask.dim() != 1:
            raise ValueError("reset_mask must be a 1-D bool tensor (True = reset that stream).")
        if reset_mask.shape[0] != self.batch_size:
            raise ValueError(f"reset_mask has {reset_mask.shape[0]} rows, state batch size is {self.batch_size}.")

        def _reset(tensor: torch.Tensor) -> torch.Tensor:
            shape = [1] * tensor.dim()
            shape[self.batch_dim] = tensor.shape[self.batch_dim]
            mask = reset_mask.to(tensor.device).view(shape)
            return torch.where(mask, torch.zeros_like(tensor), tensor)

        self._state = _map_state(self._state, _reset)
        return self._state

    def reset_indices(self, indices: Sequence[int]) -> Any:
        """Convenience form of :meth:`reset_where` using batch row indices."""
        if self._state is None:
            return None
        mask = torch.zeros(self.batch_size, dtype=torch.bool)  # type: ignore[arg-type]
        mask[list(indices)] = True
        return self.reset_where(mask)

    def select(self, indices: Union[torch.Tensor, Sequence[int]]) -> Any:
        """Keep/reorder batch rows given by ``indices``."""

        def _select(tensor: torch.Tensor) -> torch.Tensor:
            index = torch.as_tensor(indices, dtype=torch.long, device=tensor.device)
            return tensor.index_select(self.batch_dim, index)

        self._state = _map_state(self._state, _select)
        return self._state

    def to(self, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None) -> Any:
        """Move/cast the stored state explicitly."""
        self._state = _map_state(self._state, lambda t: t.to(device=device, dtype=dtype))
        return self._state
