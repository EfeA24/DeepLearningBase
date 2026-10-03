# Purpose: Detaches nested recurrent states (tensor/tuple/list/dict) from the autograd graph between truncated-BPTT chunks.
"""Recursive hidden-state detaching.

Supports None, tensors, and arbitrarily nested tuples/lists/dicts of tensors (e.g. LSTM
``(h, c)`` or per-layer lists). Structure and container types are preserved; values are
unchanged, only the autograd history is cut. Use between TBPTT chunks or when carrying state
across batches so backward does not reach into previous chunks.

Requires: torch only. No sibling files.
"""
from __future__ import annotations

from typing import Any

import torch


def detach_hidden(state: Any) -> Any:
    """Return ``state`` with every tensor detached; raises TypeError for unsupported leaves."""
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, tuple) and hasattr(state, "_fields"):
        return type(state)(*(detach_hidden(item) for item in state))
    if isinstance(state, (tuple, list)):
        return type(state)(detach_hidden(item) for item in state)
    if isinstance(state, dict):
        return {key: detach_hidden(value) for key, value in state.items()}
    raise TypeError(f"Unsupported hidden-state element type: {type(state).__name__}.")
