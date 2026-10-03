# Purpose: Provides an opt-in activation checkpointing wrapper (torch.utils.checkpoint, non-reentrant) for decoder blocks, applied explicitly by the caller.
"""Activation (gradient) checkpointing for decoder stacks.

``CheckpointWrapper(block)`` recomputes the block's forward during backward instead of storing
its activations: lower memory, ~1 extra forward of compute. It uses
``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`` (supports keyword arguments,
non-tensor outputs such as ``present_kv=None``, and inputs that do not require grad).

Checkpointing is active only when ``self.training`` and grad mode are enabled; in eval /
``torch.no_grad()`` the block runs normally. ``preserve_rng_state=True`` (default) replays the
same dropout masks during recomputation.

Nothing is patched at import time; the caller applies it explicitly, e.g.
    apply_gradient_checkpointing(model.layers)     # nn.ModuleList of blocks
    ...
    remove_gradient_checkpointing(model.layers)

Caveats: wrapped blocks get an extra ``.module`` level in parameter names (state_dict keys
change: ``layers.0.module.*``) — save/load unwrapped models or strip the prefix. Do not use
``use_cache=True`` while training with checkpointing (caches are pointless there).

Requires: torch.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class CheckpointWrapper(nn.Module):
    """Wraps a module so its forward is checkpointed during training."""

    def __init__(self, module: nn.Module, use_reentrant: bool = False, preserve_rng_state: bool = True) -> None:
        super().__init__()
        if isinstance(module, CheckpointWrapper):
            raise ValueError("Module is already wrapped")
        self.module = module
        self.use_reentrant = use_reentrant
        self.preserve_rng_state = preserve_rng_state

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.training and torch.is_grad_enabled():
            return checkpoint(
                self.module,
                *args,
                use_reentrant=self.use_reentrant,
                preserve_rng_state=self.preserve_rng_state,
                **kwargs,
            )
        return self.module(*args, **kwargs)


def apply_gradient_checkpointing(
    layers: nn.ModuleList, use_reentrant: bool = False, preserve_rng_state: bool = True
) -> nn.ModuleList:
    """Wrap every (not yet wrapped) module in ``layers`` in place; returns ``layers``."""
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("layers must be an nn.ModuleList (e.g. model.layers)")
    for i, layer in enumerate(layers):
        if not isinstance(layer, CheckpointWrapper):
            layers[i] = CheckpointWrapper(layer, use_reentrant=use_reentrant, preserve_rng_state=preserve_rng_state)
    return layers


def remove_gradient_checkpointing(layers: nn.ModuleList) -> nn.ModuleList:
    """Unwrap every ``CheckpointWrapper`` in ``layers`` in place; returns ``layers``."""
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("layers must be an nn.ModuleList")
    for i, layer in enumerate(layers):
        if isinstance(layer, CheckpointWrapper):
            layers[i] = layer.module
    return layers
