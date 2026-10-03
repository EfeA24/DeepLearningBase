# Purpose: Runs one training (or evaluation) pass of a recurrent sequence model with an explicit hidden-state carry/reset policy.
"""Sequence training / evaluation epoch.

Batch contract: ``(inputs, targets)`` tuples/lists, or dicts with ``input_key``/``target_key``
(default "inputs"/"targets", matching padded_collator.py). Batch-first tensors (B, T, ...).
Model contract: ``model(inputs, state) -> (outputs, new_state)``.
``loss_fn(outputs, targets)`` returns a MEAN scalar over its counted targets.

State policy (``state_policy``):
- "reset": every batch starts from ``state=None`` (independent sequences; any loader order).
- "carry": the detached final state of batch i is the initial state of batch i+1. Only valid
  for stream-ordered loaders (``shuffle=False``, row b continues stream b, e.g. batchified
  text). If the batch size changes, the state is reset with a warning.

Metric: ``loss`` is the target-count-weighted mean of per-batch mean losses, where a batch's
count is the number of integer targets != ``ignore_index`` (all elements for float targets).

Requires: torch only. No sibling files.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import nn


def _detach(state: Any) -> Any:
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, (tuple, list)):
        return type(state)(_detach(item) for item in state)
    raise TypeError(f"Unsupported state element type: {type(state).__name__}.")


def _state_batch_size(state: Any) -> Optional[int]:
    if isinstance(state, torch.Tensor):
        return state.shape[1]
    if isinstance(state, (tuple, list)) and state:
        return _state_batch_size(state[0])
    return None


def _unpack(batch: Any, input_key: str, target_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch[input_key], batch[target_key]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError("Batches must be (inputs, targets) tuples or dicts with input/target keys.")


def _count_targets(targets: torch.Tensor, ignore_index: int) -> int:
    if targets.dtype.is_floating_point:
        return targets.numel()
    return int((targets != ignore_index).sum().item())


def _run_epoch(
    model: nn.Module,
    loader: Iterable[Any],
    loss_fn: Callable[[Any, torch.Tensor], torch.Tensor],
    device: Union[str, torch.device],
    optimizer: Optional[torch.optim.Optimizer],
    state_policy: str,
    max_grad_norm: Optional[float],
    ignore_index: int,
    input_key: str,
    target_key: str,
) -> Dict[str, float]:
    if state_policy not in ("reset", "carry"):
        raise ValueError(f"state_policy must be 'reset' or 'carry', got {state_policy!r}.")
    state: Any = None
    loss_sum = 0.0
    target_count = 0
    num_batches = 0
    for batch in loader:
        inputs, targets = _unpack(batch, input_key, target_key)
        inputs, targets = inputs.to(device), targets.to(device)
        if state_policy == "reset":
            state = None
        elif state is not None and _state_batch_size(state) != inputs.shape[0]:
            warnings.warn("Batch size changed; resetting carried hidden state.", stacklevel=3)
            state = None

        outputs, new_state = model(inputs, state)
        loss = loss_fn(outputs, targets)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        state = _detach(new_state) if state_policy == "carry" else None

        count = _count_targets(targets, ignore_index)
        loss_sum += loss.item() * count
        target_count += count
        num_batches += 1

    return {
        "loss": loss_sum / target_count if target_count else math.nan,
        "num_targets": float(target_count),
        "num_batches": float(num_batches),
    }


def train_sequence_epoch(
    model: nn.Module,
    loader: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[Any, torch.Tensor], torch.Tensor],
    device: Union[str, torch.device],
    state_policy: str = "reset",
    max_grad_norm: Optional[float] = None,
    ignore_index: int = -100,
    input_key: str = "inputs",
    target_key: str = "targets",
) -> Dict[str, float]:
    """One training epoch (model.train(), zero_grad/backward/[clip]/step per batch)."""
    model.train()
    return _run_epoch(
        model, loader, loss_fn, device, optimizer, state_policy, max_grad_norm, ignore_index, input_key, target_key
    )


def evaluate_sequence_epoch(
    model: nn.Module,
    loader: Iterable[Any],
    loss_fn: Callable[[Any, torch.Tensor], torch.Tensor],
    device: Union[str, torch.device],
    state_policy: str = "reset",
    ignore_index: int = -100,
    input_key: str = "inputs",
    target_key: str = "targets",
) -> Dict[str, float]:
    """One evaluation pass (model.eval(), no_grad, no optimizer) with the same metric contract."""
    model.eval()
    with torch.no_grad():
        return _run_epoch(
            model, loader, loss_fn, device, None, state_policy, None, ignore_index, input_key, target_key
        )
