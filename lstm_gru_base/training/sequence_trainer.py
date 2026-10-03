# Purpose: Orchestrates one training or evaluation epoch for LSTM/GRU models with a supplied loss and an explicit hidden-state policy (reset or carry).
"""Minimal sequence epoch runner.

Contract:
    - ``forward_fn(model, batch, state) -> (predictions, new_state)``. Default: ``model(batch[input_key], state)``;
      if the model returns a non-tuple it is treated as predictions with ``new_state = None``.
      Supply a custom ``forward_fn`` to pass lengths/masks (e.g. ``model(x, state, lengths=batch["lengths"])``).
    - ``loss_fn(predictions, targets) -> scalar`` MEAN loss.
    - ``state_policy``:
        ``"reset"`` - every batch starts from ``None`` (independent sequences; the usual case);
        ``"carry"`` - the detached state from batch ``i`` initializes batch ``i+1`` (stateful streams; caller must
        keep batch size and stream-to-row mapping constant, e.g. ``drop_last=True`` and no shuffling).
    - Loss aggregation: weighted by ``targets.shape[0]`` (samples) by default; if ``ignore_index`` is given,
      weighted by the count of targets != ignore_index (token-weighted, matches per-token mean losses).
    - Training when ``optimizer`` is given (``model.train()``), otherwise evaluation under ``torch.inference_mode``.

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, Union

import torch
from torch import nn

ForwardFn = Callable[[nn.Module, Mapping[str, Any], Any], Tuple[torch.Tensor, Any]]


def _detach(state: Any) -> Any:
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, (tuple, list)):
        return type(state)(_detach(s) for s in state)
    raise TypeError(f"Unsupported state type {type(state).__name__}.")


def _to_device(batch: Mapping[str, Any], device: Union[str, torch.device]) -> Dict[str, Any]:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def make_default_forward(input_key: str = "inputs") -> ForwardFn:
    def forward(model: nn.Module, batch: Mapping[str, Any], state: Any) -> Tuple[torch.Tensor, Any]:
        out = model(batch[input_key], state)
        if isinstance(out, tuple) and len(out) == 2:
            return out[0], out[1]
        return out, None

    return forward


def run_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: Union[str, torch.device],
    optimizer: Optional[torch.optim.Optimizer] = None,
    forward_fn: Optional[ForwardFn] = None,
    target_key: str = "targets",
    state_policy: str = "reset",
    clip_grad_norm: Optional[float] = None,
    ignore_index: Optional[int] = None,
) -> Dict[str, float]:
    """Run one epoch and return ``{"loss": weighted mean, "num_items": total weight, "num_batches": n}``."""
    if state_policy not in ("reset", "carry"):
        raise ValueError(f"state_policy must be 'reset' or 'carry', got {state_policy!r}.")
    forward_fn = forward_fn or make_default_forward()
    training = optimizer is not None
    model.train(training)
    total_loss, total_items, num_batches = 0.0, 0, 0
    state: Any = None
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            targets = batch[target_key]
            if state_policy == "reset":
                state = None
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions, new_state = forward_fn(model, batch, state)
            loss = loss_fn(predictions, targets)
            if training:
                loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
            state = _detach(new_state) if state_policy == "carry" else None
            weight = int((targets != ignore_index).sum()) if ignore_index is not None else int(targets.shape[0])
            total_loss += float(loss.detach()) * weight
            total_items += weight
            num_batches += 1
    return {
        "loss": total_loss / total_items if total_items else float("nan"),
        "num_items": float(total_items),
        "num_batches": float(num_batches),
    }
