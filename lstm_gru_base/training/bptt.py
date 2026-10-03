# Purpose: Provides a full-sequence backpropagation-through-time training step for LSTM/GRU models and documents its memory implications.
"""Full BPTT training step.

Behaviour: the whole padded sequence is processed in one forward pass and gradients flow through every
time step. Activation memory grows linearly with ``T`` (roughly ``O(B * T * H * num_layers)`` saved
tensors), so very long sequences may need training/truncated_bptt.py instead.

Contract:
    - ``forward_fn(model, batch) -> predictions``; default calls ``model(batch["inputs"])`` and, if the model
      returns a tuple ``(outputs, state)``, uses the first element.
    - ``loss_fn(predictions, targets) -> scalar`` mean loss; targets come from ``batch[target_key]``.
      For per-token losses, flatten inside ``loss_fn`` and use ``ignore_index=-100`` for padding.
    - The batch is moved to ``device`` explicitly (tensors only); the model must already be on ``device``.
    - Returns ``{"loss": float, "grad_norm": float | None}``.

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Union

import torch
from torch import nn


def move_batch(batch: Mapping[str, Any], device: Union[str, torch.device]) -> Dict[str, Any]:
    """Move tensor fields of a dict batch to ``device``; other fields are passed through."""
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def default_forward(model: nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
    out = model(batch["inputs"])
    return out[0] if isinstance(out, tuple) else out


def full_bptt_step(
    model: nn.Module,
    batch: Mapping[str, Any],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: Union[str, torch.device],
    target_key: str = "targets",
    forward_fn: Optional[Callable[[nn.Module, Mapping[str, Any]], torch.Tensor]] = None,
    clip_grad_norm: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Run forward, backward through the full sequence, optional clipping, and one optimizer step."""
    if target_key not in batch:
        raise KeyError(f"Batch is missing target key {target_key!r}.")
    forward_fn = forward_fn or default_forward
    model.train()
    batch = move_batch(batch, device)
    optimizer.zero_grad(set_to_none=True)
    predictions = forward_fn(model, batch)
    loss = loss_fn(predictions, batch[target_key])
    if loss.dim() != 0:
        raise ValueError("loss_fn must return a scalar loss.")
    loss.backward()
    grad_norm = None
    if clip_grad_norm is not None:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm))
    optimizer.step()
    return {"loss": float(loss.detach()), "grad_norm": grad_norm}
