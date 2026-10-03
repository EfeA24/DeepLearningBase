# Purpose: Runs sequence classification with an already initialized LSTM/GRU model on a prepared padded batch, returning logits, probabilities and predicted labels.
"""Sequence classification inference.

Contract:
    - The caller loads weights, moves the model to ``device`` and calls ``model.eval()``; this function
      neither loads checkpoints nor moves the model (it warns if the model is still in training mode).
    - ``batch`` is a dict from data/collators/padded_sequence_collator.py (``inputs`` ``(B, T[, F])``,
      ``lengths`` ``(B,)``, ``mask`` ``(B, T)`` True = valid). Tensors are moved to ``device``.
    - ``forward_fn(model, batch) -> logits (B, C)``; default ``model(batch["inputs"], lengths=batch["lengths"])``
      if lengths exist, else ``model(batch["inputs"])``. A tuple result uses its first element.
    - Output: ``{"logits": (B, C), "probabilities": (B, C), "predictions": (B,)}``; softmax for multi-class,
      sigmoid with ``threshold`` when ``C == 1`` (binary), all computed outside the model.

Requires: torch only.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Mapping, Optional, Union

import torch
from torch import nn


def _default_forward(model: nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
    if "lengths" in batch:
        out = model(batch["inputs"], lengths=batch["lengths"])
    else:
        out = model(batch["inputs"])
    return out[0] if isinstance(out, tuple) else out


def classify_batch(
    model: nn.Module,
    batch: Mapping[str, Any],
    device: Union[str, torch.device],
    forward_fn: Optional[Callable[[nn.Module, Mapping[str, Any]], torch.Tensor]] = None,
    threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Classify one prepared batch; results stay on ``device``."""
    if model.training:
        warnings.warn("classify_batch called with model in training mode; call model.eval() first.", stacklevel=2)
    forward_fn = forward_fn or _default_forward
    moved = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    with torch.inference_mode():
        logits = forward_fn(model, moved)
    if logits.dim() != 2:
        raise ValueError(f"Expected logits (B, C), got {tuple(logits.shape)}.")
    if logits.shape[1] == 1:
        probs = torch.sigmoid(logits)
        preds = (probs.squeeze(1) >= threshold).long()
    else:
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
    return {"logits": logits, "probabilities": probs, "predictions": preds}


def map_predictions(predictions: torch.Tensor, id_to_label: Mapping[int, str]) -> list:
    """Convert predicted ids ``(B,)`` to label names with an explicit mapping (raises on unknown ids)."""
    return [id_to_label[int(i)] for i in predictions.tolist()]
