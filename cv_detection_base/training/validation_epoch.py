# Purpose: Runs object-detection validation (inference, injected post-processing, metric accumulation, optional loss) without mutating training state.
"""Detection validation epoch.

Flow per batch (under ``torch.inference_mode()``):
    images, targets = move_batch_fn(batch, device)
    outputs = model(images)                                  # raw predictions
    predictions = postprocess_fn(outputs, images, targets)   # list of {"boxes","scores","labels"}
    metric.update(predictions, targets)
    optional: losses = loss_fn(outputs, targets)             # needs raw outputs in eval mode

Boxes in predictions and targets must use the same coordinate frame (e.g. both
in the resized network-input frame, or both mapped back to original images).

The model is put in eval mode and its previous mode is restored afterwards; no
optimizer, scaler or gradient is touched. ``metric`` must provide
``reset()``, ``update(predictions, targets)`` and ``compute() -> dict``
(e.g. metrics/map.py). The metric is reset at the start.

Returns: {"loss" (if loss_fn), "<component>" means, metric scalars (non-scalar
entries such as per-class tensors are returned under the same keys), "num_images"}.

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn


def _move_batch(batch: Any, device: torch.device) -> Tuple[Any, List[Dict[str, Any]]]:
    images, targets = batch
    images = images.to(device) if isinstance(images, Tensor) else [img.to(device) for img in images]
    targets = [{k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in t.items()} for t in targets]
    return images, targets


def validate_one_epoch(
    model: nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    postprocess_fn: Callable[[Any, Any, List[Dict[str, Any]]], List[Dict[str, Tensor]]],
    metric: Any,
    loss_fn: Optional[Callable[[Any, List[Dict[str, Any]]], Mapping[str, Tensor]]] = None,
    autocast_dtype: Optional[torch.dtype] = None,
    move_batch_fn: Optional[Callable[[Any, torch.device], Tuple[Any, List[Dict[str, Any]]]]] = None,
) -> Dict[str, Any]:
    """Evaluate ``model`` over ``loader``. See module docstring for the contracts."""
    device = torch.device(device)
    move = move_batch_fn or _move_batch
    was_training = model.training
    model.eval()
    metric.reset()
    loss_sums: Dict[str, float] = {}
    num_images = 0
    try:
        with torch.inference_mode():
            for batch in loader:
                images, targets = move(batch, device)
                with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    outputs = model(images)
                    losses = loss_fn(outputs, targets) if loss_fn is not None else None
                predictions = postprocess_fn(outputs, images, targets)
                if len(predictions) != len(targets):
                    raise ValueError(f"postprocess_fn returned {len(predictions)} results for {len(targets)} images")
                metric.update(predictions, targets)
                if losses is not None:
                    for name, value in losses.items():
                        key = "loss" if name == "total" else name
                        loss_sums[key] = loss_sums.get(key, 0.0) + float(value) * len(targets)
                num_images += len(targets)
    finally:
        model.train(was_training)

    result: Dict[str, Any] = {k: v / max(num_images, 1) for k, v in loss_sums.items()}
    for key, value in metric.compute().items():
        result[key] = float(value) if isinstance(value, Tensor) and value.ndim == 0 else value
    result["num_images"] = num_images
    return result
