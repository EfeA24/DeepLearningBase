# Purpose: Evaluates an image classifier for one epoch under inference mode without optimizer or training-state mutation.
"""Validation / test epoch for image classification.

Batch contract: same as training/train_epoch.py: ``(images, targets, ...)`` with images
``(B, C, H, W)`` and int64 targets ``(B,)`` (or float multi-label targets; accuracy then omitted).
``loss_fn`` returns a scalar batch mean; the epoch loss is sample-weighted.

Behavior: switches the model to ``eval()`` for the pass and RESTORES its previous train/eval mode
afterwards; runs under ``torch.inference_mode()``; never touches an optimizer or scheduler.
Optional ``metric_updaters`` are callables ``(logits, targets) -> None`` (e.g.
``ConfusionMatrix.update``) so extra accumulators can observe the same forward pass.

Requires: torch.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import torch
from torch import nn


def _split_batch(batch: Any) -> tuple:
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    if isinstance(batch, dict) and "image" in batch and "label" in batch:
        return batch["image"], batch["label"]
    raise TypeError("batch must be (images, targets, ...) or a dict with 'image' and 'label' keys")


def validate_one_epoch(
    model: nn.Module,
    loader: Iterable[Any],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    *,
    autocast_dtype: Optional[torch.dtype] = None,
    metric_updaters: Sequence[Callable[[torch.Tensor, torch.Tensor], None]] = (),
    non_blocking: bool = False,
) -> Dict[str, float]:
    """Return ``{"loss", "accuracy" (single-label only), "num_samples"}`` over ``loader``."""
    device = torch.device(device)
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    correct = 0
    num_samples = 0
    single_label = True
    try:
        with torch.inference_mode():
            for batch in loader:
                images, targets = _split_batch(batch)
                images = images.to(device, non_blocking=non_blocking)
                targets = targets.to(device, non_blocking=non_blocking)
                with torch.amp.autocast(
                    device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None
                ):
                    logits = model(images)
                    loss = loss_fn(logits, targets)
                batch_size = images.shape[0]
                loss_sum += float(loss) * batch_size
                num_samples += batch_size
                if single_label and targets.dim() == 1 and not targets.is_floating_point() and logits.dim() == 2:
                    correct += int((logits.argmax(dim=1) == targets).sum())
                else:
                    single_label = False
                for update in metric_updaters:
                    update(logits.float(), targets)
    finally:
        model.train(was_training)

    if num_samples == 0:
        raise ValueError("loader produced no batches")
    metrics: Dict[str, float] = {"loss": loss_sum / num_samples, "num_samples": num_samples}
    if single_label:
        metrics["accuracy"] = correct / num_samples
    return metrics
