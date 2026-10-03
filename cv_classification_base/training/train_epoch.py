# Purpose: Runs one image-classification training epoch with optional AMP, gradient clipping and per-update scheduling, returning sample-weighted metrics.
"""One training epoch for image classification.

Batch contract: the loader yields ``(images, targets)`` (tuple/list; extra trailing items are
ignored) with images ``(B, C, H, W)`` and targets int64 ``(B,)`` (single-label) or float
``(B, L)`` (multi-label; accuracy is then not computed).
``loss_fn(logits, targets)`` must return a scalar MEAN over the batch; the epoch loss is the
sample-weighted mean ``sum(loss_i * B_i) / sum(B_i)``.

Options (all injected, all off by default):
    autocast_dtype: e.g. ``torch.float16`` / ``torch.bfloat16`` -> forward under
        ``torch.amp.autocast(device_type=device.type, dtype=...)``.
    scaler: a ``torch.amp.GradScaler`` (needed for float16; pass None for bf16/fp32).
    max_grad_norm: clip total grad norm after backward (unscaled first when a scaler is used).
    scheduler: stepped after EVERY optimizer step (per-update schedulers such as OneCycle /
        warmup). Leave None for per-epoch schedulers and step them in the caller's loop.

The model is NOT moved; the caller places it on ``device`` beforehand.

Requires: torch.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional

import torch
from torch import nn


def _split_batch(batch: Any) -> tuple:
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    if isinstance(batch, dict) and "image" in batch and "label" in batch:
        return batch["image"], batch["label"]
    raise TypeError("batch must be (images, targets, ...) or a dict with 'image' and 'label' keys")


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    *,
    autocast_dtype: Optional[torch.dtype] = None,
    scaler: Optional["torch.amp.GradScaler"] = None,
    max_grad_norm: Optional[float] = None,
    scheduler: Optional[Any] = None,
    non_blocking: bool = False,
) -> Dict[str, float]:
    """Train for one pass over ``loader``.

    Returns:
        ``{"loss": float, "accuracy": float, "num_samples": int}``; "accuracy" is present only for
        single-label int targets. Raises ValueError if the loader is empty.
    """
    device = torch.device(device)
    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError(f"max_grad_norm must be > 0 when given, got {max_grad_norm}")
    model.train()
    loss_sum = 0.0
    correct = 0
    num_samples = 0
    single_label = True

    for batch in loader:
        images, targets = _split_batch(batch)
        images = images.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None
        ):
            logits = model(images)
            loss = loss_fn(logits, targets)
        if loss.dim() != 0:
            raise ValueError("loss_fn must return a scalar (use reduction='mean')")

        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = images.shape[0]
        loss_sum += float(loss.detach()) * batch_size
        num_samples += batch_size
        if single_label and targets.dim() == 1 and not targets.is_floating_point() and logits.dim() == 2:
            correct += int((logits.detach().argmax(dim=1) == targets).sum())
        else:
            single_label = False

    if num_samples == 0:
        raise ValueError("loader produced no batches")
    metrics: Dict[str, float] = {"loss": loss_sum / num_samples, "num_samples": num_samples}
    if single_label:
        metrics["accuracy"] = correct / num_samples
    return metrics
