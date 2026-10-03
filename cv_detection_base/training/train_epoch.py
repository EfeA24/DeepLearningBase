# Purpose: Runs one object-detection training epoch for a supplied detector, loss function and optimizer with optional AMP, gradient clipping and per-step scheduling.
"""One detection training epoch.

Batch contract: the loader yields ``(images, targets)`` where images is a Tensor
(B, C, H, W) (or a list of (C, H, W) tensors if your model accepts lists) and
targets is a list of B target dicts (see data/collators/detection_collator.py).
Override with ``move_batch_fn(batch, device) -> (images, targets)`` for other layouts.

Loss contract: ``loss_fn(outputs, targets)`` returns a dict with a differentiable
scalar "total" plus optional scalar components (logged as means). Target
assignment lives inside loss_fn (e.g. losses/detection_loss.py with an assign_fn).

AMP: pass ``autocast_dtype`` (torch.float16 / torch.bfloat16) to enable
``torch.amp.autocast(device.type)``; pass a ``torch.amp.GradScaler`` for float16.
Gradients are unscaled before clipping.

Returns plain floats: {"loss": image-weighted mean total, "<component>": mean,
"num_images": int, "num_batches": int, "skipped_steps": int (non-finite losses)}.

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn


def move_detection_batch(batch: Any, device: torch.device, non_blocking: bool = False) -> Tuple[Any, List[Dict[str, Any]]]:
    """Move ``(images, targets)`` to ``device``; tensors inside target dicts are moved, others kept."""
    images, targets = batch
    if isinstance(images, Tensor):
        images = images.to(device, non_blocking=non_blocking)
    else:
        images = [img.to(device, non_blocking=non_blocking) for img in images]
    moved = [
        {k: (v.to(device, non_blocking=non_blocking) if isinstance(v, Tensor) else v) for k, v in t.items()} for t in targets
    ]
    return images, moved


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[Any, List[Dict[str, Any]]], Mapping[str, Tensor]],
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    autocast_dtype: Optional[torch.dtype] = None,
    max_grad_norm: Optional[float] = None,
    scheduler: Optional[Any] = None,
    move_batch_fn: Optional[Callable[[Any, torch.device], Tuple[Any, List[Dict[str, Any]]]]] = None,
    skip_non_finite: bool = True,
) -> Dict[str, float]:
    """Train ``model`` for one pass over ``loader``.

    Args:
        model: Detector already on ``device``.
        loader: Iterable of batches (see module docstring).
        optimizer: Optimizer over the model parameters.
        loss_fn: ``(outputs, targets) -> {"total": Tensor, ...}``.
        device: Device for the batch tensors (the model is NOT moved).
        scaler: Optional GradScaler (float16 AMP).
        autocast_dtype: Optional autocast dtype; None disables autocast.
        max_grad_norm: Optional global gradient-norm clip.
        scheduler: Optional LR scheduler stepped after every optimizer step
            (use for per-iteration schedules; step per-epoch schedulers outside).
        move_batch_fn: Custom batch transfer.
        skip_non_finite: Skip the optimizer step when the loss is NaN/Inf
            (counted in "skipped_steps"); if False, raise FloatingPointError.
    """
    device = torch.device(device)
    move = move_batch_fn or move_detection_batch
    model.train()
    sums: Dict[str, float] = {}
    num_images = 0
    num_batches = 0
    skipped = 0

    for batch in loader:
        images, targets = move(batch, device)
        batch_size = len(targets)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            outputs = model(images)
            losses = loss_fn(outputs, targets)
        if "total" not in losses:
            raise KeyError("loss_fn must return a dict containing 'total'")
        total = losses["total"]
        if not torch.isfinite(total.detach()):
            if not skip_non_finite:
                raise FloatingPointError(f"non-finite loss {float(total)} at batch {num_batches}")
            skipped += 1
            num_batches += 1
            continue

        if scaler is not None:
            scaler.scale(total).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        for name, value in losses.items():
            key = "loss" if name == "total" else name
            sums[key] = sums.get(key, 0.0) + float(value.detach()) * batch_size
        num_images += batch_size
        num_batches += 1

    result = {k: v / max(num_images, 1) for k, v in sums.items()}
    result.setdefault("loss", float("nan"))
    result.update({"num_images": num_images, "num_batches": num_batches, "skipped_steps": skipped})
    return result
