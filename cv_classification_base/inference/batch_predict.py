# Purpose: Runs order-preserving batched prediction over caller-supplied image batches or tensors with an already loaded image-classification model.
"""Batched prediction.

``predict_batches``: iterate over caller-provided batches (e.g. a DataLoader with
``shuffle=False``); each item is a ``(B, C, H, W)`` tensor or a tuple/list whose first element is
that tensor (labels etc. are ignored). Results are concatenated in iteration order.

``predict_tensors``: chunk an in-memory sequence of preprocessed ``(C, H, W)`` tensors (all the same
size) into batches of ``batch_size``; output row ``i`` corresponds to input ``i``.

Output: CPU float32 ``(N, num_classes)`` logits / softmax probabilities / sigmoid scores.
This module never loads data, models or checkpoints and never moves the model (device mismatch
raises). Runs under ``torch.inference_mode()``, restoring the model's previous train/eval mode.

Requires: torch.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch
from torch import nn

OUTPUTS = ("logits", "probabilities", "sigmoid")


def _postprocess(logits: torch.Tensor, output: str) -> torch.Tensor:
    if output == "probabilities":
        return logits.softmax(dim=-1)
    if output == "sigmoid":
        return torch.sigmoid(logits)
    return logits


def predict_batches(
    model: nn.Module,
    batches: Iterable[Any],
    device: torch.device,
    output: str = "logits",
    non_blocking: bool = False,
) -> torch.Tensor:
    """Predict every batch in order; returns ``(N, num_classes)`` on CPU."""
    if output not in OUTPUTS:
        raise ValueError(f"output must be one of {OUTPUTS}, got {output!r}")
    device = torch.device(device)
    param = next(model.parameters(), None)
    if param is not None and param.device.type != device.type:
        raise ValueError(f"model is on {param.device} but device={device}; move it explicitly")

    results = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in batches:
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                if not isinstance(images, torch.Tensor) or images.dim() != 4:
                    raise ValueError("each batch must be (B, C, H, W) or (images, ...) with 4D images")
                logits = model(images.to(device, non_blocking=non_blocking))
                results.append(_postprocess(logits.float(), output).cpu())
    finally:
        model.train(was_training)
    if not results:
        raise ValueError("no batches were provided")
    return torch.cat(results, dim=0)


def predict_tensors(
    model: nn.Module,
    images: Sequence[torch.Tensor],
    device: torch.device,
    batch_size: int = 32,
    output: str = "logits",
) -> torch.Tensor:
    """Predict a sequence of preprocessed ``(C, H, W)`` tensors in chunks of ``batch_size``."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if len(images) == 0:
        raise ValueError("images is empty")
    shape = images[0].shape
    for i, img in enumerate(images):
        if img.shape != shape:
            raise ValueError(f"image {i} has shape {tuple(img.shape)}, expected {tuple(shape)}")
    chunks = (
        torch.stack(list(images[start : start + batch_size]))
        for start in range(0, len(images), batch_size)
    )
    return predict_batches(model, chunks, device, output=output)
