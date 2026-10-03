# Purpose: Predicts class logits or probabilities for one image with an already loaded image-classification model on an explicit device.
"""Single-image prediction.

Inputs:
    model: already constructed, weights loaded, and placed on ``device`` by the caller (this
        module never moves the model or loads checkpoints; a device mismatch raises).
    image: a preprocessed float tensor ``(C, H, W)``, or any object accepted by ``preprocess``
        (a callable returning ``(C, H, W)``, e.g. inference/preprocessing.py or validation
        transforms).
    output: "logits" (raw), "probabilities" (softmax, single-label) or "sigmoid" (multi-label).
Output: CPU float32 tensor ``(num_classes,)``.

Runs under ``torch.inference_mode()``; the model is set to eval for the call and its previous
mode is restored.

Requires: torch.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch
from torch import nn

OUTPUTS = ("logits", "probabilities", "sigmoid")


def _check_model_device(model: nn.Module, device: torch.device) -> None:
    param = next(model.parameters(), None)
    if param is not None and param.device.type != device.type:
        raise ValueError(
            f"model parameters are on {param.device} but device={device}; move the model explicitly"
        )


def _postprocess(logits: torch.Tensor, output: str) -> torch.Tensor:
    if output == "probabilities":
        return logits.softmax(dim=-1)
    if output == "sigmoid":
        return torch.sigmoid(logits)
    return logits


def predict_image(
    model: nn.Module,
    image: Any,
    device: torch.device,
    preprocess: Optional[Callable[[Any], torch.Tensor]] = None,
    output: str = "logits",
) -> torch.Tensor:
    """Return ``(num_classes,)`` scores for one image."""
    if output not in OUTPUTS:
        raise ValueError(f"output must be one of {OUTPUTS}, got {output!r}")
    device = torch.device(device)
    _check_model_device(model, device)
    tensor = preprocess(image) if preprocess is not None else image
    if not isinstance(tensor, torch.Tensor) or tensor.dim() != 3:
        raise ValueError("image (after preprocess) must be a (C, H, W) tensor")
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            logits = model(tensor.unsqueeze(0).to(device=device, dtype=torch.float32))
    finally:
        model.train(was_training)
    if logits.dim() != 2 or logits.shape[0] != 1:
        raise ValueError(f"model must return (1, num_classes) logits, got {tuple(logits.shape)}")
    return _postprocess(logits[0].float(), output).cpu()


def top_k_predictions(
    probabilities: torch.Tensor, k: int, class_names: Optional[Sequence[str]] = None
) -> List[Tuple[Any, float]]:
    """Return ``[(class_id_or_name, score), ...]`` for the ``k`` highest scores of a ``(C,)`` tensor."""
    if probabilities.dim() != 1:
        raise ValueError("expected a (num_classes,) tensor")
    if not 1 <= k <= probabilities.numel():
        raise ValueError(f"k must be in [1, {probabilities.numel()}], got {k}")
    if class_names is not None and len(class_names) != probabilities.numel():
        raise ValueError("class_names length must equal the number of classes")
    values, indices = probabilities.topk(k)
    return [
        (class_names[i] if class_names is not None else i, float(v))
        for v, i in zip(values.tolist(), indices.tolist())
    ]
