# Purpose: Provides optional detection mixup (blend two images, concatenate their targets with per-box weights) for object-detection training.
"""Detection mixup (optional; training only).

Operation:
    image = lam * image1 + (1 - lam) * image2   (both padded bottom/right with
            ``pad_value`` to the common max size, so box coordinates are unchanged)
    boxes/labels = concat(target1, target2)
    weights      = concat(lam * w1, (1 - lam) * w2), w_i = existing "weights" or ones

Loss compatibility (important):
    Unlike classification mixup, detection mixup keeps ALL boxes as hard targets.
    The per-box "weights" are only meaningful if your assigner/loss reads them
    (e.g. scales each matched box's loss). Losses that ignore "weights" treat
    both images' objects as fully present, which is the common YOLO-style
    practice but changes the semantics. Do not use with losses that assume
    one-hot image-level labels.

Contract:
    image1, image2: float Tensors (C, H_i, W_i) with equal C.
    target1, target2: dicts with "boxes" (N_i, 4) float xyxy pixels, "labels" (N_i,).
    Returns (mixed image (C, max H, max W), target with "boxes", "labels",
    "weights" (N1 + N2,) float, plus "iscrowd"/"difficult" if present in both).

Requires: torch only.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def _sample_gamma(shape: float, generator: torch.Generator) -> float:
    """Gamma(shape, 1) sample via Marsaglia-Tsang, drawing only from ``generator``."""
    if shape < 1.0:
        u = float(torch.rand((), generator=generator, dtype=torch.float64))
        return _sample_gamma(shape + 1.0, generator) * max(u, 1e-300) ** (1.0 / shape)
    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = float(torch.randn((), generator=generator, dtype=torch.float64))
        v = (1.0 + c * x) ** 3
        if v <= 0:
            continue
        u = float(torch.rand((), generator=generator, dtype=torch.float64))
        if math.log(max(u, 1e-300)) < 0.5 * x * x + d - d * v + d * math.log(v):
            return d * v


def sample_mixup_lambda(alpha: float, generator: torch.Generator) -> float:
    """Draw lam ~ Beta(alpha, alpha) from the supplied generator (ratio of two Gamma draws)."""
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    a = _sample_gamma(alpha, generator)
    b = _sample_gamma(alpha, generator)
    return a / max(a + b, 1e-300)


def _pad_to(image: Tensor, height: int, width: int, value: float) -> Tensor:
    pad_h, pad_w = height - image.shape[-2], width - image.shape[-1]
    if pad_h == 0 and pad_w == 0:
        return image
    return F.pad(image, (0, pad_w, 0, pad_h), value=value)


def mixup_detection(
    image1: Tensor,
    target1: Dict[str, Any],
    image2: Tensor,
    target2: Dict[str, Any],
    lam: Optional[float] = None,
    alpha: float = 8.0,
    generator: Optional[torch.Generator] = None,
    pad_value: float = 0.0,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Blend two detection samples. Pass ``lam`` explicitly or ``alpha`` + ``generator``."""
    if image1.ndim != 3 or image2.ndim != 3 or image1.shape[0] != image2.shape[0]:
        raise ValueError("images must be (C, H, W) with the same number of channels")
    if not (image1.is_floating_point() and image2.is_floating_point()):
        raise TypeError("mixup expects float images")
    if lam is None:
        if generator is None:
            raise ValueError("pass either lam or a torch.Generator to sample it")
        lam = sample_mixup_lambda(alpha, generator)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")

    height = max(image1.shape[-2], image2.shape[-2])
    width = max(image1.shape[-1], image2.shape[-1])
    mixed = lam * _pad_to(image1, height, width, pad_value) + (1.0 - lam) * _pad_to(image2, height, width, pad_value)

    def _weights(target: Dict[str, Any]) -> Tensor:
        n = target["boxes"].shape[0]
        existing = target.get("weights")
        return existing.float() if isinstance(existing, Tensor) else torch.ones(n, device=target["boxes"].device)

    out: Dict[str, Tensor] = {
        "boxes": torch.cat([target1["boxes"], target2["boxes"]], dim=0).reshape(-1, 4),
        "labels": torch.cat([target1["labels"], target2["labels"]], dim=0),
        "weights": torch.cat([lam * _weights(target1), (1.0 - lam) * _weights(target2)], dim=0),
    }
    for key in ("iscrowd", "difficult"):
        if key in target1 and key in target2:
            out[key] = torch.cat([target1[key], target2[key]], dim=0)
    return mixed, out
