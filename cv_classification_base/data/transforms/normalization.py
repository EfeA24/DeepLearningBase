# Purpose: Provides torch-only normalize/denormalize helpers with explicit mean/std and value range for image-classification tensors.
"""Image tensor normalization (no torchvision needed).

Contract:
    - Input images are float tensors ``(C, H, W)`` or ``(B, C, H, W)`` in the value range that
      ``mean``/``std`` were computed for (usually [0, 1] after ``uint8 / 255``).
    - ``normalize``: ``(x - mean) / std`` per channel. ``denormalize`` inverts it (e.g. for
      visualizing augmented batches), optionally clamping back to [0, 1].
    - ``to_unit_range`` converts uint8 [0, 255] to float32 [0, 1].
Compute dataset statistics on the TRAIN split only (``compute_mean_std``).

Requires: torch.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
from torch import nn

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def _stats(x: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.dim() not in (3, 4):
        raise ValueError(f"expected (C, H, W) or (B, C, H, W), got {tuple(x.shape)}")
    if not x.is_floating_point():
        raise TypeError(f"expected a float tensor, got {x.dtype}; convert with to_unit_range first")
    channels = x.shape[-3]
    if len(mean) != channels or len(std) != channels:
        raise ValueError(f"mean/std must have {channels} values")
    if any(s <= 0 for s in std):
        raise ValueError("std values must be > 0")
    shape = (-1, 1, 1)
    m = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(shape)
    s = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(shape)
    return m, s


def to_unit_range(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.uint8:
        raise TypeError(f"to_unit_range expects uint8, got {x.dtype}")
    return x.float() / 255.0


def normalize(x: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    m, s = _stats(x, mean, std)
    return (x - m) / s


def denormalize(
    x: torch.Tensor, mean: Sequence[float], std: Sequence[float], clamp: bool = True
) -> torch.Tensor:
    m, s = _stats(x, mean, std)
    out = x * s + m
    return out.clamp(0.0, 1.0) if clamp else out


class Normalize(nn.Module):
    """Module form of ``normalize`` (e.g. as the first layer of an exported model)."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        if len(mean) != len(std) or any(s <= 0 for s in std):
            raise ValueError("mean/std must have equal length and std > 0")
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(-1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)


@torch.no_grad()
def compute_mean_std(images: Iterable[torch.Tensor]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Per-channel mean/std over an iterable of float ``(C, H, W)`` or ``(B, C, H, W)`` tensors.

    Pixel-weighted (each pixel counts equally), computed in float64 with running sums.
    """
    total = None
    total_sq = None
    count = 0
    for x in images:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4 or not x.is_floating_point():
            raise ValueError("images must be float (C, H, W) or (B, C, H, W) tensors")
        x = x.double()
        s = x.sum(dim=(0, 2, 3))
        sq = (x * x).sum(dim=(0, 2, 3))
        total = s if total is None else total + s
        total_sq = sq if total_sq is None else total_sq + sq
        count += x.shape[0] * x.shape[2] * x.shape[3]
    if count == 0 or total is None or total_sq is None:
        raise ValueError("no images provided")
    mean = total / count
    std = (total_sq / count - mean * mean).clamp_min(0).sqrt()
    return tuple(mean.tolist()), tuple(std.tolist())
