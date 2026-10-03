# Purpose: Defines the light forward/output contract (abstract base) shared by object detectors so training, post-processing and export code can rely on it.
"""Detector contract (ABC only; no architecture).

Contract:
    ``forward(images)``:
        images: FloatTensor (B, C, H, W), normalized as the backbone expects;
                H and W should be multiples of ``max_stride`` when the model needs it.
        returns RAW predictions only (no sigmoid/softmax, no decoding, no NMS),
        e.g. a list of per-level maps (B, A*(5+C), H_i, W_i) as produced by
        models/heads/detection_head.py.
    Loss computation (training) and decoding + NMS (inference) are separate,
    caller-supplied steps (see losses/ and inference/postprocess.py). Keeping
    forward raw makes the same module usable for training, evaluation and export.

Subclasses declare ``num_classes``, ``strides`` (per output level, high-res first)
and implement ``forward``.

Requires: torch only.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Sequence

import torch
from torch import Tensor, nn


class DetectorBase(nn.Module, ABC):
    """Abstract detector module.

    Args:
        num_classes: Number of foreground classes C.
        strides: Output strides per level (high-res first).
    """

    def __init__(self, num_classes: int, strides: Sequence[int]) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be > 0")
        if not strides or any(s <= 0 for s in strides):
            raise ValueError("strides must be non-empty positive ints")
        self.num_classes = int(num_classes)
        self.strides: List[int] = [int(s) for s in strides]

    @property
    def max_stride(self) -> int:
        return max(self.strides)

    def check_images(self, images: Tensor, require_divisible: bool = False) -> None:
        """Validate the input batch shape (and optional divisibility by ``max_stride``)."""
        if not isinstance(images, Tensor) or images.ndim != 4:
            raise ValueError(f"images must be a Tensor (B, C, H, W), got {getattr(images, 'shape', type(images))}")
        if not images.is_floating_point():
            raise TypeError("images must be floating point (convert and normalize first)")
        if require_divisible and (images.shape[-2] % self.max_stride or images.shape[-1] % self.max_stride):
            raise ValueError(f"image size {tuple(images.shape[-2:])} must be divisible by max stride {self.max_stride}")

    @abstractmethod
    def forward(self, images: Tensor) -> Any:
        """Return raw predictions for ``images`` (B, C, H, W). See module docstring."""
        raise NotImplementedError
