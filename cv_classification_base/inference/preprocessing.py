# Purpose: Converts one PIL/ndarray/tensor image into a normalized model-input tensor for image-classification inference using torch-only resizing.
"""Single-image preprocessing for inference (deterministic; mirrors validation transforms).

Pipeline: image -> uint8/float tensor ``(C, H, W)`` -> float in [0, 1] -> resize shorter side to
``resize_size`` (bilinear, antialias) -> center crop ``crop_size`` -> normalize with ``mean``/``std``.
Output: float32 ``(C, crop_size, crop_size)`` (add a batch dim with ``unsqueeze(0)``).

Accepted inputs:
    PIL.Image: converted to ``image_mode`` (default "RGB") without numpy (raw bytes path).
    numpy.ndarray: ``(H, W)`` or ``(H, W, C)`` HWC, uint8 [0, 255] or float [0, 1].
        ``channel_order`` declares RGB vs BGR (OpenCV images are BGR -> pass "BGR").
    torch.Tensor: ``(C, H, W)`` uint8 [0, 255] or float [0, 1], assumed RGB.

Torch resizing differs slightly from PIL/torchvision resampling; use the SAME preprocessing at
validation time as at inference (or use data/transforms/validation_transforms.py for both).
Statistics must match training (e.g. ImageNet mean/std for torchvision pretrained weights).

Requires: torch; Pillow only for PIL inputs; numpy only for ndarray inputs.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F


def image_to_tensor(image: Any, image_mode: str = "RGB", channel_order: str = "RGB") -> torch.Tensor:
    """Convert PIL / ndarray / tensor to a ``(C, H, W)`` tensor in RGB order (dtype preserved)."""
    if channel_order not in ("RGB", "BGR"):
        raise ValueError(f"channel_order must be 'RGB' or 'BGR', got {channel_order!r}")
    if isinstance(image, torch.Tensor):
        tensor = image
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 3:
            raise ValueError(f"tensor image must be (C, H, W), got {tuple(tensor.shape)}")
    elif type(image).__module__.startswith("numpy"):
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("Install numpy to preprocess ndarray images: pip install numpy") from exc
        array = np.ascontiguousarray(image)
        if array.ndim == 2:
            array = array[:, :, None]
        if array.ndim != 3:
            raise ValueError(f"ndarray image must be (H, W) or (H, W, C), got shape {array.shape}")
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
    elif hasattr(image, "convert") and hasattr(image, "tobytes"):
        pil = image.convert(image_mode)
        width, height = pil.size
        channels = len(pil.getbands())
        buffer = bytearray(pil.tobytes())
        tensor = torch.frombuffer(buffer, dtype=torch.uint8).view(height, width, channels)
        tensor = tensor.permute(2, 0, 1).contiguous()
    else:
        raise TypeError(f"unsupported image type {type(image).__name__}; expected PIL, ndarray or tensor")
    if channel_order == "BGR":
        if tensor.shape[0] != 3:
            raise ValueError("BGR conversion requires 3 channels")
        tensor = tensor.flip(0)
    return tensor


def to_float_unit_range(tensor: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] -> float32 [0, 1]; float inputs are assumed already in [0, 1]."""
    if tensor.dtype == torch.uint8:
        return tensor.float().div_(255.0)
    if tensor.is_floating_point():
        return tensor.float()
    raise TypeError(f"unsupported image dtype {tensor.dtype}")


def resize_shorter_side(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """Resize ``(C, H, W)`` float so the shorter side equals ``size`` (aspect ratio kept)."""
    _, h, w = tensor.shape
    if h <= w:
        new_h, new_w = size, max(1, round(w * size / h))
    else:
        new_h, new_w = max(1, round(h * size / w)), size
    out = F.interpolate(
        tensor.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True
    )
    return out.squeeze(0)


def center_crop(tensor: torch.Tensor, size: int) -> torch.Tensor:
    _, h, w = tensor.shape
    if h < size or w < size:
        raise ValueError(f"cannot center-crop {size}x{size} from {h}x{w}")
    top = (h - size) // 2
    left = (w - size) // 2
    return tensor[:, top : top + size, left : left + size]


def preprocess_image(
    image: Any,
    crop_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    resize_size: Optional[int] = None,
    image_mode: str = "RGB",
    channel_order: str = "RGB",
) -> torch.Tensor:
    """Full deterministic preprocessing; ``resize_size`` defaults to ``crop_size`` (no crop margin)."""
    if crop_size < 1:
        raise ValueError(f"crop_size must be >= 1, got {crop_size}")
    resize_size = crop_size if resize_size is None else resize_size
    if resize_size < crop_size:
        raise ValueError("resize_size must be >= crop_size")
    tensor = to_float_unit_range(image_to_tensor(image, image_mode, channel_order))
    if len(mean) != tensor.shape[0] or len(std) != tensor.shape[0]:
        raise ValueError(f"mean/std length must equal channel count {tensor.shape[0]}")
    if any(s <= 0 for s in std):
        raise ValueError("std values must be > 0")
    tensor = center_crop(resize_shorter_side(tensor, resize_size), crop_size)
    mean_t = torch.tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype).view(-1, 1, 1)
    return (tensor - mean_t) / std_t
