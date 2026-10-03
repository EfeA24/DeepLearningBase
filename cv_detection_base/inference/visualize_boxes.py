# Purpose: Draws detection boxes, labels and scores onto a copy of an image (PIL, numpy HWC or torch CHW) with explicit RGB/BGR channel-order handling.
"""Box visualization.

Supported inputs (the input is never modified; a new image of the same kind is returned):
    - PIL.Image (any mode; drawn in RGB)       -> new PIL.Image (RGB)
    - numpy uint8 array (H, W, 3)              -> new numpy array (H, W, 3), same channel order
    - torch Tensor (3, H, W), uint8 [0, 255] or float [0, 1] -> new Tensor of the same dtype/range
``channel_order`` ("RGB" or "BGR") states how numpy/tensor inputs are ordered
(OpenCV images are BGR). Colors are always given as RGB tuples.

Boxes: (N, 4) xyxy absolute pixels in the image's coordinate frame.

Requires: Pillow (``pip install pillow``); numpy only when numpy arrays are passed.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import torch
from torch import Tensor

Color = Tuple[int, int, int]


def color_for_class(class_id: int) -> Color:
    """Deterministic, well-spread RGB color for a class id (golden-ratio hue walk)."""
    hue = (class_id * 0.618033988749895) % 1.0
    h6 = hue * 6.0
    sector = int(h6) % 6
    f = h6 - int(h6)
    v, s = 1.0, 0.75
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    rgb = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][sector]
    return tuple(int(255 * c) for c in rgb)  # type: ignore[return-value]


def _to_pil(image: Any, channel_order: str) -> Tuple[Any, str]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Install Pillow to draw boxes: pip install pillow") from exc
    if isinstance(image, Image.Image):
        return image.convert("RGB"), "pil"
    if isinstance(image, Tensor):
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("tensor images must be (3, H, W)")
        data = image.detach().cpu()
        kind = "tensor_uint8" if data.dtype == torch.uint8 else "tensor_float"
        if data.dtype != torch.uint8:
            data = (data.float().clamp(0, 1) * 255.0).round().to(torch.uint8)
        if channel_order == "BGR":
            data = data.flip(0)
        hwc = data.permute(1, 2, 0).contiguous()
        h, w = hwc.shape[:2]
        try:
            raw = hwc.numpy().tobytes()
        except (RuntimeError, ImportError):
            raw = bytes(hwc.flatten().tolist())
        return Image.frombytes("RGB", (w, h), raw), kind
    try:
        import numpy as np
    except ImportError as exc:
        raise TypeError("unsupported image type; pass a PIL image, a torch tensor or a numpy array") from exc
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("numpy images must be uint8 (H, W, 3)")
        array = image[..., ::-1] if channel_order == "BGR" else image
        return Image.fromarray(np.ascontiguousarray(array), mode="RGB"), "numpy"
    raise TypeError(f"unsupported image type {type(image).__name__}")


def _from_pil(pil_image: Any, kind: str, channel_order: str, original: Any) -> Any:
    if kind == "pil":
        return pil_image
    if kind == "numpy":
        import numpy as np

        array = np.asarray(pil_image, dtype=np.uint8)
        return np.ascontiguousarray(array[..., ::-1]) if channel_order == "BGR" else array.copy()
    w, h = pil_image.size
    data = torch.frombuffer(bytearray(pil_image.tobytes()), dtype=torch.uint8).view(h, w, 3).permute(2, 0, 1)
    if channel_order == "BGR":
        data = data.flip(0)
    data = data.contiguous().to(original.device)
    if kind == "tensor_uint8":
        return data
    return (data.float() / 255.0).to(original.dtype)


def draw_boxes(
    image: Any,
    boxes: Tensor,
    labels: Optional[Tensor] = None,
    scores: Optional[Tensor] = None,
    class_names: Optional[Sequence[str]] = None,
    channel_order: str = "RGB",
    colors: Optional[Sequence[Color]] = None,
    line_width: int = 2,
    score_threshold: Optional[float] = None,
    font: Any = None,
) -> Any:
    """Return a new image with boxes (and optional "name score" captions) drawn.

    Args:
        image: PIL image, numpy uint8 (H, W, 3) or tensor (3, H, W); see module docstring.
        boxes: (N, 4) xyxy pixels.
        labels: Optional (N,) class ids (select color and caption).
        scores: Optional (N,) scores appended to captions.
        class_names: Optional names indexed by label.
        channel_order: "RGB" or "BGR" for numpy/tensor inputs.
        colors: Optional RGB color per class id (falls back to ``color_for_class``).
        line_width: Box outline width in pixels.
        score_threshold: Skip boxes with score < threshold (needs ``scores``).
        font: Optional PIL ImageFont; default PIL bitmap font otherwise.
    """
    if channel_order not in ("RGB", "BGR"):
        raise ValueError("channel_order must be 'RGB' or 'BGR'")
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes must be (N, 4), got {tuple(boxes.shape)}")
    n = boxes.shape[0]
    for name, value in (("labels", labels), ("scores", scores)):
        if value is not None and value.shape[0] != n:
            raise ValueError(f"{name} must have {n} entries")
    from PIL import ImageDraw

    pil_image, kind = _to_pil(image, channel_order)
    canvas = pil_image.copy()
    draw = ImageDraw.Draw(canvas)
    box_list = boxes.detach().cpu().float().tolist()
    label_list = labels.detach().cpu().tolist() if labels is not None else [None] * n
    score_list = scores.detach().cpu().float().tolist() if scores is not None else [None] * n
    for box, label, score in zip(box_list, label_list, score_list):
        if score_threshold is not None and score is not None and score < score_threshold:
            continue
        cls = int(label) if label is not None else 0
        color = tuple(colors[cls]) if colors is not None and cls < len(colors) else color_for_class(cls)
        draw.rectangle(box, outline=color, width=line_width)
        parts = []
        if label is not None:
            parts.append(class_names[cls] if class_names is not None and cls < len(class_names) else str(cls))
        if score is not None:
            parts.append(f"{score:.2f}")
        if parts:
            caption = " ".join(parts)
            x1, y1 = box[0], box[1]
            text_box = draw.textbbox((x1, y1), caption, font=font)
            text_h = text_box[3] - text_box[1]
            top = y1 - text_h - 2 if y1 - text_h - 2 >= 0 else y1
            text_box = draw.textbbox((x1, top), caption, font=font)
            draw.rectangle(text_box, fill=color)
            draw.text((x1, top), caption, fill=(0, 0, 0), font=font)
    return _from_pil(canvas, kind, channel_order, image)
