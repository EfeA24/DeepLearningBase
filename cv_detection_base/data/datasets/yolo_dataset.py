# Purpose: Reads YOLO-format detection datasets (explicit image/label directories, normalized cxcywh text labels) into the xyxy-pixel target schema.
"""YOLO directory dataset.

Layout (paths passed explicitly by the caller):
    image_dir/<stem>.<ext>      images
    label_dir/<stem>.txt        rows "class xc yc w h" normalized to [0, 1]
A missing label file means "image without objects" unless ``require_labels=True``.

Item contract: ``(image, target)`` with
    image:  FloatTensor (3, H, W) RGB in [0, 1] from the default loader (or
            whatever the injected ``image_loader`` returns; it must expose H, W
            as the last two dims of ``.shape``, or as PIL ``.size``)
    target: {"boxes": (N, 4) float32 xyxy absolute pixels (clipped to the image),
             "labels": (N,) int64 0-based class ids, "image_id": int64 scalar
             (index in sorted file order), "orig_size": (2,) int64 (H, W)}

The file scan happens in ``__init__`` (never at import). Self-contained: the
YOLO row parsing duplicates data/annotations/yolo_parser.py on purpose.

Requires: Pillow for the default image loader (``pip install pillow``); torch.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

DEFAULT_IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def default_image_loader(path: str) -> Tensor:
    """Load an image file as FloatTensor (3, H, W), RGB, [0, 1]. EXIF orientation is NOT applied."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Install Pillow to use the default image loader: pip install pillow") from exc
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        data = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
    return data.view(height, width, 3).permute(2, 0, 1).float().div_(255.0)


def _image_hw(image: Any) -> Tuple[int, int]:
    if hasattr(image, "shape") and len(image.shape) >= 2:
        return int(image.shape[-2]), int(image.shape[-1])
    if hasattr(image, "size") and isinstance(image.size, tuple):
        width, height = image.size
        return int(height), int(width)
    raise TypeError("image_loader must return a tensor (..., H, W) or a PIL-like image with .size")


class YoloDetectionDataset(Dataset):
    """Detection dataset over a YOLO image/label directory pair.

    Args:
        image_dir: Directory containing images (non-recursive).
        label_dir: Directory containing ``<stem>.txt`` label files.
        class_names: Ordered class names; label ids index into this list.
        transforms: Optional joint transform ``(image, target) -> (image, target)``.
        image_loader: Callable ``path -> image``; defaults to ``default_image_loader``.
        image_extensions: Lower-case extensions treated as images.
        require_labels: Raise if an image has no label file (else treat as empty).
        tolerance: Allowed slack outside [0, 1] for normalized coordinates.
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        class_names: Sequence[str],
        transforms: Optional[Callable[[Any, Dict[str, Any]], Tuple[Any, Dict[str, Any]]]] = None,
        image_loader: Optional[Callable[[str], Any]] = None,
        image_extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
        require_labels: bool = False,
        tolerance: float = 1e-6,
    ) -> None:
        from pathlib import Path

        if not class_names:
            raise ValueError("class_names must not be empty")
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"image_dir does not exist: {self.image_dir}")
        if not self.label_dir.is_dir():
            raise FileNotFoundError(f"label_dir does not exist: {self.label_dir}")
        self.class_names = list(class_names)
        self.transforms = transforms
        self.image_loader = image_loader or default_image_loader
        self.tolerance = tolerance
        extensions = {e.lower() for e in image_extensions}
        self.image_paths: List[Any] = sorted(
            p for p in self.image_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions
        )
        if require_labels:
            missing = [p.name for p in self.image_paths if not self._label_path(p).is_file()]
            if missing:
                raise FileNotFoundError(f"{len(missing)} images have no label file, e.g. {missing[:5]}")

    def _label_path(self, image_path: Any) -> Any:
        return self.label_dir / f"{image_path.stem}.txt"

    def __len__(self) -> int:
        return len(self.image_paths)

    def _parse_labels(self, label_path: Any) -> Tuple[Tensor, Tensor]:
        if not label_path.is_file():
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, 4, dtype=torch.float32)
        labels: List[int] = []
        boxes: List[List[float]] = []
        tol = self.tolerance
        for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"{label_path}:{line_number}: expected 'class xc yc w h', got {len(parts)} values")
            class_value = float(parts[0])
            values = [float(v) for v in parts[1:]]
            if class_value != int(class_value) or not 0 <= int(class_value) < len(self.class_names):
                raise ValueError(f"{label_path}:{line_number}: invalid class id {parts[0]!r}")
            if any(v < -tol or v > 1.0 + tol for v in values) or values[2] <= 0 or values[3] <= 0:
                raise ValueError(f"{label_path}:{line_number}: coordinates must be normalized to [0, 1] with w, h > 0")
            labels.append(int(class_value))
            boxes.append(values)
        return torch.tensor(labels, dtype=torch.long), torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)

    def __getitem__(self, index: int) -> Tuple[Any, Dict[str, Any]]:
        image_path = self.image_paths[index]
        image = self.image_loader(str(image_path))
        height, width = _image_hw(image)
        labels, cxcywh = self._parse_labels(self._label_path(image_path))
        cx, cy, w, h = cxcywh.unbind(-1)
        boxes = torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)
        boxes = boxes * boxes.new_tensor([width, height, width, height])
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
        target: Dict[str, Any] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(index, dtype=torch.long),
            "orig_size": torch.tensor([height, width], dtype=torch.long),
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
