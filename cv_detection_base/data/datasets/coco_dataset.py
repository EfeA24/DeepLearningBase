# Purpose: Adapts a local COCO-style JSON annotation file plus image directory into the xyxy-pixel detection target schema (no pycocotools, no downloads).
"""COCO-style detection dataset.

The caller passes an explicit ``image_dir`` and ``annotation_file``; the JSON
is loaded in ``__init__`` (json imported lazily). pycocotools is not required.

Item contract: ``(image, target)``
    image:  FloatTensor (3, H, W) RGB in [0, 1] (default loader) or the output
            of the injected ``image_loader``.
    target: {"boxes": (N, 4) float32 xyxy absolute pixels, "labels": (N,) int64
             contiguous ids, "image_id": int64 scalar (original COCO id),
             "orig_size": (2,) int64 (H, W) from the JSON record,
             "iscrowd": (N,) int64, "area": (N,) float32}

Label mapping: COCO category ids are sparse; they are remapped to contiguous ids
in ascending category-id order (``category_id_to_label``). Use
``label_to_category_id`` to convert predictions back for official evaluation.
Crowd annotations are skipped by default (``skip_crowd=True``).

Requires: Pillow for the default image loader (``pip install pillow``); torch.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


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


class CocoDetectionDataset(Dataset):
    """Detection dataset over a COCO-format JSON file.

    Args:
        image_dir: Directory that ``file_name`` entries are relative to.
        annotation_file: Path to the COCO JSON file.
        transforms: Optional joint transform ``(image, target) -> (image, target)``.
        image_loader: Callable ``path -> image``; defaults to ``default_image_loader``.
        skip_crowd: Drop ``iscrowd=1`` annotations (else keep them flagged).
        drop_images_without_annotations: Exclude images with no usable boxes
            (common for training; keep them for evaluation).
        category_ids: Optional subset of COCO category ids to keep (others skipped).
        min_box_size: Drop boxes whose width or height (pixels) is <= this.
    """

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        transforms: Optional[Callable[[Any, Dict[str, Any]], Tuple[Any, Dict[str, Any]]]] = None,
        image_loader: Optional[Callable[[str], Any]] = None,
        skip_crowd: bool = True,
        drop_images_without_annotations: bool = False,
        category_ids: Optional[Sequence[int]] = None,
        min_box_size: float = 1e-3,
    ) -> None:
        import json
        from pathlib import Path

        self.image_dir = Path(image_dir)
        annotation_path = Path(annotation_file)
        if not annotation_path.is_file():
            raise FileNotFoundError(f"annotation_file not found: {annotation_path}")
        with annotation_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for key in ("images", "annotations", "categories"):
            if key not in data:
                raise ValueError(f"COCO JSON is missing the {key!r} section")

        categories = sorted(data["categories"], key=lambda c: int(c["id"]))
        if category_ids is not None:
            wanted = {int(c) for c in category_ids}
            categories = [c for c in categories if int(c["id"]) in wanted]
        if not categories:
            raise ValueError("no categories selected")
        self.category_id_to_label: Dict[int, int] = {int(c["id"]): i for i, c in enumerate(categories)}
        self.label_to_category_id: Dict[int, int] = {v: k for k, v in self.category_id_to_label.items()}
        self.class_names: List[str] = [str(c.get("name", c["id"])) for c in categories]

        self.transforms = transforms
        self.image_loader = image_loader or default_image_loader
        self.skip_crowd = skip_crowd
        self.min_box_size = min_box_size

        self._annotations: Dict[int, List[Dict[str, Any]]] = {}
        for ann in data["annotations"]:
            self._annotations.setdefault(int(ann["image_id"]), []).append(ann)
        images = sorted(data["images"], key=lambda r: int(r["id"]))
        if drop_images_without_annotations:
            images = [r for r in images if self._usable_annotations(int(r["id"]))]
        self.images: List[Dict[str, Any]] = images

    def _usable_annotations(self, image_id: int) -> List[Dict[str, Any]]:
        usable = []
        for ann in self._annotations.get(image_id, []):
            if self.skip_crowd and int(ann.get("iscrowd", 0)):
                continue
            if int(ann["category_id"]) not in self.category_id_to_label:
                continue
            bbox = ann.get("bbox")
            if bbox is None or len(bbox) != 4 or bbox[2] <= self.min_box_size or bbox[3] <= self.min_box_size:
                continue
            usable.append(ann)
        return usable

    def __len__(self) -> int:
        return len(self.images)

    def load_target(self, index: int) -> Dict[str, Any]:
        """Target for ``index`` without loading the image (useful for metrics/statistics)."""
        record = self.images[index]
        image_id = int(record["id"])
        width, height = int(record["width"]), int(record["height"])
        boxes: List[List[float]] = []
        labels: List[int] = []
        crowd: List[int] = []
        areas: List[float] = []
        for ann in self._usable_annotations(image_id):
            x, y, w, h = (float(v) for v in ann["bbox"])
            x1, y1 = min(max(x, 0.0), width), min(max(y, 0.0), height)
            x2, y2 = min(max(x + w, 0.0), width), min(max(y + h, 0.0), height)
            if x2 - x1 <= self.min_box_size or y2 - y1 <= self.min_box_size:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.category_id_to_label[int(ann["category_id"])])
            crowd.append(int(ann.get("iscrowd", 0)))
            areas.append(float(ann.get("area", w * h)))
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "orig_size": torch.tensor([height, width], dtype=torch.long),
            "iscrowd": torch.tensor(crowd, dtype=torch.long),
            "area": torch.tensor(areas, dtype=torch.float32),
        }

    def __getitem__(self, index: int) -> Tuple[Any, Dict[str, Any]]:
        record = self.images[index]
        image = self.image_loader(str(self.image_dir / record["file_name"]))
        target = self.load_target(index)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
