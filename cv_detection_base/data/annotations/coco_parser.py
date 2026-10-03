# Purpose: Converts COCO-style image/category/annotation records ([x, y, w, h] pixel boxes) into the xyxy-pixel detection target schema.
"""COCO record conversion (no pycocotools required).

Input records follow the COCO JSON layout (already loaded by the caller):
    image:      {"id": int, "width": int, "height": int, "file_name": str, ...}
    category:   {"id": int, "name": str, ...}
    annotation: {"image_id": int, "category_id": int, "bbox": [x, y, w, h],
                 "iscrowd": 0/1, "area": float, ...}
    ``bbox`` is in absolute pixels, top-left origin, continuous coordinates.

Output target (warehouse schema):
    {"boxes": FloatTensor (N, 4) xyxy pixels, "labels": LongTensor (N,),
     "image_id": LongTensor scalar, "orig_size": LongTensor (2,) = (height, width),
     "iscrowd": LongTensor (N,), "area": FloatTensor (N,)}
    Labels are contiguous ids from ``build_category_mapping`` (COCO category ids
    are sparse, e.g. 1..90 with gaps); keep the inverse mapping for evaluation.

Requires: torch only.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor


def build_category_mapping(
    categories: Sequence[Mapping[str, Any]],
    start_id: int = 0,
) -> Tuple[Dict[int, int], Dict[int, int], List[str]]:
    """Map sparse COCO category ids to contiguous labels in ascending category-id order.

    Returns:
        (category_id_to_label, label_to_category_id, class_names ordered by label)
    """
    if not categories:
        raise ValueError("categories must not be empty")
    ordered = sorted(categories, key=lambda c: int(c["id"]))
    ids = [int(c["id"]) for c in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate category ids in categories")
    cat_to_label = {cid: start_id + i for i, cid in enumerate(ids)}
    label_to_cat = {label: cid for cid, label in cat_to_label.items()}
    names = [str(c.get("name", c["id"])) for c in ordered]
    return cat_to_label, label_to_cat, names


def index_annotations_by_image(annotations: Sequence[Mapping[str, Any]]) -> Dict[int, List[Mapping[str, Any]]]:
    """Group annotation records by ``image_id``."""
    grouped: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for ann in annotations:
        grouped[int(ann["image_id"])].append(ann)
    return dict(grouped)


def coco_annotations_to_target(
    annotations: Sequence[Mapping[str, Any]],
    image_record: Mapping[str, Any],
    category_id_to_label: Mapping[int, int],
    skip_crowd: bool = True,
    min_box_size: float = 1e-3,
    clip_to_image: bool = True,
    unknown_category: str = "error",
) -> Dict[str, Tensor]:
    """Convert the annotations of one image into the target schema.

    Args:
        annotations: Records belonging to ``image_record`` only.
        image_record: COCO image record with "id", "width", "height".
        category_id_to_label: Mapping from COCO category id to contiguous label.
        skip_crowd: Drop ``iscrowd=1`` regions (RLE crowd regions are not box
            instances). If False they are kept and flagged in "iscrowd" so metrics
            can ignore them.
        min_box_size: Boxes with width or height <= this (after clipping) are dropped.
        clip_to_image: Clip boxes to [0, width] x [0, height].
        unknown_category: "error" or "skip" for category ids missing in the mapping.
    """
    if unknown_category not in ("error", "skip"):
        raise ValueError("unknown_category must be 'error' or 'skip'")
    image_id = int(image_record["id"])
    width, height = int(image_record["width"]), int(image_record["height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"image {image_id} has invalid size {width}x{height}")

    boxes: List[List[float]] = []
    labels: List[int] = []
    crowd: List[int] = []
    areas: List[float] = []
    for ann in annotations:
        if int(ann["image_id"]) != image_id:
            raise ValueError(f"annotation for image {ann['image_id']} passed with image {image_id}")
        is_crowd = int(ann.get("iscrowd", 0))
        if is_crowd and skip_crowd:
            continue
        category_id = int(ann["category_id"])
        if category_id not in category_id_to_label:
            if unknown_category == "skip":
                continue
            raise KeyError(f"category_id {category_id} not in category_id_to_label")
        bbox = ann.get("bbox")
        if bbox is None or len(bbox) != 4:
            raise ValueError(f"annotation {ann.get('id')} has invalid bbox {bbox!r}")
        x, y, w, h = (float(v) for v in bbox)
        x1, y1, x2, y2 = x, y, x + w, y + h
        if clip_to_image:
            x1, x2 = min(max(x1, 0.0), width), min(max(x2, 0.0), width)
            y1, y2 = min(max(y1, 0.0), height), min(max(y2, 0.0), height)
        if (x2 - x1) <= min_box_size or (y2 - y1) <= min_box_size:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(int(category_id_to_label[category_id]))
        crowd.append(is_crowd)
        areas.append(float(ann.get("area", (x2 - x1) * (y2 - y1))))

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.long),
        "image_id": torch.tensor(image_id, dtype=torch.long),
        "orig_size": torch.tensor([height, width], dtype=torch.long),
        "iscrowd": torch.tensor(crowd, dtype=torch.long),
        "area": torch.tensor(areas, dtype=torch.float32),
    }


def target_to_coco_detections(
    boxes_xyxy: Tensor,
    scores: Tensor,
    labels: Tensor,
    image_id: int,
    label_to_category_id: Mapping[int, int],
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Convert predictions back to COCO result records ([x, y, w, h], original category ids)."""
    if not (boxes_xyxy.shape[0] == scores.shape[0] == labels.shape[0]):
        raise ValueError("boxes, scores and labels must have the same length")
    results = []
    for box, score, label in zip(boxes_xyxy.tolist(), scores.tolist(), labels.tolist()):
        if score_threshold is not None and score < score_threshold:
            continue
        x1, y1, x2, y2 = box
        results.append(
            {
                "image_id": int(image_id),
                "category_id": int(label_to_category_id[int(label)]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            }
        )
    return results
