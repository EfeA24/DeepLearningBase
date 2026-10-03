# Purpose: Parses Pascal VOC XML annotations (1-based pixel boxes, difficult flags) and converts them to the xyxy-pixel detection target schema.
"""Pascal VOC XML parsing.

VOC conventions:
    - ``<bndbox>`` values xmin/ymin/xmax/ymax are 1-based, inclusive pixel
      indices: a box covering pixel columns 1..10 has xmin=1, xmax=10.
    - ``zero_based=True`` (default) converts to the warehouse's continuous
      0-based coordinates via x1 = xmin - 1, y1 = ymin - 1, x2 = xmax, y2 = ymax,
      so width = xmax - xmin + 1 (the Detectron2 convention).
    - ``<difficult>1</difficult>`` marks objects that the official VOC
      evaluation ignores (neither TP nor FN). Keep them with ``keep_difficult=True``
      and pass the "difficult" flags to metrics as ignore masks, or drop them.

Output target:
    {"boxes": FloatTensor (N, 4) xyxy pixels, "labels": LongTensor (N,),
     "difficult": BoolTensor (N,), "orig_size": LongTensor (2,) = (height, width)}

xml.etree is not hardened against malicious XML; parse untrusted files with
defusedxml instead.

Requires: standard library xml.etree (imported lazily); torch.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import torch
from torch import Tensor


def _text(node: Any, tag: str, default: Optional[str] = None) -> Optional[str]:
    child = node.find(tag)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def parse_voc_xml_string(xml_text: str) -> Dict[str, Any]:
    """Parse VOC XML content into a plain dict.

    Returns:
        {"filename": str | None, "width": int, "height": int,
         "objects": [{"name": str, "bbox": (xmin, ymin, xmax, ymax) raw VOC floats,
                      "difficult": bool, "truncated": bool}, ...]}
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    size = root.find("size")
    if size is None:
        raise ValueError("VOC XML is missing <size>")
    width = int(float(_text(size, "width", "0")))
    height = int(float(_text(size, "height", "0")))
    objects: List[Dict[str, Any]] = []
    for obj in root.findall("object"):
        name = _text(obj, "name")
        bndbox = obj.find("bndbox")
        if name is None or bndbox is None:
            raise ValueError("VOC <object> requires <name> and <bndbox>")
        try:
            bbox = tuple(float(_text(bndbox, tag)) for tag in ("xmin", "ymin", "xmax", "ymax"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"object {name!r} has an incomplete or non-numeric <bndbox>") from exc
        objects.append(
            {
                "name": name,
                "bbox": bbox,
                "difficult": _text(obj, "difficult", "0") == "1",
                "truncated": _text(obj, "truncated", "0") == "1",
            }
        )
    return {"filename": _text(root, "filename"), "width": width, "height": height, "objects": objects}


def parse_voc_xml_file(path: str) -> Dict[str, Any]:
    """Read and parse a VOC XML file from an explicit path."""
    from pathlib import Path

    return parse_voc_xml_string(Path(path).read_text(encoding="utf-8"))


def voc_to_target(
    parsed: Mapping[str, Any],
    class_to_id: Union[Mapping[str, int], Callable[[str], Optional[int]]],
    keep_difficult: bool = False,
    zero_based: bool = True,
) -> Dict[str, Tensor]:
    """Convert a parsed VOC dict into the target schema.

    Args:
        parsed: Output of ``parse_voc_xml_string``.
        class_to_id: Name->id mapping, or a callable (e.g. ``ClassMapper.name_to_id``)
            returning None to skip an object. Unknown names in a plain mapping raise KeyError.
        keep_difficult: Keep difficult objects (flagged in "difficult").
        zero_based: Convert VOC 1-based inclusive indices to 0-based continuous coordinates.
    """
    boxes: List[List[float]] = []
    labels: List[int] = []
    difficult: List[bool] = []
    for obj in parsed["objects"]:
        if obj["difficult"] and not keep_difficult:
            continue
        if callable(class_to_id):
            class_id = class_to_id(obj["name"])
            if class_id is None:
                continue
        else:
            if obj["name"] not in class_to_id:
                raise KeyError(f"Unknown VOC class {obj['name']!r}")
            class_id = class_to_id[obj["name"]]
        xmin, ymin, xmax, ymax = obj["bbox"]
        if zero_based:
            xmin, ymin = xmin - 1.0, ymin - 1.0
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"degenerate VOC box for {obj['name']!r}: {obj['bbox']}")
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(int(class_id))
        difficult.append(bool(obj["difficult"]))
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.long),
        "difficult": torch.tensor(difficult, dtype=torch.bool),
        "orig_size": torch.tensor([int(parsed["height"]), int(parsed["width"])], dtype=torch.long),
    }
