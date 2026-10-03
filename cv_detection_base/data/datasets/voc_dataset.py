# Purpose: Adapts Pascal VOC-style XML annotations plus an image directory into the xyxy-pixel detection target schema.
"""Pascal VOC-style detection dataset.

The caller passes explicit directories and the list of sample ids (or an explicit
image-set file such as ``ImageSets/Main/train.txt``). XML parsing uses
``xml.etree`` imported lazily inside the parser.

Coordinates: VOC boxes are 1-based inclusive pixel indices; they are converted
to 0-based continuous xyxy via (xmin - 1, ymin - 1, xmax, ymax).

Item contract: ``(image, target)``
    image:  FloatTensor (3, H, W) RGB in [0, 1] (default loader) or the output
            of the injected ``image_loader``.
    target: {"boxes": (N, 4) float32 xyxy pixels, "labels": (N,) int64 ids
             (index into ``class_names``), "difficult": (N,) bool,
             "image_id": int64 scalar (position in ``ids``), "orig_size": (2,) int64 (H, W)}
    Difficult objects are dropped unless ``keep_difficult=True``; when kept, pass
    "difficult" to the metric as an ignore mask (VOC protocol).

xml.etree is not hardened against malicious XML; use defusedxml for untrusted data.

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


def read_image_set_file(path: str) -> List[str]:
    """Read sample ids (first token of each non-empty line) from an explicit image-set file."""
    from pathlib import Path

    ids = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.append(line.split()[0])
    return ids


class VocDetectionDataset(Dataset):
    """Detection dataset over VOC XML annotations.

    Args:
        image_dir: Directory with ``<id><image_extension>`` images.
        annotation_dir: Directory with ``<id>.xml`` annotations.
        ids: Sample ids (file stems). Use ``read_image_set_file`` to load a split.
        class_names: Ordered class names (e.g. the 20 VOC classes); ids index this list.
        transforms: Optional joint transform ``(image, target) -> (image, target)``.
        image_loader: Callable ``path -> image``; defaults to ``default_image_loader``.
        keep_difficult: Keep objects marked difficult (flagged in target["difficult"]).
        image_extension: Image file extension including the dot.
        unknown_class: "error" or "skip" for object names not in ``class_names``.
    """

    def __init__(
        self,
        image_dir: str,
        annotation_dir: str,
        ids: Sequence[str],
        class_names: Sequence[str],
        transforms: Optional[Callable[[Any, Dict[str, Any]], Tuple[Any, Dict[str, Any]]]] = None,
        image_loader: Optional[Callable[[str], Any]] = None,
        keep_difficult: bool = False,
        image_extension: str = ".jpg",
        unknown_class: str = "error",
    ) -> None:
        from pathlib import Path

        if not class_names:
            raise ValueError("class_names must not be empty")
        if unknown_class not in ("error", "skip"):
            raise ValueError("unknown_class must be 'error' or 'skip'")
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.ids = list(ids)
        self.class_names = list(class_names)
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}
        self.transforms = transforms
        self.image_loader = image_loader or default_image_loader
        self.keep_difficult = keep_difficult
        self.image_extension = image_extension
        self.unknown_class = unknown_class

    def __len__(self) -> int:
        return len(self.ids)

    def load_target(self, index: int) -> Dict[str, Any]:
        """Parse the XML annotation for ``index`` into the target schema."""
        import xml.etree.ElementTree as ET

        sample_id = self.ids[index]
        xml_path = self.annotation_dir / f"{sample_id}.xml"
        root = ET.parse(str(xml_path)).getroot()
        size = root.find("size")
        if size is None:
            raise ValueError(f"{xml_path}: missing <size>")
        width = int(float(size.findtext("width", "0")))
        height = int(float(size.findtext("height", "0")))

        boxes: List[List[float]] = []
        labels: List[int] = []
        difficult: List[bool] = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            is_difficult = (obj.findtext("difficult") or "0").strip() == "1"
            if is_difficult and not self.keep_difficult:
                continue
            if name not in self.class_to_id:
                if self.unknown_class == "skip":
                    continue
                raise KeyError(f"{xml_path}: unknown class {name!r}")
            bndbox = obj.find("bndbox")
            if bndbox is None:
                raise ValueError(f"{xml_path}: object {name!r} has no <bndbox>")
            xmin, ymin, xmax, ymax = (float(bndbox.findtext(tag)) for tag in ("xmin", "ymin", "xmax", "ymax"))
            box = [xmin - 1.0, ymin - 1.0, xmax, ymax]
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"{xml_path}: degenerate box for {name!r}")
            boxes.append(box)
            labels.append(self.class_to_id[name])
            difficult.append(is_difficult)
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
            "difficult": torch.tensor(difficult, dtype=torch.bool),
            "image_id": torch.tensor(index, dtype=torch.long),
            "orig_size": torch.tensor([height, width], dtype=torch.long),
        }

    def __getitem__(self, index: int) -> Tuple[Any, Dict[str, Any]]:
        sample_id = self.ids[index]
        image = self.image_loader(str(self.image_dir / f"{sample_id}{self.image_extension}"))
        target = self.load_target(index)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
