# Purpose: Reads image paths and single-label targets from a CSV with caller-specified column names for image classification.
"""CSV-indexed image dataset.

CSV schema is explicit: ``path_column`` and ``label_column`` must be passed (no inference).
Relative paths are resolved against ``image_root`` when given.

Labels:
    class_to_idx given -> label cells are class NAMES mapped through it (unknown names raise).
    class_to_idx None  -> label cells must be integer class ids (validated against
        ``num_classes`` when given).

The CSV is read once in ``__init__`` (stdlib ``csv`` imported lazily); images are opened lazily
with Pillow in ``__getitem__`` (or a caller ``loader(path)``).

Requires: Pillow (unless a custom ``loader`` is supplied).
"""
from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional, Tuple

from torch.utils.data import Dataset


def _pil_loader(path: str, image_mode: str) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Install Pillow to read image files: pip install pillow") from exc
    with Image.open(path) as img:
        return img.convert(image_mode)


class CsvImageDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        path_column: str,
        label_column: str,
        image_root: Optional[str] = None,
        class_to_idx: Optional[Mapping[str, int]] = None,
        num_classes: Optional[int] = None,
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[int], Any]] = None,
        delimiter: str = ",",
        encoding: str = "utf-8",
        image_mode: str = "RGB",
        loader: Optional[Callable[[str], Any]] = None,
    ) -> None:
        import csv
        import os

        with open(csv_path, newline="", encoding=encoding) as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            fields = reader.fieldnames or []
            for column in (path_column, label_column):
                if column not in fields:
                    raise KeyError(f"column {column!r} not in CSV header {fields}")
            rows = list(reader)

        samples: List[Tuple[str, int]] = []
        for line_no, row in enumerate(rows, start=2):
            raw_path = (row[path_column] or "").strip()
            raw_label = (row[label_column] or "").strip()
            if not raw_path:
                raise ValueError(f"line {line_no}: empty path")
            if class_to_idx is not None:
                if raw_label not in class_to_idx:
                    raise ValueError(f"line {line_no}: unknown class name {raw_label!r}")
                label = int(class_to_idx[raw_label])
            else:
                try:
                    label = int(raw_label)
                except ValueError as exc:
                    raise ValueError(
                        f"line {line_no}: label {raw_label!r} is not an int; pass class_to_idx for names"
                    ) from exc
            if label < 0 or (num_classes is not None and label >= num_classes):
                raise ValueError(f"line {line_no}: label {label} out of range")
            full_path = os.path.join(image_root, raw_path) if image_root is not None else raw_path
            samples.append((full_path, label))
        if not samples:
            raise ValueError(f"CSV {csv_path!r} contains no rows")

        self.samples = samples
        self.targets = [label for _, label in samples]
        self.class_to_idx = dict(class_to_idx) if class_to_idx is not None else None
        self.transform = transform
        self.target_transform = target_transform
        self.image_mode = image_mode
        self.loader = loader

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, label = self.samples[index]
        image = self.loader(path) if self.loader is not None else _pil_loader(path, self.image_mode)
        if self.transform is not None:
            image = self.transform(image)
        target: Any = label if self.target_transform is None else self.target_transform(label)
        return image, target
