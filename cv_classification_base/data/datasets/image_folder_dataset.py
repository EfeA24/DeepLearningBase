# Purpose: Reads class-per-folder image datasets (root/<class>/<image>) with a lazily imported PIL backend for single-label image classification.
"""Class-folder image dataset.

Layout: ``root/<class_name>/**/<image>.<ext>`` (images found recursively inside each class
folder). The directory scan happens in ``__init__`` (never at import); images are opened lazily in
``__getitem__`` with Pillow and converted to ``image_mode`` (default "RGB").

Class mapping:
    class_to_idx=None -> sorted subfolder names map to ``0..K-1``.
    class_to_idx given -> used as-is (share the TRAIN mapping with val/test datasets so ids
        agree). Subfolders not in the mapping raise unless ``ignore_unmapped_dirs=True``;
        mapped classes without a folder raise unless ``allow_missing_classes=True``.

Returns ``(transform(image), target_transform(class_id))``; without a transform the image is a
PIL.Image in ``image_mode``.

Requires: Pillow (``pip install pillow``).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from torch.utils.data import Dataset

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")


def pil_loader(path: str, image_mode: str = "RGB") -> Any:
    """Open ``path`` with Pillow and convert to ``image_mode`` (file handle closed afterwards)."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Install Pillow to read image files: pip install pillow") from exc
    with Image.open(path) as img:
        return img.convert(image_mode)


class ImageFolderClassificationDataset(Dataset):
    def __init__(
        self,
        root: str,
        class_to_idx: Optional[Mapping[str, int]] = None,
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[int], Any]] = None,
        extensions: Sequence[str] = DEFAULT_EXTENSIONS,
        image_mode: str = "RGB",
        ignore_unmapped_dirs: bool = False,
        allow_missing_classes: bool = False,
        loader: Optional[Callable[[str], Any]] = None,
    ) -> None:
        from pathlib import Path

        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError(f"dataset root {root!r} is not a directory")
        exts = tuple(e.lower() for e in extensions)
        class_dirs = sorted(p for p in root_path.iterdir() if p.is_dir())

        if class_to_idx is None:
            mapping: Dict[str, int] = {p.name: i for i, p in enumerate(class_dirs)}
        else:
            mapping = dict(class_to_idx)
            ids = sorted(mapping.values())
            if len(set(ids)) != len(ids) or any(i < 0 for i in ids):
                raise ValueError("class_to_idx values must be unique non-negative ints")
            unmapped = [p.name for p in class_dirs if p.name not in mapping]
            if unmapped and not ignore_unmapped_dirs:
                raise ValueError(f"folders not in class_to_idx: {unmapped}")
            missing = sorted(set(mapping) - {p.name for p in class_dirs})
            if missing and not allow_missing_classes:
                raise ValueError(f"classes in class_to_idx without a folder: {missing}")
        if not mapping:
            raise ValueError(f"no class folders found under {root!r}")

        samples: List[Tuple[str, int]] = []
        for class_dir in class_dirs:
            if class_dir.name not in mapping:
                continue
            label = mapping[class_dir.name]
            for file in sorted(class_dir.rglob("*")):
                if file.is_file() and file.suffix.lower() in exts:
                    samples.append((str(file), label))
        if not samples:
            raise ValueError(f"no images with extensions {exts} found under {root!r}")

        self.root = str(root_path)
        self.class_to_idx = mapping
        self.classes = [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]
        self.samples = samples
        self.targets = [label for _, label in samples]
        self.transform = transform
        self.target_transform = target_transform
        self.image_mode = image_mode
        self.loader = loader

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, label = self.samples[index]
        image = self.loader(path) if self.loader is not None else pil_loader(path, self.image_mode)
        if self.transform is not None:
            image = self.transform(image)
        target: Any = label
        if self.target_transform is not None:
            target = self.target_transform(label)
        return image, target
