# Purpose: Provides optional caller-selectable augmentation recipes (light/standard/RandAugment/TrivialAugment/AutoAugment) that preserve single-label classification semantics.
"""Augmentation recipes (PIL-level, inserted before ToTensor in the train pipeline).

``build_augmentation(recipe)`` returns a LIST of torchvision transforms operating on PIL RGB
images; pass it as ``extra_pil_transforms`` to train_transforms.py or compose it yourself.
Recipes:
    "none"           -> []
    "light"          -> small ColorJitter
    "standard"       -> ColorJitter + RandomGrayscale(0.05) + small RandomRotation (rotation_deg)
    "randaugment"    -> RandAugment(num_ops, magnitude)
    "trivialaugment" -> TrivialAugmentWide()
    "autoaugment"    -> AutoAugment(IMAGENET policy)

All recipes keep the label unchanged. Geometric ops (rotation) and color ops are only valid if
the class does not depend on orientation/color (e.g. digits 6/9 or color-coded classes); the
caller is responsible for that choice. Label-mixing methods (MixUp/CutMix) are intentionally not
included because they change the target format.

Requires: torchvision (>= 0.11 for RandAugment/TrivialAugmentWide).
"""
from __future__ import annotations

from typing import Any, List

RECIPES = ("none", "light", "standard", "randaugment", "trivialaugment", "autoaugment")


def build_augmentation(
    recipe: str,
    num_ops: int = 2,
    magnitude: int = 9,
    rotation_deg: float = 10.0,
) -> List[Any]:
    if recipe not in RECIPES:
        raise ValueError(f"recipe must be one of {RECIPES}, got {recipe!r}")
    if recipe == "none":
        return []
    try:
        import torchvision.transforms as T
    except ImportError as exc:
        raise ImportError("Install torchvision to use augmentation recipes: pip install torchvision") from exc
    if recipe == "light":
        return [T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)]
    if recipe == "standard":
        if rotation_deg < 0:
            raise ValueError("rotation_deg must be >= 0")
        return [
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            T.RandomGrayscale(p=0.05),
            T.RandomRotation(rotation_deg),
        ]
    if recipe == "randaugment":
        if num_ops < 1 or not 0 <= magnitude <= 30:
            raise ValueError("RandAugment requires num_ops >= 1 and magnitude in [0, 30]")
        return [T.RandAugment(num_ops=num_ops, magnitude=magnitude)]
    if recipe == "trivialaugment":
        return [T.TrivialAugmentWide()]
    return [T.AutoAugment(T.AutoAugmentPolicy.IMAGENET)]
