# Purpose: Wraps a torchvision ResNet (lazy import) as a classification feature extractor with the fc layer removed.
"""Thin torchvision ResNet backbone wrapper.

Output contract:
    return_feature_map=False -> pooled features ``(B, out_features)`` (512 for resnet18/34,
                                2048 for resnet50/101/152)
    return_feature_map=True  -> layer4 feature map ``(B, out_features, H/32, W/32)``
Input: ``(B, 3, H, W)`` float RGB normalized with the statistics the weights were trained with
(ImageNet mean/std for torchvision weights).

Weights: ``weights=None`` (default) builds a randomly initialized network and never downloads.
Pretrained weights are used only if the caller passes them explicitly, e.g. ``"DEFAULT"``,
``"IMAGENET1K_V1"`` or a ``torchvision.models.ResNet50_Weights`` member; torchvision may then
download/cache them at construction time (never at import).

Pair with models/heads/classification_head.py: ``ClassificationHead(backbone.out_features, C)``.

Requires: torchvision (>= 0.14 for ``torchvision.models.get_model``).
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

RESNET_ARCHS = ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")


def _import_torchvision_models() -> Any:
    try:
        import torchvision.models as tv_models
    except ImportError as exc:
        raise ImportError("Install torchvision to use ResNetBackbone: pip install torchvision") from exc
    return tv_models


class ResNetBackbone(nn.Module):
    """ResNet trunk (conv1..layer4) plus optional global average pooling; no classifier."""

    def __init__(
        self,
        arch: str = "resnet18",
        weights: Optional[Any] = None,
        return_feature_map: bool = False,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if arch not in RESNET_ARCHS:
            raise ValueError(f"arch must be one of {RESNET_ARCHS}, got {arch!r}")
        tv_models = _import_torchvision_models()
        net = tv_models.get_model(arch, weights=weights)

        self.out_features: int = net.fc.in_features
        self.return_feature_map = return_feature_map
        self.body = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        if not trainable:
            for p in self.body.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.body(x)
        if self.return_feature_map:
            return features
        return self.pool(features).flatten(1)
