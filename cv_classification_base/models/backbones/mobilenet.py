# Purpose: Adapts torchvision MobileNetV2/V3 (lazy import) as a lightweight classification feature extractor with explicit pretrained and output options.
"""MobileNet backbone adapter.

Supported archs: "mobilenet_v2" (1280 features), "mobilenet_v3_small" (576),
"mobilenet_v3_large" (960). Channel counts refer to the output of ``net.features`` (before the
torchvision classifier MLP, which is dropped).

Output contract:
    return_feature_map=False -> pooled features ``(B, out_features)``
    return_feature_map=True  -> last feature map ``(B, out_features, H/32, W/32)``
Input: ``(B, 3, H, W)`` float RGB normalized like the chosen weights (ImageNet stats for
torchvision weights).

Weights: ``weights=None`` -> random init and no download. Pass an explicit weights spec
("DEFAULT", "IMAGENET1K_V1", or an enum member) to load pretrained weights (torchvision may
download/cache them when the module is constructed).

Requires: torchvision (>= 0.14).
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

MOBILENET_ARCHS = ("mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large")


def _last_conv_out_channels(module: nn.Module) -> int:
    convs = [m for m in module.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("could not infer output channels: no Conv2d found")
    return convs[-1].out_channels


class MobileNetBackbone(nn.Module):
    def __init__(
        self,
        arch: str = "mobilenet_v3_small",
        weights: Optional[Any] = None,
        return_feature_map: bool = False,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if arch not in MOBILENET_ARCHS:
            raise ValueError(f"arch must be one of {MOBILENET_ARCHS}, got {arch!r}")
        try:
            import torchvision.models as tv_models
        except ImportError as exc:
            raise ImportError("Install torchvision to use MobileNetBackbone: pip install torchvision") from exc
        net = tv_models.get_model(arch, weights=weights)
        self.features = net.features
        self.out_features = _last_conv_out_channels(net.features)
        self.return_feature_map = return_feature_map
        self.pool = nn.AdaptiveAvgPool2d(1)
        if not trainable:
            for p in self.features.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.features(x)
        if self.return_feature_map:
            return fmap
        return self.pool(fmap).flatten(1)
