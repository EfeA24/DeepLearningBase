# Purpose: Wraps a torchvision or timm EfficientNet (lazy import) as a classification feature extractor without its classifier.
"""EfficientNet backbone adapter (torchvision or timm).

Output contract:
    return_feature_map=False -> pooled features ``(B, out_features)`` (1280 for B0)
    return_feature_map=True  -> last feature map ``(B, out_features, H/32, W/32)``
Input: ``(B, 3, H, W)`` float RGB normalized with the statistics expected by the chosen weights.
Each EfficientNet variant has a recommended input resolution (e.g. B0: 224); check its weights.

Sources:
    "torchvision": ``torchvision.models.get_model(arch, weights=weights)``; ``weights`` is None
        (random init, no download) or an explicit weights spec ("DEFAULT", enum member).
    "timm": ``timm.create_model(arch, pretrained=pretrained, num_classes=0)``; downloads only when
        the caller sets ``pretrained=True``. ``weights`` is ignored for timm.

Requires: torchvision (>= 0.14) or timm, depending on ``source``.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn


def _last_conv_out_channels(module: nn.Module) -> int:
    convs = [m for m in module.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("could not infer output channels: no Conv2d found in feature extractor")
    return convs[-1].out_channels


class EfficientNetBackbone(nn.Module):
    """EfficientNet feature extractor returning pooled features or the last feature map."""

    def __init__(
        self,
        arch: str = "efficientnet_b0",
        source: str = "torchvision",
        weights: Optional[Any] = None,
        pretrained: bool = False,
        return_feature_map: bool = False,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.source = source
        self.return_feature_map = return_feature_map
        if source == "torchvision":
            try:
                import torchvision.models as tv_models
            except ImportError as exc:
                raise ImportError("Install torchvision to use this backbone: pip install torchvision") from exc
            if not arch.startswith("efficientnet"):
                raise ValueError(f"expected a torchvision efficientnet_* arch, got {arch!r}")
            net = tv_models.get_model(arch, weights=weights)
            self.features = net.features
            self.out_features = _last_conv_out_channels(net.features)
            self._timm_model = None
        elif source == "timm":
            try:
                import timm
            except ImportError as exc:
                raise ImportError("Install timm to use source='timm': pip install timm") from exc
            net = timm.create_model(arch, pretrained=pretrained, num_classes=0)
            self._timm_model = net
            self.features = None
            self.out_features = int(net.num_features)
        else:
            raise ValueError(f"source must be 'torchvision' or 'timm', got {source!r}")
        self.pool = nn.AdaptiveAvgPool2d(1)
        if not trainable:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._timm_model is not None:
            fmap = self._timm_model.forward_features(x)
        else:
            fmap = self.features(x)
        if self.return_feature_map:
            return fmap
        return self.pool(fmap).flatten(1)
