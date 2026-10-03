# Purpose: Maps pooled image features to raw single-label class logits for image-classification models.
"""Single-label classification head.

Input: pooled features ``(B, in_features)``. If ``spatial_pool`` is "avg" or "max", the head also
accepts feature maps ``(B, in_features, H, W)`` and pools them globally first.
Output: raw logits ``(B, num_classes)``. No softmax is applied; use ``logits.softmax(-1)`` for
probabilities and a logits-based loss (e.g. ``F.cross_entropy``) for training.

Optional hidden layer: Linear(in, hidden) -> activation -> dropout -> Linear(hidden, classes).

Requires: torch.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn

SPATIAL_POOLS = ("none", "avg", "max")


class ClassificationHead(nn.Module):
    """Dropout + Linear (optionally with one hidden layer) producing class logits."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        dropout: float = 0.0,
        hidden_features: Optional[int] = None,
        activation_layer: Callable[[], nn.Module] = nn.ReLU,
        spatial_pool: str = "none",
    ) -> None:
        super().__init__()
        if in_features < 1 or num_classes < 1:
            raise ValueError("in_features and num_classes must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if spatial_pool not in SPATIAL_POOLS:
            raise ValueError(f"spatial_pool must be one of {SPATIAL_POOLS}, got {spatial_pool!r}")
        self.in_features = in_features
        self.num_classes = num_classes
        self.spatial_pool = spatial_pool

        layers = []
        if hidden_features is not None:
            if hidden_features < 1:
                raise ValueError("hidden_features must be >= 1 when given")
            layers += [nn.Linear(in_features, hidden_features), activation_layer()]
            last_in = hidden_features
        else:
            last_in = in_features
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(last_in, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 4:
            if self.spatial_pool == "avg":
                features = features.mean(dim=(2, 3))
            elif self.spatial_pool == "max":
                features = features.amax(dim=(2, 3))
            else:
                raise ValueError(
                    "Got a 4D feature map but spatial_pool='none'; pool features first or set "
                    "spatial_pool='avg'/'max'"
                )
        if features.dim() != 2 or features.shape[1] != self.in_features:
            raise ValueError(
                f"expected features (B, {self.in_features}), got shape {tuple(features.shape)}"
            )
        return self.mlp(features)
