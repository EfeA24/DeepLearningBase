# Purpose: Maps pooled image features to independent per-label logits for multi-label image classification.
"""Multi-label classification head.

Input: pooled features ``(B, in_features)`` (or ``(B, in_features, H, W)`` with ``spatial_pool``
set to "avg"/"max"). Output: raw logits ``(B, num_labels)``; column ``j`` corresponds to label
``j`` of the caller's fixed class ordering (see data/datasets/multilabel_dataset.py).

No sigmoid in forward. Train with ``BCEWithLogitsLoss``; at inference use
``torch.sigmoid(logits) >= threshold`` with a caller-chosen threshold (scalar or per-label).

Requires: torch.
"""
from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn

SPATIAL_POOLS = ("none", "avg", "max")


class MultilabelHead(nn.Module):
    """Dropout + Linear producing one independent logit per label."""

    def __init__(
        self,
        in_features: int,
        num_labels: int,
        dropout: float = 0.0,
        hidden_features: Optional[int] = None,
        spatial_pool: str = "none",
    ) -> None:
        super().__init__()
        if in_features < 1 or num_labels < 1:
            raise ValueError("in_features and num_labels must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if spatial_pool not in SPATIAL_POOLS:
            raise ValueError(f"spatial_pool must be one of {SPATIAL_POOLS}, got {spatial_pool!r}")
        self.in_features = in_features
        self.num_labels = num_labels
        self.spatial_pool = spatial_pool

        layers = []
        last_in = in_features
        if hidden_features is not None:
            if hidden_features < 1:
                raise ValueError("hidden_features must be >= 1 when given")
            layers += [nn.Linear(in_features, hidden_features), nn.ReLU(inplace=True)]
            last_in = hidden_features
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(last_in, num_labels))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 4:
            if self.spatial_pool == "avg":
                features = features.mean(dim=(2, 3))
            elif self.spatial_pool == "max":
                features = features.amax(dim=(2, 3))
            else:
                raise ValueError("Got a 4D feature map but spatial_pool='none'")
        if features.dim() != 2 or features.shape[1] != self.in_features:
            raise ValueError(
                f"expected features (B, {self.in_features}), got shape {tuple(features.shape)}"
            )
        return self.mlp(features)


def logits_to_multihot(
    logits: torch.Tensor, threshold: Union[float, torch.Tensor] = 0.5
) -> torch.Tensor:
    """Convert logits ``(B, L)`` to a {0,1} float multi-hot tensor via sigmoid >= threshold.

    ``threshold`` is a probability in (0, 1): a scalar or a ``(L,)`` per-label tensor.
    """
    threshold_t = torch.as_tensor(threshold, dtype=logits.dtype, device=logits.device)
    if torch.any(threshold_t <= 0) or torch.any(threshold_t >= 1):
        raise ValueError("threshold must be a probability strictly between 0 and 1")
    return (torch.sigmoid(logits) >= threshold_t).to(logits.dtype)
