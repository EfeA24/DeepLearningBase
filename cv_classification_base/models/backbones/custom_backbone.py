# Purpose: Provides an interface scaffold for a project-specific image-classification backbone returning feature maps or pooled features.
"""Custom backbone interface (scaffold, not a working network).

Contract for implementations:
    - ``out_features``: channel count C of the returned features.
    - ``forward_features(x)``: ``(B, in_channels, H, W)`` -> feature map ``(B, C, H', W')``.
    - ``forward(x)``: pooled features ``(B, C)`` (global average pooling by default) or the
      feature map when ``return_feature_map=True``.

Copy this file, subclass ``CustomBackbone`` and implement ``forward_features``. The pooling and
shape checks below are real; the network itself is intentionally left unimplemented.

Requires: torch.
"""
from __future__ import annotations

from abc import abstractmethod

import torch
from torch import nn


class CustomBackbone(nn.Module):
    """Base class for project-specific backbones."""

    def __init__(self, out_features: int, return_feature_map: bool = False) -> None:
        super().__init__()
        if out_features < 1:
            raise ValueError(f"out_features must be >= 1, got {out_features}")
        self.out_features = out_features
        self.return_feature_map = return_feature_map

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return a feature map ``(B, out_features, H', W')``."""
        raise NotImplementedError(
            "Implement forward_features: build your layers in __init__ and return a "
            "(B, out_features, H', W') feature map here."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.forward_features(x)
        if fmap.dim() != 4 or fmap.shape[1] != self.out_features:
            raise ValueError(
                f"forward_features must return (B, {self.out_features}, H, W), got {tuple(fmap.shape)}"
            )
        if self.return_feature_map:
            return fmap
        return fmap.mean(dim=(2, 3))
