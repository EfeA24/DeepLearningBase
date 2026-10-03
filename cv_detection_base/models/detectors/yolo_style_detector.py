# Purpose: Provides an illustrative one-stage detector assembly point (injected backbone, neck and head) following the raw-output detector contract; not a clone of any YOLO release.
"""Illustrative one-stage detector assembly ("YOLO-style" only in spirit).

This is an extension point, NOT a reproduction of any YOLO version: there is
no specific architecture, assigner, loss or training recipe here. It wires
caller-supplied components together:

    images (B, C, H, W)
      -> backbone(images)  -> list of L maps, high-res first (or a mapping; values in order)
      -> neck(features)    -> list of L' maps (optional; identity if None)
      -> head(features)    -> list of raw maps, e.g. (B, A*(5+C), H_i, W_i)

Forward returns the head's RAW output (no activation, decoding or NMS), matching
models/detectors/detector_base.py's contract. Decode with
inference/postprocess.py and train with a caller-supplied assignment + loss.

Example (all components from this template, built by the caller)::

    backbone = BackboneAdapter.from_sequential_children(net, ["layer2", "layer3", "layer4"], [128, 256, 512], [8, 16, 32])
    neck = PAN([128, 256, 512], 128)
    head = DetectionHead(128, num_classes=20, num_anchors=1, num_levels=3)
    model = YoloStyleDetector(backbone, neck, head, num_classes=20, strides=[8, 16, 32])

Requires: torch only (components are injected).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import List, Optional, Sequence

import torch
from torch import Tensor, nn


class YoloStyleDetector(nn.Module):
    """Backbone -> optional neck -> head, returning raw per-level predictions.

    Args:
        backbone: Module mapping images to multi-scale features (list or mapping).
        neck: Optional module mapping a feature list to a feature list.
        head: Module mapping a feature list to raw prediction maps.
        num_classes: Declared number of classes (for downstream decoders).
        strides: Declared output strides per level (high-res first).
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: Optional[nn.Module],
        head: nn.Module,
        num_classes: int,
        strides: Sequence[int],
    ) -> None:
        super().__init__()
        if num_classes <= 0 or not strides:
            raise ValueError("num_classes must be > 0 and strides non-empty")
        self.backbone = backbone
        self.neck = neck if neck is not None else nn.Identity()
        self.head = head
        self.num_classes = int(num_classes)
        self.strides = [int(s) for s in strides]

    def extract_features(self, images: Tensor) -> List[Tensor]:
        """Backbone + neck features (list, high-res first)."""
        if images.ndim != 4:
            raise ValueError(f"images must be (B, C, H, W), got {tuple(images.shape)}")
        features = self.backbone(images)
        if isinstance(features, Mapping):
            features = list(features.values())
        elif isinstance(features, Tensor):
            features = [features]
        features = self.neck(list(features))
        return list(features)

    def forward(self, images: Tensor) -> List[Tensor]:
        features = self.extract_features(images)
        outputs = self.head(features)
        if len(outputs) != len(self.strides):
            raise ValueError(f"head produced {len(outputs)} levels but {len(self.strides)} strides are declared")
        return list(outputs)
