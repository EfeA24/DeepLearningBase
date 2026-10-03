# Purpose: Provides a dense per-location detection head producing raw box/objectness/class predictions in a documented (B, A*(5+C), H, W) layout.
"""Dense detection head (anchor-based with ``num_anchors`` >= 1; anchor-free when 1).

Output layout per level: Tensor (B, A * (5 + C), H, W), where for anchor a the
channel block ``[a*(5+C) : (a+1)*(5+C)]`` holds, in order:
    0..3  box regression raw values (tx, ty, tw, th)  - meaning defined by the decoder
          (anchor offsets for anchor-based, grid offsets/log-sizes for anchor-free;
          see inference/postprocess.py)
    4     objectness logit
    5..   C class logits (no sigmoid/softmax applied)
All values are raw; activations belong to the loss or post-processing.

``flatten_level_outputs`` converts each level to (B, H*W*A, 5+C) ordered
row-major over (y, x, anchor), which matches models/anchors/anchor_generator.py
anchor ordering, and concatenates levels in input order.

Requires: torch only.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Union

import torch
from torch import Tensor, nn


class DetectionHead(nn.Module):
    """Per-level conv stem + 1x1 prediction conv.

    Args:
        in_channels: Channels of each input level (int if all equal).
        num_levels: Number of input levels (needed when ``in_channels`` is an int).
        num_classes: C (number of foreground classes).
        num_anchors: A anchors per location (1 = anchor-free).
        stem_convs: 3x3 Conv-BN-SiLU layers before the prediction conv (per level, unshared).
        prior_prob: Initial objectness probability used to set the objectness bias
            (-log((1 - p) / p)); stabilizes early training with many negatives.
    """

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        num_classes: int,
        num_anchors: int = 1,
        num_levels: int = 3,
        stem_convs: int = 1,
        prior_prob: float = 0.01,
    ) -> None:
        super().__init__()
        if isinstance(in_channels, int):
            in_channels = [in_channels] * num_levels
        if not in_channels or any(c <= 0 for c in in_channels):
            raise ValueError("in_channels must be positive")
        if num_classes <= 0 or num_anchors <= 0 or stem_convs < 0:
            raise ValueError("num_classes and num_anchors must be > 0, stem_convs >= 0")
        if not 0.0 < prior_prob < 1.0:
            raise ValueError("prior_prob must be in (0, 1)")
        self.in_channels = list(in_channels)
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_outputs = 5 + num_classes

        self.stems = nn.ModuleList()
        self.predictors = nn.ModuleList()
        for channels in self.in_channels:
            layers: List[nn.Module] = []
            for _ in range(stem_convs):
                layers += [
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=True),
                ]
            self.stems.append(nn.Sequential(*layers))
            self.predictors.append(nn.Conv2d(channels, num_anchors * self.num_outputs, kernel_size=1))
        self._init_prediction_bias(prior_prob)

    def _init_prediction_bias(self, prior_prob: float) -> None:
        objectness_bias = -math.log((1.0 - prior_prob) / prior_prob)
        for predictor in self.predictors:
            nn.init.normal_(predictor.weight, std=0.01)
            bias = predictor.bias.detach().view(self.num_anchors, self.num_outputs)
            with torch.no_grad():
                bias.zero_()
                bias[:, 4] = objectness_bias
                bias[:, 5:] = objectness_bias

    def forward(self, features: Sequence[Tensor]) -> List[Tensor]:
        """Return one raw prediction map (B, A*(5+C), H_i, W_i) per input level."""
        if len(features) != len(self.in_channels):
            raise ValueError(f"expected {len(self.in_channels)} feature maps, got {len(features)}")
        outputs = []
        for i, (feat, stem, predictor) in enumerate(zip(features, self.stems, self.predictors)):
            if feat.ndim != 4 or feat.shape[1] != self.in_channels[i]:
                raise ValueError(f"feature {i} must be (B, {self.in_channels[i]}, H, W), got {tuple(feat.shape)}")
            outputs.append(predictor(stem(feat)))
        return outputs


def flatten_level_outputs(level_outputs: Sequence[Tensor], num_anchors: int, num_classes: int) -> Tensor:
    """(B, A*(5+C), H, W) per level -> (B, sum_i H_i*W_i*A, 5+C), ordered (level, y, x, anchor)."""
    num_outputs = 5 + num_classes
    flat = []
    for i, out in enumerate(level_outputs):
        b, ch, h, w = out.shape
        if ch != num_anchors * num_outputs:
            raise ValueError(f"level {i}: expected {num_anchors * num_outputs} channels, got {ch}")
        flat.append(out.view(b, num_anchors, num_outputs, h, w).permute(0, 3, 4, 1, 2).reshape(b, h * w * num_anchors, num_outputs))
    return torch.cat(flat, dim=1)
