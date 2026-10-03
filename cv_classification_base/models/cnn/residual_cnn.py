# Purpose: Composes injected residual blocks into a small configurable residual image classifier.
"""Small residual CNN classifier.

Architecture:
    stem: Conv3x3(in_channels -> stem_channels) -> BN -> ReLU (stride 1; CIFAR-style, no max-pool)
    stages: for stage i, ``blocks_per_stage[i]`` blocks with ``stage_channels[i]`` output channels;
            the first block of every stage except the first uses stride 2.
    head: global average pooling -> Dropout -> Linear(stage_channels[-1], num_classes)

Block injection: ``block_factory(in_channels, out_channels, stride) -> nn.Module`` must return a
module mapping ``(B, in, H, W) -> (B, out, ceil(H/stride), ceil(W/stride))``. If omitted, the
module lazily imports ``BasicResidualBlock`` from models/blocks/residual_block.py, which requires
the template root on ``sys.path``.

Input: ``(B, in_channels, H, W)``; output: raw logits ``(B, num_classes)``.

Requires: torch; models/blocks/residual_block.py only when ``block_factory`` is not supplied.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
from torch import nn

BlockFactory = Callable[[int, int, int], nn.Module]


def _default_block_factory() -> BlockFactory:
    try:
        from models.blocks.residual_block import BasicResidualBlock
    except ImportError as exc:
        raise ImportError(
            "ResidualCNN default needs models/blocks/residual_block.py importable from the template "
            "root (add the root to sys.path), or pass block_factory explicitly."
        ) from exc

    def make(in_channels: int, out_channels: int, stride: int) -> nn.Module:
        return BasicResidualBlock(in_channels, out_channels, stride=stride)

    return make


class ResidualCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        stem_channels: int = 32,
        stage_channels: Sequence[int] = (32, 64, 128),
        blocks_per_stage: Sequence[int] = (2, 2, 2),
        block_factory: Optional[BlockFactory] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes < 1 or in_channels < 1 or stem_channels < 1:
            raise ValueError("num_classes, in_channels and stem_channels must be >= 1")
        if len(stage_channels) == 0 or len(stage_channels) != len(blocks_per_stage):
            raise ValueError("stage_channels and blocks_per_stage must be non-empty and equal length")
        if any(n < 1 for n in blocks_per_stage) or any(c < 1 for c in stage_channels):
            raise ValueError("stage_channels and blocks_per_stage entries must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        factory = block_factory if block_factory is not None else _default_block_factory()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        stages = []
        prev = stem_channels
        for stage_idx, (ch, n_blocks) in enumerate(zip(stage_channels, blocks_per_stage)):
            blocks = []
            for block_idx in range(n_blocks):
                stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
                blocks.append(factory(prev, ch, stride))
                prev = ch
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(prev, num_classes)
        self.out_channels = prev

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected (B, C, H, W) input, got shape {tuple(x.shape)}")
        return self.stages(self.stem(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.forward_features(x)).flatten(1)
        return self.classifier(self.dropout(x))
