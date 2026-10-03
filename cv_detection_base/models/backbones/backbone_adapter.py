# Purpose: Adapts a caller-supplied backbone nn.Module into a detector-compatible multi-scale feature extractor with declared channels and strides.
"""Multi-scale backbone adapter for detectors.

The adapter never builds or downloads a network: the caller supplies an
already constructed ``nn.Module`` (pretrained weights loaded by the caller if
wanted) and names the intermediate outputs to expose.

Output contract:
    ``forward(images (B, C, H, W))`` -> list of feature maps ordered exactly as
    ``return_layers`` (should be high resolution -> low resolution), map i has
    shape (B, out_channels[i], H / strides[i], W / strides[i]) (rounded by the
    backbone's own padding rules). ``out_channels`` and ``strides`` are
    declarations used by necks/heads; use ``probe_feature_shapes`` once to
    verify them.

Two construction paths:
    ``BackboneAdapter.from_sequential_children(model, ...)``
        Runs ``model``'s direct children in registration order and captures the
        requested ones. Valid only when ``forward`` is equivalent to calling the
        children sequentially (true for e.g. torchvision ResNet up to ``layer4``,
        not for models with branches or functional ops between children).
        Children after the last requested layer (avgpool/fc) are dropped.
    ``BackboneAdapter.from_torchvision(model, return_nodes, ...)``
        Uses ``torchvision.models.feature_extraction.create_feature_extractor``
        (torch.fx tracing, imported lazily) for arbitrary graphs.

Requires: torch; torchvision only for ``from_torchvision``.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


class _SequentialChildrenExtractor(nn.Module):
    """Runs selected direct children sequentially and returns {layer_name: output}."""

    def __init__(self, model: nn.Module, return_layers: Sequence[str]) -> None:
        super().__init__()
        children = OrderedDict(model.named_children())
        missing = [name for name in return_layers if name not in children]
        if missing:
            raise ValueError(f"return_layers {missing} are not direct children of the model; available: {list(children)}")
        remaining = set(return_layers)
        kept: "OrderedDict[str, nn.Module]" = OrderedDict()
        for name, module in children.items():
            kept[name] = module
            remaining.discard(name)
            if not remaining:
                break
        self.body = nn.ModuleDict(kept)
        self.return_layers = list(return_layers)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        outputs: Dict[str, Tensor] = {}
        wanted = set(self.return_layers)
        for name, module in self.body.items():
            x = module(x)
            if name in wanted:
                outputs[name] = x
        return outputs


class BackboneAdapter(nn.Module):
    """Wrap a feature extractor returning ``{name: Tensor}`` into an ordered list of maps.

    Args:
        extractor: Module whose forward returns a mapping containing ``output_names``.
        output_names: Keys to return, in output order (high-res -> low-res).
        out_channels: Declared channels per output.
        strides: Declared total stride per output (input pixels per feature cell).
    """

    def __init__(
        self,
        extractor: nn.Module,
        output_names: Sequence[str],
        out_channels: Sequence[int],
        strides: Sequence[int],
    ) -> None:
        super().__init__()
        if not (len(output_names) == len(out_channels) == len(strides)) or not output_names:
            raise ValueError("output_names, out_channels and strides must be non-empty and equally long")
        if any(c <= 0 for c in out_channels) or any(s <= 0 for s in strides):
            raise ValueError("out_channels and strides must be positive")
        if list(strides) != sorted(strides):
            raise ValueError(f"strides must be non-decreasing (high-res first), got {list(strides)}")
        self.extractor = extractor
        self.output_names = list(output_names)
        self.out_channels = [int(c) for c in out_channels]
        self.strides = [int(s) for s in strides]

    @classmethod
    def from_sequential_children(
        cls,
        model: nn.Module,
        return_layers: Sequence[str],
        out_channels: Sequence[int],
        strides: Sequence[int],
    ) -> "BackboneAdapter":
        """Capture outputs of direct children (see module docstring for the validity condition).

        Example (torchvision ResNet-50, caller-built)::

            BackboneAdapter.from_sequential_children(
                resnet50(weights=None), ["layer2", "layer3", "layer4"], [512, 1024, 2048], [8, 16, 32])
        """
        return cls(_SequentialChildrenExtractor(model, return_layers), return_layers, out_channels, strides)

    @classmethod
    def from_torchvision(
        cls,
        model: nn.Module,
        return_nodes: Mapping[str, str],
        out_channels: Sequence[int],
        strides: Sequence[int],
    ) -> "BackboneAdapter":
        """Use torchvision's fx-based ``create_feature_extractor``.

        Args:
            model: Caller-built model.
            return_nodes: {graph node name: output name}; output order follows this mapping's order.
        """
        try:
            from torchvision.models.feature_extraction import create_feature_extractor
        except ImportError as exc:
            raise ImportError("Install torchvision to use from_torchvision: pip install torchvision") from exc
        extractor = create_feature_extractor(model, return_nodes=dict(return_nodes))
        return cls(extractor, list(return_nodes.values()), out_channels, strides)

    def forward(self, images: Tensor) -> List[Tensor]:
        if images.ndim != 4:
            raise ValueError(f"images must be (B, C, H, W), got {tuple(images.shape)}")
        features = self.extractor(images)
        if not isinstance(features, Mapping):
            raise TypeError("extractor must return a mapping {name: Tensor}")
        missing = [n for n in self.output_names if n not in features]
        if missing:
            raise KeyError(f"extractor did not return {missing}; got {list(features)}")
        return [features[name] for name in self.output_names]


@torch.no_grad()
def probe_feature_shapes(
    backbone: nn.Module,
    image_size: Tuple[int, int],
    in_channels: int = 3,
    device: Optional[torch.device] = None,
) -> List[Tuple[int, int, int, float]]:
    """Run one dummy forward and report the shape of every output map.

    Returns:
        List of (channels, feature_height, feature_width, effective_stride) where
        effective_stride = image_height / feature_height. Temporarily switches
        the backbone to eval mode (restored afterwards) so BatchNorm statistics
        are not updated. Call explicitly; nothing runs at import or construction.
    """
    was_training = backbone.training
    backbone.eval()
    try:
        dummy = torch.zeros(1, in_channels, image_size[0], image_size[1], device=device)
        outputs = backbone(dummy)
        if isinstance(outputs, Mapping):
            outputs = list(outputs.values())
        report = []
        for feature in outputs:
            report.append(
                (int(feature.shape[1]), int(feature.shape[-2]), int(feature.shape[-1]), image_size[0] / feature.shape[-2])
            )
        return report
    finally:
        backbone.train(was_training)
