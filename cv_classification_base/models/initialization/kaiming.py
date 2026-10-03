# Purpose: Applies Kaiming (He) initialization with an explicit nonlinearity to conv/linear layers of image-classification models.
"""Kaiming initialization.

Supported modules: ``nn.Conv1d/2d/3d``, ``nn.ConvTranspose1d/2d/3d``, ``nn.Linear`` (weights get
Kaiming normal/uniform; biases are zeroed when ``zero_bias``). Normalization layers with affine
parameters (BatchNorm/GroupNorm/InstanceNorm/LayerNorm) are optionally reset to weight=1, bias=0.
Other modules are left untouched.

``nonlinearity`` must match the activation following the layer ("relu" or "leaky_relu"; ``a`` is
the negative slope for leaky_relu). ``mode="fan_out"`` preserves backward-pass variance and is the
common choice for conv nets (torchvision ResNet); "fan_in" preserves forward variance.

Requires: torch.
"""
from __future__ import annotations

import torch
from torch import nn

_WEIGHTED = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
    nn.Linear,
)
_NORMS = (
    nn.modules.batchnorm._BatchNorm,
    nn.GroupNorm,
    nn.modules.instancenorm._InstanceNorm,
    nn.LayerNorm,
)


@torch.no_grad()
def kaiming_init_(
    model: nn.Module,
    nonlinearity: str = "relu",
    mode: str = "fan_out",
    distribution: str = "normal",
    a: float = 0.0,
    zero_bias: bool = True,
    reset_norm_layers: bool = True,
) -> int:
    """Initialize ``model`` in place; return the number of conv/linear modules initialized."""
    if nonlinearity not in ("relu", "leaky_relu"):
        raise ValueError("nonlinearity must be 'relu' or 'leaky_relu' for Kaiming init")
    if mode not in ("fan_in", "fan_out"):
        raise ValueError(f"mode must be 'fan_in' or 'fan_out', got {mode!r}")
    if distribution not in ("normal", "uniform"):
        raise ValueError(f"distribution must be 'normal' or 'uniform', got {distribution!r}")
    init_fn = nn.init.kaiming_normal_ if distribution == "normal" else nn.init.kaiming_uniform_

    count = 0
    for module in model.modules():
        if isinstance(module, _WEIGHTED):
            init_fn(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            if zero_bias and module.bias is not None:
                nn.init.zeros_(module.bias)
            count += 1
        elif reset_norm_layers and isinstance(module, _NORMS):
            weight = getattr(module, "weight", None)
            bias = getattr(module, "bias", None)
            if weight is not None:
                nn.init.ones_(weight)
            if bias is not None:
                nn.init.zeros_(bias)
    return count
