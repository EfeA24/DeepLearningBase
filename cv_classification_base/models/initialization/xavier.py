# Purpose: Applies Xavier (Glorot) initialization with an explicit gain to conv/linear layers of image-classification models.
"""Xavier initialization.

Supported modules: ``nn.Conv1d/2d/3d``, ``nn.ConvTranspose1d/2d/3d``, ``nn.Linear``. Weights get
Xavier uniform/normal with the caller's ``gain``; biases are zeroed when ``zero_bias``. Other modules
are untouched.

``gain`` is explicit. To match an activation use ``nn.init.calculate_gain("tanh")`` etc.
(``xavier_gain_for`` is a thin convenience wrapper). Xavier assumes roughly linear/symmetric
activations; for ReLU networks prefer kaiming.py.

Requires: torch.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

_WEIGHTED = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
    nn.Linear,
)


def xavier_gain_for(nonlinearity: str, param: Optional[float] = None) -> float:
    """Return ``nn.init.calculate_gain(nonlinearity, param)``."""
    return float(nn.init.calculate_gain(nonlinearity, param))


@torch.no_grad()
def xavier_init_(
    model: nn.Module,
    gain: float = 1.0,
    distribution: str = "uniform",
    zero_bias: bool = True,
) -> int:
    """Initialize ``model`` in place; return the number of conv/linear modules initialized."""
    if gain <= 0:
        raise ValueError(f"gain must be > 0, got {gain}")
    if distribution not in ("uniform", "normal"):
        raise ValueError(f"distribution must be 'uniform' or 'normal', got {distribution!r}")
    init_fn = nn.init.xavier_uniform_ if distribution == "uniform" else nn.init.xavier_normal_

    count = 0
    for module in model.modules():
        if isinstance(module, _WEIGHTED):
            init_fn(module.weight, gain=gain)
            if zero_bias and module.bias is not None:
                nn.init.zeros_(module.bias)
            count += 1
    return count
