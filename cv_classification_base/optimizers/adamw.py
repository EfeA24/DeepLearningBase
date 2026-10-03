# Purpose: Builds a torch.optim.AdamW optimizer with decoupled weight decay and an opt-in decay/no-decay group splitter for image-classification training.
"""Explicit AdamW construction (template-local copy of common_base/optimizers/adamw.py semantics).

Decoupled weight decay (Loshchilov & Hutter): AdamW shrinks weights directly
(``p -= lr * weight_decay * p``) separately from the adaptive gradient step, unlike Adam's L2 term
that is rescaled by the second-moment estimate.

``build_adamw`` applies ``weight_decay`` to whatever params/groups the caller passes and never
splits groups implicitly. ``split_decay_parameters`` is an explicit opt-in helper that excludes
biases and normalization/embedding weights from decay (common for CNNs with BatchNorm too).

Requires: torch.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, List, Tuple, Type, Union

import torch
from torch import nn

ParamsLike = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

_DEFAULT_NO_DECAY_MODULES: Tuple[Type[nn.Module], ...] = (
    nn.LayerNorm,
    nn.GroupNorm,
    nn.LocalResponseNorm,
    nn.modules.batchnorm._BatchNorm,
    nn.modules.instancenorm._InstanceNorm,
    nn.Embedding,
)


def build_adamw(
    params: ParamsLike,
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    amsgrad: bool = False,
    **kwargs: Any,
) -> torch.optim.AdamW:
    """Create ``torch.optim.AdamW`` (``weight_decay`` default matches torch's 0.01).

    Per-group ``weight_decay`` values inside param-group dicts override the default.
    """
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    if len(betas) != 2 or not all(0.0 <= b < 1.0 for b in betas):
        raise ValueError(f"betas must be two values in [0, 1), got {betas}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
    return torch.optim.AdamW(
        params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, **kwargs
    )


def split_decay_parameters(
    model: nn.Module,
    weight_decay: float,
    no_decay_module_types: Tuple[Type[nn.Module], ...] = _DEFAULT_NO_DECAY_MODULES,
    no_decay_bias: bool = True,
) -> List[Dict[str, Any]]:
    """Opt-in: build decay / no-decay parameter groups.

    Parameters owned by ``no_decay_module_types`` and (if ``no_decay_bias``) any parameter named
    ``bias`` get ``weight_decay=0``. Frozen parameters are skipped; shared parameters are assigned
    once. Empty groups are omitted.
    """
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    seen: set = set()
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            if isinstance(module, no_decay_module_types) or (no_decay_bias and name == "bias"):
                no_decay.append(param)
            else:
                decay.append(param)
    groups: List[Dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups
