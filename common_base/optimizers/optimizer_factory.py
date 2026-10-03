# Purpose: Maps an explicit optimizer name and hyperparameter dict to a constructed PyTorch optimizer, rejecting unknown names.
"""Small explicit optimizer factory.

Supported names: "sgd", "adam", "adamw", "rmsprop", "adagrad" (all torch built-ins), plus an
optional "lion" branch that lazily imports the third-party ``lion-pytorch`` package only when
selected. Hyperparameters are passed through unchanged; there are no hidden defaults beyond
torch's own (except SGD, which requires an explicit ``lr``).

Example::

    optimizer = build_optimizer("adamw", model.parameters(), {"lr": 3e-4, "weight_decay": 0.05})

Requires: torch; optional ``lion-pytorch`` for "lion".
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Dict, Optional, Union

import torch

ParamsLike = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

SUPPORTED_OPTIMIZERS = ("sgd", "adam", "adamw", "rmsprop", "adagrad", "lion")


def build_optimizer(
    name: str,
    params: ParamsLike,
    config: Optional[Mapping[str, Any]] = None,
) -> torch.optim.Optimizer:
    """Construct an optimizer by name.

    Args:
        name: One of ``SUPPORTED_OPTIMIZERS`` (case-insensitive).
        params: Parameters or param-group dicts.
        config: Keyword arguments for the optimizer constructor.

    Raises:
        ValueError: Unknown name, or SGD without ``lr``.
        ImportError: Optional backend not installed.
    """
    key = name.lower().strip()
    kwargs: Dict[str, Any] = dict(config or {})
    if key == "sgd":
        if "lr" not in kwargs:
            raise ValueError("SGD requires an explicit 'lr' in config")
        if kwargs.get("nesterov") and kwargs.get("momentum", 0.0) <= 0:
            raise ValueError("nesterov=True requires momentum > 0")
        return torch.optim.SGD(params, **kwargs)
    if key == "adam":
        return torch.optim.Adam(params, **kwargs)
    if key == "adamw":
        return torch.optim.AdamW(params, **kwargs)
    if key == "rmsprop":
        return torch.optim.RMSprop(params, **kwargs)
    if key == "adagrad":
        return torch.optim.Adagrad(params, **kwargs)
    if key == "lion":
        try:
            from lion_pytorch import Lion
        except ImportError as exc:
            raise ImportError("Install lion-pytorch to use the 'lion' optimizer: pip install lion-pytorch") from exc
        return Lion(params, **kwargs)
    raise ValueError(f"Unknown optimizer {name!r}; supported: {', '.join(SUPPORTED_OPTIMIZERS)}")
