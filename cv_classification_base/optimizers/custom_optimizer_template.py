# Purpose: Provides a skeleton torch.optim.Optimizer subclass showing param-group, state and step structure for a project-specific optimizer (scaffold).
"""Custom optimizer template (scaffold; the update rule is intentionally unimplemented).

What is real here: hyperparameter validation, ``defaults`` handling, closure support, per-parameter
state initialization and iteration over param groups under ``torch.no_grad()``.
What you must write: ``_update_parameter`` (the actual math). It raises NotImplementedError so the
template can never silently train with a fake rule. No claims about correctness or performance.

Sketch of an implementation (plain SGD with momentum, for orientation)::

    def _update_parameter(self, param, grad, state, group):
        buf = state["momentum_buffer"]
        buf.mul_(group["momentum"]).add_(grad)
        param.add_(buf, alpha=-group["lr"])

Requires: torch.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional

import torch
from torch.optim import Optimizer


class CustomOptimizerTemplate(Optimizer):
    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if momentum < 0:
            raise ValueError(f"momentum must be >= 0, got {momentum}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _init_state(self, param: torch.Tensor, state: Dict[str, Any]) -> None:
        """Create per-parameter buffers lazily on first step (so they live on param's device)."""
        state["step"] = 0
        state["momentum_buffer"] = torch.zeros_like(param, memory_format=torch.preserve_format)

    def _update_parameter(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        state: Dict[str, Any],
        group: Dict[str, Any],
    ) -> None:
        """Apply the in-place update to ``param``. Implement your rule here."""
        raise NotImplementedError(
            "CustomOptimizerTemplate._update_parameter is a scaffold: implement the update rule "
            "(modify `param` in place using `grad`, `state` buffers and `group` hyperparameters)."
        )

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("CustomOptimizerTemplate does not support sparse gradients")
                state = self.state[param]
                if len(state) == 0:
                    self._init_state(param, state)
                state["step"] += 1
                self._update_parameter(param, param.grad, state, group)
        return loss
