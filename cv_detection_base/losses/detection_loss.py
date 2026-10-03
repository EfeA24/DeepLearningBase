# Purpose: Composes injected detection loss components (box, objectness, classification, ...) into a weighted total plus per-component values, using caller-supplied target assignment.
"""Weighted composition of detection loss components.

This module does NOT perform target assignment (anchor/grid matching). The
caller supplies either:
    - ``compute(component_inputs)`` with already matched tensors, or
    - an ``assign_fn(outputs, targets) -> component_inputs`` at construction,
      which makes the instance usable as ``loss_fn(outputs, targets)`` in
      training/train_epoch.py.

``component_inputs`` maps each component name to the keyword arguments of that
component, e.g.::

    {
      "box": {"pred": matched_pred_boxes, "target": matched_gt_boxes, "normalizer": num_pos},
      "objectness": {"logits": obj_logits, "targets": obj_targets,
                     "positive_mask": pos, "negative_mask": neg},
      "classification": {"logits": matched_cls_logits, "labels": matched_labels},
    }

Return value: {"total": weighted sum (differentiable), "<name>": component value
(detached, for logging), ...}. Every component must return a scalar tensor.

Requires: torch only (components are injected, e.g. from losses/bbox_loss.py,
losses/objectness_loss.py, losses/classification_loss.py).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

import torch
from torch import Tensor, nn

AssignFn = Callable[[Any, Any], Mapping[str, Mapping[str, Any]]]


class DetectionLoss(nn.Module):
    """Weighted sum of named loss components.

    Args:
        components: {name: callable(**kwargs) -> scalar Tensor}. nn.Module
            components are registered as submodules (so buffers move with ``.to``).
        weights: {name: float}; missing names default to 1.0.
        assign_fn: Optional caller-supplied assignment producing component inputs
            from (raw model outputs, list of targets).
    """

    def __init__(
        self,
        components: Mapping[str, Callable[..., Tensor]],
        weights: Optional[Mapping[str, float]] = None,
        assign_fn: Optional[AssignFn] = None,
    ) -> None:
        super().__init__()
        if not components:
            raise ValueError("components must not be empty")
        if "total" in components:
            raise ValueError("'total' is reserved for the combined loss")
        weights = dict(weights or {})
        unknown = set(weights) - set(components)
        if unknown:
            raise ValueError(f"weights given for unknown components: {sorted(unknown)}")
        self.component_names = list(components)
        self._callables: Dict[str, Callable[..., Tensor]] = {}
        self.modules_dict = nn.ModuleDict()
        for name, fn in components.items():
            if isinstance(fn, nn.Module):
                self.modules_dict[name] = fn
            elif not callable(fn):
                raise TypeError(f"component {name!r} is not callable")
            else:
                self._callables[name] = fn
        self.weights = {name: float(weights.get(name, 1.0)) for name in self.component_names}
        self.assign_fn = assign_fn

    def _component(self, name: str) -> Callable[..., Tensor]:
        return self.modules_dict[name] if name in self.modules_dict else self._callables[name]

    def compute(self, component_inputs: Mapping[str, Mapping[str, Any]]) -> Dict[str, Tensor]:
        """Evaluate every component on its matched inputs and combine them."""
        missing = [n for n in self.component_names if n not in component_inputs]
        if missing:
            raise KeyError(f"component_inputs is missing {missing}")
        total: Optional[Tensor] = None
        result: Dict[str, Tensor] = {}
        for name in self.component_names:
            value = self._component(name)(**component_inputs[name])
            if not isinstance(value, Tensor) or value.ndim != 0:
                raise ValueError(f"component {name!r} must return a scalar tensor")
            weighted = value * self.weights[name]
            total = weighted if total is None else total + weighted
            result[name] = value.detach()
        assert total is not None
        result["total"] = total
        return result

    def forward(self, outputs: Any, targets: Any) -> Dict[str, Tensor]:
        """``loss_fn(outputs, targets)`` interface; requires ``assign_fn``."""
        if self.assign_fn is None:
            raise ValueError(
                "DetectionLoss has no assign_fn; call compute(component_inputs) with matched tensors "
                "or construct it with a caller-supplied assign_fn(outputs, targets)"
            )
        return self.compute(self.assign_fn(outputs, targets))
