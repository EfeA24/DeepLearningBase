# Purpose: Defines a minimal base abstraction for caller-owned training loops in any PyTorch project.
"""Minimal trainer protocol and base class.

This is deliberately thin: the caller constructs the model, optimizer, data loaders,
loss function and device and passes them in. Subclasses implement ``train_step`` and
``eval_step`` which receive one batch and return a dict of Python floats plus the
number of samples (or tokens) the batch represents, so epoch means can be weighted
correctly. No dataset/model construction, logging service, or checkpoint policy lives here.

Contracts:
    - ``train_step(batch) -> StepOutput``: performs forward/backward/optimizer update.
    - ``eval_step(batch) -> StepOutput``: forward only; called under ``torch.inference_mode()``.
    - ``StepOutput.metrics`` values are per-sample means for that batch; ``StepOutput.weight``
      is the batch size (or token count) used to weight the epoch average.

Requires: torch.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, runtime_checkable

import torch
from torch import nn


@dataclass
class StepOutput:
    """Result of a single train/eval step.

    Attributes:
        metrics: Mapping of metric name to the batch-mean value (Python float).
        weight: Number of samples/tokens the batch represents (> 0).
    """

    metrics: Dict[str, float] = field(default_factory=dict)
    weight: float = 1.0


@runtime_checkable
class TrainerProtocol(Protocol):
    """Structural interface any trainer-like object can satisfy."""

    def train_step(self, batch: Any) -> StepOutput: ...

    def eval_step(self, batch: Any) -> StepOutput: ...


class TrainerBase(ABC):
    """Base class providing weighted epoch loops around caller-defined steps.

    Args:
        model: Already constructed model, already moved to ``device`` by the caller.
        device: Device batches should be moved to inside the step implementations.
    """

    def __init__(self, model: nn.Module, device: torch.device | str) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
        self.model = model
        self.device = torch.device(device)

    @abstractmethod
    def train_step(self, batch: Any) -> StepOutput:
        """Run forward, backward and optimizer update for one batch."""

    @abstractmethod
    def eval_step(self, batch: Any) -> StepOutput:
        """Run a forward pass for one batch without mutating training state."""

    def train_epoch(self, loader: Iterable[Any]) -> Dict[str, float]:
        """Run ``train_step`` over ``loader`` and return weighted metric means."""
        self.model.train()
        return self._run(loader, self.train_step)

    def evaluate(self, loader: Iterable[Any]) -> Dict[str, float]:
        """Run ``eval_step`` over ``loader`` in eval mode under inference mode."""
        self.model.eval()
        with torch.inference_mode():
            return self._run(loader, self.eval_step)

    @staticmethod
    def _run(loader: Iterable[Any], step_fn) -> Dict[str, float]:
        sums: Dict[str, float] = {}
        weights: Dict[str, float] = {}
        for batch in loader:
            out = step_fn(batch)
            if not isinstance(out, StepOutput):
                raise TypeError(f"step must return StepOutput, got {type(out).__name__}")
            if out.weight <= 0:
                raise ValueError(f"StepOutput.weight must be > 0, got {out.weight}")
            for name, value in out.metrics.items():
                sums[name] = sums.get(name, 0.0) + float(value) * out.weight
                weights[name] = weights.get(name, 0.0) + out.weight
        return {name: sums[name] / weights[name] for name in sums}
