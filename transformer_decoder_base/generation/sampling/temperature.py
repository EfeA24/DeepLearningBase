# Purpose: Applies validated temperature scaling to next-token logits for decoder LM sampling.
"""Temperature scaling: ``logits / temperature``.

Contract: logits (B, V) or (..., V) raw logits -> scaled logits, same shape. ``-inf`` entries
(already filtered tokens) stay ``-inf``.

temperature must be > 0. ``temperature == 0`` is REJECTED with a ValueError (dividing by zero
is undefined); use greedy decoding (argmax) instead — that is the limit of temperature -> 0.
temperature < 1 sharpens, > 1 flattens; 1.0 is a no-op. Apply temperature BEFORE top-k/top-p
when you want the filters to see the scaled distribution (the HF default order).

Processor contract used by generation/generate.py: ``processor(input_ids, logits) -> logits``.

Requires: torch.
"""
from __future__ import annotations

import math

import torch


def _validate_temperature(temperature: float) -> None:
    if not isinstance(temperature, (int, float)) or math.isnan(temperature) or math.isinf(temperature):
        raise ValueError(f"temperature must be a finite number, got {temperature!r}")
    if temperature <= 0:
        raise ValueError(
            f"temperature must be > 0, got {temperature}; use greedy decoding for temperature 0"
        )


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return ``logits / temperature`` (no-op for 1.0)."""
    _validate_temperature(temperature)
    if temperature == 1.0:
        return logits
    return logits / temperature


class TemperatureLogitsProcessor:
    """Logits processor wrapper around :func:`apply_temperature`."""

    def __init__(self, temperature: float) -> None:
        _validate_temperature(temperature)
        self.temperature = float(temperature)

    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return apply_temperature(logits, self.temperature)
