# Purpose: Selects each sequence's last VALID recurrent output using lengths (not the final padded step) for LSTM/GRU sequence classifiers.
"""Last-valid-step pooling.

Contract:
    - ``outputs``: ``(B, T, H)`` batch-first RNN outputs; ``lengths``: ``(B,)`` int64, each in ``[1, T]``.
    - ``padding_side="right"``: valid steps are ``0..len-1`` -> picks ``outputs[b, len_b - 1]``.
      ``padding_side="left"``: valid steps are ``T-len..T-1`` -> picks ``outputs[b, T - 1]``.
    - Bidirectional outputs (``H = 2 * H_dir``, forward first): the forward half's summary is at the last
      valid step, but the backward half's summary is at the FIRST valid step. ``bidirectional=True``
      gathers each half from its correct position and concatenates them -> ``(B, 2 * H_dir)``.
    - For packed nn.LSTM/nn.GRU, ``h_n[-1]`` (unidirectional) already equals this result.

Requires: torch only.
"""
from __future__ import annotations

import torch
from torch import nn


def _gather_steps(outputs: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    index = steps.view(-1, 1, 1).expand(-1, 1, outputs.shape[-1])
    return outputs.gather(1, index).squeeze(1)


def last_valid_output(
    outputs: torch.Tensor,
    lengths: torch.Tensor,
    padding_side: str = "right",
    bidirectional: bool = False,
) -> torch.Tensor:
    """Return ``(B, H)`` features taken at each sequence's last valid position."""
    if outputs.dim() != 3:
        raise ValueError(f"outputs must be (B, T, H), got {tuple(outputs.shape)}.")
    if padding_side not in ("right", "left"):
        raise ValueError(f"padding_side must be 'right' or 'left', got {padding_side!r}.")
    bsz, steps, hidden = outputs.shape
    lengths = lengths.to(device=outputs.device, dtype=torch.long)
    if lengths.shape != (bsz,):
        raise ValueError(f"lengths must be ({bsz},), got {tuple(lengths.shape)}.")
    if bool((lengths < 1).any()) or bool((lengths > steps).any()):
        raise ValueError(f"lengths must lie in [1, {steps}].")
    if padding_side == "right":
        last = lengths - 1
        first = torch.zeros_like(lengths)
    else:
        last = torch.full_like(lengths, steps - 1)
        first = steps - lengths
    if not bidirectional:
        return _gather_steps(outputs, last)
    if hidden % 2 != 0:
        raise ValueError("bidirectional=True requires an even feature dimension (forward | backward).")
    half = hidden // 2
    forward = _gather_steps(outputs[..., :half], last)
    backward = _gather_steps(outputs[..., half:], first)
    return torch.cat([forward, backward], dim=-1)


class LastHiddenPooling(nn.Module):
    """Module wrapper around ``last_valid_output``."""

    def __init__(self, padding_side: str = "right", bidirectional: bool = False) -> None:
        super().__init__()
        self.padding_side = padding_side
        self.bidirectional = bidirectional

    def forward(self, outputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return last_valid_output(outputs, lengths, self.padding_side, self.bidirectional)
