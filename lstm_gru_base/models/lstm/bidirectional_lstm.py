# Purpose: Wraps a bidirectional nn.LSTM with explicit output and final-state direction-merge policies (concat/sum/mean) for sequence encoders.
"""Bidirectional LSTM wrapper.

Contract:
    - ``x`` ``(B, T, I)`` batch-first; optional ``lengths`` ``(B,)`` for RIGHT-padded batches (packed,
      so the backward direction starts at each sequence's last valid step, not at padding).
    - Raw nn.LSTM outputs are ``(B, T, 2H)`` with ``[..., :H]`` = forward and ``[..., H:]`` = backward.
    - ``merge``: ``"concat"`` -> ``(B, T, 2H)``; ``"sum"``/``"mean"`` -> ``(B, T, H)``.
    - Final states ``(L*2, B, H)`` are reshaped to ``(L, 2, B, H)`` (dim 1: 0 = forward, 1 = backward)
      and merged with the same policy: concat -> ``(L, B, 2H)``, sum/mean -> ``(L, B, H)``.
      The forward final state summarizes the sequence end; the backward one summarizes its start.

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

MERGE_MODES = ("concat", "sum", "mean")


def merge_direction_outputs(outputs: torch.Tensor, hidden_size: int, merge: str) -> torch.Tensor:
    """Merge ``(B, T, 2H)`` bidirectional outputs."""
    if merge not in MERGE_MODES:
        raise ValueError(f"merge must be one of {MERGE_MODES}, got {merge!r}.")
    if merge == "concat":
        return outputs
    fwd, bwd = outputs[..., :hidden_size], outputs[..., hidden_size:]
    return fwd + bwd if merge == "sum" else 0.5 * (fwd + bwd)


def merge_direction_states(state: torch.Tensor, num_layers: int, merge: str) -> torch.Tensor:
    """Merge a ``(L*2, B, H)`` state into ``(L, B, 2H)`` (concat) or ``(L, B, H)`` (sum/mean)."""
    if merge not in MERGE_MODES:
        raise ValueError(f"merge must be one of {MERGE_MODES}, got {merge!r}.")
    if state.shape[0] != num_layers * 2:
        raise ValueError(f"Expected first dim {num_layers * 2}, got {state.shape[0]}.")
    split = state.view(num_layers, 2, *state.shape[1:])
    fwd, bwd = split[:, 0], split[:, 1]
    if merge == "concat":
        return torch.cat([fwd, bwd], dim=-1)
    return fwd + bwd if merge == "sum" else 0.5 * (fwd + bwd)


class BidirectionalLSTM(nn.Module):
    """nn.LSTM(bidirectional=True) with documented merge behaviour."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        merge: str = "concat",
    ) -> None:
        super().__init__()
        if merge not in MERGE_MODES:
            raise ValueError(f"merge must be one of {MERGE_MODES}, got {merge!r}.")
        self.merge = merge
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout,
                            bidirectional=True, batch_first=True)
        self.output_size = 2 * hidden_size if merge == "concat" else hidden_size

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return merged outputs and merged ``(h_n, c_n)``; pass raw ``(L*2, B, H)`` initial state."""
        if lengths is None:
            outputs, (h_n, c_n) = self.lstm(x, state)
        else:
            packed = pack_padded_sequence(x, lengths.detach().to("cpu", torch.long), batch_first=True, enforce_sorted=False)
            packed_out, (h_n, c_n) = self.lstm(packed, state)
            outputs, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])
        merged_out = merge_direction_outputs(outputs, self.hidden_size, self.merge)
        return merged_out, (
            merge_direction_states(h_n, self.num_layers, self.merge),
            merge_direction_states(c_n, self.num_layers, self.merge),
        )
