# Purpose: Wraps a bidirectional nn.GRU with explicit output and final-state direction-merge policies (concat/sum/mean) for sequence encoders.
"""Bidirectional GRU wrapper.

Contract:
    - ``x`` ``(B, T, I)`` batch-first; optional ``lengths`` ``(B,)`` for RIGHT-padded batches (packed, so the
      backward direction starts at each sequence's last valid step).
    - Raw outputs ``(B, T, 2H)``: ``[..., :H]`` forward, ``[..., H:]`` backward.
    - ``merge``: ``"concat"`` -> ``(B, T, 2H)``; ``"sum"``/``"mean"`` -> ``(B, T, H)``.
    - ``h_n`` ``(L*2, B, H)`` is viewed as ``(L, 2, B, H)`` (0 = forward, 1 = backward) and merged with the
      same policy: concat -> ``(L, B, 2H)``, sum/mean -> ``(L, B, H)``.

Requires: torch only.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

MERGE_MODES = ("concat", "sum", "mean")


def _merge(fwd: torch.Tensor, bwd: torch.Tensor, merge: str) -> torch.Tensor:
    if merge == "concat":
        return torch.cat([fwd, bwd], dim=-1)
    if merge == "sum":
        return fwd + bwd
    if merge == "mean":
        return 0.5 * (fwd + bwd)
    raise ValueError(f"merge must be one of {MERGE_MODES}, got {merge!r}.")


def merge_direction_outputs(outputs: torch.Tensor, hidden_size: int, merge: str) -> torch.Tensor:
    """Merge ``(B, T, 2H)`` bidirectional outputs."""
    return _merge(outputs[..., :hidden_size], outputs[..., hidden_size:], merge)


def merge_direction_states(h_n: torch.Tensor, num_layers: int, merge: str) -> torch.Tensor:
    """Merge ``(L*2, B, H)`` into ``(L, B, 2H)`` (concat) or ``(L, B, H)`` (sum/mean)."""
    if h_n.shape[0] != num_layers * 2:
        raise ValueError(f"Expected first dim {num_layers * 2}, got {h_n.shape[0]}.")
    split = h_n.view(num_layers, 2, *h_n.shape[1:])
    return _merge(split[:, 0], split[:, 1], merge)


class BidirectionalGRU(nn.Module):
    """nn.GRU(bidirectional=True) with documented merge behaviour."""

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
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, dropout=dropout,
                          bidirectional=True, batch_first=True)
        self.output_size = 2 * hidden_size if merge == "concat" else hidden_size

    def forward(
        self,
        x: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return merged outputs and merged ``h_n``; pass raw ``(L*2, B, H)`` initial state."""
        if lengths is None:
            outputs, h_n = self.gru(x, h_0)
        else:
            packed = pack_padded_sequence(x, lengths.detach().to("cpu", torch.long), batch_first=True, enforce_sorted=False)
            packed_out, h_n = self.gru(packed, h_0)
            outputs, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])
        return merge_direction_outputs(outputs, self.hidden_size, self.merge), merge_direction_states(
            h_n, self.num_layers, self.merge
        )
