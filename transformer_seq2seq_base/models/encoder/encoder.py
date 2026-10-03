# Purpose: Stacks seq2seq encoder layers over source embeddings and returns encoder memory (B,S,D) plus optional intermediate hidden states.
"""Transformer encoder stack.

Input:  x (B, S, d_model) source embeddings (token + position already added by the caller),
        attention_mask (B, S) 1/True = real token, 0/False = pad.
Output: EncoderOutput(last_hidden_state (B, S, d_model) = memory after the optional final
        norm, hidden_states = (input, layer_1_out, ..., layer_N_out) when requested;
        hidden_states are taken BEFORE the final norm).

Layers come from ``block_factory(layer_index) -> nn.Module`` with the contract
``block(x, attention_mask) -> x``; default builds models/encoder/encoder_block.py blocks.
``final_norm`` defaults to ``norm_first`` (pre-norm stacks need a closing LayerNorm).

Requires: torch; sibling models/encoder/encoder_block.py when ``block_factory`` is None
(imported lazily in __init__; it in turn needs attention and feed-forward siblings).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class EncoderOutput:
    last_hidden_state: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None


class TransformerEncoder(nn.Module):
    """Stack of encoder blocks with optional final norm."""

    def __init__(
        self,
        num_layers: int,
        *,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_first: bool = True,
        final_norm: Optional[bool] = None,
        layer_norm_eps: float = 1e-5,
        block_factory: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if block_factory is None:
            from models.encoder.encoder_block import EncoderBlock

            def block_factory(_: int) -> nn.Module:
                return EncoderBlock(
                    d_model, num_heads, d_ff, dropout=dropout, activation=activation,
                    norm_first=norm_first, layer_norm_eps=layer_norm_eps,
                )

        self.layers = nn.ModuleList(block_factory(i) for i in range(num_layers))
        use_final_norm = norm_first if final_norm is None else final_norm
        self.final_norm = nn.LayerNorm(d_model, eps=layer_norm_eps) if use_final_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> EncoderOutput:
        if x.dim() != 3:
            raise ValueError(f"x must be (B, S, d_model), got {tuple(x.shape)}")
        hidden_states = [x] if output_hidden_states else None
        for layer in self.layers:
            x = layer(x, attention_mask)
            if hidden_states is not None:
                hidden_states.append(x)
        return EncoderOutput(
            last_hidden_state=self.final_norm(x),
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
        )
