# Purpose: Stacks seq2seq decoder layers over target embeddings and encoder memory with explicit masks and per-layer incremental caches.
"""Transformer decoder stack.

Inputs:
    x                      (B, T, d_model) target embeddings (token + position added by the
                           caller; with a cache, positions start at ``past_length(past)``)
    memory                 (B, S, d_model) encoder output
    decoder_attention_mask optional (B, past + T), 1/True = real target token
    memory_attention_mask  optional (B, S), 1/True = real source token
    past_key_values        optional sequence (one per layer) of {"self": (k, v), "cross": (k, v)}
Output: DecoderOutput(last_hidden_state (B, T, d_model) after optional final norm,
        past_key_values tuple (if use_cache), hidden_states (input + each layer, pre final norm)).
The decoder returns hidden states, not logits; the output projection lives in the model.

Layers come from ``block_factory(layer_index)`` with the DecoderBlock call contract.

Requires: torch; sibling models/decoder/decoder_block.py when ``block_factory`` is None
(imported lazily in __init__).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

LayerCache = Dict[str, Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class DecoderOutput:
    last_hidden_state: torch.Tensor
    past_key_values: Optional[Tuple[LayerCache, ...]] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None


def past_length(past_key_values: Optional[Sequence[Any]]) -> int:
    """Number of cached target positions (0 when there is no cache)."""
    if not past_key_values:
        return 0
    first = past_key_values[0]
    if not first or first.get("self") is None:
        return 0
    return int(first["self"][0].size(2))


class TransformerDecoder(nn.Module):
    """Stack of decoder blocks with optional final norm."""

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
            from models.decoder.decoder_block import DecoderBlock

            def block_factory(_: int) -> nn.Module:
                return DecoderBlock(
                    d_model, num_heads, d_ff, dropout=dropout, activation=activation,
                    norm_first=norm_first, layer_norm_eps=layer_norm_eps,
                )

        self.layers = nn.ModuleList(block_factory(i) for i in range(num_layers))
        use_final_norm = norm_first if final_norm is None else final_norm
        self.final_norm = nn.LayerNorm(d_model, eps=layer_norm_eps) if use_final_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor],
        decoder_attention_mask: Optional[torch.Tensor] = None,
        memory_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[LayerCache]] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> DecoderOutput:
        if x.dim() != 3:
            raise ValueError(f"x must be (B, T, d_model), got {tuple(x.shape)}")
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError(f"expected {len(self.layers)} layer caches, got {len(past_key_values)}")
        presents = [] if use_cache else None
        hidden_states = [x] if output_hidden_states else None
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, present = layer(
                x, memory, decoder_attention_mask, memory_attention_mask,
                past_key_value=layer_past, use_cache=use_cache,
            )
            if presents is not None:
                presents.append(present)
            if hidden_states is not None:
                hidden_states.append(x)
        return DecoderOutput(
            last_hidden_state=self.final_norm(x),
            past_key_values=tuple(presents) if presents is not None else None,
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
        )
