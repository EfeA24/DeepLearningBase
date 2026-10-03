# Purpose: Stacks transformer encoder blocks (injected block factory or self-contained torch default) over embedded inputs with a padding mask, returning hidden states (B, T, D).
"""Transformer encoder stack.

Input: embedded tokens x (B, T, D) (see models/embeddings/transformer_embedding.py).
Mask: ``attention_mask`` (B, T) bool or integer 0/1, True/1 = valid token, False/0 = padding.
    Passed unchanged to every block as ``block(x, attention_mask=mask)``.
Output:
    ``output_hidden_states=False`` -> last_hidden_state (B, T, D)
    ``output_hidden_states=True``  -> (last_hidden_state, hidden_states) where
        hidden_states = (input, block_1_out, ..., block_N_out) taken before ``final_norm``.
Hidden states at padded positions are not meaningful; pool/score them with the same mask.

Block construction:
    ``TransformerEncoder(block_factory, num_layers)`` calls ``block_factory(layer_index)`` once per
    layer, e.g. ``lambda i: TransformerEncoderBlock(d_model, num_heads, d_ff, norm_first=True)``
    from models/blocks/encoder_block.py.
    ``TransformerEncoder.build_default(...)`` is self-contained: it wraps
    ``torch.nn.TransformerEncoderLayer`` (batch_first) and converts the True=valid mask to PyTorch's
    True=pad ``src_key_padding_mask`` internally.
Pre-norm stacks should pass ``final_norm=nn.LayerNorm(d_model)``.

Requires: torch only (blocks are injected).
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

__all__ = ["TransformerEncoder", "TorchEncoderLayerAdapter"]


class TorchEncoderLayerAdapter(nn.Module):
    """Adapts ``nn.TransformerEncoderLayer`` to the ``block(x, attention_mask=True-is-valid)`` contract.

    Only (B, T) key padding masks are supported.
    """

    def __init__(self, layer: nn.TransformerEncoderLayer) -> None:
        super().__init__()
        if not getattr(layer.self_attn, "batch_first", False):
            raise ValueError("nn.TransformerEncoderLayer must be created with batch_first=True")
        self.layer = layer

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError("TorchEncoderLayerAdapter only supports (B, T) padding masks")
            if attention_mask.dtype.is_floating_point:
                raise TypeError("padding mask must be bool or integer 0/1 (True/1 = valid)")
            valid = attention_mask if attention_mask.dtype == torch.bool else attention_mask != 0
            key_padding_mask = ~valid
        return self.layer(x, src_key_padding_mask=key_padding_mask)


class TransformerEncoder(nn.Module):
    """N encoder blocks applied in sequence, optional final normalization."""

    def __init__(
        self,
        block_factory: Callable[[int], nn.Module],
        num_layers: int,
        final_norm: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        self.layers = nn.ModuleList(block_factory(index) for index in range(num_layers))
        self.final_norm = final_norm

    @classmethod
    def build_default(
        cls,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = False,
        layer_norm_eps: float = 1e-5,
    ) -> "TransformerEncoder":
        """Self-contained encoder built from ``nn.TransformerEncoderLayer`` blocks.

        A final LayerNorm is added only when ``norm_first=True`` (pre-norm).
        """

        def factory(_: int) -> nn.Module:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=True,
                norm_first=norm_first,
            )
            return TorchEncoderLayerAdapter(layer)

        final_norm = nn.LayerNorm(d_model, eps=layer_norm_eps) if norm_first else None
        return cls(factory, num_layers, final_norm=final_norm)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        if x.dim() != 3:
            raise ValueError(f"x must be (B, T, D), got shape {tuple(x.shape)}")
        if attention_mask is not None and attention_mask.dim() == 2 and attention_mask.shape != x.shape[:2]:
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} does not match (B, T)={tuple(x.shape[:2])}"
            )
        hidden_states = [x] if output_hidden_states else None
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
            if hidden_states is not None:
                hidden_states.append(x)
        if self.final_norm is not None:
            x = self.final_norm(x)
        if hidden_states is not None:
            return x, tuple(hidden_states)
        return x
