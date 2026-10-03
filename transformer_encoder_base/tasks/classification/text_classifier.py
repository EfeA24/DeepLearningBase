# Purpose: Text classification task assembly that wires an injected encoder (plus optional embedding) to an injected classification head and returns logits with an optional loss.
"""Text classifier assembly (example of composition; all parts injected).

Encoder call contract (either form):
    a) ``embedding`` given: hidden = encoder(embedding(input_ids, token_type_ids=...), attention_mask=mask)
       e.g. models/embeddings/transformer_embedding.py + models/encoder/transformer_encoder.py.
    b) ``embedding`` is None: hidden = encoder(input_ids=..., attention_mask=..., token_type_ids=...)
       e.g. a Hugging Face AutoModel. Outputs may be a tensor, a tuple (first item used) or an object
       with ``last_hidden_state``.
Head contract: ``head(hidden (B, T, D), attention_mask) -> logits (B, C)`` (models/heads/classification_head.py).

forward(input_ids (B, T) long, attention_mask (B, T) bool/0-1 with True/1 = valid, token_type_ids,
labels) -> {"logits": (B, C)[, "loss": scalar]}. Logits are raw (no softmax).
Loss is computed only when labels are passed: ``loss_fn(logits, labels)`` if injected, else
cross-entropy with ignore_index=-100 for (B,) long labels, or BCE-with-logits for (B, C) float labels.

Requires: torch; encoder/head modules are injected.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["TextClassifier", "extract_hidden_states"]


def extract_hidden_states(output: Any) -> torch.Tensor:
    """Get (B, T, D) hidden states from a tensor, tuple/list, or object with ``last_hidden_state``."""
    if isinstance(output, torch.Tensor):
        return output
    hidden = getattr(output, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return hidden
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"cannot extract hidden states from encoder output of type {type(output).__name__}")


class TextClassifier(nn.Module):
    """[embedding ->] encoder -> head -> logits."""

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        embedding: Optional[nn.Module] = None,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.head = head
        self.loss_fn = loss_fn

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return encoder hidden states (B, T, D)."""
        if self.embedding is not None:
            embedded = self.embedding(input_ids, token_type_ids=token_type_ids)
            return extract_hidden_states(self.encoder(embedded, attention_mask=attention_mask))
        kwargs: Dict[str, torch.Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        return extract_hidden_states(self.encoder(**kwargs))

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.loss_fn is not None:
            return self.loss_fn(logits, labels)
        if labels.dtype.is_floating_point and labels.shape == logits.shape:
            return F.binary_cross_entropy_with_logits(logits.float(), labels.float())
        return F.cross_entropy(logits.float(), labels.long(), ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        hidden = self.encode(input_ids, attention_mask, token_type_ids)
        logits = self.head(hidden, attention_mask)
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = self.compute_loss(logits, labels)
        return output
