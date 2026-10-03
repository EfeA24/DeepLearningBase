# Purpose: Sentence embedding extraction from encoder hidden states via masked mean, [CLS] or masked max pooling with optional L2 normalization, output (B, D).
"""Sentence embeddings.

``pool_hidden_states(hidden (B, T, D), attention_mask (B, T), strategy, normalize) -> (B, D)``
    attention_mask: bool or integer 0/1, True/1 = valid token (None = all valid).
    "mean": average over valid tokens; "cls": first position (requires right padding / [CLS] first);
    "max": element-wise max over valid tokens (rows without valid tokens -> zeros).
    normalize=True applies L2 normalization (cosine similarity then equals the dot product).
``SentenceEmbedder`` wraps an injected encoder (same call contract as tasks/classification/text_classifier.py:
    with ``embedding`` -> encoder(embedding(ids, token_type_ids=...), attention_mask=mask), else
    encoder(input_ids=..., attention_mask=..., token_type_ids=...)) and pools its output.
``embed_batch`` runs it under torch.inference_mode(); no device moves, the caller sets eval mode.

Requires: torch only.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["pool_hidden_states", "SentenceEmbedder", "embed_batch", "cosine_similarity_matrix"]


def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    strategy: str = "mean",
    normalize: bool = False,
) -> torch.Tensor:
    """Pool (B, T, D) token states into (B, D) sentence vectors."""
    if strategy not in ("mean", "cls", "max"):
        raise ValueError(f"strategy must be 'mean', 'cls' or 'max', got {strategy!r}")
    if hidden_states.dim() != 3:
        raise ValueError(f"hidden_states must be (B, T, D), got shape {tuple(hidden_states.shape)}")
    if attention_mask is None:
        valid = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
    else:
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("attention_mask must be (B, T) matching hidden_states")
        if attention_mask.dtype.is_floating_point:
            raise TypeError("attention_mask must be bool or integer 0/1")
        valid = attention_mask != 0

    if strategy == "cls":
        pooled = hidden_states[:, 0]
    elif strategy == "mean":
        weights = valid.unsqueeze(-1).to(hidden_states.dtype)
        pooled = (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
    else:
        filled = hidden_states.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        pooled = filled.max(dim=1).values
        pooled = torch.where(valid.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled))
    return F.normalize(pooled, p=2, dim=-1) if normalize else pooled


def _hidden(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    hidden = getattr(output, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return hidden
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"cannot extract hidden states from {type(output).__name__}")


class SentenceEmbedder(nn.Module):
    """[embedding ->] encoder -> pooling (-> L2 normalize)."""

    def __init__(
        self,
        encoder: nn.Module,
        embedding: Optional[nn.Module] = None,
        strategy: str = "mean",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if strategy not in ("mean", "cls", "max"):
            raise ValueError(f"strategy must be 'mean', 'cls' or 'max', got {strategy!r}")
        self.encoder = encoder
        self.embedding = embedding
        self.strategy = strategy
        self.normalize = normalize

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.embedding is not None:
            output = self.encoder(self.embedding(input_ids, token_type_ids=token_type_ids), attention_mask=attention_mask)
        else:
            kwargs: Dict[str, torch.Tensor] = {"input_ids": input_ids}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            output = self.encoder(**kwargs)
        return pool_hidden_states(_hidden(output), attention_mask, self.strategy, self.normalize)


def embed_batch(embedder: SentenceEmbedder, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Embed one prepared batch under inference mode -> (B, D) on the batch's device."""
    with torch.inference_mode():
        return embedder(
            batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            token_type_ids=batch.get("token_type_ids"),
        )


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(N, D) x (M, D) -> (N, M) cosine similarities."""
    if a.dim() != 2 or b.dim() != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("expected (N, D) and (M, D) tensors with equal D")
    return F.normalize(a, dim=-1, eps=eps) @ F.normalize(b, dim=-1, eps=eps).T
