# Purpose: Masked-language-model task assembly combining an injected encoder, MLM head and loss, with masking kept as a separately injected step.
"""MLM task assembly.

``MaskedLanguageModel(encoder, mlm_head, embedding=None, loss_fn=None, project_only_masked=False)``
    forward(input_ids (B, T) corrupted ids, attention_mask (B, T) True/1 = valid, token_type_ids,
            labels (B, T) original ids at masked positions, -100 elsewhere)
    -> {"logits": (B, T, V) or (N, V), ["loss": scalar], ["target": (N,) when project_only_masked]}
    Encoder contract: as in tasks/classification/text_classifier.py (embedding + encoder(x, attention_mask=...)
    or encoder(input_ids=..., attention_mask=..., token_type_ids=...)).
    Head contract: ``mlm_head(hidden)`` -> (B, T, V); with ``project_only_masked=True`` the head is
    called as ``mlm_head(hidden, selection_mask=labels != -100)`` -> (N, V) (models/heads/mlm_head.py supports it).
    Default loss: token-count-normalized CE ignoring -100 (zero, graph-connected, when nothing is masked);
    inject losses/mlm_loss.py's ``mlm_loss`` or any ``loss_fn(logits, labels)`` instead.
``mlm_step``: applies an injected ``masking_fn(input_ids, attention_mask) -> (corrupted, labels)``
    (e.g. data/masking/mlm_mask.py ``mask_tokens`` via functools.partial) and runs the model.
Weight tying is explicit: ``tie_mlm_head_to_embedding(head, embedding_weight)``.

Requires: torch; encoder, head, loss and masking are injected.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["MaskedLanguageModel", "mlm_step", "tie_mlm_head_to_embedding"]

IGNORE_INDEX = -100


def _hidden(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    hidden = getattr(output, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return hidden
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"cannot extract hidden states from {type(output).__name__}")


def _default_mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    flat_labels = labels.reshape(-1)
    loss_sum = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), flat_labels, ignore_index=IGNORE_INDEX, reduction="sum"
    )
    count = (flat_labels != IGNORE_INDEX).sum().clamp(min=1)
    return loss_sum / count.to(loss_sum.dtype)


class MaskedLanguageModel(nn.Module):
    """[embedding ->] encoder -> MLM head (-> loss)."""

    def __init__(
        self,
        encoder: nn.Module,
        mlm_head: nn.Module,
        embedding: Optional[nn.Module] = None,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        project_only_masked: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.mlm_head = mlm_head
        self.loss_fn = loss_fn or _default_mlm_loss
        self.project_only_masked = project_only_masked

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.embedding is not None:
            embedded = self.embedding(input_ids, token_type_ids=token_type_ids)
            return _hidden(self.encoder(embedded, attention_mask=attention_mask))
        kwargs: Dict[str, torch.Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        return _hidden(self.encoder(**kwargs))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        hidden = self.encode(input_ids, attention_mask, token_type_ids)
        output: Dict[str, torch.Tensor] = {}
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")
        if self.project_only_masked and labels is not None:
            selection = labels != IGNORE_INDEX
            logits = self.mlm_head(hidden, selection_mask=selection)
            target = labels[selection]
            output["target"] = target
        else:
            logits = self.mlm_head(hidden)
            target = labels
        output["logits"] = logits
        if target is not None:
            output["loss"] = self.loss_fn(logits, target)
        return output


def mlm_step(
    model: MaskedLanguageModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    masking_fn: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    token_type_ids: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Corrupt clean ``input_ids`` with ``masking_fn`` and run the model. Adds "labels" to the output.

    Backward/optimizer steps stay with the caller.
    """
    corrupted, labels = masking_fn(input_ids, attention_mask)
    output = model(corrupted, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
    output["labels"] = labels
    return output


def tie_mlm_head_to_embedding(mlm_head: nn.Module, embedding_weight: nn.Parameter) -> None:
    """Share the head's vocabulary projection with a (V, D) token embedding Parameter."""
    if hasattr(mlm_head, "tie_weights"):
        mlm_head.tie_weights(embedding_weight)
        return
    decoder = getattr(mlm_head, "decoder", None)
    if not isinstance(decoder, nn.Linear) or decoder.weight.shape != embedding_weight.shape:
        raise ValueError("mlm_head needs tie_weights() or a .decoder nn.Linear with weight shape (V, D)")
    decoder.weight = embedding_weight
