# Purpose: NER inference over an already initialized token-classification model and tokenizer output: per-token argmax, first-subword word aggregation and BIO decoding into entity spans.
"""NER inference.

Inputs (prepared by the caller, already on the model's device; nothing is moved or loaded here):
    model: called as ``model(**{k: batch[k] for k in input_keys if k in batch})`` returning logits
        (B, T, C) as a tensor, a dict with "logits", or an object with ``.logits``.
    batch: dict with "input_ids" (B, T), "attention_mask" (B, T; 1/True = valid), optional "token_type_ids".
    word_ids_batch: per example, token -> word index or None (special/pad tokens).
    id_to_label: {label_id: "B-PER" | "I-PER" | "O" | ...} (IOB2 / BIO scheme).
Aggregation: first-subword policy (the label of a word is the prediction at its first subword).
Words with no tokens (truncated away) are labeled ``outside_label`` with score 0.
Output: per example a list of ``EntitySpan(label, start_word, end_word (exclusive), score, text)``,
    score = mean of the per-word max softmax probabilities inside the span.
The caller is responsible for ``model.eval()``; a warning is emitted if the model is in training mode.

Requires: torch only.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch import nn

__all__ = ["EntitySpan", "bio_decode", "predict_token_logits", "predict_entities"]


@dataclass(frozen=True)
class EntitySpan:
    """Entity over words [start_word, end_word)."""

    label: str
    start_word: int
    end_word: int
    score: float
    text: Optional[str] = None


def bio_decode(
    word_labels: Sequence[str],
    word_scores: Optional[Sequence[float]] = None,
    words: Optional[Sequence[str]] = None,
    outside_label: str = "O",
    stray_inside: str = "start",
) -> List[EntitySpan]:
    """Decode BIO word labels into spans.

    ``stray_inside`` handles "I-X" not preceded by B-X/I-X: "start" opens a new entity, "ignore" drops it.
    """
    if stray_inside not in ("start", "ignore"):
        raise ValueError(f"stray_inside must be 'start' or 'ignore', got {stray_inside!r}")
    if word_scores is not None and len(word_scores) != len(word_labels):
        raise ValueError("word_scores must align with word_labels")
    if words is not None and len(words) != len(word_labels):
        raise ValueError("words must align with word_labels")

    spans: List[EntitySpan] = []
    current: Optional[str] = None
    start = 0

    def close(end: int) -> None:
        nonlocal current
        if current is not None:
            score = sum(word_scores[start:end]) / (end - start) if word_scores is not None else 1.0
            text = " ".join(words[start:end]) if words is not None else None
            spans.append(EntitySpan(current, start, end, float(score), text))
        current = None

    for index, label in enumerate(word_labels):
        if label == outside_label:
            close(index)
            continue
        if len(label) < 3 or label[1] != "-" or label[0] not in "BI":
            raise ValueError(f"label {label!r} is not in BIO format (B-X / I-X / {outside_label})")
        prefix, entity = label[0], label[2:]
        if prefix == "B" or (current is not None and entity != current):
            close(index)
            if prefix == "I" and stray_inside == "ignore":
                continue
            current, start = entity, index
        elif current is None:
            if stray_inside == "start":
                current, start = entity, index
    close(len(word_labels))
    return spans


def _logits_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping) and "logits" in output:
        return output["logits"]
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    raise TypeError(f"cannot find logits in model output of type {type(output).__name__}")


def predict_token_logits(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    input_keys: Sequence[str] = ("input_ids", "attention_mask", "token_type_ids"),
) -> torch.Tensor:
    """Run the model under inference mode and return token logits (B, T, C)."""
    if model.training:
        warnings.warn("model is in training mode; call model.eval() for deterministic NER inference", stacklevel=2)
    inputs = {key: batch[key] for key in input_keys if key in batch}
    with torch.inference_mode():
        logits = _logits_from_output(model(**inputs))
    if logits.dim() != 3:
        raise ValueError(f"expected token logits (B, T, C), got shape {tuple(logits.shape)}")
    return logits


def predict_entities(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    word_ids_batch: Sequence[Sequence[Optional[int]]],
    id_to_label: Mapping[int, str],
    words_batch: Optional[Sequence[Sequence[str]]] = None,
    outside_label: str = "O",
    stray_inside: str = "start",
) -> List[List[EntitySpan]]:
    """Full pipeline: logits -> first-subword word labels -> BIO spans, per example."""
    logits = predict_token_logits(model, batch)
    if len(word_ids_batch) != logits.shape[0]:
        raise ValueError("word_ids_batch length must equal the batch size")
    probs = torch.softmax(logits.float(), dim=-1)
    best_scores, best_ids = probs.max(dim=-1)
    best_scores, best_ids = best_scores.cpu().tolist(), best_ids.cpu().tolist()

    results: List[List[EntitySpan]] = []
    for row, word_ids in enumerate(word_ids_batch):
        if len(word_ids) > logits.shape[1]:
            raise ValueError(f"example {row}: more word_ids than tokens")
        words = list(words_batch[row]) if words_batch is not None else None
        seen = [w for w in word_ids if w is not None]
        num_words = len(words) if words is not None else ((max(seen) + 1) if seen else 0)
        labels: List[str] = [outside_label] * num_words
        scores: List[float] = [0.0] * num_words
        assigned = [False] * num_words
        for position, word_id in enumerate(word_ids):
            if word_id is None or assigned[word_id]:
                continue
            label_id = best_ids[row][position]
            if label_id not in id_to_label:
                raise KeyError(f"predicted label id {label_id} missing from id_to_label")
            labels[word_id] = id_to_label[label_id]
            scores[word_id] = best_scores[row][position]
            assigned[word_id] = True
        results.append(bio_decode(labels, scores, words, outside_label, stray_inside))
    return results
