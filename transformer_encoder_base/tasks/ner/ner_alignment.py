# Purpose: Aligns word-level NER/POS labels to subword tokens via tokenizer word_ids (explicit first-subword or all-subword policy) and aggregates subword predictions back to words.
"""Word <-> subword label alignment for token classification.

``word_ids``: one entry per token, the index of the source word or None for special/pad tokens
(Hugging Face fast tokenizers: ``encoding.word_ids(i)``; reference tokenizer:
``WordPieceTokenizer.encode_words(words).word_ids``).

Label alignment policies (``align_labels_to_tokens``):
    "first": the first subword of each word gets the word label; other subwords and special tokens
             get ``ignore_index`` (-100). Standard for BERT NER fine-tuning/evaluation.
    "all":   every subword gets a label. With ``continuation_label_map`` (e.g. B-PER -> I-PER, see
             ``build_bio_continuation_map``) non-first subwords use the mapped label so BIO stays valid.
    Special tokens always get ``ignore_index``.

Prediction aggregation (subword -> word):
    ``aggregate_token_predictions`` (int ids, first-subword policy) and
    ``aggregate_token_scores`` ((T, C) scores; "first" / "mean" / "max" over each word's subwords;
    pass probabilities when averaging so the semantics are clear).
    Words without any token (e.g. truncated away) are reported as missing (None / present=False).

Requires: torch only.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

__all__ = [
    "align_labels_to_tokens",
    "align_labels_batch",
    "build_bio_continuation_map",
    "aggregate_token_predictions",
    "aggregate_token_scores",
]

WordIds = Sequence[Optional[int]]


def build_bio_continuation_map(label_to_id: Mapping[str, int]) -> Dict[int, int]:
    """Map id("B-X") -> id("I-X") for every X where both exist."""
    mapping: Dict[int, int] = {}
    for name, index in label_to_id.items():
        if name.startswith("B-") and f"I-{name[2:]}" in label_to_id:
            mapping[index] = label_to_id[f"I-{name[2:]}"]
    return mapping


def align_labels_to_tokens(
    word_ids: WordIds,
    word_labels: Sequence[int],
    policy: str = "first",
    ignore_index: int = -100,
    continuation_label_map: Optional[Mapping[int, int]] = None,
) -> List[int]:
    """Return token-level labels (len == len(word_ids)) under the chosen policy."""
    if policy not in ("first", "all"):
        raise ValueError(f"policy must be 'first' or 'all', got {policy!r}")
    labels: List[int] = []
    previous: Optional[int] = None
    for word_id in word_ids:
        if word_id is None:
            labels.append(ignore_index)
            previous = None
            continue
        if not 0 <= word_id < len(word_labels):
            raise ValueError(f"word id {word_id} out of range for {len(word_labels)} word labels")
        is_first = word_id != previous
        label = word_labels[word_id]
        if is_first:
            labels.append(label)
        elif policy == "first":
            labels.append(ignore_index)
        elif continuation_label_map is not None and label != ignore_index:
            labels.append(continuation_label_map.get(label, label))
        else:
            labels.append(label)
        previous = word_id
    return labels


def align_labels_batch(
    word_ids_batch: Sequence[WordIds],
    word_labels_batch: Sequence[Sequence[int]],
    policy: str = "first",
    ignore_index: int = -100,
    continuation_label_map: Optional[Mapping[int, int]] = None,
) -> List[List[int]]:
    """Batch version of ``align_labels_to_tokens`` (lists stay unpadded)."""
    if len(word_ids_batch) != len(word_labels_batch):
        raise ValueError("word_ids_batch and word_labels_batch must have the same length")
    return [
        align_labels_to_tokens(w, l, policy, ignore_index, continuation_label_map)
        for w, l in zip(word_ids_batch, word_labels_batch)
    ]


def _num_words(word_ids: WordIds, num_words: Optional[int]) -> int:
    seen = [w for w in word_ids if w is not None]
    inferred = (max(seen) + 1) if seen else 0
    if num_words is None:
        return inferred
    if num_words < inferred:
        raise ValueError(f"num_words={num_words} smaller than max word id + 1 = {inferred}")
    return num_words


def aggregate_token_predictions(
    word_ids: WordIds,
    token_predictions: Sequence[int],
    num_words: Optional[int] = None,
) -> List[Optional[int]]:
    """First-subword aggregation of token label ids -> per-word label ids (None = word has no token)."""
    if len(word_ids) != len(token_predictions):
        raise ValueError("word_ids and token_predictions must have the same length")
    result: List[Optional[int]] = [None] * _num_words(word_ids, num_words)
    for word_id, prediction in zip(word_ids, token_predictions):
        if word_id is not None and result[word_id] is None:
            result[word_id] = int(prediction)
    return result


def aggregate_token_scores(
    word_ids: WordIds,
    token_scores: torch.Tensor,
    policy: str = "first",
    num_words: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate (T, C) token scores to (W, C) word scores.

    Returns (word_scores (W, C), present (W,) bool). Missing words have zero scores and present=False.
    """
    if policy not in ("first", "mean", "max"):
        raise ValueError(f"policy must be 'first', 'mean' or 'max', got {policy!r}")
    if token_scores.dim() != 2 or token_scores.shape[0] != len(word_ids):
        raise ValueError("token_scores must be (T, C) with T == len(word_ids)")
    total = _num_words(word_ids, num_words)
    word_scores = torch.zeros(total, token_scores.shape[1], dtype=token_scores.dtype, device=token_scores.device)
    present = torch.zeros(total, dtype=torch.bool, device=token_scores.device)
    groups: Dict[int, List[int]] = {}
    for position, word_id in enumerate(word_ids):
        if word_id is not None:
            groups.setdefault(word_id, []).append(position)
    for word_id, positions in groups.items():
        rows = token_scores[positions]
        if policy == "first":
            word_scores[word_id] = rows[0]
        elif policy == "mean":
            word_scores[word_id] = rows.mean(dim=0)
        else:
            word_scores[word_id] = rows.max(dim=0).values
        present[word_id] = True
    return word_scores, present
