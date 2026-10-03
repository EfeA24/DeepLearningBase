# Purpose: Computes seq2seq validation loss/perplexity (token-weighted) and opt-in generation metrics via caller-supplied generate_fn and metric_fn.
"""Seq2seq validation.

``evaluate_loss``: token-weighted cross-entropy over labels != ignore_index (-100) and
perplexity = exp(loss). Runs under ``torch.no_grad()`` with ``model.eval()`` and restores
the previous train/eval mode afterwards. No optimizer or parameter mutation.

``validate``: loss metrics plus OPTIONAL generation metrics, only if the caller supplies
    generate_fn(model, batch) -> list of predictions (e.g. decoded strings)
    reference_fn(batch)       -> list of references aligned with predictions
    metric_fn(predictions, references) -> dict[str, float]  (e.g. BLEU/ROUGE from a library
                                          the caller imports; nothing is fabricated here).
Batch contract and model call match training/seq2seq_trainer.py.

Requires: torch only (metric libraries are the caller's responsibility).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

ForwardFn = Callable[[nn.Module, Mapping[str, Any]], torch.Tensor]


def _default_forward(model: nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
    return model(
        input_ids=batch["input_ids"],
        decoder_input_ids=batch["decoder_input_ids"],
        attention_mask=batch.get("attention_mask"),
        decoder_attention_mask=batch.get("decoder_attention_mask"),
    )


def _to_device(batch: Mapping[str, Any], device: Union[str, torch.device]) -> Dict[str, Any]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: Union[str, torch.device],
    forward_fn: Optional[ForwardFn] = None,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """Return {"loss", "perplexity", "num_tokens"} (token-weighted over the whole loader)."""
    forward_fn = forward_fn or _default_forward
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    try:
        for batch in loader:
            batch = _to_device(batch, device)
            logits = forward_fn(model, batch)
            labels = batch["labels"]
            total_loss += float(
                F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1),
                    ignore_index=ignore_index, reduction="sum",
                )
            )
            total_tokens += int((labels != ignore_index).sum())
    finally:
        model.train(was_training)
    if total_tokens == 0:
        return {"loss": float("nan"), "perplexity": float("nan"), "num_tokens": 0}
    loss = total_loss / total_tokens
    perplexity = math.exp(loss) if loss < 700 else float("inf")
    return {"loss": loss, "perplexity": perplexity, "num_tokens": total_tokens}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: Union[str, torch.device],
    forward_fn: Optional[ForwardFn] = None,
    ignore_index: int = -100,
    generate_fn: Optional[Callable[[nn.Module, Mapping[str, Any]], List[Any]]] = None,
    reference_fn: Optional[Callable[[Mapping[str, Any]], List[Any]]] = None,
    metric_fn: Optional[Callable[[List[Any], List[Any]], Mapping[str, float]]] = None,
) -> Dict[str, float]:
    """Loss metrics, plus generation metrics when all three generation callables are given.

    ``loader`` is iterated twice when generation is enabled (once for loss, once for
    generation), so it must be re-iterable (e.g. a DataLoader, not a generator).
    """
    results: Dict[str, float] = dict(
        evaluate_loss(model, loader, device=device, forward_fn=forward_fn, ignore_index=ignore_index)
    )
    generation_args = (generate_fn, reference_fn, metric_fn)
    if all(a is None for a in generation_args):
        return results
    if any(a is None for a in generation_args):
        raise ValueError("generate_fn, reference_fn and metric_fn must be supplied together")
    was_training = model.training
    model.eval()
    predictions: List[Any] = []
    references: List[Any] = []
    try:
        for batch in loader:
            batch = _to_device(batch, device)
            preds = list(generate_fn(model, batch))
            refs = list(reference_fn(batch))
            if len(preds) != len(refs):
                raise ValueError(f"generate_fn returned {len(preds)} predictions for {len(refs)} references")
            predictions.extend(preds)
            references.extend(refs)
    finally:
        model.train(was_training)
    for name, value in metric_fn(predictions, references).items():
        results[f"gen_{name}"] = float(value)
    return results
