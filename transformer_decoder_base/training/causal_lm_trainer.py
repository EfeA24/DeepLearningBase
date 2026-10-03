# Purpose: Provides causal-LM train/eval steps with token-count-weighted next-token loss and optional loss mask for decoder models.
"""Causal-LM train / eval steps.

Batch contract (e.g. from data/collators/causal_lm_collator.py):
    input_ids      LongTensor (B, T)                         required
    labels         LongTensor (B, T), ALIGNED with input_ids, -100 = ignored   required
    attention_mask (B, T) 1/True = real token, or (B, 1, T, T) bool       optional
    loss_mask      (B, T) 1/True = include label at this position       optional
    position_ids   LongTensor (B, T)                                     optional (packing)

Loss (exact shift): the logit at position t predicts the token at t+1:
    shift_logits = logits[:, :-1, :]     shift_labels = labels[:, 1:]
    shift_labels[loss_mask[:, 1:] == 0] = -100
    loss_sum     = sum of token CE over shift_labels != -100   (float32)
    token_count  = number of such tokens
The optimized loss is ``loss_sum / token_count`` (token-weighted mean). Report epoch metrics as
``sum(loss_sum) / sum(token_count)`` — never the mean of per-batch means.

Model contract: ``model(input_ids, attention_mask=..., [position_ids=...])`` returns logits
(B, T, V) or an object with ``.logits``. The model must already be on ``device``; batches are
moved to ``device`` here.

Requires: torch.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def causal_lm_loss_sum(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (summed next-token CE as float32 scalar, number of target tokens as long scalar)."""
    if logits.dim() != 3 or labels.shape != logits.shape[:2]:
        raise ValueError(f"Expected logits (B, T, V) and labels (B, T); got {tuple(logits.shape)}, {tuple(labels.shape)}")
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    if loss_mask is not None:
        if loss_mask.shape != labels.shape:
            raise ValueError("loss_mask must have the same shape as labels")
        shift_labels = shift_labels.masked_fill(~loss_mask[:, 1:].to(torch.bool), ignore_index)
    vocab = shift_logits.shape[-1]
    loss_sum = F.cross_entropy(
        shift_logits.reshape(-1, vocab).float(),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )
    token_count = (shift_labels != ignore_index).sum()
    return loss_sum, token_count


def _move(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def forward_logits(model: nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
    """Call the model with the optional batch fields and return logits (B, T, V)."""
    kwargs: Dict[str, Any] = {}
    if batch.get("attention_mask") is not None:
        kwargs["attention_mask"] = batch["attention_mask"]
    if batch.get("position_ids") is not None:
        kwargs["position_ids"] = batch["position_ids"]
    out = model(batch["input_ids"], **kwargs)
    return out.logits if hasattr(out, "logits") else out


def causal_lm_train_step(
    model: nn.Module,
    batch: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: Optional[float] = None,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """One optimizer step on one batch; returns {"loss_sum", "token_count", "loss"}."""
    model.train()
    batch = _move(batch, device)
    optimizer.zero_grad(set_to_none=True)
    logits = forward_logits(model, batch)
    loss_sum, token_count = causal_lm_loss_sum(logits, batch["labels"], batch.get("loss_mask"), ignore_index)
    count = int(token_count)
    if count == 0:
        return {"loss_sum": 0.0, "token_count": 0, "loss": float("nan")}
    (loss_sum / count).backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    return {"loss_sum": float(loss_sum.detach()), "token_count": count, "loss": float(loss_sum.detach()) / count}


@torch.no_grad()
def causal_lm_eval_step(
    model: nn.Module, batch: Mapping[str, Any], device: torch.device, ignore_index: int = -100
) -> Dict[str, float]:
    """Loss sum and token count on one batch without gradient or parameter updates."""
    model.eval()
    batch = _move(batch, device)
    logits = forward_logits(model, batch)
    loss_sum, token_count = causal_lm_loss_sum(logits, batch["labels"], batch.get("loss_mask"), ignore_index)
    count = int(token_count)
    return {
        "loss_sum": float(loss_sum),
        "token_count": count,
        "loss": float(loss_sum) / count if count else float("nan"),
    }


def evaluate_causal_lm(
    model: nn.Module, batches: Iterable[Mapping[str, Any]], device: torch.device, ignore_index: int = -100
) -> Dict[str, float]:
    """Token-weighted mean loss and perplexity over all batches."""
    total_loss, total_tokens = 0.0, 0
    for batch in batches:
        stats = causal_lm_eval_step(model, batch, device, ignore_index)
        total_loss += stats["loss_sum"]
        total_tokens += stats["token_count"]
    if total_tokens == 0:
        return {"loss": float("nan"), "perplexity": float("nan"), "token_count": 0}
    mean = total_loss / total_tokens
    return {"loss": mean, "perplexity": math.exp(mean) if mean < 700 else float("inf"), "token_count": total_tokens}
