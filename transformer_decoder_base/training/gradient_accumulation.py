# Purpose: Demonstrates causal-LM gradient accumulation with exact token normalization across micro-batches (sum of losses / total tokens).
"""Gradient accumulation for causal LMs.

Why: averaging per-micro-batch MEAN losses weights each micro-batch equally, so tokens in
short (heavily padded) micro-batches count more. The correct objective for one optimizer step is

    loss = (sum over micro-batches of token-CE sums) / (total target tokens in the step)

This helper first counts target tokens over all micro-batches of the step (labels[:, 1:] != -100,
also honoring ``loss_mask``), then back-propagates ``loss_sum_i / total_tokens`` for each
micro-batch, so the accumulated gradient equals the gradient of the token-weighted mean over
the whole step.

Distributed training: each rank only sees its own tokens. For an exact global mean pass
``total_tokens`` = all-reduced token count (and remember DDP averages gradients over ranks, so
multiply the loss by world_size or pass the global count accordingly). When the model has
``no_sync`` (DDP) gradient synchronization is skipped for all but the last micro-batch.

Batch contract and label shift: identical to training/causal_lm_trainer.py (aligned labels,
loss on logits[:, :-1] vs labels[:, 1:], -100 ignored). Self-contained.

Requires: torch.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _target_labels(batch: Mapping[str, Any], ignore_index: int) -> torch.Tensor:
    labels = batch["labels"][:, 1:]
    loss_mask = batch.get("loss_mask")
    if loss_mask is not None:
        labels = labels.masked_fill(~loss_mask[:, 1:].to(torch.bool), ignore_index)
    return labels


def count_step_tokens(micro_batches: Sequence[Mapping[str, Any]], ignore_index: int = -100) -> int:
    """Total number of target tokens over the micro-batches of one optimizer step."""
    return int(sum(int((_target_labels(mb, ignore_index) != ignore_index).sum()) for mb in micro_batches))


def accumulate_and_step(
    model: nn.Module,
    micro_batches: Sequence[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: Optional[float] = None,
    total_tokens: Optional[int] = None,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """Run forward/backward on every micro-batch, then one optimizer step.

    Returns {"loss_sum", "token_count", "loss"} for the whole step. If the step has no target
    tokens, no backward/step is performed.
    """
    import contextlib

    if len(micro_batches) == 0:
        raise ValueError("micro_batches must not be empty")
    model.train()
    step_tokens = count_step_tokens(micro_batches, ignore_index) if total_tokens is None else int(total_tokens)
    optimizer.zero_grad(set_to_none=True)
    if step_tokens == 0:
        return {"loss_sum": 0.0, "token_count": 0, "loss": float("nan")}

    total_loss = 0.0
    last = len(micro_batches) - 1
    for i, mb in enumerate(micro_batches):
        mb = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in mb.items()}
        sync_ctx = model.no_sync() if (hasattr(model, "no_sync") and i < last) else contextlib.nullcontext()
        with sync_ctx:
            kwargs = {}
            if mb.get("attention_mask") is not None:
                kwargs["attention_mask"] = mb["attention_mask"]
            if mb.get("position_ids") is not None:
                kwargs["position_ids"] = mb["position_ids"]
            out = model(mb["input_ids"], **kwargs)
            logits = out.logits if hasattr(out, "logits") else out
            targets = _target_labels(mb, ignore_index)
            loss_sum = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]).float(),
                targets.reshape(-1),
                ignore_index=ignore_index,
                reduction="sum",
            )
            (loss_sum / step_tokens).backward()
        total_loss += float(loss_sum.detach())

    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"loss_sum": total_loss, "token_count": step_tokens, "loss": total_loss / step_tokens}
