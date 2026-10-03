# Purpose: Provides seq2seq train step/epoch functions with ignored pad labels, token-count-aware loss, optional AMP, gradient clipping and per-update scheduling.
"""Seq2seq training step and epoch.

Batch contract (data/collators/seq2seq_collator.py): input_ids (B, S), attention_mask (B, S),
decoder_input_ids (B, T), decoder_attention_mask (B, T), labels (B, T) with -100 at pads.
Model call (overridable via ``forward_fn(model, batch) -> logits (B, T, V)``):
    model(input_ids=..., decoder_input_ids=..., attention_mask=..., decoder_attention_mask=...)

Loss: cross-entropy SUMMED over non-ignored label tokens, divided by that batch's token
count for backward (per-token mean). Epoch loss = total loss sum / total tokens (token-
weighted; never an average of batch means). Batches without valid tokens are skipped.
Schedulers passed here are stepped once per optimizer update.

Requires: torch only.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

ForwardFn = Callable[[nn.Module, Mapping[str, Any]], torch.Tensor]


def default_forward_fn(model: nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
    return model(
        input_ids=batch["input_ids"],
        decoder_input_ids=batch["decoder_input_ids"],
        attention_mask=batch.get("attention_mask"),
        decoder_attention_mask=batch.get("decoder_attention_mask"),
    )


def move_batch_to_device(batch: Mapping[str, Any], device: Union[str, torch.device]) -> Dict[str, Any]:
    """Move tensor values to ``device``; non-tensor values are passed through."""
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def seq2seq_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, int]:
    """Return (summed CE over valid tokens as a scalar tensor, number of valid tokens)."""
    if logits.dim() != 3 or logits.shape[:2] != labels.shape:
        raise ValueError(f"logits (B, T, V) {tuple(logits.shape)} incompatible with labels {tuple(labels.shape)}")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    loss_sum = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="sum",
        label_smoothing=label_smoothing,
    )
    return loss_sum, int((labels != ignore_index).sum())


def seq2seq_train_step(
    model: nn.Module,
    batch: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    device: Union[str, torch.device],
    forward_fn: ForwardFn = default_forward_fn,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    max_grad_norm: Optional[float] = None,
    scaler: Optional[Any] = None,
    autocast_dtype: Optional[torch.dtype] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, float]:
    """One optimizer update. Returns {"loss": per-token mean, "loss_sum", "num_tokens"}.

    ``scaler`` is a ``torch.amp.GradScaler`` (fp16); ``autocast_dtype`` enables
    ``torch.amp.autocast`` on the device type of ``device``. The caller sets ``model.train()``.
    """
    from contextlib import nullcontext

    batch = move_batch_to_device(batch, device)
    device_type = torch.device(device).type
    autocast = (
        torch.amp.autocast(device_type=device_type, dtype=autocast_dtype) if autocast_dtype is not None else nullcontext()
    )
    optimizer.zero_grad(set_to_none=True)
    with autocast:
        logits = forward_fn(model, batch)
    loss_sum, num_tokens = seq2seq_token_loss(logits, batch["labels"], ignore_index, label_smoothing)
    if num_tokens == 0:
        return {"loss": 0.0, "loss_sum": 0.0, "num_tokens": 0}
    loss = loss_sum / num_tokens
    if scaler is not None:
        scaler.scale(loss).backward()
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {"loss": float(loss.detach()), "loss_sum": float(loss_sum.detach()), "num_tokens": num_tokens}


def seq2seq_train_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    device: Union[str, torch.device],
    **step_kwargs: Any,
) -> Dict[str, float]:
    """Run one epoch; returns {"loss": token-weighted mean, "num_tokens", "num_steps"}."""
    model.train()
    total_loss, total_tokens, steps = 0.0, 0, 0
    for batch in loader:
        stats = seq2seq_train_step(model, batch, optimizer, device=device, **step_kwargs)
        if stats["num_tokens"] == 0:
            continue
        total_loss += stats["loss_sum"]
        total_tokens += stats["num_tokens"]
        steps += 1
    return {
        "loss": total_loss / total_tokens if total_tokens else float("nan"),
        "num_tokens": total_tokens,
        "num_steps": steps,
    }
