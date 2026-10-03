# Purpose: Shows an explicit AMP training step for decoder LMs (bf16 preferred, fp16 with GradScaler) with token-weighted causal-LM loss.
"""Mixed-precision causal-LM training step (explicit, opt-in).

precision:
    "bf16" -> ``torch.amp.autocast(device_type, dtype=torch.bfloat16)``, no scaler needed
              (same exponent range as fp32). Preferred on Ampere+ GPUs / recent CPUs.
              ``bf16_supported(device_type)`` reports capability; nothing falls back silently.
    "fp16" -> autocast float16 AND a ``GradScaler`` (required; raises otherwise) to avoid
              gradient underflow. Unscale before clipping (done here).
    "fp32" -> plain full precision (autocast disabled).

Parameters and optimizer state stay float32; only autocast-eligible ops run in low precision.
The loss is computed from float32 logits. Label shift / batch contract: as in
training/causal_lm_trainer.py (aligned labels; logits[:, :-1] vs labels[:, 1:]; -100 ignored).

Requires: torch (torch.amp; GradScaler via ``torch.amp.GradScaler`` when available, else
``torch.cuda.amp.GradScaler`` on older builds).
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

_PRECISIONS = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def resolve_amp_dtype(precision: str) -> Optional[torch.dtype]:
    """Map "fp32" | "bf16" | "fp16" to the autocast dtype (None for fp32)."""
    if precision not in _PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(_PRECISIONS)}, got {precision!r}")
    return _PRECISIONS[precision]


def bf16_supported(device_type: str) -> bool:
    """Capability check for bf16 autocast on ``device_type`` ("cuda" or "cpu")."""
    if device_type == "cuda":
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if device_type == "cpu":
        return True
    return False


def make_grad_scaler(device_type: str = "cuda", enabled: bool = True) -> Any:
    """Create a GradScaler for fp16 training on ``device_type``."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device_type, enabled=enabled)
    if device_type != "cuda":
        raise RuntimeError("This torch build only provides a CUDA GradScaler")
    return torch.cuda.amp.GradScaler(enabled=enabled)


def amp_causal_lm_train_step(
    model: nn.Module,
    batch: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    precision: str = "bf16",
    scaler: Optional[Any] = None,
    grad_clip_norm: Optional[float] = None,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """One AMP optimizer step; returns {"loss_sum", "token_count", "loss"}."""
    dtype = resolve_amp_dtype(precision)
    device = torch.device(device)
    if precision == "fp16" and scaler is None:
        raise ValueError("fp16 training requires a GradScaler (see make_grad_scaler)")
    if precision == "bf16" and not bf16_supported(device.type):
        raise RuntimeError(f"bf16 autocast is not supported on device type {device.type!r}")

    model.train()
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    optimizer.zero_grad(set_to_none=True)
    kwargs = {}
    if batch.get("attention_mask") is not None:
        kwargs["attention_mask"] = batch["attention_mask"]
    if batch.get("position_ids") is not None:
        kwargs["position_ids"] = batch["position_ids"]

    with torch.amp.autocast(device_type=device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
        out = model(batch["input_ids"], **kwargs)
        logits = out.logits if hasattr(out, "logits") else out
    labels = batch["labels"][:, 1:]
    if batch.get("loss_mask") is not None:
        labels = labels.masked_fill(~batch["loss_mask"][:, 1:].to(torch.bool), ignore_index)
    loss_sum = F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )
    count = int((labels != ignore_index).sum())
    if count == 0:
        return {"loss_sum": 0.0, "token_count": 0, "loss": float("nan")}
    loss = loss_sum / count

    if scaler is not None:
        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
    return {"loss_sum": float(loss_sum.detach()), "token_count": count, "loss": float(loss.detach())}
