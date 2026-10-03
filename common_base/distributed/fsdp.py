# Purpose: Provides opt-in FullyShardedDataParallel wrapping for large-model PyTorch training, importing FSDP components only inside the setup routine.
"""Opt-in FSDP (FullyShardedDataParallel, v1 API) wrapping.

Version/config sensitivity:
    - The FSDP API evolves quickly. This targets ``torch.distributed.fsdp.FullyShardedDataParallel``
      (torch >= 2.1). Newer torch releases also offer ``fully_shard`` (FSDP2), which has a
      different, per-module API; it is NOT used here.
    - Sharding strategy, auto-wrap policy, mixed-precision policy, CPU offload and state-dict
      type all interact; validate on your hardware before relying on throughput or memory numbers.
    - Saving: use FSDP state-dict APIs (``FSDP.state_dict_type``) or torch.distributed.checkpoint;
      a plain ``model.state_dict()`` on a sharded model is NOT a full checkpoint by default.
    - Create the optimizer AFTER wrapping (it must see the flattened sharded parameters).
    - Requires an initialized process group; NCCL backend with CUDA for real use.

Requires: torch (with torch.distributed); CUDA + NCCL for practical use.
"""
from __future__ import annotations

from typing import Any, Optional, Set, Type

import torch
from torch import nn


def wrap_fsdp(
    model: nn.Module,
    device_id: Optional[int] = None,
    sharding_strategy: str = "full_shard",
    transformer_layer_classes: Optional[Set[Type[nn.Module]]] = None,
    min_num_params: Optional[int] = None,
    param_dtype: Optional[torch.dtype] = None,
    reduce_dtype: Optional[torch.dtype] = None,
    buffer_dtype: Optional[torch.dtype] = None,
    cpu_offload: bool = False,
    use_orig_params: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Wrap ``model`` with FSDP using explicit options.

    Args:
        model: Unwrapped model (may still be on CPU; ``device_id`` moves shards to that GPU).
        device_id: CUDA device index for this rank.
        sharding_strategy: "full_shard", "shard_grad_op", "no_shard" or "hybrid_shard".
        transformer_layer_classes: Layer classes to wrap individually (transformer auto-wrap).
        min_num_params: Size-based auto-wrap threshold (mutually exclusive with the above).
        param_dtype / reduce_dtype / buffer_dtype: Mixed-precision policy dtypes (None = fp32).
        cpu_offload: Offload parameters to CPU (slow, saves GPU memory).
        use_orig_params: Keep original parameter names/objects (needed for per-param optimizer groups).
        **kwargs: Forwarded to FSDP.
    """
    import functools

    import torch.distributed as dist
    from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("Initialize a process group before wrapping with FSDP")
    strategies = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }
    if sharding_strategy not in strategies:
        raise ValueError(f"sharding_strategy must be one of {sorted(strategies)}, got {sharding_strategy!r}")
    if transformer_layer_classes and min_num_params is not None:
        raise ValueError("Use either transformer_layer_classes or min_num_params, not both")

    auto_wrap_policy = None
    if transformer_layer_classes:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=set(transformer_layer_classes)
        )
    elif min_num_params is not None:
        auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)

    mixed_precision = None
    if any(d is not None for d in (param_dtype, reduce_dtype, buffer_dtype)):
        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

    return FSDP(
        model,
        sharding_strategy=strategies[sharding_strategy],
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        cpu_offload=CPUOffload(offload_params=True) if cpu_offload else None,
        device_id=device_id,
        use_orig_params=use_orig_params,
        **kwargs,
    )
