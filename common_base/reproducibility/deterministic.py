# Purpose: Applies explicitly requested deterministic PyTorch backend settings (cuDNN, deterministic algorithms, cuBLAS workspace) and documents their limits.
"""Deterministic backend settings.

``configure_determinism`` changes GLOBAL torch state and only does so when called.

Settings:
    - ``torch.backends.cudnn.deterministic``: pick deterministic cuDNN conv algorithms.
    - ``torch.backends.cudnn.benchmark``: autotuning; must be False for reproducible algorithm choice.
    - ``torch.use_deterministic_algorithms(True, warn_only=...)``: raise (or warn) when an op has
      no deterministic implementation.
    - ``CUBLAS_WORKSPACE_CONFIG``: CUDA >= 10.2 needs ``":4096:8"`` or ``":16:8"`` for
      deterministic cuBLAS. It must be set BEFORE the first CUDA/cuBLAS call, ideally in the shell
      or launcher. This module only sets it when ``set_cublas_workspace_config=True`` is passed.

Limitations: determinism holds only for the same hardware, driver, CUDA/cuDNN and torch versions
and the same number of processes/workers; deterministic kernels can be slower; some ops raise.

Requires: torch.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional

import torch

_VALID_CUBLAS_CONFIGS = (":4096:8", ":16:8")


def configure_determinism(
    deterministic_algorithms: bool = True,
    warn_only: bool = False,
    cudnn_deterministic: bool = True,
    cudnn_benchmark: bool = False,
    set_cublas_workspace_config: bool = False,
    cublas_workspace_config: str = ":4096:8",
) -> Dict[str, object]:
    """Apply the requested settings and return the resulting configuration.

    Args:
        deterministic_algorithms: Call ``torch.use_deterministic_algorithms``.
        warn_only: With deterministic algorithms, warn instead of raising for unsupported ops.
        cudnn_deterministic / cudnn_benchmark: cuDNN flags.
        set_cublas_workspace_config: Explicitly set the env var (only effective before CUDA init).
        cublas_workspace_config: ":4096:8" or ":16:8".
    """
    import os

    if cublas_workspace_config not in _VALID_CUBLAS_CONFIGS:
        raise ValueError(f"cublas_workspace_config must be one of {_VALID_CUBLAS_CONFIGS}")
    if set_cublas_workspace_config:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            warnings.warn("CUDA is already initialized; CUBLAS_WORKSPACE_CONFIG may have no effect", stacklevel=2)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.use_deterministic_algorithms(deterministic_algorithms, warn_only=warn_only)
    return describe_determinism()


def describe_determinism() -> Dict[str, object]:
    """Report current determinism-related settings (reads the env var on call, never at import)."""
    import os

    cublas: Optional[str] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": cublas,
    }
