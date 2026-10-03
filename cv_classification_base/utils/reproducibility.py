# Purpose: Applies explicitly requested deterministic/benchmark backend settings for image-classification experiments and documents their limits.
"""Reproducibility settings wrapper.

``configure_reproducibility`` changes global torch backend flags only when called:
    - ``torch.backends.cudnn.deterministic`` / ``torch.backends.cudnn.benchmark``
    - ``torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)``
    - optionally ``CUBLAS_WORKSPACE_CONFIG`` (required by deterministic cuBLAS on CUDA >= 10.2);
      set only when ``set_cublas_workspace=True`` and must happen before the first CUDA call.

No guarantee across torch/CUDA/cuDNN versions, GPUs, driver versions or thread counts.
Deterministic mode can be slower and some ops (e.g. certain scatter/index ops, adaptive pooling
backward on CUDA) will raise unless ``warn_only=True``. Combine with utils/seed.py.

Requires: torch.
"""
from __future__ import annotations

from typing import Dict

import torch


def configure_reproducibility(
    deterministic: bool,
    benchmark: bool = False,
    warn_only: bool = False,
    set_cublas_workspace: bool = False,
    cublas_workspace_config: str = ":4096:8",
) -> Dict[str, object]:
    """Apply the requested settings and return a summary of what was set."""
    if deterministic and benchmark:
        raise ValueError("benchmark=True selects kernels nondeterministically; use benchmark=False")
    summary: Dict[str, object] = {}
    if set_cublas_workspace:
        if cublas_workspace_config not in (":4096:8", ":16:8"):
            raise ValueError("cublas_workspace_config must be ':4096:8' or ':16:8'")
        import os

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config
        summary["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    summary.update(
        {
            "cudnn.deterministic": deterministic,
            "cudnn.benchmark": benchmark,
            "use_deterministic_algorithms": deterministic,
            "warn_only": warn_only,
        }
    )
    return summary


def current_settings() -> Dict[str, object]:
    """Read the current global flags (no changes)."""
    return {
        "cudnn.deterministic": torch.backends.cudnn.deterministic,
        "cudnn.benchmark": torch.backends.cudnn.benchmark,
        "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
    }
