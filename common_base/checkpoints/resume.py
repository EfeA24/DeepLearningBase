# Purpose: Captures and restores training progress plus Python/NumPy/Torch/CUDA RNG state for resuming PyTorch training runs.
"""Resume support: train state + RNG state.

``capture_rng_state()`` returns a dict that can be stored under a checkpoint's ``"extra"`` or
``"metadata"`` entry; ``restore_rng_state()`` puts it back. ``build_resume_payload`` and
``apply_resume_payload`` bundle a plain train-state dict (e.g. ``TrainState.to_dict()``) with RNG.

Storage format is weights_only-friendly: torch RNG states are uint8 tensors; Python's
``random`` state is converted to nested lists/ints; NumPy's legacy state tuple is converted to a
dict holding a tensor of the key array.

Reproducibility limits:
    - Restoring RNG reproduces random draws only if code, library versions, hardware, number of
      processes and data-loader worker setup are identical.
    - DataLoader shuffling uses its own generator/seed per epoch; worker processes re-seed
      themselves. Pass an explicit ``torch.Generator`` to the DataLoader and save its state with
      ``extra_generators`` if exact data order matters.
    - Non-deterministic CUDA kernels can still differ (see reproducibility/deterministic.py).
    - CUDA RNG is per device; restoring to a different device count raises.

Requires: torch; optional numpy (handled if installed).
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch


def _python_state_to_plain(state: Any) -> Any:
    if isinstance(state, tuple):
        return [_python_state_to_plain(s) for s in state]
    return state


def _plain_to_python_state(state: Any) -> Any:
    if isinstance(state, list):
        return tuple(_plain_to_python_state(s) for s in state)
    return state


def capture_rng_state(
    include_cuda: bool = True,
    extra_generators: Optional[Mapping[str, torch.Generator]] = None,
) -> Dict[str, Any]:
    """Snapshot RNG states (python, numpy if importable, torch CPU, all CUDA devices)."""
    import random

    state: Dict[str, Any] = {
        "python": _python_state_to_plain(random.getstate()),
        "torch": torch.get_rng_state(),
    }
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        kind, keys, pos, has_gauss, cached = np.random.get_state(legacy=True)
        state["numpy"] = {
            "kind": str(kind),
            "keys": torch.from_numpy(keys.astype(np.int64)),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached),
        }
    if include_cuda and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if extra_generators:
        state["generators"] = {name: gen.get_state() for name, gen in extra_generators.items()}
    return state


def restore_rng_state(
    state: Mapping[str, Any],
    extra_generators: Optional[Mapping[str, torch.Generator]] = None,
    strict: bool = True,
) -> None:
    """Restore states captured by ``capture_rng_state``.

    With ``strict=True`` a captured NumPy/CUDA state that cannot be restored (numpy missing, no
    CUDA, device count changed) raises RuntimeError; with ``strict=False`` it is skipped.
    """
    import random

    if "python" in state:
        random.setstate(_plain_to_python_state(state["python"]))
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if "numpy" in state:
        try:
            import numpy as np
        except ImportError:
            np = None
        if np is None:
            if strict:
                raise RuntimeError("Checkpoint contains NumPy RNG state but numpy is not installed")
        else:
            ns = state["numpy"]
            np.random.set_state(
                (ns["kind"], ns["keys"].cpu().numpy().astype(np.uint32), ns["pos"], ns["has_gauss"], ns["cached_gaussian"])
            )
    if "cuda" in state:
        cuda_states = state["cuda"]
        if not torch.cuda.is_available() or torch.cuda.device_count() != len(cuda_states):
            if strict:
                raise RuntimeError("Saved CUDA RNG state does not match the available CUDA devices")
        else:
            torch.cuda.set_rng_state_all([s.cpu() for s in cuda_states])
    if extra_generators:
        saved = state.get("generators", {})
        for name, gen in extra_generators.items():
            if name in saved:
                gen.set_state(saved[name].cpu())
            elif strict:
                raise KeyError(f"No saved state for generator '{name}'")


def build_resume_payload(
    train_state: Mapping[str, Any],
    include_rng: bool = True,
    include_cuda: bool = True,
    extra_generators: Optional[Mapping[str, torch.Generator]] = None,
) -> Dict[str, Any]:
    """Combine a plain train-state dict with optional RNG state for checkpoint ``extra``."""
    payload: Dict[str, Any] = {"train_state": dict(train_state)}
    if include_rng:
        payload["rng"] = capture_rng_state(include_cuda=include_cuda, extra_generators=extra_generators)
    return payload


def apply_resume_payload(
    payload: Mapping[str, Any],
    restore_rng: bool = True,
    extra_generators: Optional[Mapping[str, torch.Generator]] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Restore RNG (if present and requested) and return the plain train-state dict."""
    if "train_state" not in payload:
        raise KeyError("Resume payload has no 'train_state' entry")
    if restore_rng and "rng" in payload:
        restore_rng_state(payload["rng"], extra_generators=extra_generators, strict=strict)
    return dict(payload["train_state"])
