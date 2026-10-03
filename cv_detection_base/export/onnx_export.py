# Purpose: Exports a supplied detection model to ONNX with explicit example inputs, I/O names, dynamic axes and opset, plus optional onnx.checker and onnxruntime validation.
"""ONNX export for detectors.

The caller provides the model (in eval mode, on the device of the example
inputs), example inputs, names, dynamic axes and the output path. Export is
done with ``torch.onnx.export``; the legacy TorchScript-based exporter vs. the
newer dynamo-based exporter is selected by ``dynamo`` (passed through only when
not None, because its default changed across torch releases and the dynamo
path needs ``onnxscript``). Exported graphs usually contain only the raw head
outputs; decoding/NMS are typically kept outside the graph for portability.

Optional validation (lazy imports):
    - ``check_model=True``           -> ``onnx.checker.check_model`` (pip install onnx)
    - ``validate_with_onnxruntime``  -> runs onnxruntime on the example inputs and
      compares every output with PyTorch's (pip install onnxruntime)

Version sensitivity: opset support, dynamic-shape behavior and supported ops
vary across torch/onnx/onnxruntime versions; validate on your target runtime.

Requires: torch and onnx (recent torch releases need ``pip install onnx`` even to
export; the dynamo exporter also needs onnxscript); onnxruntime + numpy optional
(validation only).
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn


def _flatten_outputs(outputs: Any) -> List[Tensor]:
    if isinstance(outputs, Tensor):
        return [outputs]
    if isinstance(outputs, Mapping):
        return [t for v in outputs.values() for t in _flatten_outputs(v)]
    if isinstance(outputs, (list, tuple)):
        return [t for v in outputs for t in _flatten_outputs(v)]
    raise TypeError(f"unsupported output type for ONNX comparison: {type(outputs).__name__}")


def export_onnx(
    model: nn.Module,
    example_inputs: Union[Tensor, Tuple[Tensor, ...]],
    output_path: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 17,
    do_constant_folding: bool = True,
    dynamo: Optional[bool] = None,
    check_model: bool = False,
    validate_with_onnxruntime: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> str:
    """Export ``model`` to ``output_path`` and optionally validate it.

    Args:
        model: Module to export; should already be in eval mode.
        example_inputs: Tensor or tuple of tensors fed to ``model(*inputs)``.
        output_path: Destination .onnx path (parent directory must exist).
        input_names / output_names: Graph I/O names (output count must match the flattened outputs).
        dynamic_axes: e.g. {"images": {0: "batch", 2: "height", 3: "width"}, "p3": {0: "batch"}}.
        opset_version: ONNX opset.
        do_constant_folding: Fold constants during export.
        dynamo: Exporter selection forwarded to torch.onnx.export when not None.
        check_model: Run onnx.checker.
        validate_with_onnxruntime: Compare onnxruntime outputs with PyTorch outputs.
        rtol, atol: Tolerances for the comparison.

    Returns:
        ``output_path``.
    """
    inputs = example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)
    if len(inputs) != len(input_names):
        raise ValueError(f"{len(inputs)} example inputs but {len(input_names)} input names")
    if model.training:
        warnings.warn("model is in training mode; call model.eval() before export for inference graphs", stacklevel=2)

    with torch.no_grad():
        reference = _flatten_outputs(model(*inputs))
    if len(reference) != len(output_names):
        raise ValueError(f"model produced {len(reference)} outputs but {len(output_names)} output names were given")

    kwargs: Dict[str, Any] = {}
    if dynamo is not None:
        kwargs["dynamo"] = dynamo
    torch.onnx.export(
        model,
        inputs,
        output_path,
        input_names=list(input_names),
        output_names=list(output_names),
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        **kwargs,
    )

    if check_model:
        try:
            import onnx
        except ImportError as exc:
            raise ImportError("Install onnx to check the exported model: pip install onnx") from exc
        onnx.checker.check_model(onnx.load(output_path))

    if validate_with_onnxruntime:
        validate_onnx_outputs(output_path, inputs, input_names, reference, rtol=rtol, atol=atol)
    return output_path


def validate_onnx_outputs(
    onnx_path: str,
    inputs: Sequence[Tensor],
    input_names: Sequence[str],
    reference_outputs: Sequence[Tensor],
    rtol: float = 1e-3,
    atol: float = 1e-4,
    providers: Optional[Sequence[str]] = None,
) -> None:
    """Run the ONNX file with onnxruntime and assert outputs match ``reference_outputs``."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("Install onnxruntime and numpy to validate: pip install onnxruntime numpy") from exc
    session = ort.InferenceSession(onnx_path, providers=list(providers or ["CPUExecutionProvider"]))
    feeds = {name: t.detach().cpu().numpy() for name, t in zip(input_names, inputs)}
    outputs = session.run(None, feeds)
    if len(outputs) != len(reference_outputs):
        raise AssertionError(f"onnxruntime returned {len(outputs)} outputs, expected {len(reference_outputs)}")
    for i, (got, expected) in enumerate(zip(outputs, reference_outputs)):
        expected_np = expected.detach().cpu().float().numpy()
        if got.shape != expected_np.shape:
            raise AssertionError(f"output {i}: shape {got.shape} != {expected_np.shape}")
        if not np.allclose(got, expected_np, rtol=rtol, atol=atol):
            max_diff = float(np.max(np.abs(got - expected_np)))
            raise AssertionError(f"output {i}: max abs diff {max_diff:.3e} exceeds rtol={rtol}, atol={atol}")
