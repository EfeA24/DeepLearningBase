# Purpose: Provides optional, version-sensitive TensorRT glue to build and load engines from an exported detection ONNX file with explicit workspace and precision flags.
"""Optional TensorRT helpers (deployment glue, VERSION-SENSITIVE).

Platform/version requirements:
    - NVIDIA GPU + driver, CUDA, and the TensorRT Python package matching them
      (``pip install tensorrt`` on supported Linux/Windows setups, or NVIDIA containers).
    - Written against the TensorRT 8.6 / 10.x Python API: ``set_memory_pool_limit``
      (>= 8.4) and ``build_serialized_network``. TensorRT 10 removed implicit
      batch, so networks are always created in explicit-batch mode.
    - Engines are NOT portable across GPU architectures or TensorRT versions;
      build them on (or for) the deployment target.
``tensorrt`` is imported only inside the functions; importing this module never
touches CUDA or TensorRT.

Requires: tensorrt (optional; NVIDIA platforms only).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def _import_tensorrt() -> Any:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ImportError(
            "TensorRT is not installed. Install the NVIDIA TensorRT Python package matching your CUDA/driver "
            "(e.g. pip install tensorrt) to use these helpers."
        ) from exc
    return trt


def build_engine_from_onnx(
    onnx_path: str,
    engine_path: Optional[str] = None,
    workspace_bytes: int = 1 << 30,
    fp16: bool = False,
    optimization_profiles: Optional[Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]] = None,
    verbose: bool = False,
) -> bytes:
    """Build a serialized TensorRT engine from an ONNX file.

    Args:
        onnx_path: Path to the exported ONNX model.
        engine_path: If given, the serialized engine is also written here.
        workspace_bytes: Builder workspace memory pool limit.
        fp16: Enable FP16 kernels (raises if the platform has no fast FP16).
        optimization_profiles: For dynamic inputs: {input_name: (min_shape, opt_shape, max_shape)}.
        verbose: Use a verbose TensorRT logger.

    Returns:
        Serialized engine bytes.
    """
    if workspace_bytes <= 0:
        raise ValueError("workspace_bytes must be > 0")
    trt = _import_tensorrt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as handle:
        if not parser.parse(handle.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("TensorRT failed to parse the ONNX model:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_bytes))
    if fp16:
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("fp16=True requested but this platform has no fast FP16 support")
        config.set_flag(trt.BuilderFlag.FP16)
    if optimization_profiles:
        profile = builder.create_optimization_profile()
        for name, (min_shape, opt_shape, max_shape) in optimization_profiles.items():
            profile.set_shape(name, tuple(min_shape), tuple(opt_shape), tuple(max_shape))
        config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed (see logger output)")
    engine_bytes = bytes(serialized)
    if engine_path is not None:
        with open(engine_path, "wb") as handle:
            handle.write(engine_bytes)
    return engine_bytes


def load_engine(engine_path: str, verbose: bool = False) -> Any:
    """Deserialize a TensorRT engine from an explicit file path (same TensorRT version/GPU required)."""
    trt = _import_tensorrt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as handle:
        engine = runtime.deserialize_cuda_engine(handle.read())
    if engine is None:
        raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
    return engine


def describe_engine_io(engine: Any) -> Dict[str, Dict[str, Any]]:
    """List engine I/O tensors {name: {"mode": "INPUT"/"OUTPUT", "shape": tuple, "dtype": str}} (TensorRT >= 8.5 API)."""
    trt = _import_tensorrt()
    if not hasattr(engine, "num_io_tensors"):
        raise RuntimeError("describe_engine_io needs the TensorRT >= 8.5 named-tensor API")
    info: Dict[str, Dict[str, Any]] = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        info[name] = {
            "mode": "INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT",
            "shape": tuple(engine.get_tensor_shape(name)),
            "dtype": str(engine.get_tensor_dtype(name)),
        }
    return info
