# Purpose: Provides an object-detection inference facade around an already initialized model with injected preprocess and postprocess callbacks.
"""Detection inference facade.

Pipeline (each stage stays separately replaceable):
    raw image --preprocess_fn--> (tensor (C, H, W), meta dict)
    tensors   --batch_fn------> batch (default: stack; requires equal sizes)
    batch     --model---------> raw outputs                    (inference_mode, eval)
    outputs   --postprocess_fn-> list of {"boxes", "scores", "labels"} per image
    (optional) map_back_fn(detections, meta) -> detections in original-image coordinates

The facade never loads checkpoints and never moves the model: it verifies that
the model's parameters are on ``device`` and raises otherwise. The model is
switched to eval mode for the call and its previous mode is restored.

Requires: torch only (preprocessing libraries live inside the injected callables).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

Detections = Dict[str, Tensor]


def stack_batch(tensors: Sequence[Tensor]) -> Tensor:
    """Stack equally sized (C, H, W) tensors into (B, C, H, W)."""
    sizes = {tuple(t.shape) for t in tensors}
    if len(sizes) != 1:
        raise ValueError(f"preprocessed images differ in shape {sorted(sizes)}; resize them or pass a padding batch_fn")
    return torch.stack(list(tensors), dim=0)


def scale_boxes_to_original(detections: Detections, meta: Dict[str, Any]) -> Detections:
    """Map boxes from network-input pixels back to the original image.

    Expects ``meta`` with "scale" (sx, sy) applied during preprocessing,
    optional "pad" (left, top) added after scaling, and optional "orig_size" (H, W) for clipping.
    """
    sx, sy = meta.get("scale", (1.0, 1.0))
    pad_left, pad_top = meta.get("pad", (0.0, 0.0))
    boxes = detections["boxes"].clone()
    boxes[:, 0::2] = (boxes[:, 0::2] - pad_left) / sx
    boxes[:, 1::2] = (boxes[:, 1::2] - pad_top) / sy
    if "orig_size" in meta:
        h, w = meta["orig_size"]
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
    out = dict(detections)
    out["boxes"] = boxes
    return out


class Detector:
    """Inference wrapper.

    Args:
        model: Initialized detector (weights loaded by the caller), already on ``device``.
        device: Device the inputs are moved to.
        preprocess_fn: ``raw_image -> (Tensor (C, H, W), meta dict)``.
        postprocess_fn: ``(outputs, batch) -> list of detections`` (e.g.
            inference/postprocess.py ``DetectionPostprocessor``; it is called as
            ``postprocess_fn(outputs, batch)``).
        batch_fn: Batching of preprocessed tensors (default ``stack_batch``).
        map_back_fn: Optional ``(detections, meta) -> detections`` (e.g. ``scale_boxes_to_original``).
        autocast_dtype: Optional autocast dtype for the forward pass.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        preprocess_fn: Callable[[Any], Tuple[Tensor, Dict[str, Any]]],
        postprocess_fn: Callable[[Any, Any], List[Detections]],
        batch_fn: Callable[[Sequence[Tensor]], Any] = stack_batch,
        map_back_fn: Optional[Callable[[Detections, Dict[str, Any]], Detections]] = None,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self.batch_fn = batch_fn
        self.map_back_fn = map_back_fn
        self.autocast_dtype = autocast_dtype

    def _check_model_device(self) -> None:
        param = next(self.model.parameters(), None)
        if param is None:
            return
        if param.device.type != self.device.type or (
            self.device.index is not None and param.device.index != self.device.index
        ):
            raise RuntimeError(f"model is on {param.device} but Detector device is {self.device}; move the model explicitly")

    def predict(self, images: Sequence[Any]) -> List[Detections]:
        """Run detection on a list of raw images; output order matches input order."""
        if len(images) == 0:
            return []
        self._check_model_device()
        processed = [self.preprocess_fn(img) for img in images]
        tensors = [p[0] for p in processed]
        metas = [p[1] for p in processed]
        batch = self.batch_fn(tensors)
        batch = batch.to(self.device) if isinstance(batch, Tensor) else [t.to(self.device) for t in batch]

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode(), torch.amp.autocast(
                device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None
            ):
                outputs = self.model(batch)
            detections = self.postprocess_fn(outputs, batch)
        finally:
            self.model.train(was_training)

        if len(detections) != len(images):
            raise ValueError(f"postprocess_fn returned {len(detections)} results for {len(images)} images")
        if self.map_back_fn is not None:
            detections = [self.map_back_fn(d, m) for d, m in zip(detections, metas)]
        return detections

    def predict_one(self, image: Any) -> Detections:
        return self.predict([image])[0]
