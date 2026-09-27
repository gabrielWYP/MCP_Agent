"""Python reference for app-side pre/post-processing of the exported student.

This is the executable spec the mobile app mirrors:

1. `preprocess_bgr` — the exact eval-time preprocessing of
   `YOLODataset.__getitem__` for the val/test splits (BGR->RGB, centered
   letterbox with pad 114, /255, ImageNet mean/std). It calls the dataset's
   own `letterbox` and `normalize_rgb`, so it cannot drift from training.
2. `decode_exported_outputs` — turns the RAW exported tensors
   (`boxes_raw`, `cls_logits`, see `src/export/layout.py`) into detections.
   It only re-shapes them back to per-level maps and delegates to
   `src.training.decode.decode_detections`, the decode used by training
   evaluation, so the math (sigmoid, per-class candidates, exp sizes,
   per-class NMS, max-detections cap) exists in exactly one place.
3. `unletterbox_boxes` — maps detections from the 640x640 letterboxed frame
   back to original image pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import torch

from src.export.layout import unflatten_levels
from src.training.dataset import letterbox, normalize_rgb
from src.training.decode import decode_detections


@dataclass(frozen=True)
class LetterboxInfo:
    """Geometry needed to map boxes back to the original image."""

    orig_w: int
    orig_h: int
    scale: float
    pad_x: int
    pad_y: int


def preprocess_bgr(
    image_bgr: np.ndarray,
    image_size: int = 640,
    pad_value: int = 114,
) -> tuple[torch.Tensor, LetterboxInfo]:
    """Eval-time preprocessing of one `cv2.imread` (BGR uint8) image.

    Returns:
        (1, 3, image_size, image_size) float32 NCHW tensor, and the
        letterbox geometry. Use `.permute(0, 2, 3, 1)` for the NHWC model.
    """
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("preprocess_bgr expects an (H, W, 3) BGR uint8 image.")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    padded, scale, pad_x, pad_y = letterbox(rgb, image_size, pad_value)
    tensor = normalize_rgb(padded).unsqueeze(0)
    info = LetterboxInfo(
        orig_w=int(rgb.shape[1]), orig_h=int(rgb.shape[0]),
        scale=float(scale), pad_x=int(pad_x), pad_y=int(pad_y),
    )
    return tensor, info


def decode_exported_outputs(
    boxes_raw: torch.Tensor | np.ndarray,
    cls_logits: torch.Tensor | np.ndarray,
    *,
    image_size: int,
    strides: Sequence[int],
    conf_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    per_class_candidates: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode one image's RAW exported outputs into detections.

    Args:
        boxes_raw: (1, N, 4) [dx, dy, log_w, log_h].
        cls_logits: (1, N, num_classes) raw logits.

    Returns:
        boxes: (P, 4) cxcywh in letterboxed-input pixels.
        scores: (P,) sigmoid scores.
        labels: (P,) int64 class ids.
    """
    boxes_raw = torch.as_tensor(np.asarray(boxes_raw), dtype=torch.float32)
    cls_logits = torch.as_tensor(np.asarray(cls_logits), dtype=torch.float32)
    if boxes_raw.shape[0] != 1 or cls_logits.shape[0] != 1:
        raise ValueError("decode_exported_outputs decodes a single image (batch size 1).")

    num_classes = cls_logits.shape[-1]
    reg_levels = unflatten_levels(boxes_raw, image_size, strides)
    cls_levels = unflatten_levels(cls_logits, image_size, strides)
    # `decode_detections` reads regressions from `preds[:, num_classes:]`,
    # i.e. the head's `cat([cls, reg])` layout.
    preds = [torch.cat([c, r], dim=1) for c, r in zip(cls_levels, reg_levels)]
    return decode_detections(
        preds,
        cls_levels,
        batch_idx=0,
        num_classes=num_classes,
        image_size=image_size,
        strides=tuple(strides),
        conf_threshold=conf_threshold,
        nms_iou_threshold=nms_iou_threshold,
        nms_enabled=True,
        per_class_candidates=per_class_candidates,
        max_detections=max_detections,
        normalize=False,
    )


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def unletterbox_boxes(boxes_cxcywh: torch.Tensor, info: LetterboxInfo) -> torch.Tensor:
    """Letterboxed-input cxcywh pixels -> original-image xyxy pixels, clipped."""
    xyxy = cxcywh_to_xyxy(boxes_cxcywh)
    xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - info.pad_x) / info.scale).clamp(0, info.orig_w)
    xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - info.pad_y) / info.scale).clamp(0, info.orig_h)
    return xyxy
