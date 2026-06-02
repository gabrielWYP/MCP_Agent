"""
Detection metrics: mAP@0.5, mAP@0.5:0.95, per-class AP.

Custom implementation using torchvision.ops.box_iou — no torchmetrics
or pycocotools dependency required.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import numpy as np
from torchvision.ops import box_iou


@dataclass
class LossHistory:
    """Track per-epoch loss values for plotting and analysis."""

    cls_loss: list[float] = field(default_factory=list)
    box_loss: list[float] = field(default_factory=list)
    total_loss: list[float] = field(default_factory=list)

    def update(self, cls: float, box: float, total: float):
        self.cls_loss.append(cls)
        self.box_loss.append(box)
        self.total_loss.append(total)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _compute_ap_at_iou(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float,
    num_classes: int,
) -> tuple[float, dict[int, float]]:
    """Compute Average Precision at a single IoU threshold.

    Uses all-point interpolation (VOC2010+ style).

    Args:
        pred_boxes: (P, 4) predicted boxes in xyxy format.
        pred_scores: (P,) confidence scores.
        pred_labels: (P,) predicted class IDs.
        gt_boxes: (G, 4) ground truth boxes in xyxy format.
        gt_labels: (G,) ground truth class IDs.
        iou_threshold: IoU threshold for positive match.
        num_classes: Total number of classes.

    Returns:
        mAP: Mean AP across all classes.
        per_class_ap: Dict mapping class_id → AP.
    """
    per_class_ap = {}

    for cls_id in range(num_classes):
        # Filter predictions and GT for this class
        pred_mask = pred_labels == cls_id
        gt_mask = gt_labels == cls_id

        cls_pred_boxes = pred_boxes[pred_mask]
        cls_pred_scores = pred_scores[pred_mask]
        cls_gt_boxes = gt_boxes[gt_mask]

        n_gt = len(cls_gt_boxes)
        if n_gt == 0:
            per_class_ap[cls_id] = 0.0
            continue

        if len(cls_pred_boxes) == 0:
            per_class_ap[cls_id] = 0.0
            continue

        # Sort predictions by score (descending)
        sort_idx = cls_pred_scores.argsort(descending=True)
        cls_pred_boxes = cls_pred_boxes[sort_idx]
        cls_pred_scores = cls_pred_scores[sort_idx]

        # Compute IoU between predictions and GT
        ious = box_iou(cls_pred_boxes, cls_gt_boxes)  # (P, G)

        # Match predictions to GT (greedy, highest IoU first)
        matched_gt = set()
        tp = torch.zeros(len(cls_pred_boxes))
        fp = torch.zeros(len(cls_pred_boxes))

        for i in range(len(cls_pred_boxes)):
            if ious.shape[1] == 0:
                fp[i] = 1
                continue

            best_iou, best_gt = ious[i].max(dim=0)
            if best_iou >= iou_threshold and best_gt.item() not in matched_gt:
                tp[i] = 1
                matched_gt.add(best_gt.item())
            else:
                fp[i] = 1

        # Compute precision-recall curve
        cum_tp = tp.cumsum(0)
        cum_fp = fp.cumsum(0)

        precision = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / n_gt

        # All-point interpolation (VOC2010+)
        # Prepend 0 and append 0 for proper integration
        recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
        precision = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])

        # Make precision monotonically decreasing
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = precision[i].max(precision[i + 1])

        # Compute AP as area under PR curve
        recall_diff = recall[1:] - recall[:-1]
        ap = (recall_diff * precision[1:]).sum()

        per_class_ap[cls_id] = ap.item()

    # Mean AP across all classes
    mAP = np.mean(list(per_class_ap.values())) if per_class_ap else 0.0
    return float(mAP), per_class_ap


def compute_map(
    pred_boxes: list[torch.Tensor],
    pred_scores: list[torch.Tensor],
    pred_labels: list[torch.Tensor],
    gt_boxes: list[torch.Tensor],
    gt_labels: list[torch.Tensor],
    num_classes: int = 2,
    iou_threshold: float = 0.5,
) -> dict:
    """Compute mAP metrics across a dataset.

    Args:
        pred_boxes: List of (P_i, 4) predicted boxes per image (cxcywh normalized).
        pred_scores: List of (P_i,) confidence scores per image.
        pred_labels: List of (P_i,) predicted class IDs per image.
        gt_boxes: List of (G_i, 4) GT boxes per image (cxcywh normalized).
        gt_labels: List of (G_i,) GT class IDs per image.
        num_classes: Number of classes.
        iou_threshold: IoU threshold for mAP@IoU computation.

    Returns:
        Dict with keys: map50, map_50_95, per_class_ap_50, per_class_ap_50_95.
    """
    # Concatenate all images
    all_pred_boxes = torch.cat(pred_boxes) if pred_boxes else torch.zeros(0, 4)
    all_pred_scores = torch.cat(pred_scores) if pred_scores else torch.zeros(0)
    all_pred_labels = torch.cat(pred_labels) if pred_labels else torch.zeros(0, dtype=torch.long)
    all_gt_boxes = torch.cat(gt_boxes) if gt_boxes else torch.zeros(0, 4)
    all_gt_labels = torch.cat(gt_labels) if gt_labels else torch.zeros(0, dtype=torch.long)

    if len(all_pred_boxes) > 0:
        all_pred_boxes = _cxcywh_to_xyxy(all_pred_boxes)
    if len(all_gt_boxes) > 0:
        all_gt_boxes = _cxcywh_to_xyxy(all_gt_boxes)

    # mAP@0.5
    map50, per_class_50 = _compute_ap_at_iou(
        all_pred_boxes, all_pred_scores, all_pred_labels,
        all_gt_boxes, all_gt_labels, 0.5, num_classes,
    )

    # mAP@0.5:0.95 (COCO-style, 10 thresholds)
    thresholds = [0.50 + 0.05 * i for i in range(10)]
    maps_per_thresh = []
    per_class_95 = {c: 0.0 for c in range(num_classes)}

    for thresh in thresholds:
        mAP_t, pc_t = _compute_ap_at_iou(
            all_pred_boxes, all_pred_scores, all_pred_labels,
            all_gt_boxes, all_gt_labels, thresh, num_classes,
        )
        maps_per_thresh.append(mAP_t)
        for c in range(num_classes):
            per_class_95[c] += pc_t.get(c, 0.0)

    map_50_95 = float(np.mean(maps_per_thresh))
    for c in per_class_95:
        per_class_95[c] /= len(thresholds)

    return {
        "map50": map50,
        "map_50_95": map_50_95,
        "per_class_ap_50": per_class_50,
        "per_class_ap_50_95": per_class_95,
    }
