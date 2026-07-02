"""
Detection metrics: mAP@0.5, mAP@0.5:0.95, per-class AP.

Custom implementation using torchvision.ops.box_iou — no torchmetrics
or pycocotools dependency required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import numpy as np
from torchvision.ops import box_iou

logger = logging.getLogger(__name__)


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
    pred_boxes: list[torch.Tensor],
    pred_scores: list[torch.Tensor],
    pred_labels: list[torch.Tensor],
    gt_boxes: list[torch.Tensor],
    gt_labels: list[torch.Tensor],
    iou_threshold: float,
    num_classes: int,
) -> tuple[float, dict[int, float]]:
    """Compute Average Precision at a single IoU threshold.

    Uses all-point interpolation (VOC2010+ style).

    Args:
        pred_boxes: List of (P_i, 4) predicted boxes per image in xyxy format.
        pred_scores: List of (P_i,) confidence scores per image.
        pred_labels: List of (P_i,) predicted class IDs per image.
        gt_boxes: List of (G_i, 4) ground truth boxes per image in xyxy format.
        gt_labels: List of (G_i,) ground truth class IDs per image.
        iou_threshold: IoU threshold for positive match.
        num_classes: Total number of classes.

    Returns:
        mAP: Mean AP across all classes.
        per_class_ap: Dict mapping class_id → AP.
    """
    per_class_ap = {}

    for cls_id in range(num_classes):
        class_predictions: list[tuple[float, int, torch.Tensor]] = []
        class_gt_boxes: list[torch.Tensor] = []

        for image_idx, (boxes_i, scores_i, labels_i, gt_boxes_i, gt_labels_i) in enumerate(
            zip(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
        ):
            pred_mask = labels_i == cls_id
            for box, score in zip(boxes_i[pred_mask], scores_i[pred_mask]):
                class_predictions.append((float(score.item()), image_idx, box))

            gt_mask = gt_labels_i == cls_id
            class_gt_boxes.append(gt_boxes_i[gt_mask])

        n_gt = sum(len(boxes_i) for boxes_i in class_gt_boxes)
        if n_gt == 0:
            per_class_ap[cls_id] = 0.0
            continue

        if not class_predictions:
            per_class_ap[cls_id] = 0.0
            continue

        class_predictions.sort(key=lambda item: item[0], reverse=True)

        matched_gt = set()
        tp = torch.zeros(len(class_predictions))
        fp = torch.zeros(len(class_predictions))

        for i, (_, image_idx, pred_box) in enumerate(class_predictions):
            gt_boxes_i = class_gt_boxes[image_idx]
            if len(gt_boxes_i) == 0:
                fp[i] = 1
                continue

            ious = box_iou(pred_box.unsqueeze(0), gt_boxes_i).squeeze(0)
            best_iou, best_gt = ious.max(dim=0)
            match_key = (image_idx, best_gt.item())
            if best_iou >= iou_threshold and match_key not in matched_gt:
                tp[i] = 1
                matched_gt.add(match_key)
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
    pred_boxes_xyxy = [
        _cxcywh_to_xyxy(boxes) if len(boxes) > 0 else boxes
        for boxes in pred_boxes
    ]
    gt_boxes_xyxy = [
        _cxcywh_to_xyxy(boxes) if len(boxes) > 0 else boxes
        for boxes in gt_boxes
    ]

    # mAP@0.5
    map50, per_class_50 = _compute_ap_at_iou(
        pred_boxes_xyxy, pred_scores, pred_labels,
        gt_boxes_xyxy, gt_labels, 0.5, num_classes,
    )

    # mAP@0.5:0.95 (COCO-style, 10 thresholds)
    thresholds = [0.50 + 0.05 * i for i in range(10)]
    maps_per_thresh = []
    per_class_95 = {c: 0.0 for c in range(num_classes)}

    for thresh in thresholds:
        mAP_t, pc_t = _compute_ap_at_iou(
            pred_boxes_xyxy, pred_scores, pred_labels,
            gt_boxes_xyxy, gt_labels, thresh, num_classes,
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


def generate_training_curves(history: LossHistory, output_dir: Path) -> None:
    """Generate training_curves.png from LossHistory.

    Plots 3 subplots (cls_loss, box_loss, total_loss) over epoch indices
    and saves the result as training_curves.png in output_dir.

    Args:
        history: LossHistory with per-epoch loss data.
        output_dir: Directory to save the PNG file.
    """
    if not history.total_loss:
        logger.warning("No loss data to plot — skipping training_curves.png")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable, skipping training_curves.png")
        return

    try:
        epochs = range(1, len(history.total_loss) + 1)

        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        axes[0].plot(epochs, history.cls_loss, "b-", label="cls_loss")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Classification Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, history.box_loss, "r-", label="box_loss")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Box Regression Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, history.total_loss, "g-", label="total_loss")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Loss")
        axes[2].set_title("Total Loss")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Training curves saved to %s", output_dir / "training_curves.png")

    except (RuntimeError, ValueError) as e:
        logger.warning("Could not generate training_curves.png: %s", e)
