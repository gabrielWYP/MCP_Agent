"""
YOLOv8-style detection loss with Task-Aligned Assigner (TAL).

Implements:
    - TaskAlignedAssigner: matches predictions to GT using alignment metric
    - YOLOv8Loss: class-weighted BCE + CIoU regression loss

References:
    - TOOD: Task-aligned One-stage Object Detection (ICCV 2021)
    - YOLOv8: https://github.com/ultralytics/ultralytics
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import complete_box_iou_loss, box_iou


class TaskAlignedAssigner:
    """Task-Aligned Assigner for YOLOv8-style detection.

    Computes alignment metric = cls_score^alpha * iou^beta, selects top-k
    positive anchors per ground truth box.

    Args:
        topk: Number of candidate anchors per GT.
        alpha: Exponent for classification score in alignment metric.
        beta: Exponent for IoU in alignment metric.
    """

    def __init__(self, topk: int = 13, alpha: float = 1.0, beta: float = 6.0):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def __call__(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        gt_labels: list[torch.Tensor],
        gt_bboxes: list[torch.Tensor],
        anchors: torch.Tensor,
        strides: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign predictions to ground truth targets.

        Args:
            pred_scores: (B, num_anchors, num_classes) predicted class scores (sigmoid).
            pred_bboxes: (B, num_anchors, 4) predicted bboxes in cxcywh format.
            gt_labels: List of (N_i,) GT class labels per image.
            gt_bboxes: List of (N_i, 4) GT bboxes per image in cxcywh format.
            anchors: (num_anchors, 2) anchor center points in pixel coords.
            strides: List of stride values per FPN level.

        Returns:
            target_classes: (B, num_anchors) assigned class IDs (-1 for negative).
            target_bboxes: (B, num_anchors, 4) assigned GT bboxes.
            target_scores: (B, num_anchors) alignment scores for positive samples.
            fg_mask: (B, num_anchors) boolean mask of foreground (positive) anchors.
        """
        B, num_anchors, num_classes = pred_scores.shape
        device = pred_scores.device

        target_classes = torch.full((B, num_anchors), -1, dtype=torch.long, device=device)
        target_bboxes = torch.zeros((B, num_anchors, 4), device=device)
        target_scores = torch.zeros((B, num_anchors), device=device)
        fg_mask = torch.zeros((B, num_anchors), dtype=torch.bool, device=device)

        for b in range(B):
            n_gt = len(gt_labels[b])
            if n_gt == 0:
                continue

            gt_cls = gt_labels[b]  # (N_gt,)
            gt_box = gt_bboxes[b]  # (N_gt, 4) cxcywh

            # Compute alignment metric for each GT-anchor pair
            # pred_scores[b]: (num_anchors, num_classes)
            # For each GT, get the predicted score for that GT's class
            gt_cls_scores = pred_scores[b, :, gt_cls]  # (num_anchors, N_gt)

            # Compute IoU between predicted bboxes and GT bboxes
            # Convert cxcywh to xyxy for IoU computation
            pred_xyxy = self._cxcywh_to_xyxy(pred_bboxes[b])  # (num_anchors, 4)
            gt_xyxy = self._cxcywh_to_xyxy(gt_box)  # (N_gt, 4)
            ious = box_iou(pred_xyxy, gt_xyxy)  # (num_anchors, N_gt)

            # Alignment metric: score^alpha * iou^beta
            alignment = (gt_cls_scores ** self.alpha) * (ious ** self.beta)  # (num_anchors, N_gt)

            # Select top-k anchors per GT
            topk = min(self.topk, num_anchors)
            topk_vals, topk_idx = alignment.topk(topk, dim=0)  # (topk, N_gt)

            for gt_i in range(n_gt):
                # Positive anchors for this GT
                pos_idx = topk_idx[:, gt_i]  # (topk,)
                pos_scores = topk_vals[:, gt_i]  # (topk,)

                # Filter out zero-score anchors
                valid = pos_scores > 0
                pos_idx = pos_idx[valid]
                pos_scores = pos_scores[valid]

                if len(pos_idx) == 0:
                    continue

                # Check if anchor center is inside GT box (spatial constraint)
                anchor_centers = anchors[pos_idx]  # (K, 2)
                gt_cxcy = gt_box[gt_i, :2]  # (2,)
                gt_wh = gt_box[gt_i, 2:]  # (2,)
                gt_min = gt_cxcy - gt_wh / 2
                gt_max = gt_cxcy + gt_wh / 2

                inside = (
                    (anchor_centers[:, 0] >= gt_min[0]) &
                    (anchor_centers[:, 0] <= gt_max[0]) &
                    (anchor_centers[:, 1] >= gt_min[1]) &
                    (anchor_centers[:, 1] <= gt_max[1])
                )

                pos_idx = pos_idx[inside]
                pos_scores = pos_scores[inside]

                if len(pos_idx) == 0:
                    continue

                # Handle conflicts: if an anchor is assigned to multiple GTs,
                # keep the one with highest alignment score
                for idx, score in zip(pos_idx, pos_scores):
                    if target_scores[b, idx] < score or target_classes[b, idx] == -1:
                        target_classes[b, idx] = gt_cls[gt_i]
                        target_bboxes[b, idx] = gt_box[gt_i]
                        target_scores[b, idx] = score
                        fg_mask[b, idx] = True

        return target_classes, target_bboxes, target_scores, fg_mask

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)


def _generate_anchors(
    feat_sizes: list[tuple[int, int]],
    strides: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    """Generate anchor center points for all FPN levels.

    Args:
        feat_sizes: List of (H_i, W_i) per FPN level.
        strides: List of stride values per level.
        device: Target device.

    Returns:
        anchors: (num_anchors, 2) center points in pixel coords.
        num_per_level: Number of anchors per level.
    """
    all_anchors = []
    num_per_level = []

    for (h, w), stride in zip(feat_sizes, strides):
        y = (torch.arange(h, device=device).float() + 0.5) * stride
        x = (torch.arange(w, device=device).float() + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        centers = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        all_anchors.append(centers)
        num_per_level.append(h * w)

    anchors = torch.cat(all_anchors, dim=0)
    return anchors, num_per_level


class YOLOv8Loss(nn.Module):
    """YOLOv8-style detection loss with TAL assignment.

    Combines class-weighted BCE classification loss with CIoU regression loss.

    Args:
        num_classes: Number of detection classes.
        box_weight: Weight for regression (CIoU) loss.
        cls_weight: Weight for classification (BCE) loss.
        class_weights: Per-class weights for BCE loss.
        strides: FPN level strides.
    """

    def __init__(
        self,
        num_classes: int = 2,
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        class_weights: list[float] | None = None,
        strides: list[int] | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.strides = strides or [8, 16, 32]

        if class_weights is None:
            class_weights = [1.0] * num_classes
        self.register_buffer(
            "class_weights", torch.tensor(class_weights, dtype=torch.float32)
        )

        self.assigner = TaskAlignedAssigner(topk=13, alpha=1.0, beta=6.0)

    def forward(
        self,
        predictions: list[torch.Tensor],
        targets: dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute detection loss.

        Args:
            predictions: List of (B, nc+4, H_i, W_i) per FPN level.
                First nc channels are class logits, last 4 are bbox deltas.
            targets: Dict with keys:
                bboxes: List of (N_i, 4) GT bboxes per image (cxcywh, normalized 0-1).
                labels: List of (N_i,) GT class labels per image.

        Returns:
            total_loss: Scalar loss for backward().
            loss_dict: {"box_loss": float, "cls_loss": float}.
        """
        device = predictions[0].device
        B = predictions[0].shape[0]

        # Split predictions into class scores and bbox deltas
        all_cls = []
        all_reg = []
        feat_sizes = []

        for pred in predictions:
            # pred: (B, nc+4, H, W)
            cls_pred = pred[:, :self.num_classes, :, :]  # (B, nc, H, W)
            reg_pred = pred[:, self.num_classes:, :, :]  # (B, 4, H, W)

            H, W = pred.shape[2], pred.shape[3]
            feat_sizes.append((H, W))

            # Flatten spatial: (B, nc, H*W) → (B, H*W, nc)
            cls_flat = cls_pred.flatten(2).permute(0, 2, 1)
            reg_flat = reg_pred.flatten(2).permute(0, 2, 1)

            all_cls.append(cls_flat)
            all_reg.append(reg_flat)

        # Concatenate all levels: (B, total_anchors, nc) and (B, total_anchors, 4)
        pred_cls = torch.cat(all_cls, dim=1)  # (B, A, nc)
        pred_reg = torch.cat(all_reg, dim=1)  # (B, A, 4)

        # Generate anchors
        anchors, _ = _generate_anchors(feat_sizes, self.strides, device)

        # Convert reg predictions from deltas to absolute bboxes
        # pred_reg is (cx, cy, w, h) in pixel space relative to anchors
        pred_bboxes = self._decode_bboxes(pred_reg, anchors)  # (B, A, 4) cxcywh pixel

        # Convert GT bboxes from normalized to pixel coords
        # We need the image size — assume all same size from the batch
        img_size = predictions[0].shape[2] * self.strides[0]  # approximate
        # Better: use the actual feature sizes to compute
        img_h = feat_sizes[0][0] * self.strides[0]
        img_w = feat_sizes[0][1] * self.strides[0]

        gt_bboxes_pixel = []
        gt_labels_list = targets["labels"]
        for b in range(B):
            gt_b = targets["bboxes"][b]  # (N, 4) normalized cxcywh
            if len(gt_b) > 0:
                gt_pixel = gt_b.clone()
                gt_pixel[:, 0] *= img_w  # cx
                gt_pixel[:, 1] *= img_h  # cy
                gt_pixel[:, 2] *= img_w  # w
                gt_pixel[:, 3] *= img_h  # h
                gt_bboxes_pixel.append(gt_pixel)
            else:
                gt_bboxes_pixel.append(gt_b)

        # TAL assignment
        pred_scores = pred_cls.sigmoid()
        target_classes, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores, pred_bboxes, gt_labels_list, gt_bboxes_pixel, anchors, self.strides
        )

        # --- Classification loss ---
        # BCE with logits for all anchors
        # Target: one-hot for positive, all-zero for negative
        target_cls = torch.zeros_like(pred_cls)  # (B, A, nc)
        for b in range(B):
            pos = fg_mask[b]
            if pos.any():
                pos_classes = target_classes[b, pos]  # (K,)
                # Clamp to valid range
                pos_classes = pos_classes.clamp(0, self.num_classes - 1)
                target_cls[b, pos] = F.one_hot(pos_classes, self.num_classes).float()
                # Scale by alignment score
                target_cls[b, pos] *= target_scores[b, pos].unsqueeze(1)

        # Apply class weights
        cls_weights = self.class_weights.to(device)  # (nc,)
        cls_loss = F.binary_cross_entropy_with_logits(
            pred_cls, target_cls,
            weight=cls_weights.unsqueeze(0).unsqueeze(0).expand_as(pred_cls),
            reduction="sum",
        )
        # Normalize by number of positive samples
        num_pos = fg_mask.sum().clamp(min=1)
        cls_loss = cls_loss / num_pos

        # --- Regression loss (CIoU on positives only) ---
        box_loss = torch.tensor(0.0, device=device)
        if fg_mask.any():
            for b in range(B):
                pos = fg_mask[b]
                if not pos.any():
                    continue
                pred_pos = pred_bboxes[b, pos]  # (K, 4) cxcywh
                target_pos = target_bboxes[b, pos]  # (K, 4) cxcywh

                # Convert to xyxy for CIoU
                pred_xyxy = TaskAlignedAssigner._cxcywh_to_xyxy(pred_pos)
                target_xyxy = TaskAlignedAssigner._cxcywh_to_xyxy(target_pos)

                ciou = complete_box_iou_loss(pred_xyxy, target_xyxy, reduction="sum")
                box_loss = box_loss + ciou

            box_loss = box_loss / num_pos

        # --- Total loss ---
        total_loss = self.box_weight * box_loss + self.cls_weight * cls_loss

        loss_dict = {
            "box_loss": box_loss.item(),
            "cls_loss": cls_loss.item(),
        }

        return total_loss, loss_dict

    @staticmethod
    def _decode_bboxes(
        reg_preds: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """Decode bbox regression predictions to absolute cxcywh boxes.

        reg_preds are (cx_offset, cy_offset, w, h) where offsets are relative
        to anchor centers.

        Args:
            reg_preds: (B, A, 4) raw regression outputs.
            anchors: (A, 2) anchor center points.

        Returns:
            (B, A, 4) decoded bboxes in cxcywh pixel coords.
        """
        # reg_preds: (B, A, 4) — (dx, dy, w, h)
        # Anchors: (A, 2) — (cx, cy)
        anchor_cx = anchors[:, 0].unsqueeze(0)  # (1, A)
        anchor_cy = anchors[:, 1].unsqueeze(0)  # (1, A)

        # Decode: cx = anchor_cx + dx, cy = anchor_cy + dy
        # w and h are exp-predicted
        dx = reg_preds[:, :, 0]
        dy = reg_preds[:, :, 1]
        w = reg_preds[:, :, 2].exp()
        h = reg_preds[:, :, 3].exp()

        cx = anchor_cx + dx
        cy = anchor_cy + dy

        return torch.stack([cx, cy, w, h], dim=-1)
