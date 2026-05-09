"""
Cascade R-CNN Detection Head.

Cascade R-CNN uses 3 sequential detection stages with increasing IoU thresholds:
    Stage 1: IoU threshold 0.5  (coarse proposals)
    Stage 2: IoU threshold 0.6  (refined proposals)
    Stage 3: IoU threshold 0.7  (precise proposals)

Each stage refines the bounding boxes from the previous stage, which is
especially useful for detecting subtle damage regions (small, low-contrast).

Architecture per stage:
    RoI features (via RoIAlign) → 2x FC layers → cls head + reg head

Classes for mango mechanical damage detection:
    0: background
    1: mango (healthy)
    2: mango_damaged (early mechanical damage)

Note: This is a PyTorch implementation of the Cascade R-CNN head.
For production training, consider using MMDetection's Cascade R-CNN
which has optimized CUDA kernels and battle-tested implementations.
This module is designed to be compatible with MMDetection's interface.
"""

import torch
import torch.nn as nn
import torchvision.ops as ops


# IoU thresholds for each cascade stage (standard Cascade R-CNN values)
CASCADE_IOU_THRESHOLDS = [0.5, 0.6, 0.7]

# Number of detection classes (background + healthy + damaged)
NUM_CLASSES = 3


class RoIFeatureExtractor(nn.Module):
    """
    Extracts fixed-size features from the FPN pyramid using RoIAlign.

    Assigns each proposal to the appropriate FPN level based on its size
    using the standard FPN level assignment formula from the original paper.

    Args:
        out_size (int): Output spatial size after RoIAlign (default 7x7).
        out_channels (int): FPN output channels (default 256).
        fpn_strides (list[int]): Strides of FPN levels [4, 8, 16, 32].
    """

    def __init__(
        self,
        out_size: int = 7,
        out_channels: int = 256,
        fpn_strides: list[int] = None,
    ):
        super().__init__()
        self.out_size = out_size
        self.out_channels = out_channels
        self.fpn_strides = fpn_strides or [4, 8, 16, 32]

    def _assign_level(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Assign proposals to FPN levels based on box area.
        Formula: level = floor(k0 + log2(sqrt(area) / 224))
        Clamped to [0, 3] for our 4-level FPN.
        """
        # boxes: (N, 4) in (x1, y1, x2, y2) format
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sqrt_areas = torch.sqrt(areas.clamp(min=1e-6))
        # k0=4 is standard (224 is ImageNet canonical size)
        levels = torch.floor(4 + torch.log2(sqrt_areas / 224.0))
        levels = levels.clamp(0, len(self.fpn_strides) - 1).long()
        return levels

    def forward(
        self,
        pyramid: list[torch.Tensor],
        proposals: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            pyramid: [P2, P3, P4, P5] — FPN output
            proposals: list of (M_i, 4) tensors, one per image in batch

        Returns:
            roi_features: (total_proposals, C, out_size, out_size)
        """
        all_roi_features = []

        for img_idx, boxes in enumerate(proposals):
            if boxes.shape[0] == 0:
                # No proposals for this image
                dummy = torch.zeros(
                    0, self.out_channels, self.out_size, self.out_size,
                    device=pyramid[0].device
                )
                all_roi_features.append(dummy)
                continue

            levels = self._assign_level(boxes)
            img_roi_features = torch.zeros(
                boxes.shape[0], self.out_channels, self.out_size, self.out_size,
                device=pyramid[0].device
            )

            for lvl_idx, feat_map in enumerate(pyramid):
                mask = levels == lvl_idx
                if not mask.any():
                    continue
                lvl_boxes = boxes[mask]
                # RoIAlign expects boxes as (batch_idx, x1, y1, x2, y2)
                batch_indices = torch.full(
                    (lvl_boxes.shape[0], 1), img_idx,
                    dtype=lvl_boxes.dtype, device=lvl_boxes.device
                )
                rois = torch.cat([batch_indices, lvl_boxes], dim=1)
                spatial_scale = 1.0 / self.fpn_strides[lvl_idx]
                lvl_features = ops.roi_align(
                    feat_map,
                    rois,
                    output_size=self.out_size,
                    spatial_scale=spatial_scale,
                    sampling_ratio=2,
                    aligned=True,
                )
                img_roi_features[mask] = lvl_features

            all_roi_features.append(img_roi_features)

        return torch.cat(all_roi_features, dim=0)


class CascadeStage(nn.Module):
    """
    Single stage of the Cascade R-CNN head.

    Each stage has:
        - 2 FC layers for feature processing
        - Classification head (num_classes scores)
        - Regression head (4 * num_classes bbox deltas)

    Args:
        in_channels (int): RoI feature channels * spatial_size^2.
        fc_channels (int): Hidden size of FC layers (default 1024).
        num_classes (int): Number of classes including background.
        iou_threshold (float): IoU threshold for this stage's positive samples.
    """

    def __init__(
        self,
        in_channels: int,
        fc_channels: int = 1024,
        num_classes: int = NUM_CLASSES,
        iou_threshold: float = 0.5,
    ):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes

        # Shared FC layers
        self.fc1 = nn.Linear(in_channels, fc_channels)
        self.fc2 = nn.Linear(fc_channels, fc_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

        # Classification head
        self.cls_head = nn.Linear(fc_channels, num_classes)

        # Regression head (class-agnostic: 4 deltas, not 4*num_classes)
        # Class-agnostic regression is standard in Cascade R-CNN
        self.reg_head = nn.Linear(fc_channels, 4)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_head.weight, std=0.01)
        nn.init.normal_(self.reg_head.weight, std=0.001)
        nn.init.zeros_(self.cls_head.bias)
        nn.init.zeros_(self.reg_head.bias)

    def forward(self, roi_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            roi_features: (N_rois, C, H, W)

        Returns:
            cls_scores: (N_rois, num_classes)
            bbox_deltas: (N_rois, 4)
        """
        # Flatten spatial dims
        x = roi_features.flatten(1)   # (N_rois, C*H*W)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        cls_scores = self.cls_head(x)    # (N_rois, num_classes)
        bbox_deltas = self.reg_head(x)   # (N_rois, 4)

        return cls_scores, bbox_deltas


class CascadeRCNNHead(nn.Module):
    """
    Full 3-stage Cascade R-CNN detection head.

    Also exposes intermediate features for knowledge distillation:
        - stage_features: list of 3 tensors, one per cascade stage
          These are the FC2 activations before the cls/reg heads.
          The student can distill from these.

    Args:
        roi_out_size (int): RoIAlign output spatial size (default 7).
        fpn_channels (int): FPN output channels (default 256).
        fc_channels (int): FC hidden size per stage (default 1024).
        num_classes (int): Total classes including background.
    """

    def __init__(
        self,
        roi_out_size: int = 7,
        fpn_channels: int = 256,
        fc_channels: int = 1024,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        self.roi_extractor = RoIFeatureExtractor(
            out_size=roi_out_size,
            out_channels=fpn_channels,
        )

        # Flattened RoI feature size
        roi_flat_size = fpn_channels * roi_out_size * roi_out_size

        # 3 cascade stages with increasing IoU thresholds
        self.stages = nn.ModuleList([
            CascadeStage(
                in_channels=roi_flat_size,
                fc_channels=fc_channels,
                num_classes=num_classes,
                iou_threshold=iou_thr,
            )
            for iou_thr in CASCADE_IOU_THRESHOLDS
        ])

        # Store intermediate features for distillation (set during forward)
        self.distill_features: list[torch.Tensor] = []

    def forward(
        self,
        pyramid: list[torch.Tensor],
        proposals: list[torch.Tensor],
    ) -> dict[str, list]:
        """
        Args:
            pyramid: [P2, P3, P4, P5] from DualFPN
            proposals: list of (M_i, 4) proposal boxes per image

        Returns:
            dict with:
                'cls_scores':   list of 3 tensors (one per stage)
                'bbox_deltas':  list of 3 tensors (one per stage)
                'distill_feats': list of 3 FC2 feature tensors for distillation
        """
        self.distill_features = []

        all_cls_scores = []
        all_bbox_deltas = []

        # Extract RoI features from FPN
        roi_feats = self.roi_extractor(pyramid, proposals)

        current_roi_feats = roi_feats

        for stage in self.stages:
            # Flatten for FC layers
            x = current_roi_feats.flatten(1)
            x = stage.relu(stage.fc1(x))
            x = stage.dropout(x)
            x = stage.relu(stage.fc2(x))
            x = stage.dropout(x)

            # Save FC2 features for distillation BEFORE cls/reg heads
            self.distill_features.append(x.detach())

            cls_scores = stage.cls_head(x)
            bbox_deltas = stage.reg_head(x)

            all_cls_scores.append(cls_scores)
            all_bbox_deltas.append(bbox_deltas)

            # In a full implementation, proposals would be refined here
            # using bbox_deltas before passing to the next stage.
            # For now we keep the same RoI features (simplified).
            # TODO: implement proposal refinement between stages.

        return {
            "cls_scores":    all_cls_scores,    # 3 x (N_rois, num_classes)
            "bbox_deltas":   all_bbox_deltas,   # 3 x (N_rois, 4)
            "distill_feats": self.distill_features,  # 3 x (N_rois, fc_channels)
        }
