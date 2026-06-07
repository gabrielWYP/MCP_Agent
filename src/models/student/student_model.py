"""
StudentModel — YOLOv8-Nano integration for knowledge distillation.

Composes CSPDarknet-Nano backbone → PANet neck → DecoupledHead into a
single forward pipeline that produces a 7-key output dict matching the
teacher MasterModel's distillation contract.

Input:  RGB only (N, 3, 640, 640)
Output: 7-key dict with detection predictions + intermediate features for KD

Pipeline:
    x (N,3,640,640)
        ↓
    CSPDarknetNano → stages [S1..S4], distill (S3, S4)
        ↓
    PANet → {p3, p4, p5}
        ↓
    YOLOStudentHead → preds, cls_preds, reg_preds, distill_head_cls, distill_head_reg
        ↓
    7-key output dict

Channel contract (all dims locked to ProjectionLayers defaults):
    distill_backbone:  [128, 256]       → backbone_projections student_channels
    distill_fpn:       [128, 256, 256]  → fpn_projections student_channels
    distill_head_cls:  [64, 128, 256]   → head_projections student_channels
    distill_head_reg:  [64, 128, 256]   → head_projections student_channels
"""

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from src.models.student.backbone import CSPDarknetNano
from src.models.student.neck import PANet
from src.models.student.head import YOLOStudentHead, NUM_CLASSES


class StudentModel(nn.Module):
    """
    YOLOv8-Nano student model for knowledge distillation.

    Single-modality RGB input, produces 7-key output dict with detection
    predictions and intermediate features at backbone, FPN, and head levels.

    Args:
        num_classes: Number of detection classes (default 2: sano, danado).
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.backbone = CSPDarknetNano()
        self.neck = PANet()
        self.head = YOLOStudentHead(num_classes=num_classes)

    def forward(self, x: Tensor) -> dict[str, Any]:
        """
        Forward pass: backbone → neck → head → 7-key output dict.

        Args:
            x: (N, 3, 640, 640) RGB input tensor.

        Returns:
            dict with exactly 7 keys:
                'preds':            [(N,6,80,80), (N,6,40,40), (N,6,20,20)]
                'cls_preds':        [(N,2,80,80), (N,2,40,40), (N,2,20,20)]
                'reg_preds':        [(N,4,80,80), (N,4,40,40), (N,4,20,20)]
                'distill_backbone': [(N,128,40,40), (N,256,20,20)]
                'distill_fpn':      [(N,128,80,80), (N,256,40,40), (N,256,20,20)]
                'distill_head_cls': [(N,64,80,80), (N,128,40,40), (N,256,20,20)]
                'distill_head_reg': [(N,64,80,80), (N,128,40,40), (N,256,20,20)]
        """
        # Backbone: extract multi-scale features
        stages, (s3, s4) = self.backbone(x)

        # Neck: bidirectional feature fusion (FPN + PAN)
        fpn = self.neck(stages)

        # Head: decoupled detection predictions
        head_out = self.head(fpn)

        return {
            # Detection outputs
            "preds": head_out["preds"],
            "cls_preds": head_out["cls_preds"],
            "reg_preds": head_out["reg_preds"],
            # Distillation features
            "distill_backbone": [s3, s4],
            "distill_fpn": [fpn["p3"], fpn["p4"], fpn["p5"]],
            "distill_head_cls": head_out["distill_head_cls"],
            "distill_head_reg": head_out["distill_head_reg"],
        }

    def count_parameters(self) -> dict[str, int]:
        """Returns parameter counts per module and total."""
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "backbone": count(self.backbone),
            "neck": count(self.neck),
            "head": count(self.head),
            "total": count(self),
        }
