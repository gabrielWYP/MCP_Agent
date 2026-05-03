"""
Master Model — Full architecture for multimodal mango damage detection.

Input:  RGB image (N, 3, H, W) + NIR image (N, 1, H, W)
Output: Detection results + intermediate features for knowledge distillation

Full pipeline:
    RGB + NIR
        ↓
    DualConvNeXtBackbone (shared weights, separate stems)
        ↓
    [rgb_features, nir_features]  — 4 stages each
        ↓
    CrossModalFusion (cross-attention per stage)
        ↓
    [fused_features, nir_features]
        ↓
    DualFPN (two FPNs + per-level fusion)
        ↓
    unified_pyramid [P2, P3, P4, P5]
        ↓
    CascadeRCNNHead (3 stages, IoU 0.5 / 0.6 / 0.7)
        ↓
    {cls_scores, bbox_deltas, distill_feats}

Distillation outputs exposed:
    - backbone_rgb_features:  [S1..S4] from RGB encoder
    - backbone_fused_features: [F1..F4] after cross-attention
    - fpn_pyramid:            [P2..P5] unified pyramid
    - head_distill_feats:     [FC2_s1, FC2_s2, FC2_s3] from cascade stages
"""

import torch
import torch.nn as nn

from .backbone import DualConvNeXtBackbone
from .fusion import CrossModalFusion
from .neck import DualFPN
from .head import CascadeRCNNHead


class MasterModel(nn.Module):
    """
    Multimodal master model for cross-modal knowledge distillation.

    Args:
        num_classes (int): Detection classes including background (default 3).
        pretrained_backbone (bool): Load ImageNet weights for ConvNeXt stages.
        fpn_channels (int): FPN output channels (default 256).
        fc_channels (int): Cascade R-CNN FC hidden size (default 1024).
        roi_out_size (int): RoIAlign output size (default 7).
        fusion_dropout (float): Dropout in cross-attention fusion.
        fpn_dropout (float): Dropout in FPN fusion convs.
    """

    # ConvNeXt-Small stage channels
    STAGE_CHANNELS = [96, 192, 384, 768]

    def __init__(
        self,
        num_classes: int = 3,
        pretrained_backbone: bool = True,
        fpn_channels: int = 256,
        fc_channels: int = 1024,
        roi_out_size: int = 7,
        fusion_dropout: float = 0.1,
        fpn_dropout: float = 0.1,
    ):
        super().__init__()

        # --- 1. Dual backbone (shared weights, separate stems) ---
        self.backbone = DualConvNeXtBackbone(pretrained=pretrained_backbone)

        # --- 2. Cross-modal attention fusion ---
        self.fusion = CrossModalFusion(
            stage_channels=self.STAGE_CHANNELS,
            dropout=fusion_dropout,
        )

        # --- 3. Dual FPN neck ---
        self.neck = DualFPN(
            in_channels=self.STAGE_CHANNELS,
            out_channels=fpn_channels,
            dropout=fpn_dropout,
        )

        # --- 4. Cascade R-CNN head ---
        self.head = CascadeRCNNHead(
            roi_out_size=roi_out_size,
            fpn_channels=fpn_channels,
            fc_channels=fc_channels,
            num_classes=num_classes,
        )

        # Storage for distillation features (populated during forward)
        self._distill_cache: dict[str, list[torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        rgb: torch.Tensor,
        nir: torch.Tensor,
        proposals: list[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            rgb:       (N, 3, H, W) — RGB image, normalized [0,1] or ImageNet norm
            nir:       (N, 1, H, W) — NIR grayscale image, normalized [0,1]
            proposals: list of (M_i, 4) proposal boxes per image.
                       Required for training. At inference, an RPN would generate these.
                       Format: (x1, y1, x2, y2) in pixel coordinates.

        Returns:
            dict with keys:
                'cls_scores'             : list of 3 tensors (cascade stages)
                'bbox_deltas'            : list of 3 tensors (cascade stages)
                'distill_backbone_rgb'   : [S1..S4] RGB backbone features
                'distill_backbone_fused' : [F1..F4] cross-attention fused features
                'distill_fpn'            : [P2..P5] unified FPN pyramid
                'distill_head'           : [FC2_s1, FC2_s2, FC2_s3] head features
        """
        # --- Backbone ---
        rgb_features, nir_features = self.backbone(rgb, nir)
        # rgb_features: [S1..S4], channels [96, 192, 384, 768]
        # nir_features: [S1..S4], same channels

        # --- Cross-modal fusion ---
        fused_features, nir_features_pass = self.fusion(rgb_features, nir_features)
        # fused_features: [F1..F4] — RGB enriched with NIR context

        # --- Dual FPN ---
        pyramid = self.neck(fused_features, nir_features_pass)
        # pyramid: [P2..P5] at fpn_channels (256)

        # --- Detection head ---
        if proposals is None:
            # Inference mode: return features only (RPN not implemented here)
            # In production, plug in an RPN or use pre-computed proposals
            head_output = {"cls_scores": [], "bbox_deltas": [], "distill_feats": []}
        else:
            head_output = self.head(pyramid, proposals)

        # --- Assemble output with all distillation features ---
        return {
            # Detection outputs
            "cls_scores":   head_output["cls_scores"],
            "bbox_deltas":  head_output["bbox_deltas"],

            # Distillation features — exposed for student training
            "distill_backbone_rgb":    rgb_features,      # [S1..S4]
            "distill_backbone_fused":  fused_features,    # [F1..F4]
            "distill_fpn":             pyramid,           # [P2..P5]
            "distill_head":            head_output.get("distill_feats", []),  # [FC2 x3]
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def freeze_backbone(self, freeze_stages: int = 2):
        """
        Freeze the first N stages of the shared backbone.
        Useful for fine-tuning with a small dataset — freeze early stages
        (which capture generic features) and only train later stages.

        Args:
            freeze_stages: Number of stages to freeze (0-4).
                           0 = freeze nothing, 4 = freeze all stages.
        """
        # Freeze stems
        for param in self.backbone.rgb_stem.parameters():
            param.requires_grad = False
        for param in self.backbone.nir_stem.parameters():
            param.requires_grad = False

        # Freeze shared stages up to freeze_stages
        for i, stage in enumerate(self.backbone.shared_stages):
            if i < freeze_stages:
                for param in stage.parameters():
                    param.requires_grad = False

        frozen = freeze_stages
        print(f"[MasterModel] Frozen: stems + first {frozen} shared stages.")

    def count_parameters(self) -> dict[str, int]:
        """Returns parameter counts per module."""
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "backbone":   count(self.backbone),
            "fusion":     count(self.fusion),
            "neck":       count(self.neck),
            "head":       count(self.head),
            "total":      count(self),
        }
