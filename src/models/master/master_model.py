"""
Master Model — Full architecture for multimodal mango damage detection.

Input:  RGB image (N, 3, H, W) + NIR image (N, 1, H, W)
Output: YOLO-style detections + intermediate features for knowledge distillation

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
    unified_pyramid [P3, P4, P5]
        ↓
    YOLODetectionHead (anchor-free, decoupled)
        ↓
    {preds, cls_preds, reg_preds, distill_feats}

Distillation outputs exposed:
    - backbone_rgb_features:  [S1..S4] from RGB encoder
    - backbone_fused_features: [F1..F4] after cross-attention
    - fpn_pyramid:            [P3..P5] unified pyramid
    - head_cls_feats:         [cls_stem_P3, cls_stem_P4, cls_stem_P5]
    - head_reg_feats:         [reg_stem_P3, reg_stem_P4, reg_stem_P5]
"""

import torch
import torch.nn as nn

from .backbone import DualConvNeXtBackbone
from .fusion import CrossModalFusion
from .neck import DualFPN
from .head import YOLODetectionHead, NUM_CLASSES


class MasterModel(nn.Module):
    """
    Multimodal master model for cross-modal knowledge distillation.

    Args:
        num_classes (int): Detection classes (default 2: mango, danado).
        pretrained_backbone (bool): Load ImageNet weights for ConvNeXt stages.
        fpn_channels (int): FPN output channels (default 256).
        fusion_dropout (float): Dropout in cross-attention fusion.
        fpn_dropout (float): Dropout in FPN fusion convs.
        head_strides (list[int]): Strides for YOLO head levels.
        backbone_variant (str): ConvNeXt variant — "tiny" or "small".
    """

    # ConvNeXt stage channels (identical for Tiny and Small)
    STAGE_CHANNELS = [96, 192, 384, 768]

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained_backbone: bool = True,
        fpn_channels: int = 256,
        fusion_dropout: float = 0.1,
        fpn_dropout: float = 0.1,
        head_strides: list[int] = None,
        backbone_variant: str = "tiny",
    ):
        super().__init__()

        # --- 1. Dual backbone (shared weights, separate stems) ---
        self.backbone = DualConvNeXtBackbone(
            pretrained=pretrained_backbone, variant=backbone_variant
        )

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

        # --- 4. YOLO-style detection head (compatible with YOLO Nano student) ---
        self.head = YOLODetectionHead(
            fpn_channels=fpn_channels,
            num_classes=num_classes,
            strides=head_strides or [8, 16, 32],
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        rgb: torch.Tensor,
        nir: torch.Tensor,
        return_attention: bool = False,
    ) -> dict:
        """
        Args:
            rgb: (N, 3, H, W) — RGB image, normalized
            nir: (N, 1, H, W) — NIR grayscale image, normalized
            return_attention: If True, include 'attention_maps' in output dict.

        Returns:
            dict with keys:
                'preds'              : list of (B, nc+4, H_i, W_i) per level
                'cls_preds'          : list of (B, nc, H_i, W_i) per level
                'reg_preds'          : list of (B, 4, H_i, W_i) per level
                'distill_backbone_rgb'   : [S1..S4] RGB backbone features
                'distill_backbone_fused' : [F1..F4] cross-attention fused features
                'distill_fpn'            : [P3..P5] unified FPN pyramid
                'distill_head_cls'       : [cls_stem x3] head classification features
                'distill_head_reg'       : [reg_stem x3] head regression features
                'attention_maps'         : (if return_attention) [M1..M4] each (N,H,W)
        """
        # --- Backbone ---
        rgb_features, nir_features = self.backbone(rgb, nir)
        # rgb_features: [S1..S4], channels [96, 192, 384, 768]
        # nir_features: [S1..S4], same channels

        # --- Cross-modal fusion ---
        if return_attention:
            fused_features, nir_features_pass, attn_maps = self.fusion(
                rgb_features, nir_features, return_attention=True
            )
        else:
            fused_features, nir_features_pass = self.fusion(rgb_features, nir_features)
        # fused_features: [F1..F4] — RGB enriched with NIR context

        # --- Dual FPN ---
        pyramid = self.neck(fused_features, nir_features_pass)
        # pyramid: [P3..P5] at fpn_channels (256)

        # --- YOLO Detection Head ---
        head_output = self.head(pyramid)

        # --- Assemble output with all distillation features ---
        output = {
            # Detection outputs
            "preds":      head_output["preds"],
            "cls_preds":  head_output["cls_preds"],
            "reg_preds":  head_output["reg_preds"],

            # Distillation features — exposed for student training
            "distill_backbone_rgb":   rgb_features,          # [S1..S4]
            "distill_backbone_fused": fused_features,        # [F1..F4]
            "distill_fpn":            pyramid,               # [P3..P5]
            "distill_head_cls":       head_output["distill_cls"],  # [cls_stem x3]
            "distill_head_reg":       head_output["distill_reg"],  # [reg_stem x3]
        }

        if return_attention:
            output["attention_maps"] = attn_maps  # [M1..M4] each (N, H, W)

        return output

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
        # Freeze RGB stem: it starts from ImageNet and is high-risk to overfit.
        for param in self.backbone.rgb_stem.parameters():
            param.requires_grad = False

        # Keep NIR stem trainable: it is the modality adapter for 850 nm images.
        for param in self.backbone.nir_stem.parameters():
            param.requires_grad = True

        # Freeze shared stages up to freeze_stages
        for i, stage in enumerate(self.backbone.shared_stages):
            if i < freeze_stages:
                for param in stage.parameters():
                    param.requires_grad = False

        frozen = freeze_stages
        print(
            f"[MasterModel] Frozen: RGB stem + first {frozen} shared stages. "
            "NIR stem remains trainable."
        )

    def unfreeze_backbone_stages(self, unfreeze_stages: list[int]):
        """
        Unfreeze specific backbone stages by index (0-based).
        Stems always remain frozen to preserve modality-specific preprocessing.

        Args:
            unfreeze_stages: List of stage indices to unfreeze (e.g., [2, 3] for stages 3-4).
        """
        for i, stage in enumerate(self.backbone.shared_stages):
            if i in unfreeze_stages:
                for param in stage.parameters():
                    param.requires_grad = True

        print(
            f"[MasterModel] Unfrozen stages: {unfreeze_stages}. "
            "RGB stem remains frozen; NIR stem remains trainable."
        )

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
