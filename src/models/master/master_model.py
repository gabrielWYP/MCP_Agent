"""
Master Model — Full architecture for multimodal mango damage detection.

Input:  RGB image (N, 3, H, W) + NIR image (N, 1, H, W)
Output: YOLO-style detections + intermediate features for knowledge distillation

fusion-redesign: simple early fusion. RGB and NIR are stacked into one
4-channel tensor and fed through a single-stream backbone — no cross-modal
attention module, no second FPN. `forward(rgb, nir)` keeps its two-tensor
signature so `dataset.py`, `Trainer._train_epoch`, and `KDTrainer` need no
call-site edit; the concatenation is internal to the model.

Full pipeline:
    RGB + NIR
        ↓ cat(dim=1)
    EarlyFusionBackbone (single stem, single stage stack)
        ↓
    [S1..S4] — 4 stage features
        ↓
    FPNNeck ( SingleFPN, unchanged )
        ↓
    pyramid — levels named by `head_strides`, default [P2, P3, P4, P5]
        ↓
    YOLODetectionHead (anchor-free, decoupled)
        ↓
    {preds, cls_preds, reg_preds, distill_backbone, distill_fpn,
     distill_head_cls, distill_head_reg}   — 7 keys

Distillation outputs exposed:
    - distill_backbone:  [S1..S4] from the single backbone stream
    - distill_fpn:       the emitted pyramid levels (per `head_strides`)
    - distill_head_cls:  [cls_stem per level] head classification features
    - distill_head_reg:  [reg_stem per level] head regression features
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import EarlyFusionBackbone
from .neck import FPNNeck
from .head import YOLODetectionHead, NUM_CLASSES
from src.training.strides import DEFAULT_HEAD_STRIDES, validate_strides


class MasterModel(nn.Module):
    """
    Multimodal master model for cross-modal knowledge distillation.

    Args:
        num_classes (int): Detection classes (default 2: mango, danado).
        pretrained_backbone (bool): Load ImageNet weights for ConvNeXt stages
            and inflate the stem (mean-preserving for `in_channels=4`,
            verbatim for `in_channels=3`).
        fpn_channels (int): FPN output channels (default 256).
        head_strides (list[int] | None): Pyramid strides the head is built
            over, finest-first. Default `[4, 8, 16, 32]` — includes the
            reconnected P2 level (fusion-redesign D-3). Pass `[8, 16, 32]`
            to reproduce the pre-redesign 3-level pyramid.
        backbone_variant (str): ConvNeXt variant — "tiny" or "small".
        in_channels (int): Backbone stem input channels. 4 (default) for
            the early-fused RGB+NIR input; 3 for the RGB-only H-D control
            arm. `forward` always takes `(rgb, nir)`; with `in_channels=3`,
            `nir` is accepted but ignored (control-arm convenience so call
            sites do not need a separate code path).
    """

    # ConvNeXt stage channels (identical for Tiny and Small)
    STAGE_CHANNELS = [96, 192, 384, 768]

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained_backbone: bool = True,
        fpn_channels: int = 256,
        head_strides: list[int] | None = None,
        backbone_variant: str = "tiny",
        in_channels: int = 4,
    ):
        super().__init__()

        strides = list(head_strides) if head_strides is not None else list(DEFAULT_HEAD_STRIDES)
        validate_strides(strides)
        self.head_strides = strides
        self.in_channels = in_channels

        # --- 1. Single-stream early-fusion backbone ---
        self.backbone = EarlyFusionBackbone(
            pretrained=pretrained_backbone, variant=backbone_variant, in_channels=in_channels
        )

        # --- 2. FPN neck (SingleFPN, wrapped to emit the configured levels) ---
        self.neck = FPNNeck(
            in_channels=self.STAGE_CHANNELS,
            out_channels=fpn_channels,
            strides=tuple(strides),
        )

        # --- 3. YOLO-style detection head (compatible with YOLO Nano student) ---
        self.head = YOLODetectionHead(
            fpn_channels=fpn_channels,
            num_classes=num_classes,
            strides=strides,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, rgb: torch.Tensor, nir: torch.Tensor) -> dict:
        """
        Args:
            rgb: (N, 3, H, W) — RGB image, normalized.
            nir: (N, 1, H, W) — NIR grayscale image, normalized. Ignored
                when `self.in_channels == 3` (the H-D RGB-only control arm),
                accepted anyway so call sites do not need a separate path.

        Returns:
            dict with keys:
                'preds'              : list of (B, nc+4, H_i, W_i) per level
                'cls_preds'          : list of (B, nc, H_i, W_i) per level
                'reg_preds'          : list of (B, 4, H_i, W_i) per level
                'distill_backbone'   : [S1..S4] backbone stage features
                'distill_fpn'        : emitted FPN pyramid levels
                'distill_head_cls'   : [cls_stem per level] head features
                'distill_head_reg'   : [reg_stem per level] head features
        """
        # --- Early fusion: stack RGB+NIR into one input tensor ---
        if self.in_channels == 4:
            x = torch.cat([rgb, nir], dim=1)  # (N, 4, H, W)
        else:
            x = rgb  # H-D RGB-only control arm

        # --- Backbone (single stream) ---
        backbone_features = self.backbone(x)  # [S1..S4], channels [96, 192, 384, 768]

        # --- FPN neck ---
        pyramid = self.neck(backbone_features)  # levels named by self.head_strides

        # --- YOLO Detection Head ---
        head_output = self.head(pyramid)

        return {
            # Detection outputs
            "preds":      head_output["preds"],
            "cls_preds":  head_output["cls_preds"],
            "reg_preds":  head_output["reg_preds"],

            # Distillation features — exposed for student training
            "distill_backbone": backbone_features,          # [S1..S4]
            "distill_fpn":       pyramid,                    # emitted levels
            "distill_head_cls":  head_output["distill_cls"], # [cls_stem per level]
            "distill_head_reg":  head_output["distill_reg"], # [reg_stem per level]
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def freeze_backbone(self, freeze_stages: int = 2):
        """
        Freeze the stem plus the first N stages of the backbone.

        No stem/stage asymmetry: there is only one stream now, so the stem
        is frozen or trainable exactly like the two-stream design's shared
        stages were (fusion-redesign D-4 — the previous
        `rgb_stem`-always-frozen / `nir_stem`-always-trainable split had no
        remaining rationale once there is a single input stem).

        Args:
            freeze_stages: Number of stages to freeze (0-4).
                           0 = freeze nothing but the stem is still governed
                           by this call (see below), 4 = freeze all stages.
        """
        for param in self.backbone.stem.parameters():
            param.requires_grad = freeze_stages == 0

        for i, stage in enumerate(self.backbone.stages):
            if i < freeze_stages:
                for param in stage.parameters():
                    param.requires_grad = False
            else:
                for param in stage.parameters():
                    param.requires_grad = True

        print(f"[MasterModel] Frozen: stem (if freeze_stages>0) + first {freeze_stages} stages.")

    def unfreeze_backbone_stages(
        self,
        unfreeze_stages: list[int],
        unfreeze_stem: bool = False,
    ):
        """
        Unfreeze specific backbone stages by index (0-based).

        Args:
            unfreeze_stages: List of stage indices to unfreeze (e.g., [2, 3] for stages 3-4).
            unfreeze_stem: If True, also unfreeze the stem. Renamed from the
                previous `unfreeze_rgb_stem` (fusion-redesign D-4) — there is
                only one stem now, so the RGB/NIR asymmetry no longer
                applies. Defaults to False to preserve prior call-site
                behavior for any caller that does not explicitly opt in.
        """
        for i, stage in enumerate(self.backbone.stages):
            if i in unfreeze_stages:
                for param in stage.parameters():
                    param.requires_grad = True

        if unfreeze_stem:
            for param in self.backbone.stem.parameters():
                param.requires_grad = True

        print(
            f"[MasterModel] Unfrozen stages: {unfreeze_stages}. "
            f"Stem {'unfrozen' if unfreeze_stem else 'unchanged'}."
        )

    def count_parameters(self) -> dict[str, int]:
        """Returns parameter counts per module."""
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "backbone":   count(self.backbone),
            "neck":       count(self.neck),
            "head":       count(self.head),
            "total":      count(self),
        }
