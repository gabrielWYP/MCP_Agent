"""
YOLOv8-style Detection Head — anchor-free, decoupled.

Diseñado para ser compatible con YOLO Nano (estudiante) en knowledge distillation.

Arquitectura por nivel FPN:
    P_i (256ch) ──► conv 3×3 ──► conv_cls (nc)
                  └─► conv 3×3 ──► conv_reg (4)

Output por nivel: (B, nc + 4, H_i, W_i)
    - nc = num_classes (mango, danado)
    - 4  = bbox deltas (xywh)

Clases:
    0: mango
    1: danado

Para distilación con YOLO Nano (estudiante, fijo en 3 niveles [8, 16, 32]):
    - Mismo formato de output: (B, 6, H_i, W_i) por nivel
    - El maestro (`MasterModel`) es configurable via `head_strides`
      (default [4, 8, 16, 32], fusion-redesign D-3); KD selecciona por
      stride los 3 niveles que coinciden con el estudiante — ver
      `src/training/kd_trainer.py` y `src/training/strides.py`.
    - Features intermedios expuestos para feature-level KD, computados una
      sola vez por nivel (fusion-redesign W5 — antes se recomputaban)

Referencias:
    - YOLOv8: https://github.com/ultralytics/ultralytics
    - TOOD: Task-aligned One-stage Object Detection, ICCV 2021
"""

import torch
import torch.nn as nn


# Número de clases de detección (sin background — YOLO es class-specific)
NUM_CLASSES = 2  # mango, danado


class DecoupledHead(nn.Module):
    """
    Cabeza de detección decoupled para UN nivel de la pirámide FPN.

    Separa las tareas de clasificación y regresión en ramas independientes,
    lo que mejora la convergencia y la precisión (patrón estándar en YOLOv8).

    Args:
        in_channels (int): Canales de entrada del nivel FPN (default 256).
        num_classes (int): Clases de deteccion (default 2).
    """

    def __init__(self, in_channels: int = 256, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

        # Rama de clasificación
        self.cls_stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        # Rama de regresión de bounding boxes
        self.reg_stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.reg_pred = nn.Conv2d(in_channels, 4, kernel_size=1)  # xywh

        self._init_weights()

    def _init_weights(self):
        """Inicialización consistente con YOLOv8."""
        import math
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # YOLOv8 standard: initialize classification bias to give ~0.01 prior probability
        # This prevents the model from predicting "object everywhere" at epoch 0
        prior_prob = 0.01
        bias_value = -math.log((1.0 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, in_channels, H, W) — feature map de un nivel FPN

        Returns:
            cls_out: (B, num_classes, H, W)
            reg_out: (B, 4, H, W)
            cls_feat: (B, in_channels, H, W) — cls_stem output, reused by the
                caller for distillation (fusion-redesign W5) instead of
                recomputing `cls_stem` a second time.
            reg_feat: (B, in_channels, H, W) — reg_stem output, same reuse.
        """
        cls_feat = self.cls_stem(x)
        reg_feat = self.reg_stem(x)
        cls_out = self.cls_pred(cls_feat)
        reg_out = self.reg_pred(reg_feat)
        return cls_out, reg_out, cls_feat, reg_feat


class YOLODetectionHead(nn.Module):
    """
    Head de detección completo estilo YOLOv8 sobre la pirámide FPN.

    Aplica un DecoupledHead independiente en cada nivel [P3, P4, P5].

    Args:
        fpn_channels (int): Canales de cada nivel FPN (default 256).
        num_classes (int): Clases de deteccion (default 2).
        strides (list[int]): Strides de cada nivel FPN. Requerido, sin
            default (fusion-redesign D-D) — un argumento olvidado produce un
            `TypeError` en el call site en vez de un `[8, 16, 32]` silencioso
            y potencialmente incorrecto para un pyramid de 4 niveles.

    Forward returns:
        dict con:
            'preds':        list de tensors (B, nc+4, H_i, W_i) por nivel
            'cls_preds':    list de tensors (B, nc, H_i, W_i) por nivel
            'reg_preds':    list de tensors (B, 4, H_i, W_i) por nivel
            'distill_cls':  list de features cls_stem para KD
            'distill_reg':  list de features reg_stem para KD
    """

    def __init__(
        self,
        fpn_channels: int = 256,
        num_classes: int = NUM_CLASSES,
        *,
        strides: list[int],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = list(strides)
        self.num_levels = len(self.strides)

        # Un decoupled head por nivel FPN
        self.heads = nn.ModuleList([
            DecoupledHead(in_channels=fpn_channels, num_classes=num_classes)
            for _ in range(self.num_levels)
        ])

    def forward(self, pyramid: list[torch.Tensor]) -> dict[str, list]:
        """
        Args:
            pyramid: [P3, P4, P5] — output del FPN, cada uno (B, 256, H_i, W_i)

        Returns:
            dict con predicciones y features para distilación.
        """
        assert len(pyramid) == self.num_levels, (
            f"Expected {self.num_levels} FPN levels, got {len(pyramid)}"
        )

        all_cls = []
        all_reg = []
        all_preds = []
        distill_cls_feats = []
        distill_reg_feats = []

        for i, head in enumerate(self.heads):
            feat = pyramid[i]  # (B, 256, H_i, W_i)

            # fusion-redesign W5: DecoupledHead.forward returns the stem
            # features it already computed, so they are reused directly for
            # distillation instead of calling head.cls_stem(feat) /
            # head.reg_stem(feat) again — this used to run each stem twice
            # per level, per forward, doubling head activation memory (the
            # duplicate copy retained by autograd during training).
            cls_out, reg_out, cls_feat, reg_feat = head(feat)
            # Concatenar: (B, nc + 4, H_i, W_i)
            pred = torch.cat([cls_out, reg_out], dim=1)

            all_cls.append(cls_out)
            all_reg.append(reg_out)
            all_preds.append(pred)

            distill_cls_feats.append(cls_feat)
            distill_reg_feats.append(reg_feat)

        return {
            "preds":            all_preds,        # [(B, 6, H3, W3), (B, 6, H4, W4), (B, 6, H5, W5)]
            "cls_preds":        all_cls,           # [(B, 2, H3, W3), ...]
            "reg_preds":        all_reg,           # [(B, 4, H3, W3), ...]
            "distill_cls":      distill_cls_feats, # features intermedios para KD
            "distill_reg":      distill_reg_feats, # features intermedios para KD
        }
