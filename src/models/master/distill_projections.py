"""
Projection layers para Knowledge Distillation entre Maestro y Estudiante.

El maestro (ConvNeXt + FPN 256ch) y el estudiante (YOLO Nano CSPDarknet) tienen
dimensiones de features distintas en cada nivel. Estos projection layers son
1×1 convs aprendibles que mapean las features del maestro al espacio del
estudiante para que la loss de distilación (MSE) sea computable.

Arquitectura típica de distilación:

    Maestro FPN[P3] (256ch) ──► proj_1×1 ──► (student_ch) ──┐
                                                              ├──► MSE loss
    Estudiante FPN[P3] (C_ch) ───────────────────────────────┘

Se aplican en 3 niveles:
    - Backbone features (S3 del maestro → backbone intermedio del estudiante)
    - FPN features (P3/P4/P5 del maestro → P3/P4/P5 del estudiante)
    - Head features (cls_stem/reg_stem del maestro → head del estudiante)

Los pesos de estos layers se entrenan junto con el estudiante durante KD.
"""

import torch
import torch.nn as nn


class ProjectionLayers(nn.Module):
    """
    Proyecciones 1×1 para alinear dimensiones maestro → estudiante.

    Args:
        teacher_channels (list[int]): Canales del maestro por nivel.
            Default: [256, 256, 256] para FPN P3/P4/P5.
        student_channels (list[int]): Canales del estudiante por nivel.
            Default YOLO Nano: [128, 256, 256] para P3/P4/P5.
        use_bn (bool): Si True, agrega BatchNorm después de cada proyección.
    """

    def __init__(
        self,
        teacher_channels: list[int] = None,
        student_channels: list[int] = None,
        use_bn: bool = True,
    ):
        super().__init__()

        if teacher_channels is None:
            teacher_channels = [256, 256, 256]
        if student_channels is None:
            # YOLO Nano: P3=128ch, P4=256ch, P5=256ch
            student_channels = [128, 256, 256]

        assert len(teacher_channels) == len(student_channels), (
            f"Teacher ({len(teacher_channels)}) y student ({len(student_channels)}) "
            f"deben tener el mismo número de niveles"
        )

        self.num_levels = len(teacher_channels)

        self.projections = nn.ModuleList([
            self._make_proj(t_ch, s_ch, use_bn)
            for t_ch, s_ch in zip(teacher_channels, student_channels)
        ])

    def _make_proj(self, in_ch: int, out_ch: int, use_bn: bool) -> nn.Sequential:
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        return nn.Sequential(*layers)

    def forward(self, teacher_features: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Proyecta features del maestro al espacio del estudiante.

        Args:
            teacher_features: list de tensors del maestro, uno por nivel.

        Returns:
            list de tensors proyectados, mismos shapes espaciales pero
            con canales del estudiante.
        """
        assert len(teacher_features) == self.num_levels
        return [
            proj(feat)
            for proj, feat in zip(self.projections, teacher_features)
        ]


# ── Presets para configuraciones comunes ─────────────────────────────────

def fpn_projections(use_bn: bool = True) -> ProjectionLayers:
    """Proyecciones para distilación a nivel de FPN (P3/P4/P5)."""
    return ProjectionLayers(
        teacher_channels=[256, 256, 256],   # Maestro FPN
        student_channels=[128, 256, 256],    # YOLO Nano P3/P4/P5
        use_bn=use_bn,
    )


def backbone_projections(use_bn: bool = True) -> ProjectionLayers:
    """
    Proyecciones para distilación a nivel de backbone.
    Maestro S3 (384ch) → estudiante backbone intermedio.
    """
    return ProjectionLayers(
        teacher_channels=[384, 768],          # Maestro S3, S4
        student_channels=[128, 256],          # YOLO Nano intermedios
        use_bn=use_bn,
    )


def head_projections(use_bn: bool = True) -> ProjectionLayers:
    """
    Proyecciones para distilación a nivel de head.
    Maestro cls_stem/reg_stem (256ch) → estudiante head channels.
    """
    return ProjectionLayers(
        teacher_channels=[256, 256, 256],     # Maestro head stems
        student_channels=[64, 128, 256],      # YOLO Nano head
        use_bn=use_bn,
    )
