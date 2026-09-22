"""
Projection layers (adapters) for Knowledge Distillation between Master and Student.

The master (ConvNeXt + FPN 256ch) and the student (YOLO Nano CSPDarknet) have
different feature widths at every level. These layers are learnable 1x1 convs
that map the STUDENT features into the TEACHER channel space (FitNets-style
"regressor"), so the distillation loss (MSE) is computable against the frozen
teacher features:

    Student FPN[P3] (C_ch) ──► adapter_1x1 ──► (teacher_ch) ──┐
                                                              ├──► MSE loss
    Teacher FPN[P3] (256ch, frozen, no_grad) ─────────────────┘

Direction matters: an earlier revision projected the TEACHER into the student
space. A trainable projection on the teacher side lets the optimizer shrink
the regression target itself (e.g. BatchNorm gamma -> 0 collapses both sides
of the MSE), and because it ran under the teacher's `torch.no_grad()` it
never received a gradient at all. With the adapter on the student side the
target is the fixed teacher feature and the only way to reduce the loss is
to make the student features predictive of it.

Applied at 3 levels:
    - Backbone features (student S3/S4 → teacher S3/S4)
    - FPN features (student P3/P4/P5 → teacher P3/P4/P5)
    - Head features (student head stems → teacher cls_stem/reg_stem)

The adapters are training-only: they are owned by `KDTrainer`, not by
`StudentModel`, so the student's state_dict and RGB-only inference path never
contain or require them.
"""

import torch
import torch.nn as nn


class ProjectionLayers(nn.Module):
    """
    1x1 adapters aligning student → teacher channel widths (FitNets-style).

    Args:
        teacher_channels (list[int]): Teacher channels per level (adapter
            OUTPUT). Default: [256, 256, 256] for FPN P3/P4/P5.
        student_channels (list[int]): Student channels per level (adapter
            INPUT). Default YOLO Nano: [128, 256, 256] for P3/P4/P5.
        use_bn (bool): If True, appends a BatchNorm2d after each 1x1 conv.
            Default False: the regression target is the raw (frozen) teacher
            feature, which a plain conv + bias can already match in scale
            and per-channel offset. A BN would only add a train/eval
            discrepancy and a dependence on the (often small) KD batch
            statistics, for a module that is never used at inference.
    """

    def __init__(
        self,
        teacher_channels: list[int] = None,
        student_channels: list[int] = None,
        use_bn: bool = False,
    ):
        super().__init__()

        if teacher_channels is None:
            teacher_channels = [256, 256, 256]
        if student_channels is None:
            # YOLO Nano: P3=128ch, P4=256ch, P5=256ch
            student_channels = [128, 256, 256]

        assert len(teacher_channels) == len(student_channels), (
            f"Teacher ({len(teacher_channels)}) and student ({len(student_channels)}) "
            f"must have the same number of levels"
        )

        self.num_levels = len(teacher_channels)

        self.projections = nn.ModuleList([
            self._make_proj(s_ch, t_ch, use_bn)
            for t_ch, s_ch in zip(teacher_channels, student_channels)
        ])

    def _make_proj(self, in_ch: int, out_ch: int, use_bn: bool) -> nn.Sequential:
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        return nn.Sequential(*layers)

    def forward(self, student_features: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Project student features into the teacher channel space.

        Args:
            student_features: list of student tensors, one per level.

        Returns:
            list of projected tensors, same spatial shapes but with the
            teacher's channel counts.

        Raises:
            AssertionError: if `len(student_features) != self.num_levels`.
                fusion-redesign design.md §5 flagged the un-guarded `zip`
                below as a silent-truncation hazard once the teacher's
                pyramid/head level count became configurable (4 levels by
                default, vs this class's fixed 3-level presets). This
                explicit length check, with a message identifying both
                counts, makes a silent level misalignment impossible; the
                caller (`KDTrainer`) is responsible for slicing the teacher's
                features down to the student's levels by stride
                (`src/training/strides.select_by_strides`).
        """
        assert len(student_features) == self.num_levels, (
            f"ProjectionLayers expected {self.num_levels} feature "
            f"level(s), got {len(student_features)}. A length mismatch here "
            "would otherwise silently misalign levels via zip() — the "
            "caller must select the matching levels by stride before "
            "calling forward()."
        )
        return [
            proj(feat)
            for proj, feat in zip(self.projections, student_features)
        ]


# ── Presets for the student/teacher pair ─────────────────────────────────

def fpn_projections(use_bn: bool = False) -> ProjectionLayers:
    """Adapters for FPN-level distillation (P3/P4/P5): student → teacher."""
    return ProjectionLayers(
        teacher_channels=[256, 256, 256],   # Master FPN
        student_channels=[128, 256, 256],    # YOLO Nano P3/P4/P5
        use_bn=use_bn,
    )


def backbone_projections(use_bn: bool = False) -> ProjectionLayers:
    """
    Adapters for backbone-level distillation.
    Student intermediate (S3, S4) → master S3 (384ch), S4 (768ch).
    """
    return ProjectionLayers(
        teacher_channels=[384, 768],          # Master S3, S4
        student_channels=[128, 256],          # YOLO Nano intermediates
        use_bn=use_bn,
    )


def head_projections(use_bn: bool = False) -> ProjectionLayers:
    """
    Adapters for head-level distillation.
    Student head stems → master cls_stem/reg_stem (256ch).
    """
    return ProjectionLayers(
        teacher_channels=[256, 256, 256],     # Master head stems
        student_channels=[64, 128, 256],      # YOLO Nano head
        use_bn=use_bn,
    )
