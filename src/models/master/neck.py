"""
FPN necks for the two `MasterModel` fusion modes.

- `FPNNeck` — `fusion_mode="early"` (default). One `SingleFPN` over the
  single early-fused backbone stream, emitting the levels named by
  `head_strides`.
- `DualFPN` — `fusion_mode="cross_attention"`. Two `SingleFPN`s (one per
  modality) fused per level via a 1x1 conv, emitting [P3, P4, P5].

fusion-redesign (design.md D-D) originally deleted `DualFPN`: with only one
backbone stream there is only one FPN. It is restored here for the
cross-attention path alone, verbatim — its `fusion_convs` count and module
names are part of the two-stream checkpoint schema.

`SingleFPN` is shared by both and kept unchanged — it already builds the
full [P2, P3, P4, P5] pyramid. `DualFPN` computes P2 in both of its internal
FPNs and discards both copies before fusion (`fused_features[0].grad is
None`, verified in proposal.md D3) because `YOLODetectionHead` only consumes
strides [8, 16, 32] on that path — a known cost of the cross-attention
architecture, kept as-is so its checkpoints stay loadable.

`FPNNeck` reconnects P2 by default: it wraps
`SingleFPN` and emits exactly the levels named by the configured
`head_strides`, finest-first. Every level `FPNNeck` returns is consumed
downstream by `YOLODetectionHead`, so — unlike the old design — no computed
pyramid level is discarded without a gradient path to the loss.

Output channels: 256 per level (unchanged).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.strides import (
    CROSS_ATTENTION_HEAD_STRIDES,
    STRIDE_TO_LEVEL,
    validate_strides,
)


class SingleFPN(nn.Module):
    """
    Standard Feature Pyramid Network for one modality stream.

    Takes 4 stage features [S1, S2, S3, S4] with channels [96, 192, 384, 768]
    and produces 4 pyramid levels [P2, P3, P4, P5] all with `out_channels`.

    The naming follows FPN convention:
        P2 ← S1 (stride 4)
        P3 ← S2 (stride 8)
        P4 ← S3 (stride 16)
        P5 ← S4 (stride 32)

    Args:
        in_channels (list[int]): Input channels per stage [96, 192, 384, 768].
        out_channels (int): Output channels for all pyramid levels (default 256).
    """

    def __init__(
        self,
        in_channels: list[int] = None,
        out_channels: int = 256,
    ):
        super().__init__()

        if in_channels is None:
            in_channels = [96, 192, 384, 768]

        # Lateral 1x1 convs: project each stage to out_channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, kernel_size=1)
            for ch in in_channels
        ])

        # Output 3x3 convs: smooth after upsampling addition
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            features: [S1, S2, S3, S4] — backbone stage outputs

        Returns:
            pyramid: [P2, P3, P4, P5] — all at out_channels
        """
        # Step 1: lateral projections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Step 2: top-down pathway (upsample and add)
        # Start from the deepest level (S4 → P5) and go up
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample deeper level to match spatial size of shallower level
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode="nearest",
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Step 3: output convolutions
        pyramid = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]

        return pyramid  # [P2, P3, P4, P5]


class FPNNeck(nn.Module):
    """
    Thin wrapper over `SingleFPN` that emits a configurable subset of the
    pyramid, selected by stride rather than position.

    `SingleFPN` always computes the full [P2, P3, P4, P5] pyramid; `FPNNeck`
    selects which of those levels to return, in finest-first order, driven
    by `strides`. Every returned level MUST reach the head (design.md
    "Every Emitted Pyramid Level Is Consumed") — `FPNNeck` never drops a
    level after the fact.

    Args:
        in_channels (list[int]): Stage channels from the backbone.
        out_channels (int): Output channels per pyramid level (default 256).
        strides (tuple[int, ...]): Which strides to emit, finest-first.
            Default `(4, 8, 16, 32)` — all 4 levels, including P2.
    """

    def __init__(
        self,
        in_channels: list[int] = None,
        out_channels: int = 256,
        strides: tuple[int, ...] = (4, 8, 16, 32),
    ):
        super().__init__()

        validate_strides(strides)
        self.strides = tuple(strides)
        self.emit_levels = tuple(STRIDE_TO_LEVEL[s] for s in strides)

        self.fpn = SingleFPN(in_channels, out_channels)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            features: [S1, S2, S3, S4] — backbone stage outputs.

        Returns:
            pyramid: the levels named by `self.strides`, finest-first, each
                at `out_channels`. Index `k` corresponds to `self.strides[k]`.
        """
        pyramid = self.fpn(features)  # always [P2, P3, P4, P5]
        return [pyramid[i] for i in self.emit_levels]


class DualFPN(nn.Module):
    """
    Two parallel FPNs (one per modality) with per-level fusion —
    `fusion_mode="cross_attention"` only.

    Fusion strategy: concatenate along channel dim → 1x1 conv → out_channels.
    This is lightweight (1x1 conv) but fully learnable, letting the model
    decide how to weight RGB vs NIR information at each pyramid level.

    The emitted pyramid is fixed at 3 levels
    (`src.training.strides.CROSS_ATTENTION_HEAD_STRIDES`): P2 is dropped
    before fusion and there are exactly 3 `fusion_convs`. `MasterModel`
    rejects any other `head_strides` under this mode rather than letting the
    mismatch surface as a `zip` that silently truncates.

    Args:
        in_channels (list[int]): Stage channels from backbone.
        out_channels (int): Output channels per pyramid level (default 256).
        dropout (float): Dropout after fusion conv (regularization).
    """

    def __init__(
        self,
        in_channels: list[int] = None,
        out_channels: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        if in_channels is None:
            in_channels = [96, 192, 384, 768]

        self.fpn_rgb = SingleFPN(in_channels, out_channels)
        self.fpn_nir = SingleFPN(in_channels, out_channels)

        # Fusion: concat (2 * out_channels) → out_channels per pyramid level
        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
            )
            # one per emitted pyramid level (P3, P4, P5 — P2 dropped)
            for _ in CROSS_ATTENTION_HEAD_STRIDES
        ])

    def forward(
        self,
        fused_features: list[torch.Tensor],
        nir_features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Args:
            fused_features: [F1, F2, F3, F4] — cross-attention fused RGB+NIR
            nir_features:   [S1, S2, S3, S4] — original NIR backbone features

        Returns:
            pyramid: [P3, P4, P5] — unified multi-scale feature pyramid
        """
        # Build per-modality pyramids
        rgb_pyramid = self.fpn_rgb(fused_features)   # [P2..P5] at 256ch
        nir_pyramid = self.fpn_nir(nir_features)     # [P2..P5] at 256ch

        # Fuse per level (skip P2 — head expects CROSS_ATTENTION_HEAD_STRIDES)
        unified_pyramid = []
        for rgb_level, nir_level, fusion_conv in zip(
            rgb_pyramid[1:], nir_pyramid[1:], self.fusion_convs
        ):
            # Concatenate along channel dim: (N, 512, H, W)
            combined = torch.cat([rgb_level, nir_level], dim=1)
            # Fuse to (N, 256, H, W)
            fused_level = fusion_conv(combined)
            unified_pyramid.append(fused_level)

        return unified_pyramid  # [P3, P4, P5] at 256ch
