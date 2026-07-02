"""
Dual-stream ConvNeXt backbone for RGB + NIR inputs.

Supports ConvNeXt-Tiny and ConvNeXt-Small variants (both have identical
stage channels [96, 192, 384, 768], differing only in block depth).

Design decisions:
- Shared weights between RGB and NIR encoders (regularization for small dataset ~2000 imgs)
- Only the first stem layer differs: RGB accepts 3ch, NIR accepts 1ch
- Both streams produce 4 stage feature maps: [S1, S2, S3, S4]
  with channels [96, 192, 384, 768]
"""

import torch
import torch.nn as nn
from torchvision.models import (
    convnext_small,
    convnext_tiny,
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
)
from torchvision.models.convnext import ConvNeXt

# Supported backbone variants — both have identical stage channels
SUPPORTED_VARIANTS = {"tiny", "small"}


def _build_convnext_tiny_body(pretrained: bool = True) -> nn.ModuleList:
    """
    Extract the 4 stages of ConvNeXt-Tiny (without the stem and classifier).
    Returns them as a ModuleList so we can iterate per-stage.
    """
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = convnext_tiny(weights=weights)
    stages = nn.ModuleList([
        model.features[1],   # stage 1  → 96ch,  H/4  x W/4
        nn.Sequential(model.features[2], model.features[3]),   # down + stage 2 → 192ch, H/8  x W/8
        nn.Sequential(model.features[4], model.features[5]),   # down + stage 3 → 384ch, H/16 x W/16
        nn.Sequential(model.features[6], model.features[7]),   # down + stage 4 → 768ch, H/32 x W/32
    ])
    return stages


def _build_convnext_small_body(pretrained: bool = True) -> nn.ModuleList:
    """
    Extract the 4 stages of ConvNeXt-Small (without the stem and classifier).
    Returns them as a ModuleList so we can iterate per-stage.
    """
    weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = convnext_small(weights=weights)
    # model.features is a Sequential with:
    #   [0] stem (Conv2d 4x4 s4 + LayerNorm)
    #   [1] stage 1
    #   [2] downsampling
    #   [3] stage 2
    #   [4] downsampling
    #   [5] stage 3
    #   [6] downsampling
    #   [7] stage 4
    stages = nn.ModuleList([
        model.features[1],   # stage 1  → 96ch,  H/4  x W/4
        nn.Sequential(model.features[2], model.features[3]),   # down + stage 2 → 192ch, H/8  x W/8
        nn.Sequential(model.features[4], model.features[5]),   # down + stage 3 → 384ch, H/16 x W/16
        nn.Sequential(model.features[6], model.features[7]),   # down + stage 4 → 768ch, H/32 x W/32
    ])
    return stages


def _build_stem(in_channels: int) -> nn.Sequential:
    """
    ConvNeXt stem: Conv2d (4x4, stride 4) + LayerNorm.
    in_channels=3 for RGB, in_channels=1 for NIR.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, 96, kernel_size=4, stride=4),
        # ConvNeXt uses LayerNorm on (N, H, W, C) — we use a wrapper
        _LayerNorm2d(96),
    )


class _LayerNorm2d(nn.Module):
    """LayerNorm for channels-first tensors (N, C, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W) → (N, H, W, C) → norm → (N, C, H, W)
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DualConvNeXtBackbone(nn.Module):
    """
    Dual-stream ConvNeXt backbone (Tiny or Small).

    The two streams share the 4 ConvNeXt stages (shared weights act as
    regularization for the small dataset). Only the input stems differ
    to handle 3-channel RGB and 1-channel NIR independently.

    Args:
        pretrained (bool): Load ImageNet-1K weights for the shared stages.
            The NIR stem is always initialized from scratch.
        variant (str): Backbone variant — "tiny" (28M params) or "small" (50M params).
            Both produce identical stage channels [96, 192, 384, 768].

    Returns (forward):
        rgb_features: list of 4 tensors [S1, S2, S3, S4]
        nir_features: list of 4 tensors [S1, S2, S3, S4]

        Channel dims: [96, 192, 384, 768]
        Spatial dims (for 640x640 input): [160x160, 80x80, 40x40, 20x20]
    """

    # Both Tiny and Small have identical stage channel dimensions
    STAGE_CHANNELS = [96, 192, 384, 768]

    def __init__(self, pretrained: bool = True, variant: str = "tiny"):
        super().__init__()

        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported variant '{variant}'. Choose from {SUPPORTED_VARIANTS}"
            )

        self.variant = variant

        # --- Stems (different per modality) ---
        self.rgb_stem = _build_stem(in_channels=3)
        self.nir_stem = _build_stem(in_channels=1)  # NIR is grayscale

        # --- Shared stages ---
        if variant == "tiny":
            self.shared_stages = _build_convnext_tiny_body(pretrained=pretrained)
        else:
            self.shared_stages = _build_convnext_small_body(pretrained=pretrained)

        if pretrained:
            self._load_pretrained_rgb_stem()
            self._load_pretrained_nir_stem()
        else:
            self._init_nir_stem()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, rgb: torch.Tensor, nir: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Args:
            rgb: (N, 3, H, W)
            nir: (N, 1, H, W)  — grayscale NIR image

        Returns:
            rgb_features: [S1, S2, S3, S4]
            nir_features: [S1, S2, S3, S4]
        """
        # ConvNeXt stage 4 is numerically fragile under CUDA autocast in this
        # dual-stream setup. Run the backbone in FP32 so the teacher produces
        # stable features for both detection and future distillation.
        device_type = rgb.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            # Pass through modality-specific stems
            rgb_x = self.rgb_stem(rgb.float())   # (N, 96, H/4, W/4)
            nir_x = self.nir_stem(nir.float())   # (N, 96, H/4, W/4)

            rgb_features, nir_features = [], []

            # Pass through shared stages: same weights, different activations
            for stage in self.shared_stages:
                rgb_x = stage(rgb_x)
                nir_x = stage(nir_x)
                rgb_features.append(rgb_x)
                nir_features.append(nir_x)

        return rgb_features, nir_features

    # ------------------------------------------------------------------
    # Weight initialization helpers
    # ------------------------------------------------------------------

    def _init_nir_stem(self):
        """Initialize NIR stem with Kaiming normal (standard for conv layers)."""
        nn.init.kaiming_normal_(
            self.nir_stem[0].weight, mode="fan_out", nonlinearity="relu"
        )
        if self.nir_stem[0].bias is not None:
            nn.init.zeros_(self.nir_stem[0].bias)

    def _load_pretrained_rgb_stem(self):
        """
        Copy the pretrained ConvNeXt stem weights into rgb_stem.
        The pretrained stem expects 3-channel input, which matches RGB.
        Works for both Tiny and Small variants (identical stem architecture).
        """
        if self.variant == "tiny":
            pretrained_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            pretrained_model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
        pretrained_stem_state = {
            "0.weight": pretrained_model.features[0][0].weight,
            "0.bias":   pretrained_model.features[0][0].bias,
            "1.norm.weight": pretrained_model.features[0][1].weight,
            "1.norm.bias":   pretrained_model.features[0][1].bias,
        }
        self.rgb_stem.load_state_dict(pretrained_stem_state, strict=False)

    def _load_pretrained_nir_stem(self):
        """Initialize the 1-channel NIR stem from RGB ImageNet stem weights.

        Averaging pretrained RGB kernels gives the NIR stream low-level edge and
        texture priors while keeping the stem trainable for 850 nm reflectance.
        """
        if self.variant == "tiny":
            pretrained_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            pretrained_model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)

        rgb_weight = pretrained_model.features[0][0].weight
        pretrained_stem_state = {
            "0.weight": rgb_weight.mean(dim=1, keepdim=True),
            "0.bias": pretrained_model.features[0][0].bias,
            "1.norm.weight": pretrained_model.features[0][1].weight,
            "1.norm.bias": pretrained_model.features[0][1].bias,
        }
        self.nir_stem.load_state_dict(pretrained_stem_state, strict=False)
