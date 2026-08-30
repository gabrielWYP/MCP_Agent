"""Tests for `EarlyFusionBackbone` (fusion-redesign W1).

Covers:
- Mean-preserving stem inflation values (D-1/D-B).
- Single forward pass, no second stream (no fusion module in the call graph).
- H-B: NIR-reachable trainable capacity (>= 27M parameters).
"""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.models.master.backbone import EarlyFusionBackbone
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


@pytest.fixture(scope="module")
def pretrained_reference_weight() -> torch.Tensor:
    """Freshly loaded ConvNeXt-Tiny ImageNet stem weight, (96, 3, 4, 4)."""
    ref = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    return ref.features[0][0].weight.detach().clone()


class TestStemInflation:
    """D-1/D-B: mean-preserving inflation, not zero-init, not naive copy."""

    def test_inflated_weight_is_mean_preserving(self, pretrained_reference_weight):
        backbone = EarlyFusionBackbone(pretrained=True, variant="tiny", in_channels=4)
        w_new = backbone.stem[0].weight.detach()
        w_ref = pretrained_reference_weight

        assert torch.allclose(w_new[:, :3], w_ref * 0.75, atol=1e-6)
        assert torch.allclose(w_new[:, 3], w_ref.mean(dim=1) * 0.75, atol=1e-6)

    def test_bias_and_layernorm_copied_verbatim(self, pretrained_reference_weight):
        ref = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        backbone = EarlyFusionBackbone(pretrained=True, variant="tiny", in_channels=4)

        assert torch.equal(backbone.stem[0].bias, ref.features[0][0].bias)
        assert torch.equal(backbone.stem[1].norm.weight, ref.features[0][1].weight)
        assert torch.equal(backbone.stem[1].norm.bias, ref.features[0][1].bias)

    def test_three_channel_control_arm_is_verbatim_copy(self, pretrained_reference_weight):
        """H-D control: in_channels=3 gets an unscaled verbatim copy, not 3/4 scaling."""
        backbone = EarlyFusionBackbone(pretrained=True, variant="tiny", in_channels=3)
        assert torch.equal(backbone.stem[0].weight.detach(), pretrained_reference_weight)

    def test_unsupported_in_channels_raises(self):
        with pytest.raises(ValueError, match="in_channels"):
            EarlyFusionBackbone(pretrained=True, variant="tiny", in_channels=5)


class TestSingleStreamForward:
    """No second stream, no fusion module — one stem, one stage stack."""

    def test_forward_shapes(self):
        backbone = EarlyFusionBackbone(pretrained=False, in_channels=4)
        backbone.eval()
        x = torch.randn(2, 4, 640, 640)
        with torch.no_grad():
            features = backbone(x)

        assert len(features) == 4
        expected_channels = [96, 192, 384, 768]
        expected_spatial = [160, 80, 40, 20]
        for feat, ch, hw in zip(features, expected_channels, expected_spatial):
            assert feat.shape == (2, ch, hw, hw)

    def test_forward_returns_single_list_not_a_tuple_of_two_streams(self):
        """The old DualConvNeXtBackbone returned (rgb_features, nir_features)
        — a 2-tuple of lists. EarlyFusionBackbone must return one list."""
        backbone = EarlyFusionBackbone(pretrained=False, in_channels=4)
        x = torch.randn(1, 4, 64, 64)
        out = backbone(x)
        assert isinstance(out, list)
        assert all(isinstance(t, torch.Tensor) for t in out)


class TestNIRCapacity:
    """H-B: early fusion gives NIR real trainable capacity."""

    def test_nir_only_perturbation_reaches_at_least_27m_params(self):
        """Gradient audit: count parameters receiving non-zero grad from a
        NIR-only input perturbation. Bar: >= 27,000,000 (vs the measured
        1,824 in the removed dual-stream design).

        `eval()`, not `train()`: this is a static architectural-reachability
        claim, not a training-behavior claim, and ConvNeXt's stochastic
        depth would otherwise randomly zero some blocks per forward pass,
        making the exact reachable-parameter count flaky. A full 640x640
        input is used so every stage's receptive field is exercised (a
        64x64 probe undercounts reachable stage-4 parameters purely from
        spatial coverage, not from an architectural limitation).
        """
        backbone = EarlyFusionBackbone(pretrained=False, in_channels=4)
        backbone.eval()

        x = torch.zeros(1, 4, 640, 640, requires_grad=False)
        # Perturb only the NIR channel (index 3).
        x = x.clone()
        x[:, 3] = torch.randn(640, 640)

        features = backbone(x)
        loss = sum(f.float().pow(2).sum() for f in features)
        loss.backward()

        nonzero_params = sum(
            p.numel()
            for p in backbone.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert nonzero_params >= 27_000_000, (
            f"Only {nonzero_params:,} params reachable from NIR — expected >= 27,000,000"
        )
