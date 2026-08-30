"""Tests for `FPNNeck` (fusion-redesign W2/D-D).

Covers:
- FPNNeck emits exactly the levels named by `strides`, finest-first, 256ch.
- Every emitted level has a live gradient path to the loss (D3 regression —
  the old `DualFPN` computed P2 twice and discarded both copies).
- Invalid strides raise at construction, not at some later silent failure.
"""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.models.master.neck import FPNNeck, SingleFPN


def _backbone_features(batch=1, hw=64):
    """Synthetic [S1..S4] backbone features at the standard channel counts."""
    return [
        torch.randn(batch, 96, hw, hw, requires_grad=True),
        torch.randn(batch, 192, hw // 2, hw // 2, requires_grad=True),
        torch.randn(batch, 384, hw // 4, hw // 4, requires_grad=True),
        torch.randn(batch, 768, hw // 8, hw // 8, requires_grad=True),
    ]


class TestEmittedLevels:
    @pytest.mark.parametrize("strides", [(4, 8, 16, 32), (8, 16, 32)])
    def test_emits_exactly_the_configured_levels(self, strides):
        neck = FPNNeck(strides=strides)
        features = _backbone_features()
        pyramid = neck(features)

        assert len(pyramid) == len(strides)
        for level in pyramid:
            assert level.shape[1] == 256

    def test_levels_are_finest_first(self):
        neck = FPNNeck(strides=(4, 8, 16, 32))
        features = _backbone_features(hw=64)
        pyramid = neck(features)
        spatial_sizes = [level.shape[-1] for level in pyramid]
        assert spatial_sizes == sorted(spatial_sizes, reverse=True), (
            "Pyramid levels must be ordered finest (largest spatial size) first"
        )

    def test_three_level_ablation_excludes_p2(self):
        neck = FPNNeck(strides=(8, 16, 32))
        features = _backbone_features(hw=64)
        pyramid = neck(features)
        # 3 levels means the coarsest 3 of SingleFPN's 4 — none should equal
        # the S1-derived P2 spatial size (64), which only the 4-level config emits.
        assert len(pyramid) == 3
        assert all(level.shape[-1] != 64 for level in pyramid)


class TestGradientReachability:
    """D3 regression: every emitted level must receive a non-None gradient."""

    def test_every_emitted_level_has_gradient_path_to_loss(self):
        neck = FPNNeck(strides=(4, 8, 16, 32))
        features = _backbone_features()
        pyramid = neck(features)

        loss = sum(level.float().pow(2).sum() for level in pyramid)
        loss.backward()

        for lateral_conv in neck.fpn.lateral_convs:
            assert lateral_conv.weight.grad is not None
            assert lateral_conv.weight.grad.abs().sum() > 0


class TestInvalidStrides:
    def test_unsupported_stride_raises(self):
        with pytest.raises(ValueError):
            FPNNeck(strides=(8, 16, 64))

    def test_non_increasing_strides_raises(self):
        with pytest.raises(ValueError):
            FPNNeck(strides=(16, 8, 32))

    def test_empty_strides_raises(self):
        with pytest.raises(ValueError):
            FPNNeck(strides=())


class TestSingleFPNUnchanged:
    """SingleFPN itself is kept unchanged — it already emits [P2, P3, P4, P5]."""

    def test_single_fpn_always_emits_four_levels(self):
        fpn = SingleFPN()
        features = _backbone_features(hw=64)
        pyramid = fpn(features)
        assert len(pyramid) == 4
        assert all(level.shape[1] == 256 for level in pyramid)
