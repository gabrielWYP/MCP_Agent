"""Tests for `MasterModel` (fusion-redesign W3/W4).

Covers:
- D3 regression: forward+backward on CPU; output dict has exactly 7 keys;
  every emitted pyramid level receives a non-None gradient (the old DualFPN
  computed P2 and discarded it before the head, with no gradient path at
  all — this is that defect's regression test).
- H-C: no `adaptive_avg_pool2d` anywhere in the model (the old fusion
  module pooled to a fixed token grid, destroying the textural NIR signal);
  finest emitted level's cell size <= 8px at 640px input.
- The single-stream call graph contains no cross-modal fusion module.
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.models.master.master_model import MasterModel

EXPECTED_KEYS = {
    "preds", "cls_preds", "reg_preds",
    "distill_backbone", "distill_fpn", "distill_head_cls", "distill_head_reg",
}


def _build_model(head_strides=None):
    return MasterModel(
        num_classes=2, pretrained_backbone=False, backbone_variant="tiny",
        head_strides=head_strides,
    )


class TestOutputContract:
    def test_output_dict_has_exactly_seven_keys(self):
        model = _build_model()
        rgb = torch.randn(1, 3, 128, 128)
        nir = torch.randn(1, 1, 128, 128)
        out = model(rgb, nir)
        assert set(out.keys()) == EXPECTED_KEYS

    def test_no_attention_maps_key(self):
        """The old 8-key dict's `attention_maps` key must not exist — there
        is no attention module left to produce one."""
        model = _build_model()
        rgb = torch.randn(1, 3, 128, 128)
        nir = torch.randn(1, 1, 128, 128)
        out = model(rgb, nir)
        assert "attention_maps" not in out
        assert "distill_backbone_fused" not in out
        assert "distill_backbone_rgb" not in out


class TestGradientReachability:
    """D3 regression: every emitted pyramid level must receive a gradient."""

    def test_every_emitted_level_receives_nonnone_grad(self):
        model = _build_model(head_strides=[4, 8, 16, 32])
        model.train()
        rgb = torch.randn(1, 3, 128, 128)
        nir = torch.randn(1, 1, 128, 128)

        out = model(rgb, nir)
        loss = sum(p.float().pow(2).sum() for p in out["preds"])
        loss.backward()

        for i, lateral_conv in enumerate(model.neck.fpn.lateral_convs):
            assert lateral_conv.weight.grad is not None, (
                f"Lateral conv {i} has no gradient — an emitted level was "
                "computed but never reached the loss (D3 regression)."
            )

    def test_backbone_receives_gradient(self):
        model = _build_model()
        model.train()
        rgb = torch.randn(1, 3, 128, 128)
        nir = torch.randn(1, 1, 128, 128)

        out = model(rgb, nir)
        loss = sum(p.float().pow(2).sum() for p in out["preds"])
        loss.backward()

        assert model.backbone.stem[0].weight.grad is not None
        assert model.backbone.stem[0].weight.grad.abs().sum() > 0


class TestNoPoolingBetweenBackboneAndHead:
    """H-C: no adaptive_avg_pool2d anywhere in the assembled model."""

    def test_no_adaptive_avg_pool2d_module(self):
        model = _build_model()
        for module in model.modules():
            assert not isinstance(module, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d))

    def test_finest_emitted_cell_at_most_eight_pixels(self):
        """At 640px input, the finest emitted level (P2, stride 4) has an
        8px-or-smaller receptive cell — vs 32px in the removed design."""
        model = _build_model(head_strides=[4, 8, 16, 32])
        model.eval()
        rgb = torch.randn(1, 3, 640, 640)
        nir = torch.randn(1, 1, 640, 640)
        with torch.no_grad():
            out = model(rgb, nir)
        finest_level = out["distill_fpn"][0]
        cell_size_px = 640 / finest_level.shape[-1]
        assert cell_size_px <= 8


class TestSingleStreamNoFusionModule:
    def test_no_fusion_attribute(self):
        model = _build_model()
        assert not hasattr(model, "fusion")

    def test_no_multihead_attention_in_call_graph(self):
        model = _build_model()
        for module in model.modules():
            assert not isinstance(module, nn.MultiheadAttention)
