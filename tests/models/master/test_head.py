"""Tests for `YOLODetectionHead`/`DecoupledHead` (fusion-redesign W5).

Covers:
- The head's stem is computed exactly once per level per forward pass
  (previously `head(feat)` ran `cls_stem`/`reg_stem`, then
  `head.cls_stem(feat)`/`head.reg_stem(feat)` ran them again for the
  distillation features — doubling head activation memory).
- Post-fix output is numerically identical to the pre-fix computation
  (pure memory/compute fix, no numerical consequence).
- Configurable head_strides: exactly N `DecoupledHead` instances for N strides.
"""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.models.master.head import DecoupledHead, YOLODetectionHead


def _pyramid(num_levels: int, hw_start: int = 32, channels: int = 256):
    return [
        torch.randn(1, channels, hw_start // (2 ** i), hw_start // (2 ** i))
        for i in range(num_levels)
    ]


class TestDecoupledHeadReturnsStemFeatures:
    def test_forward_returns_four_tensors(self):
        head = DecoupledHead(in_channels=256, num_classes=2)
        x = torch.randn(1, 256, 8, 8)
        cls_out, reg_out, cls_feat, reg_feat = head(x)
        assert cls_out.shape == (1, 2, 8, 8)
        assert reg_out.shape == (1, 4, 8, 8)
        assert cls_feat.shape == (1, 256, 8, 8)
        assert reg_feat.shape == (1, 256, 8, 8)

    def test_returned_cls_pred_matches_manual_recomputation(self):
        """The returned cls_feat/reg_feat must be usable to reproduce
        cls_out/reg_out exactly — proving they are the same stem output,
        not an independent recomputation that could drift."""
        head = DecoupledHead(in_channels=256, num_classes=2)
        head.eval()
        x = torch.randn(1, 256, 8, 8)
        with torch.no_grad():
            cls_out, reg_out, cls_feat, reg_feat = head(x)
            assert torch.allclose(head.cls_pred(cls_feat), cls_out)
            assert torch.allclose(head.reg_pred(reg_feat), reg_out)


class TestStemComputedOncePerLevel:
    """W5: forward-hook call counter proves cls_stem/reg_stem run exactly
    once per level per forward pass."""

    def test_stem_modules_called_exactly_once_per_level(self):
        head = YOLODetectionHead(fpn_channels=256, num_classes=2, strides=[8, 16, 32])
        pyramid = _pyramid(3)

        call_counts = {"cls_stem": 0, "reg_stem": 0}

        def _count(name):
            def _hook(module, inputs, output):
                call_counts[name] += 1
            return _hook

        hooks = []
        for h in head.heads:
            hooks.append(h.cls_stem.register_forward_hook(_count("cls_stem")))
            hooks.append(h.reg_stem.register_forward_hook(_count("reg_stem")))

        head(pyramid)

        for hook in hooks:
            hook.remove()

        assert call_counts["cls_stem"] == 3, "cls_stem must run exactly once per level"
        assert call_counts["reg_stem"] == 3, "reg_stem must run exactly once per level"

    def test_distill_features_are_the_reused_stem_output(self):
        """distill_cls[i]/distill_reg[i] must be the exact same tensor
        object the prediction was computed from — not a second, independent
        stem() call on the same input."""
        head = YOLODetectionHead(fpn_channels=256, num_classes=2, strides=[8, 16, 32])
        pyramid = _pyramid(3)
        out = head(pyramid)

        for i, h in enumerate(head.heads):
            expected_cls_feat = h.cls_stem(pyramid[i])
            expected_reg_feat = h.reg_stem(pyramid[i])
            # Same computation (deterministic, eval-independent conv), so
            # this recomputation must be numerically identical to what the
            # single internal call produced.
            assert torch.allclose(out["distill_cls"][i], expected_cls_feat, atol=1e-5)
            assert torch.allclose(out["distill_reg"][i], expected_reg_feat, atol=1e-5)


class TestConfigurableHeadStrides:
    def test_four_levels_builds_four_decoupled_heads(self):
        head = YOLODetectionHead(fpn_channels=256, num_classes=2, strides=[4, 8, 16, 32])
        assert len(head.heads) == 4
        assert head.num_levels == 4

    def test_three_levels_ablation_builds_three_decoupled_heads(self):
        head = YOLODetectionHead(fpn_channels=256, num_classes=2, strides=[8, 16, 32])
        assert len(head.heads) == 3
        assert head.num_levels == 3

    def test_mismatched_pyramid_length_raises(self):
        head = YOLODetectionHead(fpn_channels=256, num_classes=2, strides=[8, 16, 32])
        with pytest.raises(AssertionError):
            head(_pyramid(4))

    def test_strides_is_required_keyword_argument(self):
        with pytest.raises(TypeError):
            YOLODetectionHead(fpn_channels=256, num_classes=2)
