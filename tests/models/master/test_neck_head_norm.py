"""Neck/head norm selection (`src/models/master/norms.py`).

"gn" must replace every BatchNorm2d in the neck and head (and touch nothing
else), fall back to a channel divisor for the group count, and a BatchNorm
checkpoint must be refused by a GroupNorm model with a clear error.
"""

import sys
from pathlib import Path

import pytest
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.models.master.master_model import MasterModel
from src.models.master.neck import DualFPN
from src.models.master.norms import check_norm_compatible, make_norm, resolve_gn_groups
from src.training.config import TrainingConfig


def _count(module: nn.Module, kind: type) -> int:
    return sum(isinstance(m, kind) for m in module.modules())


def _two_stream(norm: str) -> MasterModel:
    return MasterModel(pretrained_backbone=False, fusion_mode="cross_attention", neck_head_norm=norm)


class TestGroupCountFallback:
    @pytest.mark.parametrize(
        "channels, requested, expected",
        [(256, 32, 32), (48, 32, 24), (30, 32, 30), (7, 32, 7), (13, 4, 1), (256, 1, 1)],
    )
    def test_largest_divisor_not_above_request(self, channels, requested, expected):
        assert resolve_gn_groups(channels, requested) == expected

    def test_make_norm_uses_fallback(self):
        norm = make_norm("gn", 48, 32)
        assert isinstance(norm, nn.GroupNorm)
        assert norm.num_groups == 24

    def test_make_norm_bn(self):
        assert isinstance(make_norm("bn", 16), nn.BatchNorm2d)


class TestSubstitution:
    def test_dual_fpn_gn_has_no_batchnorm(self):
        neck = DualFPN(emit_strides=(4, 8, 16, 32), norm="gn")
        assert _count(neck, nn.BatchNorm2d) == 0
        assert _count(neck, nn.GroupNorm) == 4

    def test_master_model_gn_neck_and_head_have_no_batchnorm(self):
        bn_model = _two_stream("bn")
        gn_model = _two_stream("gn")
        n_bn = _count(bn_model.neck, nn.BatchNorm2d) + _count(bn_model.head, nn.BatchNorm2d)
        assert n_bn > 0
        assert _count(gn_model.neck, nn.BatchNorm2d) == 0
        assert _count(gn_model.head, nn.BatchNorm2d) == 0
        assert _count(gn_model.neck, nn.GroupNorm) + _count(gn_model.head, nn.GroupNorm) == n_bn
        # The backbone keeps its own norms untouched.
        assert _count(gn_model.backbone, nn.GroupNorm) == _count(bn_model.backbone, nn.GroupNorm)
        assert _count(gn_model.backbone, nn.BatchNorm2d) == _count(bn_model.backbone, nn.BatchNorm2d)

    def test_default_is_bn(self):
        assert _two_stream("bn").neck_head_norm == MasterModel(
            pretrained_backbone=False, fusion_mode="cross_attention"
        ).neck_head_norm == "bn"


class TestCheckpointSchema:
    def test_gn_model_refuses_bn_state_dict_with_clear_error(self):
        bn_state = _two_stream("bn").state_dict()
        gn_model = _two_stream("gn")
        with pytest.raises(ValueError, match="neck_head_norm='bn'"):
            check_norm_compatible(gn_model, bn_state)
        # And strict loading on its own never mis-loads silently.
        with pytest.raises(RuntimeError):
            gn_model.load_state_dict(bn_state, strict=True)

    def test_bn_model_refuses_gn_state_dict(self):
        with pytest.raises(ValueError, match="neck_head_norm='gn'"):
            check_norm_compatible(_two_stream("bn"), _two_stream("gn").state_dict())

    def test_matching_schema_passes(self):
        gn_model = _two_stream("gn")
        check_norm_compatible(gn_model, gn_model.state_dict())
        gn_model.load_state_dict(_two_stream("gn").state_dict(), strict=True)


class TestConfigValidation:
    def test_defaults(self):
        config = TrainingConfig()
        assert (config.neck_head_norm, config.gn_groups) == ("bn", 32)

    def test_rejects_unknown_norm(self):
        with pytest.raises(ValueError, match="neck_head_norm"):
            TrainingConfig(neck_head_norm="ln")

    @pytest.mark.parametrize("groups", [0, -1])
    def test_rejects_non_positive_groups(self, groups):
        with pytest.raises(ValueError, match="gn_groups"):
            TrainingConfig(gn_groups=groups)
