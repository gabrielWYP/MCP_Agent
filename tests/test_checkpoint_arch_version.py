"""Tests for the checkpoint `arch_version` hard-break guard (fusion-redesign D-C).

Every state_dict key changes under the redesign
(`backbone.rgb_stem.*`/`backbone.nir_stem.*`/`backbone.shared_stages.*` ->
`backbone.stem.*`/`backbone.stages.*`; `fusion.*` disappears;
`neck.fpn_rgb.*`/`neck.fpn_nir.*`/`neck.fusion_convs.*` -> `neck.fpn.*`), so a
v1 (dual-stream fusion) checkpoint is unloadable into the v2 MasterModel.
These tests verify the `arch_version` check raises a readable error BEFORE
`load_state_dict` is attempted, rather than a 200-line missing/unexpected-key
dump.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.config import TrainingConfig
from src.training.loop import CHECKPOINT_ARCH_VERSION
from src.training.strides import CROSS_ATTENTION_HEAD_STRIDES


def _write_synthetic_v1_checkpoint(path: Path) -> None:
    """A synthetic v1-shaped checkpoint: dual-stream state_dict keys, no
    `arch_version` (the pre-D-C schema never wrote one)."""
    torch.save(
        {
            "epoch": 68,
            "phase": 1,
            "model_state_dict": {
                "backbone.rgb_stem.0.weight": torch.zeros(96, 3, 4, 4),
                "backbone.nir_stem.0.weight": torch.zeros(96, 1, 4, 4),
                "fusion.fusion_stages.0.norm_rgb.weight": torch.zeros(96),
            },
            "metrics": {},
            "best_map50": 0.0643,
            "config": {"head_strides": [8, 16, 32]},
        },
        path,
    )


@pytest.fixture
def v1_checkpoint_path(tmp_path) -> Path:
    path = tmp_path / "v1_checkpoint.pt"
    _write_synthetic_v1_checkpoint(path)
    return path


@pytest.fixture
def master_config(tmp_path) -> TrainingConfig:
    return TrainingConfig(
        model_type="master",
        output_dir=str(tmp_path),
        num_classes=2,
        backbone_variant="tiny",
    )


class TestEvaluateCheckpointArchVersionGuard:
    def test_v1_checkpoint_raises_before_load_state_dict(self, v1_checkpoint_path, master_config):
        from scripts.evaluate_checkpoint import _load_model

        device = torch.device("cpu")
        with pytest.raises(ValueError, match="arch_version"):
            _load_model("master", str(v1_checkpoint_path), master_config, device)

    def test_error_message_names_v1_and_v2(self, v1_checkpoint_path, master_config):
        from scripts.evaluate_checkpoint import _load_model

        device = torch.device("cpu")
        with pytest.raises(ValueError, match="v1.*v2|v2.*v1|MasterModel v2"):
            _load_model("master", str(v1_checkpoint_path), master_config, device)


class TestVisualizeDamagePredictionsArchVersionGuard:
    def test_v1_checkpoint_raises_before_load_state_dict(self, v1_checkpoint_path, master_config):
        from scripts.visualize_damage_predictions import load_model

        device = torch.device("cpu")
        with pytest.raises(ValueError, match="arch_version"):
            load_model("master", str(v1_checkpoint_path), master_config, device)


class TestV2CheckpointCarriesArchVersion:
    def test_current_checkpoint_schema_version_is_two(self):
        assert CHECKPOINT_ARCH_VERSION == 2


class TestCrossAttentionModeAcceptsV1Checkpoints:
    """`fusion_mode="cross_attention"` is the mode a v1 checkpoint belongs
    to, so the same synthetic checkpoint that must be rejected above must be
    accepted here — otherwise the restored two-stream path could never load
    `checkpoints/mastermodel_mango` (arch_version absent)."""

    @pytest.fixture
    def cross_attention_config(self, tmp_path) -> TrainingConfig:
        return TrainingConfig(
            model_type="master",
            output_dir=str(tmp_path),
            num_classes=2,
            backbone_variant="tiny",
            fusion_mode="cross_attention",
            head_strides=list(CROSS_ATTENTION_HEAD_STRIDES),
            assigner_level_ranges=[64.0, 128.0],
        )

    def test_untagged_checkpoint_passes_the_guard(self, v1_checkpoint_path, cross_attention_config):
        from scripts.evaluate_checkpoint import _check_arch_version

        # Reaches load_state_dict rather than raising on arch_version. (The
        # synthetic state_dict holds only 3 keys, so the load itself fails —
        # what matters is that it is no longer the arch_version guard.)
        _check_arch_version(str(v1_checkpoint_path), None, cross_attention_config.fusion_mode)

    def test_v2_checkpoint_is_rejected_by_the_cross_attention_path(self, cross_attention_config):
        from scripts.evaluate_checkpoint import _check_arch_version

        with pytest.raises(ValueError, match="arch_version"):
            _check_arch_version("dummy.pt", CHECKPOINT_ARCH_VERSION, cross_attention_config.fusion_mode)

    def test_head_strides_resolve_without_a_recorded_config(self, cross_attention_config):
        """Pre-tag checkpoints record no `config.head_strides`; the
        cross-attention pyramid is fixed by DualFPN, so there is nothing to
        read and nothing to guess."""
        from scripts.evaluate_checkpoint import _resolve_head_strides

        resolved = _resolve_head_strides({"config": {}}, cross_attention_config.fusion_mode)
        assert resolved == list(CROSS_ATTENTION_HEAD_STRIDES)
