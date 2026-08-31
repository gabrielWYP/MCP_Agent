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
