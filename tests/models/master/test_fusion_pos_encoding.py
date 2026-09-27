"""Tests for the opt-in positional encoding in `StageAttentionFusion`.

The trained two-stream checkpoint's cross-attention collapsed to
near-uniform weights: RGB and NIR are pixel-aligned, but the attention
logits carried no positional signal (reports/fusion-gradient-collapse/
diagnosis.md). `fusion_pos_encoding` adds a fixed 2D sinusoidal encoding to
the normalized query and key.

What these tests pin down:
- Flag off is bit-identical to a module built without the kwarg — every
  existing config and checkpoint keeps its behaviour.
- Flag on puts position into the logits: spatially constant RGB and NIR
  give exactly uniform attention without it, and non-uniform with it.
- The encoding adds no state_dict key, so the pre-redesign two-stream
  checkpoint still loads strict with the flag on — which is exactly why
  loaders must read the flag back from the checkpoint's recorded config.
- The config field validates, round-trips, and its experiment file differs
  from `twostream_p2.yaml` only where it should.
"""

from pathlib import Path
import sys

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.models.master.fusion import (
    CrossModalFusion,
    StageAttentionFusion,
    sinusoidal_2d_position_encoding,
)
from src.models.master.master_model import MasterModel
from src.training.config import TrainingConfig
from src.training.fusion_modes import (
    accepted_arch_versions,
    fusion_pos_encoding_from_checkpoint,
)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_CROSS_ATTENTION_CHECKPOINT = "checkpoints/mastermodel_mango/best_model.pt"
_P2_CONFIG = _REPO_ROOT / "configs/experiment/twostream_p2.yaml"
_P2_POSENC_CONFIG = _REPO_ROOT / "configs/experiment/twostream_p2_posenc.yaml"

CHANNELS = 96
HEADS = 4
SIDE = 8


def _stage(seed: int = 0, **kwargs) -> StageAttentionFusion:
    torch.manual_seed(seed)
    return StageAttentionFusion(channels=CHANNELS, num_heads=HEADS, **kwargs).eval()


def _attention_weights(stage: StageAttentionFusion, rgb, nir) -> torch.Tensor:
    """Head-averaged softmax weights (N, L, S) of the stage's attention."""
    captured = {}
    handle = stage.attn.register_forward_hook(
        lambda _module, _inputs, output: captured.__setitem__("weights", output[1])
    )
    try:
        with torch.no_grad():
            stage(rgb, nir)
    finally:
        handle.remove()
    return captured["weights"]


def _spatially_constant(seed: int) -> torch.Tensor:
    """Same channel vector at every position: LayerNorm maps every token to
    the same sequence element, so content alone cannot tell keys apart."""
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(1, CHANNELS, 1, 1, generator=generator)
    return vector.expand(1, CHANNELS, SIDE, SIDE).contiguous()


class TestEncoding:
    def test_shape_and_range(self):
        pos = sinusoidal_2d_position_encoding(20, CHANNELS)
        assert pos.shape == (400, CHANNELS)
        assert pos.dtype == torch.float32
        assert pos.abs().max() <= 1.0

    def test_every_grid_position_is_distinct(self):
        pos = sinusoidal_2d_position_encoding(20, CHANNELS)
        assert torch.unique(pos, dim=0).shape[0] == 400

    def test_row_major_token_order(self):
        """Token index y * side + x must match `flatten(2)`: moving along x
        changes only the column half, moving along y only the row half."""
        side = 5
        pos = sinusoidal_2d_position_encoding(side, CHANNELS).reshape(side, side, CHANNELS)
        half = CHANNELS // 2
        assert torch.equal(pos[2, 0, :half], pos[2, 4, :half])
        assert torch.equal(pos[0, 3, half:], pos[4, 3, half:])

    def test_channels_must_split_into_sin_cos_pairs_per_axis(self):
        with pytest.raises(AssertionError, match="divisible by 4"):
            sinusoidal_2d_position_encoding(4, 98)


class TestFlagOffIsUnchanged:
    def test_stage_output_matches_a_module_built_without_the_kwarg(self):
        rgb = torch.randn(2, CHANNELS, 32, 32)
        nir = torch.randn(2, CHANNELS, 32, 32)

        legacy = _stage()
        flagged_off = _stage(pos_encoding=False)
        with torch.no_grad():
            legacy_fused, legacy_map = legacy(rgb, nir)
            fused, attn_map = flagged_off(rgb, nir)

        assert torch.equal(fused, legacy_fused)
        assert torch.equal(attn_map, legacy_map)

    def test_master_model_output_matches_a_model_built_without_the_kwarg(self):
        rgb = torch.randn(1, 3, 128, 128)
        nir = torch.randn(1, 1, 128, 128)

        torch.manual_seed(0)
        legacy = MasterModel(num_classes=2, pretrained_backbone=False, fusion_mode="cross_attention").eval()
        torch.manual_seed(0)
        flagged_off = MasterModel(
            num_classes=2, pretrained_backbone=False, fusion_mode="cross_attention",
            fusion_pos_encoding=False,
        ).eval()

        with torch.no_grad():
            expected = legacy(rgb, nir)["preds"]
            actual = flagged_off(rgb, nir)["preds"]
        for a, b in zip(actual, expected):
            assert torch.equal(a, b)

    def test_flag_on_changes_the_output(self):
        """Guards the test above against a flag that is never read."""
        rgb = torch.randn(1, CHANNELS, 16, 16)
        nir = torch.randn(1, CHANNELS, 16, 16)
        with torch.no_grad():
            off, _ = _stage()(rgb, nir)
            on, _ = _stage(pos_encoding=True)(rgb, nir)
        assert not torch.allclose(off, on)


class TestPositionReachesTheLogits:
    def test_constant_inputs_give_exactly_uniform_attention_without_the_flag(self):
        weights = _attention_weights(_stage(), _spatially_constant(1), _spatially_constant(2))
        assert weights.shape == (1, SIDE * SIDE, SIDE * SIDE)
        assert torch.allclose(weights, torch.full_like(weights, 1.0 / (SIDE * SIDE)), atol=1e-6)

    def test_constant_inputs_give_non_uniform_attention_with_the_flag(self):
        weights = _attention_weights(
            _stage(pos_encoding=True), _spatially_constant(1), _spatially_constant(2)
        )
        uniform = torch.full_like(weights, 1.0 / (SIDE * SIDE))
        # At random init the positional term in the logits is small (the
        # projections learn its scale), so the bar is "well above float
        # noise" — the flag-off case above is uniform to 1e-6.
        assert (weights - uniform).abs().max() > 1e-4
        # Different query positions prefer different keys: the output can now
        # vary across positions even though the NIR content is constant.
        assert (weights[0, 0] - weights[0, -1]).abs().max() > 1e-4

    def test_encoding_is_sized_to_the_pooled_grid(self):
        """At 64x64 with max_tokens_side=20 the stage pools to 20x20; the
        encoding must follow the pooled grid, not the input resolution."""
        stage = _stage(pos_encoding=True)
        rgb = torch.randn(1, CHANNELS, 64, 64)
        weights = _attention_weights(stage, rgb, torch.randn(1, CHANNELS, 64, 64))
        assert weights.shape == (1, 400, 400)


class TestStateDictSchema:
    def test_the_flag_adds_no_state_dict_key_or_buffer(self):
        off = CrossModalFusion(pos_encoding=False)
        on = CrossModalFusion(pos_encoding=True)
        assert {k: v.shape for k, v in on.state_dict().items()} == {
            k: v.shape for k, v in off.state_dict().items()
        }
        assert not list(on.buffers())

    def test_master_model_state_dict_keys_are_unchanged(self):
        def keys(**kwargs):
            return set(MasterModel(
                num_classes=2, pretrained_backbone=False, fusion_mode="cross_attention", **kwargs
            ).state_dict())

        assert keys(fusion_pos_encoding=True) == keys()

    def test_pre_redesign_two_stream_checkpoint_loads_strict_with_the_flag_on(self):
        path = _REPO_ROOT / _CROSS_ATTENTION_CHECKPOINT
        if not path.exists():
            pytest.skip(f"local checkpoint not present: {_CROSS_ATTENTION_CHECKPOINT}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint.get("arch_version") in accepted_arch_versions("cross_attention")

        model = MasterModel(
            num_classes=2, pretrained_backbone=False, fusion_mode="cross_attention",
            fusion_pos_encoding=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        # Strict loading cannot tell the two apart — the recorded config can.
        assert fusion_pos_encoding_from_checkpoint(checkpoint) is False


class TestCheckpointResolution:
    @pytest.mark.parametrize(
        "checkpoint,expected",
        [
            ({}, False),
            ({"config": None}, False),
            ({"config": {"head_strides": [8, 16, 32]}}, False),
            ({"config": {"fusion_pos_encoding": False}}, False),
            ({"config": {"fusion_pos_encoding": True}}, True),
        ],
    )
    def test_absent_means_false(self, checkpoint, expected):
        assert fusion_pos_encoding_from_checkpoint(checkpoint) is expected

    def test_evaluate_checkpoint_prefers_the_checkpoint_over_the_config(self):
        from scripts.evaluate_checkpoint import _resolve_fusion_pos_encoding

        config = TrainingConfig(fusion_mode="cross_attention", fusion_pos_encoding=True)
        assert _resolve_fusion_pos_encoding({"config": {}}, config) is False

        config = TrainingConfig(fusion_mode="cross_attention")
        checkpoint = {"config": {"fusion_pos_encoding": True}}
        assert _resolve_fusion_pos_encoding(checkpoint, config) is True


class TestConfig:
    def test_default_is_off(self):
        assert TrainingConfig().fusion_pos_encoding is False

    def test_accepted_on_the_cross_attention_path(self):
        config = TrainingConfig(fusion_mode="cross_attention", fusion_pos_encoding=True)
        assert config.fusion_pos_encoding is True

    def test_rejected_on_the_early_path(self):
        with pytest.raises(ValueError, match="fusion_pos_encoding"):
            TrainingConfig(fusion_mode="early", fusion_pos_encoding=True)

    def test_master_model_rejects_it_on_the_early_path(self):
        with pytest.raises(ValueError, match="fusion_pos_encoding"):
            MasterModel(pretrained_backbone=False, fusion_mode="early", fusion_pos_encoding=True)

    @pytest.mark.parametrize("value", ["false", "true", 1, 0, None])
    def test_non_bool_values_are_rejected(self, value):
        """A YAML string "false" is truthy — it must not silently enable it."""
        with pytest.raises(ValueError, match="fusion_pos_encoding must be a bool"):
            TrainingConfig(fusion_mode="cross_attention", fusion_pos_encoding=value)

    def test_yaml_round_trip(self, tmp_path):
        config = TrainingConfig(fusion_mode="cross_attention", fusion_pos_encoding=True)
        path = tmp_path / "config.yaml"
        config.to_yaml(path)
        assert TrainingConfig.from_yaml(path).fusion_pos_encoding is True

    def test_experiment_file_enables_it(self):
        config = TrainingConfig.from_yaml(_P2_POSENC_CONFIG)
        assert config.fusion_pos_encoding is True
        assert config.fusion_mode == "cross_attention"
        assert config.head_strides == [4, 8, 16, 32]

    def test_experiment_file_differs_from_twostream_p2_only_where_intended(self):
        """experiment_sha256 hashes file bytes, hence the separate file; its
        parsed content must still differ from the P2 arm in nothing else."""
        with open(_P2_CONFIG) as f:
            p2 = yaml.safe_load(f)
        with open(_P2_POSENC_CONFIG) as f:
            posenc = yaml.safe_load(f)

        changed = {k for k in p2.keys() | posenc.keys() if p2.get(k) != posenc.get(k)}
        assert changed == {"fusion_pos_encoding", "output_dir"}
        assert "fusion_pos_encoding" not in p2
