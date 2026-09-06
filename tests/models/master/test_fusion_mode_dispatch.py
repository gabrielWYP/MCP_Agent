"""Tests for `MasterModel`'s `fusion_mode` dispatch.

`fusion_mode="early"` (default) is the single-stream 4-channel early-fusion
architecture; `fusion_mode="cross_attention"` is the restored two-stream
path (two stems over shared stages -> per-stage cross-attention -> DualFPN).

What these tests pin down:
- Both modes forward to the SAME 7-key output dict, with per-level shapes
  matching their own `head_strides`.
- The early path's state_dict key set is unchanged by the restoration —
  the four existing `arch_version=2` checkpoints must keep loading.
- The two key sets are disjoint, which is exactly why `arch_version` maps
  1:1 to a fusion mode.
- Every trainable parameter reaches an optimizer parameter group in both
  modes (the cross-attention path adds a whole `fusion` submodule that a
  module-name-based grouping would silently drop).
"""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.models.master.master_model import MasterModel
from src.training.config import TrainingConfig
from src.training.fusion_modes import (
    accepted_arch_versions,
    arch_version_for_mode,
    validate_fusion_mode,
)
from src.training.loop import CHECKPOINT_ARCH_VERSION, Trainer
from src.training.strides import CROSS_ATTENTION_HEAD_STRIDES

IMAGE_SIZE = 128
NUM_CLASSES = 2

OUTPUT_KEYS = {
    "preds",
    "cls_preds",
    "reg_preds",
    "distill_backbone",
    "distill_fpn",
    "distill_head_cls",
    "distill_head_reg",
}


def _build(fusion_mode: str, head_strides=None) -> MasterModel:
    return MasterModel(
        num_classes=NUM_CLASSES,
        pretrained_backbone=False,
        head_strides=head_strides,
        fusion_mode=fusion_mode,
    )


def _inputs(batch: int = 1):
    return (
        torch.randn(batch, 3, IMAGE_SIZE, IMAGE_SIZE),
        torch.randn(batch, 1, IMAGE_SIZE, IMAGE_SIZE),
    )


class TestForwardPerMode:
    @pytest.mark.parametrize(
        "fusion_mode,head_strides,expected_strides",
        [
            ("early", None, [4, 8, 16, 32]),
            ("early", [8, 16, 32], [8, 16, 32]),
            ("cross_attention", None, list(CROSS_ATTENTION_HEAD_STRIDES)),
        ],
    )
    def test_output_keys_and_level_shapes(self, fusion_mode, head_strides, expected_strides):
        model = _build(fusion_mode, head_strides).eval()
        assert model.head_strides == expected_strides

        rgb, nir = _inputs()
        with torch.no_grad():
            out = model(rgb, nir)

        assert set(out) == OUTPUT_KEYS

        # One entry per emitted pyramid level, spatially at that level's stride.
        for key in ("preds", "cls_preds", "reg_preds", "distill_fpn",
                    "distill_head_cls", "distill_head_reg"):
            assert len(out[key]) == len(expected_strides), key

        for i, stride in enumerate(expected_strides):
            side = IMAGE_SIZE // stride
            assert out["preds"][i].shape == (1, NUM_CLASSES + 4, side, side)
            assert out["cls_preds"][i].shape == (1, NUM_CLASSES, side, side)
            assert out["reg_preds"][i].shape == (1, 4, side, side)
            assert out["distill_fpn"][i].shape == (1, 256, side, side)

        # `distill_backbone` is always the 4-stage feature set carrying both
        # modalities — [S1..S4] on the early path, [F1..F4] (post
        # cross-attention) on the two-stream one — so `KDTrainer`'s
        # `distill_backbone[2:]` projection means the same thing in both.
        assert len(out["distill_backbone"]) == 4
        for i, channels in enumerate(MasterModel.STAGE_CHANNELS):
            side = IMAGE_SIZE // (4 * 2 ** i)
            assert out["distill_backbone"][i].shape == (1, channels, side, side)

    def test_cross_attention_gradients_reach_the_fusion_module(self):
        """The cross-attention module is only worth restoring if it trains."""
        model = _build("cross_attention")
        rgb, nir = _inputs()
        model(rgb, nir)["preds"][0].sum().backward()

        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.fusion.parameters()
        )


class TestStateDictSchema:
    def test_early_mode_key_set_is_the_pre_restoration_one(self):
        """The four existing arch_version=2 checkpoints load by key name; if
        any early-path key moved, they would stop loading."""
        keys = set(_build("early").state_dict())
        prefixes = {k.split(".", 2)[0] + "." + k.split(".", 2)[1] for k in keys}
        assert prefixes == {"backbone.stem", "backbone.stages", "neck.fpn", "head.heads"}

    def test_cross_attention_mode_restores_the_two_stream_key_set(self):
        keys = set(_build("cross_attention").state_dict())
        prefixes = {k.split(".", 2)[0] + "." + k.split(".", 2)[1] for k in keys}
        assert prefixes == {
            "backbone.rgb_stem",
            "backbone.nir_stem",
            "backbone.shared_stages",
            "fusion.fusion_stages",
            "neck.fpn_rgb",
            "neck.fpn_nir",
            "neck.fusion_convs",
            "head.heads",
        }

    def test_the_two_modes_share_no_backbone_fusion_or_neck_key(self):
        """The detection head is the only module the two modes have in
        common (same class, same names). Everything upstream of it is
        disjoint, which is why loading either checkpoint into the other
        model can only ever fail — hence the arch_version guard."""
        def upstream(model):
            return {
                k for k in model.state_dict()
                if k.startswith(("backbone.", "fusion.", "neck."))
            }

        assert not (upstream(_build("early")) & upstream(_build("cross_attention")))


class TestInvalidCombinations:
    def test_unknown_fusion_mode_raises(self):
        with pytest.raises(ValueError, match="fusion_mode"):
            _build("cross-attention")

    def test_config_rejects_unknown_fusion_mode(self):
        with pytest.raises(ValueError, match="fusion_mode"):
            TrainingConfig(fusion_mode="dual")

    def test_config_rejects_cross_attention_with_a_four_level_pyramid(self):
        with pytest.raises(ValueError, match="head_strides"):
            TrainingConfig(fusion_mode="cross_attention")

    def test_cross_attention_rejects_unsupported_head_strides(self):
        """DualFPN drops P2 and owns exactly 3 fusion convs — a 4-level
        request must raise, not silently truncate via zip."""
        with pytest.raises(ValueError, match="head_strides"):
            _build("cross_attention", [4, 8, 16, 32])

    def test_cross_attention_rejects_the_rgb_only_control_arm(self):
        with pytest.raises(ValueError, match="in_channels"):
            MasterModel(pretrained_backbone=False, in_channels=3, fusion_mode="cross_attention")


class TestArchVersionMapping:
    def test_early_is_two_and_cross_attention_is_one(self):
        assert arch_version_for_mode("early") == 2
        assert arch_version_for_mode("cross_attention") == 1

    def test_default_constant_still_names_the_early_schema(self):
        assert CHECKPOINT_ARCH_VERSION == 2

    def test_untagged_checkpoints_are_accepted_only_by_cross_attention(self):
        """Checkpoints predating the tag (e.g. checkpoints/mastermodel_mango)
        are two-stream by construction; None can never mean early fusion."""
        assert None in accepted_arch_versions("cross_attention")
        assert None not in accepted_arch_versions("early")

    def test_no_arch_version_is_shared_between_modes(self):
        assert not (accepted_arch_versions("early") & accepted_arch_versions("cross_attention"))

    def test_validate_fusion_mode_accepts_both_modes(self):
        validate_fusion_mode("early")
        validate_fusion_mode("cross_attention")


class TestOptimizerCoverage:
    """`_discriminative_param_groups` must reach every trainable parameter.

    The cross-attention path adds a top-level `fusion` submodule; a grouping
    built from an explicit module list would leave it out of the optimizer
    entirely and never train it — silently.
    """

    @pytest.mark.parametrize(
        "fusion_mode,head_strides",
        [("early", None), ("cross_attention", list(CROSS_ATTENTION_HEAD_STRIDES))],
    )
    def test_every_trainable_parameter_lands_in_a_group(self, fusion_mode, head_strides):
        model = _build(fusion_mode, head_strides)
        config = TrainingConfig(
            model_type="master",
            fusion_mode=fusion_mode,
            head_strides=head_strides or [4, 8, 16, 32],
            assigner_level_ranges=[64.0, 128.0] if head_strides else [32.0, 64.0, 128.0],
            split_manifest=None,
        )
        groups = Trainer._discriminative_param_groups(
            SimpleNamespace(model=model, config=config), lr=1e-3
        )

        grouped = {id(p) for group in groups for p in group["params"]}
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        assert grouped == trainable

        # Two groups, and the pretrained stages are the discriminated one.
        assert len(groups) == 2
        stage_ids = {id(p) for p in model.backbone.stages.parameters()}
        assert {id(p) for p in groups[0]["params"]} == stage_ids

    @pytest.mark.parametrize("fusion_mode", ["early", "cross_attention"])
    def test_grad_norm_instrumentation_covers_every_stem(self, fusion_mode):
        model = _build(fusion_mode)
        norms = Trainer._module_grad_norms(SimpleNamespace(model=model))

        expected_stems = {f"backbone_{name}" for name in model.backbone.stem_modules}
        assert expected_stems <= set(norms)
        assert ("fusion" in norms) == (fusion_mode == "cross_attention")


# Local-only artifacts (not tracked in git): skipped when absent rather than
# failing, so the suite still runs on a fresh clone.
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_EARLY_CHECKPOINTS = {
    "v1_seed42": ("checkpoints/fusion_redesign/v1_seed42/best_model.pt", [4, 8, 16, 32]),
    "he_seed42": ("checkpoints/fusion_redesign/he_seed42/best_model.pt", [8, 16, 32]),
    "v1_clean_seed42": ("checkpoints/fusion_redesign/v1_clean_seed42/best_model.pt", [4, 8, 16, 32]),
    "v1_seed1337": ("checkpoints/fusion_redesign/v1_seed1337/best_model.pt", [4, 8, 16, 32]),
}
_CROSS_ATTENTION_CHECKPOINT = "checkpoints/mastermodel_mango/best_model.pt"


def _load_local_checkpoint(relative_path: str):
    path = _REPO_ROOT / relative_path
    if not path.exists():
        pytest.skip(f"local checkpoint not present: {relative_path}")
    return torch.load(path, map_location="cpu", weights_only=False)


class TestRealCheckpointsStillLoad:
    """The restoration must not shift a single key on either path."""

    @pytest.mark.parametrize("name", sorted(_EARLY_CHECKPOINTS))
    def test_existing_early_fusion_checkpoint_loads_strict(self, name):
        relative_path, head_strides = _EARLY_CHECKPOINTS[name]
        checkpoint = _load_local_checkpoint(relative_path)
        assert checkpoint["arch_version"] == arch_version_for_mode("early")

        model = MasterModel(
            num_classes=NUM_CLASSES,
            pretrained_backbone=False,
            head_strides=head_strides,
            fusion_mode="early",
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    def test_pre_redesign_two_stream_checkpoint_loads_strict(self):
        checkpoint = _load_local_checkpoint(_CROSS_ATTENTION_CHECKPOINT)
        # Written before the arch_version tag existed.
        assert checkpoint.get("arch_version") in accepted_arch_versions("cross_attention")

        model = MasterModel(
            num_classes=NUM_CLASSES,
            pretrained_backbone=False,
            fusion_mode="cross_attention",
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
