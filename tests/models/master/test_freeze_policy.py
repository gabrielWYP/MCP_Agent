"""Tests for MasterModel backbone freezing policy.

fusion-redesign D-4: the previous `rgb_stem`/`nir_stem` freeze asymmetry
(`rgb_stem` always frozen regardless of `freeze_stages`, `nir_stem` always
trainable) no longer has a rationale — there is a single stem now. This
test file is rewritten against the unified `stem`/`stages` names.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.models.master.master_model import MasterModel


def _all_frozen(parameters) -> bool:
    """Return True when every parameter is frozen."""
    return all(not param.requires_grad for param in parameters)


def _all_trainable(parameters) -> bool:
    """Return True when every parameter is trainable."""
    return all(param.requires_grad for param in parameters)


def test_freeze_backbone_freezes_stem_and_stages_below_threshold() -> None:
    """freeze_backbone(freeze_stages=2): stem and stages[0:2] frozen, stages[2:] trainable."""
    model = MasterModel(pretrained_backbone=False)

    model.freeze_backbone(freeze_stages=2)

    assert _all_frozen(model.backbone.stem.parameters())
    assert _all_frozen(model.backbone.stages[0].parameters())
    assert _all_frozen(model.backbone.stages[1].parameters())
    assert _all_trainable(model.backbone.stages[2].parameters())
    assert _all_trainable(model.backbone.stages[3].parameters())


def test_freeze_backbone_zero_leaves_stem_and_stages_trainable() -> None:
    """fusion-redesign D-4: freeze_stages=0 must train everything, including
    the stem — the end-to-end schedule's defining property."""
    model = MasterModel(pretrained_backbone=False)

    model.freeze_backbone(freeze_stages=0)

    assert _all_trainable(model.backbone.stem.parameters())
    assert all(
        _all_trainable(stage.parameters())
        for stage in model.backbone.stages
    )


def test_freeze_backbone_all_stages() -> None:
    model = MasterModel(pretrained_backbone=False)

    model.freeze_backbone(freeze_stages=4)

    assert _all_frozen(model.backbone.stem.parameters())
    assert all(
        _all_frozen(stage.parameters())
        for stage in model.backbone.stages
    )


def test_unfreeze_backbone_stages_default_leaves_stem_frozen() -> None:
    """Backward-compat default: unfreeze_stem=False changes nothing about the stem."""
    model = MasterModel(pretrained_backbone=False)
    model.freeze_backbone(freeze_stages=4)

    model.unfreeze_backbone_stages([2, 3])

    assert _all_frozen(model.backbone.stem.parameters())
    assert _all_trainable(model.backbone.stages[2].parameters())
    assert _all_trainable(model.backbone.stages[3].parameters())


def test_unfreeze_backbone_stages_can_unfreeze_stem() -> None:
    """unfreeze_stem=True must actually let the stem adapt during a
    partial-unfreeze call (renamed from unfreeze_rgb_stem, D-4)."""
    model = MasterModel(pretrained_backbone=False)
    model.freeze_backbone(freeze_stages=4)

    model.unfreeze_backbone_stages([2, 3], unfreeze_stem=True)

    assert _all_trainable(model.backbone.stem.parameters())
