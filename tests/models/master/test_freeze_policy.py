"""Tests for MasterModel backbone freezing policy."""

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


def test_freeze_backbone_keeps_nir_stem_trainable() -> None:
    """NIR stem must adapt even when shared ConvNeXt stages are frozen."""
    model = MasterModel(pretrained_backbone=False)

    model.freeze_backbone(freeze_stages=4)

    assert _all_frozen(model.backbone.rgb_stem.parameters())
    assert _all_trainable(model.backbone.nir_stem.parameters())
    assert all(
        _all_frozen(stage.parameters())
        for stage in model.backbone.shared_stages
    )


def test_unfreeze_backbone_stages_default_leaves_rgb_stem_frozen() -> None:
    """Backward-compat default: unfreeze_rgb_stem=False changes nothing about the stem."""
    model = MasterModel(pretrained_backbone=False)
    model.freeze_backbone(freeze_stages=4)

    model.unfreeze_backbone_stages([2, 3])

    assert _all_frozen(model.backbone.rgb_stem.parameters())


def test_unfreeze_backbone_stages_can_unfreeze_rgb_stem() -> None:
    """E6/Q10 production fix: unfreeze_rgb_stem=True must actually let the
    RGB stem adapt during a Phase 2 unfreeze call — the documented root cause
    behind the maestro underperforming its own student."""
    model = MasterModel(pretrained_backbone=False)
    model.freeze_backbone(freeze_stages=4)

    model.unfreeze_backbone_stages([2, 3], unfreeze_rgb_stem=True)

    assert _all_trainable(model.backbone.rgb_stem.parameters())
