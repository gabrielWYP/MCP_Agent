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
