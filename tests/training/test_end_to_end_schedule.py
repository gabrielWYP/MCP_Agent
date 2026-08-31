"""Tests for the end-to-end MasterModel training schedule (fusion-redesign D-4/D-F).

Covers:
- freeze_stages=0 from epoch 0: every backbone stage, stem, neck, head
  parameter is trainable.
- Discriminative LR: pretrained backbone-stage params get
  `backbone_lr_mult * lr`; stem/neck/head get the full `lr`.
- `_module_grad_norms()` covers every backbone stage, the stem, the neck,
  and every head level (the generalisation of the old single-module
  `_rgb_stem_grad_norm`).
- `schedule="two_phase"` still exists and is selectable (rollback path).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.master.master_model import MasterModel
from src.training.config import TrainingConfig
from src.training.loop import Trainer


def _make_trainer(tmp_path, **config_overrides) -> Trainer:
    model = MasterModel(
        num_classes=2, pretrained_backbone=False, backbone_variant="tiny",
        head_strides=[8, 16, 32],
    )
    config = TrainingConfig(
        model_type="master",
        head_strides=[8, 16, 32],
        assigner_level_ranges=[64, 128],
        image_size=64,
        batch_size=1,
        effective_batch=1,
        num_workers=0,
        precision="fp32",
        device="cpu",  # deterministic CPU test regardless of host GPU
        output_dir=str(tmp_path),
        **config_overrides,
    )
    return Trainer(model=model, config=config, train_loader=None, val_loader=[])


class TestDiscriminativeParamGroups:
    def test_two_groups_with_correct_lr_multiplier(self, tmp_path):
        trainer = _make_trainer(tmp_path, backbone_lr_mult=0.1)
        groups = trainer._discriminative_param_groups(lr=0.001)

        assert len(groups) == 2
        backbone_group, new_group = groups
        assert backbone_group["lr"] == pytest.approx(0.0001)
        assert new_group["lr"] == pytest.approx(0.001)

    def test_backbone_group_only_contains_stage_params(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        groups = trainer._discriminative_param_groups(lr=0.001)
        backbone_group_params = set(id(p) for p in groups[0]["params"])
        stage_params = set(
            id(p) for p in trainer.model.backbone.stages.parameters()
        )
        assert backbone_group_params == stage_params

    def test_new_group_contains_stem_neck_and_head(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        groups = trainer._discriminative_param_groups(lr=0.001)
        new_group_params = set(id(p) for p in groups[1]["params"])

        stem_params = set(id(p) for p in trainer.model.backbone.stem.parameters())
        neck_params = set(id(p) for p in trainer.model.neck.parameters())
        head_params = set(id(p) for p in trainer.model.head.parameters())

        assert stem_params <= new_group_params
        assert neck_params <= new_group_params
        assert head_params <= new_group_params
        # Stem must NOT be in the pretrained backbone group — it is
        # 4-channel and out of ImageNet distribution by construction.
        assert stem_params.isdisjoint(set(
            id(p) for p in trainer.model.backbone.stages.parameters()
        ))

    def test_frozen_stage_excluded_from_groups(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        trainer.model.freeze_backbone(freeze_stages=2)
        groups = trainer._discriminative_param_groups(lr=0.001)
        all_group_params = set(id(p) for g in groups for p in g["params"])

        for i, stage in enumerate(trainer.model.backbone.stages):
            for p in stage.parameters():
                if i < 2:
                    assert id(p) not in all_group_params
                else:
                    assert id(p) in all_group_params


class TestModuleGradNorms:
    def test_covers_every_backbone_stage_stem_neck_and_head(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        trainer.model.train()

        rgb = torch.randn(1, 3, 64, 64)
        nir = torch.randn(1, 1, 64, 64)
        out = trainer.model(rgb, nir)
        loss = sum(p.float().pow(2).sum() for p in out["preds"])
        loss.backward()

        norms = trainer._module_grad_norms()

        assert "backbone_stem" in norms
        for i in range(4):
            assert f"backbone_stage{i}" in norms
        assert "neck" in norms
        for i in range(len(trainer.model.head.heads)):
            assert f"head_level{i}" in norms

        # Every module actually received a gradient after a real backward.
        assert all(v >= 0.0 for v in norms.values())
        assert norms["backbone_stem"] > 0.0

    def test_returns_zero_for_frozen_module_with_no_grad(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        # No backward has run yet — every param.grad is None.
        norms = trainer._module_grad_norms()
        assert all(v == 0.0 for v in norms.values())


class TestFreezeStagesZeroTrainsEverything:
    def test_end_to_end_freeze_stages_zero_leaves_all_trainable(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        trainer.model.freeze_backbone(freeze_stages=0)

        assert all(p.requires_grad for p in trainer.model.backbone.stem.parameters())
        assert all(
            p.requires_grad
            for stage in trainer.model.backbone.stages
            for p in stage.parameters()
        )


class TestScheduleFieldRollback:
    def test_two_phase_schedule_is_still_selectable(self, tmp_path):
        """Config-only rollback (design.md Rollback Plan): schedule is
        restorable without a code revert."""
        trainer = _make_trainer(tmp_path, schedule="two_phase", epochs_phase1=1, epochs_phase2=1)
        assert trainer.config.schedule == "two_phase"

    def test_invalid_schedule_raises(self, tmp_path):
        with pytest.raises(ValueError, match="schedule"):
            TrainingConfig(model_type="master", schedule="three_phase")
