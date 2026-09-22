"""KD adapters must actually train, and must stay out of the student.

Verified defect: `KDTrainer` attached its 1x1 distillation adapters to the
student as `kd_proj_*` submodules mapping TEACHER → student channels, and
called them inside the teacher's `torch.no_grad()` block. They were in the
optimizer but never received a gradient, so they stayed at random init for
the whole run (only BatchNorm running stats moved). They also leaked into
the student's `model_state_dict`, which a bare `StudentModel` cannot load
with `strict=True`.

Fix (FitNets): the adapters map STUDENT → teacher channels, run outside
`no_grad`, regress onto the frozen (detached) teacher features, and are owned
by the trainer (`KDTrainer.kd_adapters`) rather than by the student. These
tests pin each part of that contract with a real `KDTrainer` (real student,
real untrained tiny teacher), mirroring `test_kd_trainer_guards.py`.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.master.master_model import MasterModel
from src.models.student.student_model import (
    LEGACY_KD_ADAPTER_PREFIX,
    StudentModel,
    strip_legacy_kd_adapter_keys,
)
from src.training.kd_config import KDConfig
from src.training.kd_trainer import KDTrainer

_IMAGE_SIZE = 64
_NUM_CLASSES = 2


def _fake_batch(n: int = 1) -> dict:
    return {
        "rgb": torch.randn(n, 3, _IMAGE_SIZE, _IMAGE_SIZE),
        "nir": torch.randn(n, 1, _IMAGE_SIZE, _IMAGE_SIZE),
        "bboxes": [torch.tensor([[0.5, 0.5, 0.3, 0.3]]) for _ in range(n)],
        "labels": [torch.tensor([0]) for _ in range(n)],
    }


class _FakeLoader:
    def __init__(self, n_batches: int):
        self.n_batches = n_batches

    def __iter__(self):
        return (_fake_batch(1) for _ in range(self.n_batches))


class _StopAfterOptimizerBuilt(Exception):
    pass


@pytest.fixture(scope="module")
def teacher_checkpoint_path(tmp_path_factory):
    teacher = MasterModel(
        num_classes=_NUM_CLASSES, pretrained_backbone=False, backbone_variant="tiny"
    )
    path = tmp_path_factory.mktemp("kd_adapter_ckpt") / "teacher.pt"
    torch.save({"model_state_dict": teacher.state_dict()}, path)
    return path


def _make_kd_trainer(teacher_checkpoint_path, tmp_path, n_batches: int = 1) -> KDTrainer:
    config = KDConfig(
        teacher_checkpoint=str(teacher_checkpoint_path),
        num_classes=_NUM_CLASSES,
        backbone_variant="tiny",
        image_size=_IMAGE_SIZE,
        batch_size=1,
        precision="fp32",
        num_workers=0,
        output_dir=str(tmp_path),
    )
    return KDTrainer(
        model=StudentModel(num_classes=_NUM_CLASSES),
        config=config,
        train_loader=_FakeLoader(n_batches),
        val_loader=_FakeLoader(0),
    )


def _kd_loss_backward(trainer: KDTrainer) -> None:
    """One forward/backward of the KD loss alone, through the same code path
    `_train_epoch` uses (teacher under no_grad, adapters outside it)."""
    captured = {}
    real_kd_forward = trainer.kd_criterion.forward

    def _capturing_kd_forward(teacher_feats, student_feats):
        loss, per_level = real_kd_forward(teacher_feats, student_feats)
        captured["kd_loss"] = loss
        return loss, per_level

    trainer.kd_criterion.forward = _capturing_kd_forward
    # Zero out the detection loss so the only gradient source is KD.
    trainer.criterion = lambda preds, targets: (
        torch.zeros((), requires_grad=False),
        {"cls_loss": 0.0, "box_loss": 0.0},
    )

    class _NoStepOptimizer:
        """Keeps gradients intact after backward for inspection."""

        def zero_grad(self, set_to_none=True):
            for p in list(trainer.model.parameters()) + list(trainer.kd_adapters.parameters()):
                p.grad = None

        def step(self):
            pass

    trainer._train_epoch(_NoStepOptimizer(), epoch=1, phase=1)
    assert captured["kd_loss"].requires_grad


class TestAdapterGradients:
    def test_adapters_receive_nonzero_gradients(self, teacher_checkpoint_path, tmp_path):
        trainer = _make_kd_trainer(teacher_checkpoint_path, tmp_path)
        _kd_loss_backward(trainer)

        for name, adapter in trainer.kd_adapters.items():
            for pname, p in adapter.named_parameters():
                assert p.grad is not None, f"kd_adapters.{name}.{pname} got no gradient"
                assert p.grad.abs().sum().item() > 0, (
                    f"kd_adapters.{name}.{pname} gradient is all zeros"
                )

    def test_teacher_receives_no_gradients(self, teacher_checkpoint_path, tmp_path):
        trainer = _make_kd_trainer(teacher_checkpoint_path, tmp_path)
        _kd_loss_backward(trainer)

        assert all(not p.requires_grad for p in trainer.teacher.parameters())
        assert all(p.grad is None for p in trainer.teacher.parameters())

    def test_student_features_receive_kd_gradients(self, teacher_checkpoint_path, tmp_path):
        """The KD signal must reach the student (det loss is zeroed here)."""
        trainer = _make_kd_trainer(teacher_checkpoint_path, tmp_path)
        _kd_loss_backward(trainer)

        grads = [p.grad for p in trainer.model.backbone.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum().item() > 0 for g in grads)

    def test_adapters_map_student_to_teacher_channels(self, teacher_checkpoint_path, tmp_path):
        trainer = _make_kd_trainer(teacher_checkpoint_path, tmp_path)
        in_out = {
            name: [
                (seq[0].in_channels, seq[0].out_channels) for seq in adapter.projections
            ]
            for name, adapter in trainer.kd_adapters.items()
        }
        assert in_out == {
            "backbone": [(128, 384), (256, 768)],
            "fpn": [(128, 256), (256, 256), (256, 256)],
            "head_cls": [(64, 256), (128, 256), (256, 256)],
            "head_reg": [(64, 256), (128, 256), (256, 256)],
        }


class TestAdaptersInOptimizer:
    def test_adapter_params_are_in_the_real_optimizer(self, teacher_checkpoint_path, tmp_path):
        """Capture the optimizer `Trainer._train_phase` actually builds."""
        trainer = _make_kd_trainer(teacher_checkpoint_path, tmp_path)
        captured = {}

        def _capture(optimizer, epoch, phase):
            captured["optimizer"] = optimizer
            raise _StopAfterOptimizerBuilt

        trainer._train_epoch = _capture
        with pytest.raises(_StopAfterOptimizerBuilt):
            trainer._train_phase(phase=1, epochs=1, lr=1e-3, freeze_stages=0)

        optimized = {
            id(p) for group in captured["optimizer"].param_groups for p in group["params"]
        }
        adapter_ids = {id(p) for p in trainer.kd_adapters.parameters()}
        student_ids = {id(p) for p in trainer.model.parameters() if p.requires_grad}
        teacher_ids = {id(p) for p in trainer.teacher.parameters()}

        assert adapter_ids and adapter_ids <= optimized
        assert student_ids <= optimized
        assert not (teacher_ids & optimized)


class TestStudentStaysAdapterFree:
    def test_student_has_no_adapter_submodules_or_keys(self, teacher_checkpoint_path, tmp_path):
        trainer = _make_kd_trainer(teacher_checkpoint_path, tmp_path)

        adapter_ids = {id(p) for p in trainer.kd_adapters.parameters()}
        assert not adapter_ids & {id(p) for p in trainer.model.parameters()}
        assert not any(k.startswith(LEGACY_KD_ADAPTER_PREFIX) for k in trainer.model.state_dict())

    def test_student_state_dict_loads_strictly_into_bare_student(
        self, teacher_checkpoint_path, tmp_path
    ):
        trainer = _make_kd_trainer(teacher_checkpoint_path, tmp_path)
        bare = StudentModel(num_classes=_NUM_CLASSES)
        bare.load_state_dict(trainer.model.state_dict(), strict=True)

        bare.eval()
        with torch.no_grad():
            out = bare(torch.randn(1, 3, _IMAGE_SIZE, _IMAGE_SIZE))
        assert "preds" in out

    def test_checkpoint_stores_adapters_under_separate_key(
        self, teacher_checkpoint_path, tmp_path
    ):
        trainer = _make_kd_trainer(teacher_checkpoint_path, tmp_path)
        trainer._save_checkpoint(epoch=1, phase=1, metrics={}, filename="ckpt.pt")
        ckpt = torch.load(tmp_path / "ckpt.pt", map_location="cpu", weights_only=False)

        assert "kd_adapters_state_dict" in ckpt
        assert set(ckpt["kd_adapters_state_dict"]) == set(trainer.kd_adapters.state_dict())
        assert not any(k.startswith(LEGACY_KD_ADAPTER_PREFIX) for k in ckpt["model_state_dict"])
        StudentModel(num_classes=_NUM_CLASSES).load_state_dict(
            ckpt["model_state_dict"], strict=True
        )

    def test_legacy_kd_checkpoint_keys_are_stripped_for_student_loading(self):
        """Pre-fix KD checkpoints carried `kd_proj_*` inside
        `model_state_dict`; student weights must still load strictly."""
        student = StudentModel(num_classes=_NUM_CLASSES)
        legacy = dict(student.state_dict())
        legacy["kd_proj_fpn.projections.0.0.weight"] = torch.zeros(128, 256, 1, 1)
        legacy["kd_proj_backbone.projections.0.1.running_mean"] = torch.zeros(128)

        with pytest.raises(RuntimeError, match="Unexpected key"):
            StudentModel(num_classes=_NUM_CLASSES).load_state_dict(legacy, strict=True)

        StudentModel(num_classes=_NUM_CLASSES).load_state_dict(
            strip_legacy_kd_adapter_keys(legacy), strict=True
        )
