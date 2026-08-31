"""Integration test: KDTrainer selects the teacher's matching levels by
stride, not by position (fusion-redesign §5/W7).

The teacher (`MasterModel`) defaults to 4 levels (`head_strides=[4, 8, 16,
32]`, P2 reconnected, D-3); the student is fixed at 3 (`STUDENT_STRIDES =
(8, 16, 32)`, D-2). A positional zip/truncation of the teacher's 4-level
`distill_fpn`/`distill_head_cls`/`distill_head_reg` against the student's
3-level projection presets would silently distill P2/P3/P4 into student
P3/P4/P5. This test builds a real 4-level teacher and a real student and
asserts the projections receive exactly the P3/P4/P5-shaped features, not
P2/P3/P4.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.master.master_model import MasterModel
from src.models.student.student_model import StudentModel
from src.training.kd_config import KDConfig
from src.training.kd_trainer import KDTrainer

_IMAGE_SIZE = 128
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


def test_kd_projections_receive_p3_p4_p5_shapes_not_p2_p3_p4(tmp_path):
    # 4-level teacher (default head_strides), matching the shipped default.
    teacher = MasterModel(
        num_classes=_NUM_CLASSES, pretrained_backbone=False, backbone_variant="tiny",
        head_strides=[4, 8, 16, 32],
    )
    teacher_ckpt = tmp_path / "teacher.pt"
    torch.save({"model_state_dict": teacher.state_dict()}, teacher_ckpt)

    config = KDConfig(
        teacher_checkpoint=str(teacher_ckpt),
        num_classes=_NUM_CLASSES,
        backbone_variant="tiny",
        image_size=_IMAGE_SIZE,
        batch_size=1,
        precision="fp32",
        num_workers=0,
        output_dir=str(tmp_path),
        head_strides=[4, 8, 16, 32],  # teacher architecture, carried via KDConfig
    )
    model = StudentModel(num_classes=_NUM_CLASSES)
    trainer = KDTrainer(
        model=model, config=config,
        train_loader=_FakeLoader(1), val_loader=_FakeLoader(0),
    )

    # Expected spatial sizes at strides 8/16/32 for a 128px input.
    expected_fpn_shapes = [_IMAGE_SIZE // s for s in (8, 16, 32)]

    captured = {}
    real_fpn_forward = trainer.model.kd_proj_fpn.forward

    def _capturing_fpn_forward(teacher_features):
        captured["fpn_spatial_sizes"] = [f.shape[-1] for f in teacher_features]
        return real_fpn_forward(teacher_features)

    trainer.model.kd_proj_fpn.forward = _capturing_fpn_forward

    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
    trainer._train_epoch(optimizer, epoch=1, phase=1)

    assert captured["fpn_spatial_sizes"] == expected_fpn_shapes, (
        f"KD received FPN levels at spatial sizes {captured['fpn_spatial_sizes']}, "
        f"expected the P3/P4/P5-matching sizes {expected_fpn_shapes} "
        "(a positional truncation would instead select P2/P3/P4)."
    )
