"""Evaluation decode threshold split and checkpoint-selection metric.

Two evaluation defects:

1. AP was computed on detections already decoded at the operating-point
   `conf_threshold` (0.25), truncating the PR curve. `Trainer.evaluate()`
   now decodes at `eval_conf_threshold` (default 0.001) while P/R/F1 and
   TP/FP/FN stay at the `conf_threshold` operating point.
2. best_model.pt selection and early-stopping patience followed `map50`,
   which is saturated by mango (AP=1.0). `config.selection_metric` now
   chooses the driving metric ("map50" | "damage_ap50").
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.config import TrainingConfig
from src.training.loop import Trainer, resolve_selection_value
from src.training.metrics import compute_map

IMAGE_SIZE = 64
BOX_PX = 16.0
# Damage anchors on the stride-8 level: (y, x, score). The second one is a
# true positive whose score sits below the 0.25 operating point.
DAMAGE_ANCHORS = ((1, 1, 0.9), (5, 5, 0.1))


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


class _FixedOutputModel(nn.Module):
    """Student-shaped model emitting fixed 3-level raw predictions.

    Every logit is ~0 probability except the class-1 anchors in
    `DAMAGE_ANCHORS`, which decode to 16x16 px boxes centred on their anchor.
    """

    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))

    def forward(self, rgb, nir=None):
        preds, cls_preds = [], []
        for stride in (8, 16, 32):
            size = IMAGE_SIZE // stride
            cls = torch.full((1, 2, size, size), -20.0)
            reg = torch.zeros((1, 4, size, size))
            reg[:, 2:] = math.log(BOX_PX)
            if stride == 8:
                for y, x, score in DAMAGE_ANCHORS:
                    cls[0, 1, y, x] = _logit(score)
            preds.append(torch.cat([cls, reg], dim=1).to(rgb.device))
            cls_preds.append(cls.to(rgb.device))
        return {"preds": preds, "cls_preds": cls_preds}


def _gt_box(y: int, x: int) -> list[float]:
    """Normalized cxcywh of the box decoded at stride-8 anchor (y, x)."""
    return [(x + 0.5) * 8 / IMAGE_SIZE, (y + 0.5) * 8 / IMAGE_SIZE,
            BOX_PX / IMAGE_SIZE, BOX_PX / IMAGE_SIZE]


def _val_batch() -> dict:
    return {
        "rgb": torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        "nir": torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE),
        "bboxes": [torch.tensor([_gt_box(y, x) for y, x, _ in DAMAGE_ANCHORS])],
        "labels": [torch.tensor([1, 1])],
    }


def _eval_trainer(**config_kwargs) -> Trainer:
    config = TrainingConfig(
        model_type="student",
        image_size=IMAGE_SIZE,
        batch_size=1,
        effective_batch=1,
        num_workers=0,
        precision="fp32",
        device="cpu",
        output_dir="/tmp/test_eval_threshold_unused",
        **config_kwargs,
    )
    return Trainer(
        model=_FixedOutputModel(), config=config,
        train_loader=None, val_loader=[_val_batch()],
    )


class TestEvalDecodeThreshold:
    def test_low_decode_threshold_recovers_truncated_ap(self):
        """A TP scored 0.1 is invisible to AP when decoding at 0.25."""
        full = _eval_trainer(eval_conf_threshold=0.001).evaluate()
        truncated = _eval_trainer(eval_conf_threshold=0.25).evaluate()

        assert full["per_class_ap_50"][1] >= truncated["per_class_ap_50"][1]
        assert full["per_class_ap_50"][1] == pytest.approx(1.0)
        assert truncated["per_class_ap_50"][1] == pytest.approx(0.5)

    def test_operating_point_unchanged_by_eval_decode_threshold(self):
        """P/R/F1/TP/FP/FN stay at conf_threshold regardless of the decode."""
        full = _eval_trainer(eval_conf_threshold=0.001).evaluate()
        truncated = _eval_trainer(eval_conf_threshold=0.25).evaluate()

        for metrics in (full, truncated):
            assert metrics["per_class_counts"][1] == {"tp": 1, "fp": 0, "fn": 1}
            assert metrics["per_class_recall"][1] == pytest.approx(0.5)
            assert metrics["per_class_precision"][1] == pytest.approx(1.0)

    def test_operating_point_ignores_detections_below_score_threshold(self):
        """compute_map: sub-threshold detections count toward AP only."""
        gt_boxes = [torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1]])]
        gt_labels = [torch.tensor([1, 1])]
        pred_boxes = [torch.tensor([
            [0.2, 0.2, 0.1, 0.1],  # TP, above threshold
            [0.7, 0.7, 0.1, 0.1],  # TP, below threshold
            [0.4, 0.4, 0.1, 0.1],  # FP, below threshold
        ])]
        pred_scores = [torch.tensor([0.9, 0.1, 0.05])]
        pred_labels = [torch.tensor([1, 1, 1])]

        result = compute_map(
            pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels,
            num_classes=2, score_threshold=0.25,
        )

        assert result["per_class_counts"][1] == {"tp": 1, "fp": 0, "fn": 1}
        assert result["per_class_precision"][1] == pytest.approx(1.0)
        assert result["per_class_recall"][1] == pytest.approx(0.5)
        assert result["per_class_ap_50"][1] == pytest.approx(1.0)


class TestEvalConfThresholdConfig:
    def test_default_keeps_old_configs_loadable(self):
        config = TrainingConfig(model_type="student")
        assert config.eval_conf_threshold == pytest.approx(0.001)
        assert config.selection_metric == "map50"

    def test_rejects_eval_threshold_above_operating_point(self):
        with pytest.raises(ValueError, match="eval_conf_threshold"):
            TrainingConfig(model_type="student", eval_conf_threshold=0.5)


class _StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))

    def forward(self, rgb, nir=None):
        return {"preds": None}

    def freeze_backbone(self, freeze_stages: int) -> None:
        pass


class _FakeLoader:
    def __iter__(self):
        yield {
            "rgb": torch.zeros(1, 3, 4, 4),
            "nir": torch.zeros(1, 1, 4, 4),
            "bboxes": [torch.zeros(0, 4)],
            "labels": [torch.zeros(0, dtype=torch.long)],
        }


def _val_metrics(map50: float, ap_damage: float) -> dict:
    return {
        "map50": map50,
        "map_50_95": 0.0,
        "per_class_ap_50": {0: 1.0, 1: ap_damage},
        "per_class_counts": {0: {"tp": 1, "fp": 0, "fn": 0},
                             1: {"tp": 1, "fp": 0, "fn": 1}},
    }


# mAP keeps improving (mango-driven) while damage AP peaks at epoch 1.
VAL_SEQUENCE = [_val_metrics(0.55, 0.10), _val_metrics(0.53, 0.05), _val_metrics(0.56, 0.04),
                _val_metrics(0.57, 0.03), _val_metrics(0.58, 0.02)]


def _run_phase(tmp_path: Path, selection_metric: str) -> tuple[Trainer, list[int]]:
    config = TrainingConfig(
        model_type="student",
        batch_size=1,
        effective_batch=1,
        num_workers=0,
        precision="fp32",
        device="cpu",
        warmup_epochs=1,
        save_interval=100,
        patience=2,
        selection_metric=selection_metric,
        output_dir=str(tmp_path),
    )
    trainer = Trainer(model=_StubModel(), config=config,
                      train_loader=_FakeLoader(), val_loader=[])
    trainer.criterion = lambda preds, targets: (
        torch.tensor(0.5, requires_grad=True),
        {"cls_loss": 0.25, "box_loss": 0.25},
    )
    validated: list[int] = []

    def _validate(epoch, phase):
        validated.append(epoch)
        return VAL_SEQUENCE[epoch - 1]

    trainer._validate = _validate
    trainer._train_phase(phase=1, epochs=len(VAL_SEQUENCE), lr=1e-3, freeze_stages=0)
    return trainer, validated


class TestSelectionMetric:
    def test_damage_ap50_drives_best_checkpoint_and_patience(self, tmp_path):
        trainer, validated = _run_phase(tmp_path, "damage_ap50")

        # Damage AP never improves after epoch 1 -> stop after patience=2.
        assert validated == [1, 2, 3]
        checkpoint = torch.load(tmp_path / "best_model.pt", weights_only=False)
        assert checkpoint["epoch"] == 1
        assert checkpoint["selection_metric"] == "damage_ap50"
        assert checkpoint["best_score"] == pytest.approx(0.10)
        assert checkpoint["best_map50"] == pytest.approx(0.55)

    def test_map50_keeps_legacy_behaviour(self, tmp_path):
        trainer, validated = _run_phase(tmp_path, "map50")

        # mAP dips once (epoch 2) then improves -> runs every epoch.
        assert validated == [1, 2, 3, 4, 5]
        checkpoint = torch.load(tmp_path / "best_model.pt", weights_only=False)
        assert checkpoint["epoch"] == 5
        assert checkpoint["selection_metric"] == "map50"
        assert trainer.best_map50 == pytest.approx(0.58)

    def test_rejects_unknown_selection_metric(self):
        with pytest.raises(ValueError, match="selection_metric"):
            TrainingConfig(model_type="student", selection_metric="ap50_class_1")

    def test_missing_class_1_ap_falls_back_to_map50(self):
        value, name = resolve_selection_value({"map50": 0.7}, "damage_ap50")
        assert (value, name) == (pytest.approx(0.7), "map50")

    def test_no_class_1_gt_falls_back_to_map50(self):
        metrics = {
            "map50": 0.5,
            "per_class_ap_50": {0: 1.0, 1: 0.0},
            "per_class_counts": {1: {"tp": 0, "fp": 3, "fn": 0}},
        }
        value, name = resolve_selection_value(metrics, "damage_ap50")
        assert (value, name) == (pytest.approx(0.5), "map50")

    def test_class_1_ap_selected_when_defined(self):
        value, name = resolve_selection_value(_val_metrics(0.55, 0.12), "damage_ap50")
        assert (value, name) == (pytest.approx(0.12), "damage_ap50")
