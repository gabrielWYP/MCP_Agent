"""Pooled out-of-fold aggregation in scripts/aggregate_kfold.py."""

from __future__ import annotations

import pytest
import torch

from scripts.aggregate_kfold import aggregate_seed, concat_predictions
from src.training.metrics import compute_map


def _image(gt: list[list[float]], gt_labels: list[int], preds: list[tuple[list[float], float, int]]) -> dict:
    return {
        "gt_boxes": torch.tensor(gt, dtype=torch.float32).reshape(-1, 4),
        "gt_labels": torch.tensor(gt_labels, dtype=torch.int64),
        "pred_boxes": torch.tensor([p[0] for p in preds], dtype=torch.float32).reshape(-1, 4),
        "pred_scores": torch.tensor([p[1] for p in preds], dtype=torch.float32),
        "pred_labels": torch.tensor([p[2] for p in preds], dtype=torch.int64),
    }


def _fold(images: list[dict]) -> dict[str, list]:
    keys = ("pred_boxes", "pred_scores", "pred_labels", "gt_boxes", "gt_labels")
    return {k: [img[k] for img in images] for k in keys}


BOX_A = [0.3, 0.3, 0.1, 0.1]
BOX_B = [0.7, 0.7, 0.1, 0.1]


def test_pooled_equals_ap_over_concatenated_predictions() -> None:
    # Fold 0: confident FP outranks the TP. Fold 1: TP with low score plus a missed GT.
    fold0 = _fold([
        _image([BOX_A], [1], [(BOX_A, 0.4, 1), (BOX_B, 0.9, 1)]),
        _image([BOX_B], [1], [(BOX_B, 0.8, 1)]),
    ])
    fold1 = _fold([
        _image([BOX_A, BOX_B], [1, 1], [(BOX_A, 0.3, 1)]),
        _image([], [], [(BOX_A, 0.95, 1)]),
    ])
    per_fold = {0: fold0, 1: fold1}

    result = aggregate_seed(per_fold, num_classes=2, score_threshold=0.25)

    concat = {k: fold0[k] + fold1[k] for k in fold0}
    assert concat_predictions(per_fold) == concat
    expected = compute_map(**concat, num_classes=2, score_threshold=0.25)
    assert result["pooled"]["ap50"] == pytest.approx(expected["per_class_ap_50"][1])
    assert result["pooled"]["ap50_95"] == pytest.approx(expected["per_class_ap_50_95"][1])
    assert result["pooled"]["num_images"] == 4

    for fold, preds in per_fold.items():
        m = compute_map(**preds, num_classes=2, score_threshold=0.25)
        assert result["per_fold"][fold]["ap50"] == pytest.approx(m["per_class_ap_50"][1])
    mean = (result["per_fold"][0]["ap50"] + result["per_fold"][1]["ap50"]) / 2
    assert result["per_fold_mean_std"]["ap50"]["mean"] == pytest.approx(mean)
