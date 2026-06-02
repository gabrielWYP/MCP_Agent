"""Unit test: mAP computation with hand-crafted predictions and GT."""

import torch
import pytest

from src.training.metrics import compute_map


class TestComputeMap:
    """Test mAP computation with known inputs."""

    def test_perfect_predictions(self):
        """Perfect predictions should yield mAP=1.0 for the predicted class.

        With 2 classes and only class 0 present, class 1 AP=0, so mean = 0.5.
        """
        # GT: one box at center
        gt_boxes = [torch.tensor([[0.5, 0.5, 0.2, 0.2]])]
        gt_labels = [torch.tensor([0])]

        # Prediction: exact match with high confidence
        pred_boxes = [torch.tensor([[0.5, 0.5, 0.2, 0.2]])]
        pred_scores = [torch.tensor([0.99])]
        pred_labels = [torch.tensor([0])]

        result = compute_map(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels, num_classes=2,
        )

        # Class 0 AP = 1.0, Class 1 AP = 0.0 (no GT) → mean = 0.5
        assert result["map50"] == pytest.approx(0.5, abs=0.01)
        assert result["per_class_ap_50"][0] == pytest.approx(1.0, abs=0.01)

    def test_no_predictions(self):
        """No predictions should yield mAP=0.0."""
        gt_boxes = [torch.tensor([[0.5, 0.5, 0.2, 0.2]])]
        gt_labels = [torch.tensor([0])]

        pred_boxes = [torch.zeros(0, 4)]
        pred_scores = [torch.zeros(0)]
        pred_labels = [torch.zeros(0, dtype=torch.long)]

        result = compute_map(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels, num_classes=2,
        )

        assert result["map50"] == pytest.approx(0.0, abs=0.01)

    def test_no_gt(self):
        """No ground truth should yield mAP=0.0."""
        gt_boxes = [torch.zeros(0, 4)]
        gt_labels = [torch.zeros(0, dtype=torch.long)]

        pred_boxes = [torch.tensor([[0.5, 0.5, 0.2, 0.2]])]
        pred_scores = [torch.tensor([0.99])]
        pred_labels = [torch.tensor([0])]

        result = compute_map(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels, num_classes=2,
        )

        assert result["map50"] == pytest.approx(0.0, abs=0.01)

    def test_wrong_class(self):
        """Predictions with wrong class should not count as TP."""
        gt_boxes = [torch.tensor([[0.5, 0.5, 0.2, 0.2]])]
        gt_labels = [torch.tensor([0])]

        pred_boxes = [torch.tensor([[0.5, 0.5, 0.2, 0.2]])]
        pred_scores = [torch.tensor([0.99])]
        pred_labels = [torch.tensor([1])]  # wrong class

        result = compute_map(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels, num_classes=2,
        )

        # Class 0 has GT but no correct predictions → AP=0
        # Class 1 has predictions but no GT → AP=0
        assert result["map50"] == pytest.approx(0.0, abs=0.01)

    def test_per_class_ap(self):
        """Per-class AP should be reported for each class."""
        gt_boxes = [
            torch.tensor([[0.3, 0.3, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1]])
        ]
        gt_labels = [torch.tensor([0, 1])]

        pred_boxes = [
            torch.tensor([[0.3, 0.3, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1]])
        ]
        pred_scores = [torch.tensor([0.9, 0.8])]
        pred_labels = [torch.tensor([0, 1])]

        result = compute_map(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels, num_classes=2,
        )

        assert 0 in result["per_class_ap_50"]
        assert 1 in result["per_class_ap_50"]
        assert result["per_class_ap_50"][0] == pytest.approx(1.0, abs=0.01)
        assert result["per_class_ap_50"][1] == pytest.approx(1.0, abs=0.01)

    def test_map_50_95(self):
        """mAP@0.5:0.95 should be ≤ mAP@0.5."""
        gt_boxes = [torch.tensor([[0.5, 0.5, 0.3, 0.3]])]
        gt_labels = [torch.tensor([0])]

        pred_boxes = [torch.tensor([[0.5, 0.5, 0.3, 0.3]])]
        pred_scores = [torch.tensor([0.99])]
        pred_labels = [torch.tensor([0])]

        result = compute_map(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels, num_classes=2,
        )

        assert result["map_50_95"] <= result["map50"] + 0.01
