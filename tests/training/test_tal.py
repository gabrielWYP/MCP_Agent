"""Unit test: TAL assigner with synthetic inputs — verify fg_mask and target alignment."""

import torch
import pytest

from src.training.loss import TaskAlignedAssigner, _generate_anchors


class TestTaskAlignedAssigner:
    """Test TaskAlignedAssigner with known synthetic inputs."""

    def test_zero_gt_all_negative(self):
        """Zero GT boxes → all anchors should be negative."""
        assigner = TaskAlignedAssigner(topk=13, alpha=1.0, beta=6.0)

        B, num_anchors, nc = 1, 100, 2
        pred_scores = torch.rand(B, num_anchors, nc).sigmoid()
        pred_bboxes = torch.rand(B, num_anchors, 4) * 640

        gt_labels = [torch.zeros(0, dtype=torch.long)]
        gt_bboxes = [torch.zeros(0, 4)]

        # Simple anchors
        anchors = torch.rand(num_anchors, 2) * 640
        strides = [8]

        target_classes, target_bboxes, target_scores, fg_mask = assigner(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes, anchors, strides
        )

        assert fg_mask.shape == (B, num_anchors)
        assert not fg_mask.any(), "All anchors should be negative with zero GT"
        assert (target_classes == -1).all()

    def test_fg_mask_shape(self):
        """fg_mask should have shape (B, num_anchors)."""
        assigner = TaskAlignedAssigner(topk=5)

        B, num_anchors, nc = 2, 200, 2
        pred_scores = torch.rand(B, num_anchors, nc).sigmoid()
        pred_bboxes = torch.rand(B, num_anchors, 4) * 640

        gt_labels = [torch.tensor([0, 1]), torch.tensor([0])]
        gt_bboxes = [
            torch.tensor([[320.0, 320.0, 100.0, 100.0], [160.0, 160.0, 50.0, 50.0]]),
            torch.tensor([[400.0, 400.0, 80.0, 80.0]]),
        ]

        anchors = torch.rand(num_anchors, 2) * 640
        strides = [8]

        _, _, _, fg_mask = assigner(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes, anchors, strides
        )

        assert fg_mask.shape == (B, num_anchors)
        assert fg_mask.dtype == torch.bool

    def test_target_shapes(self):
        """Output tensors should have correct shapes."""
        assigner = TaskAlignedAssigner(topk=5)

        B, num_anchors, nc = 1, 50, 2
        pred_scores = torch.rand(B, num_anchors, nc).sigmoid()
        pred_bboxes = torch.rand(B, num_anchors, 4) * 640

        gt_labels = [torch.tensor([0])]
        gt_bboxes = [torch.tensor([[320.0, 320.0, 100.0, 100.0]])]

        anchors = torch.rand(num_anchors, 2) * 640
        strides = [8]

        target_classes, target_bboxes, target_scores, fg_mask = assigner(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes, anchors, strides
        )

        assert target_classes.shape == (B, num_anchors)
        assert target_bboxes.shape == (B, num_anchors, 4)
        assert target_scores.shape == (B, num_anchors)
        assert fg_mask.shape == (B, num_anchors)

    def test_positive_assignment(self):
        """Anchors near GT should be assigned as positive."""
        assigner = TaskAlignedAssigner(topk=3, alpha=1.0, beta=6.0)

        B, nc = 1, 2
        # Create anchors where some are inside the GT box
        # GT box: cx=320, cy=320, w=200, h=200 → x:[220,420], y:[220,420]
        anchors = torch.tensor([
            [320.0, 320.0],  # inside GT
            [300.0, 300.0],  # inside GT
            [10.0, 10.0],    # outside GT
            [600.0, 600.0],  # outside GT
            [350.0, 350.0],  # inside GT
        ])
        num_anchors = len(anchors)

        # Predictions: high score for class 0 at anchors near GT
        pred_scores = torch.zeros(B, num_anchors, nc)
        pred_scores[0, 0, 0] = 0.9  # high score at anchor 0
        pred_scores[0, 1, 0] = 0.8  # high score at anchor 1
        pred_scores[0, 4, 0] = 0.7  # high score at anchor 4

        # Predicted bboxes close to GT
        pred_bboxes = torch.zeros(B, num_anchors, 4)
        pred_bboxes[0, 0] = torch.tensor([320.0, 320.0, 190.0, 190.0])
        pred_bboxes[0, 1] = torch.tensor([300.0, 300.0, 180.0, 180.0])
        pred_bboxes[0, 4] = torch.tensor([350.0, 350.0, 170.0, 170.0])

        gt_labels = [torch.tensor([0])]
        gt_bboxes = [torch.tensor([[320.0, 320.0, 200.0, 200.0]])]

        _, _, _, fg_mask = assigner(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes, anchors, [8]
        )

        # At least some anchors near GT should be positive
        assert fg_mask.sum() > 0, "Should have at least one positive anchor"


class TestGenerateAnchors:
    """Test anchor generation utility."""

    def test_anchor_count(self):
        """Total anchors should equal sum of H*W across levels."""
        feat_sizes = [(80, 80), (40, 40), (20, 20)]
        strides = [8, 16, 32]

        anchors, num_per_level = _generate_anchors(
            feat_sizes, strides, torch.device("cpu")
        )

        expected = 80 * 80 + 40 * 40 + 20 * 20  # 6400 + 1600 + 400 = 8400
        assert anchors.shape == (expected, 2)
        assert sum(num_per_level) == expected

    def test_anchor_spacing(self):
        """Anchors at stride 8 should be spaced 8 pixels apart."""
        feat_sizes = [(4, 4)]
        strides = [8]

        anchors, _ = _generate_anchors(feat_sizes, strides, torch.device("cpu"))

        # First anchor should be at (4, 4) — center of first cell
        assert anchors[0, 0] == pytest.approx(4.0)
        assert anchors[0, 1] == pytest.approx(4.0)
