"""Tests for the Task-Aligned Assigner fix (Phase 2 / W1).

Covers: sub-stride GT coverage (D6/A1), level binning (D8/A3), legacy
byte-for-byte reproduction at `center_radius=0.0` (D9), instrumentation
on/off parity (A4), and the fallback-after-filter guarantee (D7/A2).
"""

from __future__ import annotations

import torch

from src.training.loss import TaskAlignedAssigner, _generate_anchors


def _make_anchors(feat_sizes, strides):
    return _generate_anchors(feat_sizes, strides, torch.device("cpu"))


class TestSubStrideGT:
    def test_sub_stride_gt_gets_at_least_one_positive(self) -> None:
        """A 4px-wide GT at 640px is narrower than every stride (8/16/32)."""
        feat_sizes = [(80, 80), (40, 40), (20, 20)]
        strides = [8, 16, 32]
        anchors, anchor_strides, num_per_level = _make_anchors(feat_sizes, strides)
        num_anchors = anchors.shape[0]

        assigner = TaskAlignedAssigner(topk=13, center_radius=0.0)

        B, nc = 1, 2
        pred_scores = torch.rand(B, num_anchors, nc) * 0.01  # near-zero everywhere
        pred_bboxes = torch.rand(B, num_anchors, 4) * 640

        gt_labels = [torch.tensor([1])]
        # 4px-wide GT centered inside the image, narrower than any stride.
        gt_bboxes = [torch.tensor([[320.0, 320.0, 4.0, 4.0]])]

        _, _, _, fg_mask = assigner(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes, anchors, strides,
            anchor_strides=anchor_strides, num_per_level=num_per_level,
        )

        assert fg_mask.sum() >= 1, "Sub-stride GT must receive at least one positive anchor"

    def test_sub_stride_gt_not_dropped_by_spatial_filter(self) -> None:
        """The fallback guarantee must hold even when NO anchor center is
        literally inside a sub-stride GT (spatial filter fails for everyone)."""
        feat_sizes = [(10, 10)]
        strides = [32]  # anchor centers are 32px apart; GT below is 2px wide
        anchors, anchor_strides, num_per_level = _make_anchors(feat_sizes, strides)
        num_anchors = anchors.shape[0]

        assigner = TaskAlignedAssigner(topk=5, center_radius=0.0)

        pred_scores = torch.zeros(1, num_anchors, 1)
        pred_bboxes = torch.rand(1, num_anchors, 4) * 320

        gt_labels = [torch.tensor([0])]
        # Centered between anchor grid points so no anchor center falls inside.
        gt_bboxes = [torch.tensor([[100.0, 100.0, 2.0, 2.0]])]

        target_classes, _, _, fg_mask = assigner(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes, anchors, strides,
            anchor_strides=anchor_strides, num_per_level=num_per_level,
        )

        assert fg_mask.sum() >= 1
        assert (target_classes[fg_mask] == 0).all()


class TestLevelBinning:
    def test_small_gt_restricted_to_stride_8_only(self) -> None:
        """A 30px GT (< 64) must only receive positives at stride 8 (level 0)."""
        feat_sizes = [(80, 80), (40, 40), (20, 20)]
        strides = [8, 16, 32]
        anchors, anchor_strides, num_per_level = _make_anchors(feat_sizes, strides)
        num_anchors = anchors.shape[0]

        assigner = TaskAlignedAssigner(topk=13, center_radius=2.5, level_ranges=[64, 128])

        B, nc = 1, 1
        pred_scores = torch.full((B, num_anchors, nc), 0.5)
        pred_bboxes = anchors.unsqueeze(0).clone()
        pred_bboxes = torch.cat(
            [pred_bboxes, torch.full((B, num_anchors, 2), 30.0)], dim=-1
        )  # predicted boxes centered on every anchor, 30x30

        gt_labels = [torch.tensor([0])]
        gt_bboxes = [torch.tensor([[320.0, 320.0, 30.0, 30.0]])]

        level_ids = TaskAlignedAssigner._anchor_level_ids(
            num_per_level, num_anchors, torch.device("cpu")
        )

        _, _, _, fg_mask = assigner(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes, anchors, strides,
            anchor_strides=anchor_strides, num_per_level=num_per_level,
        )

        positive_levels = level_ids[fg_mask[0]]
        assert fg_mask.sum() >= 1
        assert torch.all(positive_levels == 0), "30px GT must only assign stride-8 anchors"

    def test_large_gt_restricted_to_stride_32_only(self) -> None:
        """A 200px GT (>= 128) must only receive positives at stride 32 (level 2)."""
        feat_sizes = [(80, 80), (40, 40), (20, 20)]
        strides = [8, 16, 32]
        anchors, anchor_strides, num_per_level = _make_anchors(feat_sizes, strides)
        num_anchors = anchors.shape[0]

        assigner = TaskAlignedAssigner(topk=13, center_radius=2.5, level_ranges=[64, 128])

        B, nc = 1, 1
        pred_scores = torch.full((B, num_anchors, nc), 0.5)
        pred_bboxes = anchors.unsqueeze(0).clone()
        pred_bboxes = torch.cat(
            [pred_bboxes, torch.full((B, num_anchors, 2), 200.0)], dim=-1
        )

        gt_labels = [torch.tensor([0])]
        gt_bboxes = [torch.tensor([[320.0, 320.0, 200.0, 200.0]])]

        level_ids = TaskAlignedAssigner._anchor_level_ids(
            num_per_level, num_anchors, torch.device("cpu")
        )

        _, _, _, fg_mask = assigner(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes, anchors, strides,
            anchor_strides=anchor_strides, num_per_level=num_per_level,
        )

        positive_levels = level_ids[fg_mask[0]]
        assert fg_mask.sum() >= 1
        assert torch.all(positive_levels == 2), "200px GT must only assign stride-32 anchors"


class TestLegacyReproduction:
    @staticmethod
    def _legacy_assign(pred_scores, pred_bboxes, gt_labels, gt_bboxes, anchors, strides, topk=13):
        """Reference copy of the pre-fix algorithm (post-topk containment
        filter, fallback ranked by predicted-box centers, fallback BEFORE the
        filter). Used only to prove the new implementation reproduces it
        byte-for-byte under representative conditions (task 2.6)."""
        B, num_anchors, num_classes = pred_scores.shape
        device = pred_scores.device
        target_classes = torch.full((B, num_anchors), -1, dtype=torch.long, device=device)
        target_bboxes = torch.zeros((B, num_anchors, 4), device=device)
        target_scores = torch.zeros((B, num_anchors), device=device)
        fg_mask = torch.zeros((B, num_anchors), dtype=torch.bool, device=device)

        from torchvision.ops import box_iou

        def cxcywh_to_xyxy(boxes):
            cx, cy, w, h = boxes.unbind(-1)
            return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

        for b in range(B):
            n_gt = len(gt_labels[b])
            if n_gt == 0:
                continue
            gt_cls = gt_labels[b]
            gt_box = gt_bboxes[b]
            gt_cls_scores = pred_scores[b, :, gt_cls]
            pred_xyxy = cxcywh_to_xyxy(pred_bboxes[b])
            gt_xyxy = cxcywh_to_xyxy(gt_box)
            ious = box_iou(pred_xyxy, gt_xyxy)
            alignment = (gt_cls_scores ** 1.0) * (ious ** 6.0)
            k = min(topk, num_anchors)
            topk_vals, topk_idx = alignment.topk(k, dim=0)

            for gt_i in range(n_gt):
                pos_idx = topk_idx[:, gt_i]
                pos_scores = topk_vals[:, gt_i]
                if pos_scores.max() <= 0:
                    gt_cx = gt_box[gt_i, 0].item()
                    gt_cy = gt_box[gt_i, 1].item()
                    anchor_cx = pred_bboxes[b, :, 0]
                    anchor_cy = pred_bboxes[b, :, 1]
                    dist = (anchor_cx - gt_cx) ** 2 + (anchor_cy - gt_cy) ** 2
                    _, fallback_idx = dist.topk(min(3, num_anchors), largest=False)
                    pos_idx = fallback_idx
                    pos_scores = torch.ones(len(fallback_idx), device=device) * 0.01
                else:
                    valid = pos_scores > 0
                    pos_idx = pos_idx[valid]
                    pos_scores = pos_scores[valid]
                if len(pos_idx) == 0:
                    continue
                anchor_centers = anchors[pos_idx]
                gt_cxcy = gt_box[gt_i, :2]
                gt_wh = gt_box[gt_i, 2:]
                gt_min = gt_cxcy - gt_wh / 2
                gt_max = gt_cxcy + gt_wh / 2
                inside = (
                    (anchor_centers[:, 0] >= gt_min[0]) & (anchor_centers[:, 0] <= gt_max[0]) &
                    (anchor_centers[:, 1] >= gt_min[1]) & (anchor_centers[:, 1] <= gt_max[1])
                )
                pos_idx = pos_idx[inside]
                pos_scores = pos_scores[inside]
                if len(pos_idx) == 0:
                    continue
                for idx, score in zip(pos_idx, pos_scores):
                    if target_scores[b, idx] < score or target_classes[b, idx] == -1:
                        target_classes[b, idx] = gt_cls[gt_i]
                        target_bboxes[b, idx] = gt_box[gt_i]
                        target_scores[b, idx] = score
                        fg_mask[b, idx] = True
        return target_classes, target_bboxes, target_scores, fg_mask

    def test_center_radius_zero_matches_legacy_on_representative_input(self) -> None:
        """center_radius=0.0 (D9 default) reproduces the pre-fix assignment
        byte-for-byte.

        Masking-before-topk (new) and topk-then-filtering (old/legacy) are
        only guaranteed to coincide when every anchor OUTSIDE the GT box also
        has exactly zero alignment (e.g. its predicted box does not overlap
        the GT at all) — otherwise an outside anchor could rank inside the
        old code's raw top-k and get filtered out afterward, while a
        lower-ranked inside anchor that never made the old top-k would be
        picked up by the new code's masked top-k instead, a genuine (and
        deliberate, D6-motivated) behavior difference between the two
        algorithms. This test constructs exactly that representative,
        non-degenerate condition: anchors inside the GT get predicted boxes
        that overlap it (nonzero alignment); anchors outside get predicted
        boxes placed far away with zero overlap (alignment forced to exactly
        zero in both algorithms alike).
        """
        anchors = torch.tensor([
            [320.0, 320.0],  # inside GT (below)
            [340.0, 310.0],  # inside GT
            [10.0, 10.0],    # outside GT
            [600.0, 600.0],  # outside GT
        ])
        strides = [8]

        gt_labels = [torch.tensor([0])]
        gt_bboxes = [torch.tensor([[320.0, 320.0, 100.0, 100.0]])]  # x:[270,370] y:[270,370]

        pred_scores = torch.tensor([[[0.9], [0.8], [0.7], [0.6]]])
        pred_bboxes = torch.tensor([[
            [320.0, 320.0, 90.0, 90.0],   # overlaps GT heavily
            [340.0, 310.0, 80.0, 80.0],   # overlaps GT
            [10.0, 10.0, 5.0, 5.0],       # far away, zero overlap with GT
            [600.0, 600.0, 5.0, 5.0],     # far away, zero overlap with GT
        ]])

        legacy = self._legacy_assign(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes, anchors, strides, topk=2,
        )

        assigner = TaskAlignedAssigner(topk=2, center_radius=0.0)
        new = assigner(pred_scores, pred_bboxes, gt_labels, gt_bboxes, anchors, strides)

        for legacy_t, new_t in zip(legacy, new):
            assert torch.equal(legacy_t, new_t)

        # And it is a non-trivial check: the two inside anchors are positive.
        assert new[3][0, 0] and new[3][0, 1]
        assert not new[3][0, 2] and not new[3][0, 3]


class TestInstrumentation:
    def test_instrumentation_disabled_by_default(self) -> None:
        assigner = TaskAlignedAssigner(topk=5)
        assert assigner.collect_stats is False
        assert assigner.last_stats == {}

    def test_instrumentation_does_not_alter_assignment(self) -> None:
        feat_sizes = [(20, 20), (10, 10)]
        strides = [8, 16]
        anchors, anchor_strides, num_per_level = _make_anchors(feat_sizes, strides)
        num_anchors = anchors.shape[0]

        torch.manual_seed(1)
        pred_scores = torch.rand(1, num_anchors, 2)
        pred_bboxes = torch.rand(1, num_anchors, 4) * 320
        gt_labels = [torch.tensor([0, 1])]
        gt_bboxes = [torch.tensor([[100.0, 100.0, 50.0, 50.0], [200.0, 200.0, 20.0, 20.0]])]

        off = TaskAlignedAssigner(topk=10, center_radius=2.5)
        on = TaskAlignedAssigner(topk=10, center_radius=2.5, collect_stats=True)

        result_off = off(pred_scores, pred_bboxes, gt_labels, gt_bboxes, anchors, strides,
                          anchor_strides=anchor_strides, num_per_level=num_per_level)
        result_on = on(pred_scores, pred_bboxes, gt_labels, gt_bboxes, anchors, strides,
                        anchor_strides=anchor_strides, num_per_level=num_per_level)

        for t_off, t_on in zip(result_off, result_on):
            assert torch.equal(t_off, t_on)

        assert on.last_stats != {}, "collect_stats=True must record per-class/per-level counts"

    def test_reset_stats_clears_accumulation(self) -> None:
        assigner = TaskAlignedAssigner(topk=5, collect_stats=True)
        assigner.last_stats = {(0, 0): 3}
        assigner.reset_stats()
        assert assigner.last_stats == {}


class TestFallbackOrdering:
    def test_fallback_never_empties_a_gt(self) -> None:
        """A GT whose only top-k candidates fail spatial admissibility must
        still end with len(pos_idx) >= 1 via the post-filter fallback (D7)."""
        anchors = torch.tensor([[0.0, 0.0], [640.0, 640.0]])
        strides = [8]
        anchor_strides = torch.tensor([8.0, 8.0])

        assigner = TaskAlignedAssigner(topk=2, center_radius=0.0)

        pred_scores = torch.tensor([[[0.9], [0.9]]])
        pred_bboxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [640.0, 640.0, 10.0, 10.0]]])

        gt_labels = [torch.tensor([0])]
        # GT is far from both anchors — neither is inside the box.
        gt_bboxes = [torch.tensor([[320.0, 320.0, 10.0, 10.0]])]

        _, _, _, fg_mask = assigner(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes, anchors, strides,
            anchor_strides=anchor_strides,
        )

        assert fg_mask.sum() >= 1
