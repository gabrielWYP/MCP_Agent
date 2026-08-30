"""Tests for the shared decode module (src/training/decode.py).

Covers per-class NMS (D4/E1), per-class candidate emission (D5/E3), the
legacy-threshold default, and 0/1/2-candidate emission edge cases.
"""

from __future__ import annotations

import pytest
import torch

from src.training.decode import decode_detections


def _single_level_inputs(cls_logits: torch.Tensor, reg: torch.Tensor) -> tuple[list, list]:
    """Wrap a single (nc, H, W) / (4, H, W) pair into the multi-level list format.

    Batch dim is added; only stride-8, one level is used for these unit tests.
    """
    nc, H, W = cls_logits.shape
    pred = torch.zeros((1, nc + 4, H, W))
    pred[0, nc:] = reg
    cls_pred = cls_logits.unsqueeze(0)
    return [pred], [cls_pred]


def _logit(p: float) -> float:
    """Inverse sigmoid — build a logit that sigmoids back to `p`."""
    import math
    return math.log(p / (1 - p))


class TestNMS:
    def test_same_class_duplicates_collapse(self) -> None:
        """Three overlapping same-class anchors -> NMS keeps only 1."""
        H, W = 1, 3
        cls_logits = torch.full((1, H, W), _logit(0.01))
        # Three adjacent anchors all predict class 0 with high confidence and
        # near-identical decoded boxes (small deltas, same size).
        cls_logits[0, 0, 0] = _logit(0.9)
        cls_logits[0, 0, 1] = _logit(0.8)
        cls_logits[0, 0, 2] = _logit(0.7)

        reg = torch.zeros((4, H, W))
        # All three anchors decode to large, heavily overlapping boxes so every
        # pair clears the 0.5 IoU threshold regardless of NMS processing order.
        for x in range(W):
            reg[2, 0, x] = torch.log(torch.tensor(100.0))
            reg[3, 0, x] = torch.log(torch.tensor(100.0))

        preds, cls_preds = _single_level_inputs(cls_logits, reg)

        boxes, scores, labels = decode_detections(
            preds, cls_preds, batch_idx=0, num_classes=1, image_size=64,
            strides=(8,), conf_threshold=0.25, nms_iou_threshold=0.5,
            nms_enabled=True, per_class_candidates=True, normalize=False,
        )

        assert boxes.shape[0] == 1
        assert scores.item() == pytest.approx(0.9, abs=1e-4)

    def test_cross_class_overlap_both_survive(self) -> None:
        """A damage box fully inside a mango box: NMS must not cross-suppress."""
        H, W = 1, 1
        cls_logits = torch.tensor([[[_logit(0.9)]], [[_logit(0.8)]]])  # (nc=2, 1, 1)
        reg = torch.zeros((4, H, W))
        reg[2, 0, 0] = torch.log(torch.tensor(40.0))
        reg[3, 0, 0] = torch.log(torch.tensor(40.0))

        preds, cls_preds = _single_level_inputs(cls_logits, reg)

        boxes, scores, labels = decode_detections(
            preds, cls_preds, batch_idx=0, num_classes=2, image_size=64,
            strides=(8,), conf_threshold=0.25, nms_iou_threshold=0.5,
            nms_enabled=True, per_class_candidates=True, normalize=False,
        )

        # Same anchor, same box geometry, but two different classes both above
        # threshold — both must survive per-class NMS.
        assert boxes.shape[0] == 2
        assert set(labels.tolist()) == {0, 1}


class TestPerClassCandidateEmission:
    def test_anchor_emits_two_candidates_when_both_qualify(self) -> None:
        H, W = 1, 1
        cls_logits = torch.tensor([[[_logit(0.6)]], [[_logit(0.3)]]])  # mango=0.6, damage=0.3
        reg = torch.zeros((4, H, W))
        preds, cls_preds = _single_level_inputs(cls_logits, reg)

        boxes, scores, labels = decode_detections(
            preds, cls_preds, batch_idx=0, num_classes=2, image_size=64,
            strides=(8,), conf_threshold=0.25, nms_enabled=False,
            per_class_candidates=True, normalize=False,
        )

        assert boxes.shape[0] == 2
        assert set(labels.tolist()) == {0, 1}

    def test_anchor_emits_one_candidate_when_only_one_qualifies(self) -> None:
        H, W = 1, 1
        cls_logits = torch.tensor([[[_logit(0.6)]], [[_logit(0.1)]]])  # only mango qualifies
        reg = torch.zeros((4, H, W))
        preds, cls_preds = _single_level_inputs(cls_logits, reg)

        boxes, scores, labels = decode_detections(
            preds, cls_preds, batch_idx=0, num_classes=2, image_size=64,
            strides=(8,), conf_threshold=0.25, nms_enabled=False,
            per_class_candidates=True, normalize=False,
        )

        assert boxes.shape[0] == 1
        assert labels.item() == 0

    def test_anchor_emits_zero_candidates_when_none_qualify(self) -> None:
        H, W = 1, 1
        cls_logits = torch.tensor([[[_logit(0.1)]], [[_logit(0.05)]]])
        reg = torch.zeros((4, H, W))
        preds, cls_preds = _single_level_inputs(cls_logits, reg)

        boxes, scores, labels = decode_detections(
            preds, cls_preds, batch_idx=0, num_classes=2, image_size=64,
            strides=(8,), conf_threshold=0.25, nms_enabled=False,
            per_class_candidates=True, normalize=False,
        )

        assert boxes.shape[0] == 0


class TestLegacyDefaultPreserved:
    def test_argmax_only_matches_legacy_threshold(self) -> None:
        """per_class_candidates=False, nms_enabled=False reproduces the pre-fix decode."""
        H, W = 1, 1
        cls_logits = torch.tensor([[[_logit(0.6)]], [[_logit(0.3)]]])  # both above 0.25
        reg = torch.zeros((4, H, W))
        preds, cls_preds = _single_level_inputs(cls_logits, reg)

        boxes, scores, labels = decode_detections(
            preds, cls_preds, batch_idx=0, num_classes=2, image_size=64,
            strides=(8,), conf_threshold=0.25, nms_enabled=False,
            per_class_candidates=False, normalize=False,
        )

        # Legacy argmax-only decode: only the highest-scoring class (mango=0) emitted.
        assert boxes.shape[0] == 1
        assert labels.item() == 0
        assert scores.item() == pytest.approx(0.6, abs=1e-4)

    def test_default_conf_threshold_is_quarter(self) -> None:
        H, W = 1, 1
        cls_logits = torch.tensor([[[_logit(0.24)]]])
        reg = torch.zeros((4, H, W))
        preds, cls_preds = _single_level_inputs(cls_logits, reg)

        boxes, _, _ = decode_detections(
            preds, cls_preds, batch_idx=0, num_classes=1, image_size=64, strides=(8,),
        )
        assert boxes.shape[0] == 0  # 0.24 < default 0.25 threshold


class TestNormalization:
    def test_normalize_divides_by_image_size(self) -> None:
        H, W = 1, 1
        cls_logits = torch.tensor([[[_logit(0.9)]]])
        reg = torch.zeros((4, H, W))
        reg[2, 0, 0] = torch.log(torch.tensor(32.0))
        reg[3, 0, 0] = torch.log(torch.tensor(32.0))
        preds, cls_preds = _single_level_inputs(cls_logits, reg)

        boxes, _, _ = decode_detections(
            preds, cls_preds, batch_idx=0, num_classes=1, image_size=64,
            strides=(8,), nms_enabled=False, normalize=True,
        )
        assert torch.all(boxes >= 0) and torch.all(boxes <= 1)


class TestDecodeParity:
    def test_trainer_and_visualizer_wrappers_agree(self) -> None:
        """Trainer._decode_predictions and the visualizer must produce identical output
        on identical raw tensors and identical TrainingConfig thresholds."""
        from src.training.config import TrainingConfig
        from src.training.loop import Trainer
        from scripts.visualize_damage_predictions import decode_predictions as viz_decode

        H, W = 2, 2
        torch.manual_seed(0)
        preds = [torch.randn(1, 2 + 4, H, W) for _ in range(3)]
        cls_preds = [p[:, :2] for p in preds]

        config = TrainingConfig(model_type="student", num_classes=2, image_size=64)

        # Trainer decode normalizes to [0, 1]; the visualizer decodes in
        # letterbox pixel space (it draws directly on the image). Both call
        # the same underlying decode_detections — scaling one to the other's
        # coordinate space is the correct parity check (same suppression and
        # thresholding, different output convention by design).
        t_boxes, t_scores, t_labels = Trainer._decode_predictions_static(
            preds, cls_preds, batch_idx=0, config=config,
        )

        output = {"preds": preds, "cls_preds": cls_preds}
        viz_boxes, viz_scores, viz_labels = viz_decode(
            output,
            num_classes=2,
            image_size=config.image_size,
            conf_threshold=config.conf_threshold,
            nms_iou_threshold=config.nms_iou_threshold,
            nms_enabled=config.nms_enabled,
            per_class_candidates=config.decode_per_class,
            max_detections=config.max_detections,
        )

        assert torch.allclose(
            t_boxes * config.image_size, torch.as_tensor(viz_boxes), atol=1e-4
        )
        assert torch.allclose(t_scores, torch.as_tensor(viz_scores), atol=1e-5)
        assert torch.equal(t_labels, torch.as_tensor(viz_labels))
