"""
Shape and distillation output tests for DecoupledHead.

Verifies:
- preds[i] shape (B, 6, H, W) — cls + reg concatenated
- cls_preds[i] shape (B, 2, H, W)
- reg_preds[i] shape (B, 4, H, W)
- distill_cls channels [64, 128, 256] matching head_projections
- distill_reg channels [64, 128, 256] matching head_projections
- head_projections compatibility

Run with:
    python -m pytest tests/models/student/test_head.py -v
Or as script:
    python -m tests.models.student.test_head
"""

import torch

from src.models.student.backbone import CSPDarknetNano
from src.models.student.neck import PANet
from src.models.student.head import YOLOStudentHead
from src.models.master.distill_projections import head_projections

BATCH_SIZE = 2
INPUT_H, INPUT_W = 640, 640
NUM_CLASSES = 2
EXPECTED_STEM_CHANNELS = [64, 128, 256]
EXPECTED_SPATIAL = [80, 40, 20]


def _get_head_output():
    """Helper: run backbone -> neck -> head and return head output dict."""
    backbone = CSPDarknetNano()
    neck = PANet()
    head = YOLOStudentHead()
    backbone.eval()
    neck.eval()
    head.eval()

    x = torch.randn(BATCH_SIZE, 3, INPUT_H, INPUT_W)
    with torch.no_grad():
        stages, _ = backbone(x)
        fpn = neck(stages)
        out = head(fpn)
    return out


def test_head_prediction_shapes():
    """preds[i]=(B,6,H,W), cls=(B,2,H,W), reg=(B,4,H,W)."""
    out = _get_head_output()

    assert len(out["preds"]) == 3, f"Expected 3 levels, got {len(out['preds'])}"

    for i, spatial in enumerate(EXPECTED_SPATIAL):
        pred = out["preds"][i]
        expected_pred_shape = (BATCH_SIZE, NUM_CLASSES + 4, spatial, spatial)
        assert pred.shape == expected_pred_shape, (
            f"preds[{i}]: expected {expected_pred_shape}, got {tuple(pred.shape)}"
        )

        cls = out["cls_preds"][i]
        expected_cls_shape = (BATCH_SIZE, NUM_CLASSES, spatial, spatial)
        assert cls.shape == expected_cls_shape, (
            f"cls_preds[{i}]: expected {expected_cls_shape}, got {tuple(cls.shape)}"
        )

        reg = out["reg_preds"][i]
        expected_reg_shape = (BATCH_SIZE, 4, spatial, spatial)
        assert reg.shape == expected_reg_shape, (
            f"reg_preds[{i}]: expected {expected_reg_shape}, got {tuple(reg.shape)}"
        )

    print("  Prediction shapes:", [tuple(p.shape) for p in out["preds"]])


def test_head_distill_stem_channels():
    """distill_cls and distill_reg at [64,128,256]ch matching head_projections."""
    out = _get_head_output()

    for i, (exp_ch, spatial) in enumerate(
        zip(EXPECTED_STEM_CHANNELS, EXPECTED_SPATIAL)
    ):
        dc = out["distill_head_cls"][i]
        expected_shape = (BATCH_SIZE, exp_ch, spatial, spatial)
        assert dc.shape == expected_shape, (
            f"distill_head_cls[{i}]: expected {expected_shape}, "
            f"got {tuple(dc.shape)}"
        )

        dr = out["distill_head_reg"][i]
        assert dr.shape == expected_shape, (
            f"distill_head_reg[{i}]: expected {expected_shape}, "
            f"got {tuple(dr.shape)}"
        )

    print(
        "  Distill cls channels:",
        [t.shape[1] for t in out["distill_head_cls"]],
    )
    print(
        "  Distill reg channels:",
        [t.shape[1] for t in out["distill_head_reg"]],
    )


def test_head_projection_compatibility():
    """Student head distill outputs compatible with head_projections."""
    out = _get_head_output()

    expected = [64, 128, 256]
    for i, exp_ch in enumerate(expected):
        assert out["distill_head_cls"][i].shape[1] == exp_ch, (
            f"distill_head_cls[{i}]: {out['distill_head_cls'][i].shape[1]}ch "
            f"!= head_projections student_ch {exp_ch}"
        )
        assert out["distill_head_reg"][i].shape[1] == exp_ch, (
            f"distill_head_reg[{i}]: {out['distill_head_reg'][i].shape[1]}ch "
            f"!= head_projections student_ch {exp_ch}"
        )

    # Verify ProjectionLayers can be instantiated with matching dims
    proj = head_projections()
    assert proj is not None

    print("  Head projection compatibility: OK")


if __name__ == "__main__":
    print("Testing DecoupledHead...")

    print("  test_head_prediction_shapes...")
    test_head_prediction_shapes()
    print("  PASSED")

    print("  test_head_distill_stem_channels...")
    test_head_distill_stem_channels()
    print("  PASSED")

    print("  test_head_projection_compatibility...")
    test_head_projection_compatibility()
    print("  PASSED")

    print("\nALL HEAD TESTS PASSED")
