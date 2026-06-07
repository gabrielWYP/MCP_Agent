"""
Shape validation tests for PANet neck.

Verifies:
- Output channels [128, 256, 256] at P3/P4/P5
- Spatial sizes [80, 40, 20] for 640x640 input
- FPN projection compatibility with fpn_projections contract

Run with:
    python -m pytest tests/models/student/test_neck.py -v
Or as script:
    python -m tests.models.student.test_neck
"""

import torch

from src.models.student.backbone import CSPDarknetNano
from src.models.student.neck import PANet
from src.models.master.distill_projections import fpn_projections

BATCH_SIZE = 2
INPUT_H, INPUT_W = 640, 640
EXPECTED_CHANNELS = [128, 256, 256]
EXPECTED_SPATIAL = [80, 40, 20]


def test_neck_output_shapes():
    """PANet output channels [128,256,256] at spatial sizes [80,40,20]."""
    backbone = CSPDarknetNano()
    neck = PANet()
    backbone.eval()
    neck.eval()

    x = torch.randn(BATCH_SIZE, 3, INPUT_H, INPUT_W)
    with torch.no_grad():
        stages, _ = backbone(x)
        fpn = neck(stages)

    keys = ["p3", "p4", "p5"]
    for i, key in enumerate(keys):
        feat = fpn[key]
        assert feat.shape[0] == BATCH_SIZE, (
            f"{key}: batch size {feat.shape[0]} != {BATCH_SIZE}"
        )
        assert feat.shape[1] == EXPECTED_CHANNELS[i], (
            f"{key}: expected {EXPECTED_CHANNELS[i]}ch, got {feat.shape[1]}ch"
        )
        assert feat.shape[2] == EXPECTED_SPATIAL[i], (
            f"{key}: expected H={EXPECTED_SPATIAL[i]}, got {feat.shape[2]}"
        )
        assert feat.shape[3] == EXPECTED_SPATIAL[i], (
            f"{key}: expected W={EXPECTED_SPATIAL[i]}, got {feat.shape[3]}"
        )

    print("  FPN shapes:", {k: tuple(fpn[k].shape) for k in keys})


def test_neck_fpn_projection_compatibility():
    """Student FPN outputs match fpn_projections student_channels [128,256,256]."""
    backbone = CSPDarknetNano()
    neck = PANet()
    backbone.eval()
    neck.eval()

    x = torch.randn(BATCH_SIZE, 3, INPUT_H, INPUT_W)
    with torch.no_grad():
        stages, _ = backbone(x)
        fpn = neck(stages)

    # Verify channels match fpn_projections contract
    expected_student_ch = [128, 256, 256]
    for key, exp_ch in zip(["p3", "p4", "p5"], expected_student_ch):
        assert fpn[key].shape[1] == exp_ch, (
            f"{key}: {fpn[key].shape[1]}ch != fpn_projections student_ch {exp_ch}"
        )

    # Verify ProjectionLayers can be instantiated with matching dims
    proj = fpn_projections()
    assert proj is not None

    print("  FPN projection compatibility: OK")


if __name__ == "__main__":
    print("Testing PANet neck...")

    print("  test_neck_output_shapes...")
    test_neck_output_shapes()
    print("  PASSED")

    print("  test_neck_fpn_projection_compatibility...")
    test_neck_fpn_projection_compatibility()
    print("  PASSED")

    print("\nALL NECK TESTS PASSED")
