"""Tests for `ProjectionLayers` length guard (fusion-redesign W7/§5).

design.md §5 flagged `ProjectionLayers.forward`'s `zip(self.projections,
teacher_features)` as a silent-truncation hazard once the teacher's
pyramid/head level count became configurable (default 4 levels) against
this class's fixed 3-level presets. As verified in the current codebase,
`forward` already asserted `len(teacher_features) == self.num_levels`
before the `zip` — so it was already loud, not silent. This file's tests
lock that in with an explicit message and cover both the pass and fail
paths, and `test_kd_trainer_guards.py`-adjacent slicing (`select_by_strides`)
is what makes the *correct* (non-crashing) path work — see
`tests/training/test_strides.py::TestSelectByStrides`.
"""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.models.master.distill_projections import ProjectionLayers, fpn_projections


class TestLengthGuard:
    def test_matching_length_succeeds(self):
        proj = ProjectionLayers(teacher_channels=[256, 256, 256], student_channels=[128, 256, 256])
        features = [torch.randn(1, 256, 8, 8) for _ in range(3)]
        out = proj(features)
        assert len(out) == 3

    def test_four_level_teacher_against_three_level_preset_raises(self):
        """The exact hazard: a 4-level teacher pyramid (with P2 reconnected,
        fusion-redesign D-3) passed directly into the 3-level fpn_projections
        preset must raise, not silently distill P2/P3/P4 into student
        P3/P4/P5 via zip-truncation."""
        proj = fpn_projections()
        four_level_features = [torch.randn(1, 256, s, s) for s in (160, 80, 40, 20)]
        with pytest.raises(AssertionError, match="expected 3"):
            proj(four_level_features)

    def test_too_few_levels_raises(self):
        proj = ProjectionLayers(teacher_channels=[256, 256, 256], student_channels=[128, 256, 256])
        with pytest.raises(AssertionError):
            proj([torch.randn(1, 256, 8, 8), torch.randn(1, 256, 4, 4)])
