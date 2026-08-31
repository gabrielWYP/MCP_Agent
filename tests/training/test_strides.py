"""Tests for `src/training/strides.py` (fusion-redesign D-D).

Covers the resolver, the config invariant that prevents the coarsest level
from silently starving of positive assignments, and `select_by_strides`
(the fix for the ProjectionLayers/KD silent-truncation hazard).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.config import TrainingConfig
from src.training.strides import (
    ALL_STRIDES,
    STRIDE_TO_LEVEL,
    STUDENT_STRIDES,
    resolve_active_strides,
    resolve_from_checkpoint,
    resolve_head_strides,
    select_by_strides,
    validate_strides,
)


class TestValidateStrides:
    def test_default_four_level_strides_are_valid(self):
        validate_strides([4, 8, 16, 32], [32, 64, 128])

    def test_three_level_strides_are_valid(self):
        validate_strides([8, 16, 32], [64, 128])

    def test_empty_strides_raise(self):
        with pytest.raises(ValueError, match="non-empty"):
            validate_strides([])

    def test_unsupported_stride_raises(self):
        with pytest.raises(ValueError, match="unsupported"):
            validate_strides([8, 16, 64])

    def test_duplicate_stride_raises(self):
        with pytest.raises(ValueError, match="duplicates"):
            validate_strides([8, 8, 16])

    def test_non_increasing_order_raises(self):
        with pytest.raises(ValueError, match="increasing"):
            validate_strides([16, 8, 32])

    def test_level_ranges_shorter_than_required_raises(self):
        """The one genuinely silent failure this whole module exists to
        prevent: assigner_level_ranges shorter than len(head_strides) - 1
        would leave the coarsest level with zero positive assignments."""
        with pytest.raises(ValueError, match="assigner_level_ranges"):
            validate_strides([4, 8, 16, 32], [64, 128])

    def test_level_ranges_longer_than_required_raises(self):
        with pytest.raises(ValueError):
            validate_strides([8, 16, 32], [32, 64, 128])


class TestResolveHeadStrides:
    def test_resolves_from_config(self):
        config = TrainingConfig(model_type="master", head_strides=[8, 16, 32], assigner_level_ranges=[64, 128])
        assert resolve_head_strides(config) == [8, 16, 32]

    def test_default_config_resolves_four_levels(self):
        config = TrainingConfig(model_type="master")
        assert resolve_head_strides(config) == [4, 8, 16, 32]


class TestResolveActiveStrides:
    def test_student_always_fixed_regardless_of_head_strides(self):
        config = TrainingConfig(model_type="student")
        assert resolve_active_strides(config) == list(STUDENT_STRIDES)

    def test_master_uses_configured_head_strides(self):
        config = TrainingConfig(model_type="master", head_strides=[8, 16, 32], assigner_level_ranges=[64, 128])
        assert resolve_active_strides(config) == [8, 16, 32]


class TestResolveFromCheckpoint:
    def test_reads_head_strides_from_checkpoint_config(self):
        checkpoint = {"config": {"head_strides": [4, 8, 16, 32]}}
        assert resolve_from_checkpoint(checkpoint) == [4, 8, 16, 32]

    def test_missing_config_raises_keyerror(self):
        with pytest.raises(KeyError):
            resolve_from_checkpoint({"model_state_dict": {}})

    def test_missing_head_strides_key_raises_keyerror(self):
        with pytest.raises(KeyError):
            resolve_from_checkpoint({"config": {"num_classes": 2}})


class TestConfigInvariantEnforced:
    """TrainingConfig.__post_init__ must enforce the invariant for master."""

    def test_mismatched_level_ranges_raises_at_construction(self):
        with pytest.raises(ValueError):
            TrainingConfig(model_type="master", head_strides=[4, 8, 16, 32], assigner_level_ranges=[64, 128])

    def test_student_is_not_subject_to_the_invariant(self):
        """Student always uses its own fixed 3-level strides regardless of
        head_strides/assigner_level_ranges (D-2: out of scope)."""
        TrainingConfig(model_type="student", head_strides=[4, 8, 16, 32], assigner_level_ranges=[64, 128])


class TestSelectByStrides:
    def test_selects_matching_levels_by_stride_not_position(self):
        features = ["P2", "P3", "P4", "P5"]
        source_strides = (4, 8, 16, 32)
        selected = select_by_strides(features, source_strides, (8, 16, 32))
        assert selected == ["P3", "P4", "P5"]

    def test_missing_target_stride_raises(self):
        features = ["P3", "P4", "P5"]
        with pytest.raises(ValueError, match="not present"):
            select_by_strides(features, (8, 16, 32), (4, 8, 16, 32))

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            select_by_strides(["a", "b"], (8, 16, 32), (8,))


def test_stride_to_level_mapping_matches_all_strides():
    assert STRIDE_TO_LEVEL == {stride: i for i, stride in enumerate(ALL_STRIDES)}
