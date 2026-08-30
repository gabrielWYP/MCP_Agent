"""train.py's --override list coercion (§5 of design.md, item 7).

Verified defect: `train.py:88-99` coerced bool/int/float but not list, so
`--override head_strides=[4,8,16,32]` assigned the *string*
`"[4,8,16,32]"` — `len()` of that string is 12, silently building a
12-level pyramid instead of failing at the flag. Ported
`evaluate_checkpoint.py:103-106`'s list branch verbatim so the two
entrypoints cannot drift.

`head_strides` does not exist as a config field until PR2 (fusion-redesign
Phase 6+), so these tests exercise the same generic, field-name-agnostic
coercion path (keyed only on `isinstance(current, list)`) against existing
list-typed `TrainingConfig` fields (`class_weights`, `assigner_level_ranges`).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.config import TrainingConfig


def _apply_overrides(config: TrainingConfig, overrides: list[str]) -> None:
    """Mirror of train.py's override-application loop, isolated for testing
    without invoking argparse/full main()."""
    import json

    for override in overrides:
        key, value = override.split("=", 1)
        if hasattr(config, key):
            current = getattr(config, key)
            if isinstance(current, bool):
                value = value.lower() in ("true", "1", "yes")
            elif isinstance(current, int):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)
            elif isinstance(current, list):
                value = json.loads(value) if value.strip().startswith("[") else [
                    float(v) for v in value.split(",")
                ]
            setattr(config, key, value)


class TestListOverrideCoercion:
    """4.1: a bracketed list override yields a real list, not the literal string."""

    def test_bracket_syntax_yields_list_of_ints(self):
        config = TrainingConfig()
        _apply_overrides(config, ["class_weights=[4,8,16,32]"])

        assert config.class_weights == [4, 8, 16, 32]
        assert isinstance(config.class_weights, list)
        assert all(isinstance(v, int) for v in config.class_weights)

    def test_bracket_syntax_does_not_assign_the_literal_string(self):
        config = TrainingConfig()
        _apply_overrides(config, ["class_weights=[4,8,16,32]"])

        # The historical bug: the literal string "[4,8,16,32]" has len()==12.
        assert config.class_weights != "[4,8,16,32]"
        assert len(config.class_weights) == 4

    def test_bracket_syntax_preserves_floats(self):
        config = TrainingConfig()
        _apply_overrides(config, ["class_weights=[0.5, 1.5]"])

        assert config.class_weights == [0.5, 1.5]

    def test_comma_syntax_falls_back_to_floats(self):
        config = TrainingConfig()
        _apply_overrides(config, ["assigner_level_ranges=32,64,128"])

        assert config.assigner_level_ranges == [32.0, 64.0, 128.0]
        assert all(isinstance(v, float) for v in config.assigner_level_ranges)

    def test_scalar_overrides_still_coerce_correctly(self):
        """Non-list fields must be unaffected by the new branch."""
        config = TrainingConfig()
        _apply_overrides(
            config,
            ["batch_size=4", "lr=0.01", "precision=fp16"],
        )

        assert config.batch_size == 4
        assert isinstance(config.batch_size, int)
        assert config.lr == pytest.approx(0.01)
        assert config.precision == "fp16"

    def test_unknown_override_key_is_ignored(self):
        config = TrainingConfig()
        _apply_overrides(config, ["nonexistent_field=[1,2,3]"])
        assert not hasattr(config, "nonexistent_field")
