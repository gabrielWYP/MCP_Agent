"""D-H: machine profile / experiment split, enforced rather than documented.

See openspec/changes/fusion-redesign/design.md D-H.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.config import TrainingConfig
from src.training.machine import (
    MACHINE_PROFILE_WHITELIST,
    apply_machine_profile,
    derive_grad_accum_steps,
    experiment_sha256,
    load_machine_profile,
    validate_machine_profile,
)


class TestMachineProfileWhitelist:
    """3.1: a machine profile setting a non-whitelisted key raises."""

    def test_whitelisted_keys_pass(self):
        validate_machine_profile(
            {"device": "cuda", "batch_size": 2, "num_workers": 2, "pin_memory": True, "precision": "fp32"}
        )  # must not raise

    def test_non_whitelisted_key_raises(self):
        with pytest.raises(ValueError, match="non-whitelisted"):
            validate_machine_profile({"head_strides": [4, 8, 16, 32]})

    def test_non_whitelisted_key_mixed_with_valid_keys_raises(self):
        with pytest.raises(ValueError, match="non-whitelisted"):
            validate_machine_profile({"device": "cuda", "epochs": 100})

    def test_load_machine_profile_raises_on_bad_key(self, tmp_path):
        bad_profile = tmp_path / "bad.yaml"
        bad_profile.write_text("device: cuda\nlr: 0.001\n")

        with pytest.raises(ValueError, match="non-whitelisted"):
            load_machine_profile(bad_profile)

    def test_load_machine_profile_accepts_whitelisted_keys(self, tmp_path):
        profile = tmp_path / "good.yaml"
        profile.write_text("device: cuda\nbatch_size: 4\nprecision: fp16\n")

        data = load_machine_profile(profile)
        assert data == {"device": "cuda", "batch_size": 4, "precision": "fp16"}

    def test_apply_machine_profile_sets_config_fields(self):
        config = TrainingConfig()
        apply_machine_profile(config, {"batch_size": 4, "device": "cpu"})
        assert config.batch_size == 4
        assert config.device == "cpu"

    def test_apply_machine_profile_rejects_bad_key(self):
        config = TrainingConfig()
        with pytest.raises(ValueError, match="non-whitelisted"):
            apply_machine_profile(config, {"lr": 0.5})

    def test_whitelist_is_exactly_five_keys(self):
        assert MACHINE_PROFILE_WHITELIST == {
            "device", "batch_size", "num_workers", "pin_memory", "precision"
        }


class TestGradAccumSteps:
    """3.2: effective_batch % batch_size != 0 raises; grad_accum_steps is
    derived, never read from a file."""

    def test_derives_evenly(self):
        assert derive_grad_accum_steps(effective_batch=8, batch_size=2) == 4
        assert derive_grad_accum_steps(effective_batch=8, batch_size=4) == 2
        assert derive_grad_accum_steps(effective_batch=8, batch_size=8) == 1

    def test_raises_on_non_divisible_pair(self):
        with pytest.raises(ValueError, match="not evenly divisible"):
            derive_grad_accum_steps(effective_batch=8, batch_size=3)

    def test_config_construction_raises_on_non_divisible_pair(self):
        with pytest.raises(ValueError, match="not evenly divisible"):
            TrainingConfig(effective_batch=8, batch_size=3)

    def test_config_default_is_evenly_divisible(self):
        # Defaults (effective_batch=8, batch_size=2) must not raise.
        config = TrainingConfig()
        assert derive_grad_accum_steps(config.effective_batch, config.batch_size) == 4

    def test_grad_accum_steps_is_not_a_settable_config_field(self):
        with pytest.raises(TypeError):
            TrainingConfig(grad_accum_steps=4)


class TestExperimentSha256:
    """3.3: experiment_sha256 is stable across machine profiles, changes on
    a one-byte experiment-file edit."""

    def test_stable_across_machine_profiles(self, tmp_path):
        experiment_path = tmp_path / "fusion.yaml"
        experiment_path.write_text("head_strides: [4, 8, 16, 32]\nepochs: 100\n")

        hash_with_profile_a = experiment_sha256(experiment_path)
        # Applying a different machine profile does not touch the
        # experiment file bytes at all.
        hash_with_profile_b = experiment_sha256(experiment_path)

        assert hash_with_profile_a == hash_with_profile_b

    def test_changes_on_one_byte_edit(self, tmp_path):
        experiment_path = tmp_path / "fusion.yaml"
        experiment_path.write_text("epochs: 100\n")
        original_hash = experiment_sha256(experiment_path)

        experiment_path.write_text("epochs: 101\n")
        edited_hash = experiment_sha256(experiment_path)

        assert original_hash != edited_hash

    def test_is_a_valid_sha256_hex_digest(self, tmp_path):
        experiment_path = tmp_path / "fusion.yaml"
        experiment_path.write_text("epochs: 100\n")

        digest = experiment_sha256(experiment_path)
        assert len(digest) == 64
        int(digest, 16)  # must be valid hex
