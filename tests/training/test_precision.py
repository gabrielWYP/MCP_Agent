"""W9a: precision is an explicit fp32|fp16|bf16 enum, not a silently-defaulting bool.

Verified defect (openspec/changes/fusion-redesign/design.md D-I):
`TrainingConfig.amp: bool = True` meant any config omitting the key silently
enabled fp16. `amp` is removed, not deprecated.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.config import TrainingConfig
from src.training.precision import (
    autocast_ctx,
    make_scaler,
    validate_bf16_support,
    validate_precision,
)


class TestPrecisionField:
    """2.1: precision accepts only fp32|fp16|bf16, default fp32, amp is gone."""

    def test_default_is_fp32(self):
        config = TrainingConfig()
        assert config.precision == "fp32"

    @pytest.mark.parametrize("value", ["fp32", "fp16", "bf16"])
    def test_accepts_valid_values(self, value, monkeypatch):
        # bf16 needs CUDA support validation bypassed on a CPU-only test box.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        config = TrainingConfig(precision=value)
        assert config.precision == value

    def test_rejects_invalid_value(self):
        with pytest.raises(ValueError, match="Invalid precision"):
            TrainingConfig(precision="fp64")

    def test_amp_field_is_gone(self):
        config = TrainingConfig()
        assert not hasattr(config, "amp")

    def test_amp_kwarg_rejected_at_construction(self):
        with pytest.raises(TypeError):
            TrainingConfig(amp=True)


class TestBf16DeviceGate:
    """2.2: bf16 on a monkeypatched non-bf16 device raises and names the card."""

    def test_bf16_raises_on_unsupported_device(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "GTX 1660 SUPER")

        with pytest.raises(ValueError, match="GTX 1660 SUPER"):
            TrainingConfig(precision="bf16")

    def test_bf16_allowed_on_supported_device(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

        config = TrainingConfig(precision="bf16")
        assert config.precision == "bf16"

    def test_bf16_skipped_when_no_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        config = TrainingConfig(precision="bf16")
        assert config.precision == "bf16"

    def test_validate_bf16_support_direct(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "Turing Card")

        with pytest.raises(ValueError, match="Turing Card"):
            validate_bf16_support()


class TestPrecisionModule:
    """src/training/precision.py: autocast_ctx / make_scaler policy."""

    def test_validate_precision_rejects_unknown(self):
        with pytest.raises(ValueError):
            validate_precision("int8")

    def test_autocast_ctx_fp32_is_nullcontext(self):
        from contextlib import nullcontext

        ctx = autocast_ctx("cpu", "fp32")
        assert isinstance(ctx, type(nullcontext()))

    def test_autocast_ctx_fp16_uses_float16(self):
        with autocast_ctx("cpu", "fp16"):
            assert torch.get_autocast_dtype("cpu") == torch.float16

    def test_autocast_ctx_bf16_uses_bfloat16(self):
        with autocast_ctx("cpu", "bf16"):
            assert torch.get_autocast_dtype("cpu") == torch.bfloat16

    def test_make_scaler_enabled_only_for_fp16(self):
        assert make_scaler("fp16").is_enabled()
        assert not make_scaler("fp32").is_enabled()
        assert not make_scaler("bf16").is_enabled()
