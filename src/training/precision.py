"""Precision policy: fp32 | fp16 | bf16.

Replaces the old `TrainingConfig.amp: bool = True` flag
(`src/training/config.py`), which silently enabled fp16 in any config that
omitted the key. See `openspec/changes/fusion-redesign/design.md` D-I.

- fp32: no autocast, no gradient scaling. The safe default.
- fp16: autocast + `GradScaler` (fp16 needs loss scaling to avoid gradient
  underflow).
- bf16: autocast, no `GradScaler` (bf16 carries fp32's exponent range, so
  gradient underflow from a narrow exponent is not a concern). Requires
  Ampere or newer; validated at config-load time via
  `validate_bf16_support()` so an unsupported card fails loud, not with a
  silent downgrade to fp32 that would make an H-BF16 CONFIRM meaningless.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch.amp import GradScaler

VALID_PRECISIONS = ("fp32", "fp16", "bf16")


def validate_precision(precision: str) -> None:
    """Raise ValueError if `precision` is not one of the supported values."""
    if precision not in VALID_PRECISIONS:
        raise ValueError(
            f"Invalid precision '{precision}'. Must be one of: {VALID_PRECISIONS}"
        )


def validate_bf16_support() -> None:
    """Raise if bf16 is requested but the active CUDA device cannot run it.

    No-op when CUDA is unavailable (nothing to validate against; bf16 on CPU
    is out of scope for this project's hardware-portability design).
    """
    if not torch.cuda.is_available():
        return
    if not torch.cuda.is_bf16_supported():
        name = torch.cuda.get_device_name()
        raise ValueError(
            f"precision='bf16' requested, but the active CUDA device "
            f"({name}) does not support bf16 (requires Ampere or newer)."
        )


def autocast_ctx(device_type: str, precision: str):
    """Return the autocast context manager for `precision`.

    fp32 returns `nullcontext()` (a true no-op, not a disabled autocast —
    avoids any per-op autocast bookkeeping overhead). fp16/bf16 wrap
    `torch.amp.autocast` with the matching dtype.
    """
    validate_precision(precision)
    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.amp.autocast(device_type=device_type, dtype=dtype)


def make_scaler(precision: str) -> GradScaler:
    """Build the `GradScaler` for `precision`.

    Only fp16 needs loss scaling; the returned scaler is `enabled=False`
    (and therefore a no-op) for fp32 and bf16 alike.
    """
    validate_precision(precision)
    return GradScaler("cuda", enabled=(precision == "fp16"))
