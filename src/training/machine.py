"""Machine-profile / experiment split (D-H, hardware portability).

The experiment (WHAT is measured — architecture, schedule, seeds,
`effective_batch`, ...) and the machine profile (HOW it runs — `device`,
`batch_size`, `num_workers`, `pin_memory`, `precision`) are kept in separate
YAML files so moving hardware is a profile switch, never an edit of the
experiment being measured. See
`openspec/changes/fusion-redesign/design.md` D-H.

Four enforcement mechanisms, because "must stay identical across machines"
without a mechanism is exactly how the two-phase-schedule defect (D2) went
unnoticed for 68 epochs:

1. A machine profile may only set `MACHINE_PROFILE_WHITELIST` keys — anything
   else raises at load, so a profile physically cannot alter the experiment.
2. `grad_accum_steps` is *derived* from `effective_batch // batch_size`,
   never read from a file. A non-evenly-divisible pair raises.
3. `experiment_sha256` — a SHA-256 of the experiment file's raw bytes —
   travels with every checkpoint and `stage_summary.json` entry.
4. Comparability rule (enforced by convention in the validation report, not
   by code): two runs are comparable only if `experiment_sha256` AND
   `effective_batch` both match.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance, type-checking only
    from .config import TrainingConfig

# A machine profile may set only these keys. Anything else raises at load —
# this is the mechanism that makes "the experiment stays identical across
# machines" enforced rather than merely documented.
MACHINE_PROFILE_WHITELIST = frozenset(
    {"device", "batch_size", "num_workers", "pin_memory", "precision"}
)


def validate_machine_profile(profile: dict[str, Any]) -> None:
    """Raise ValueError if `profile` sets any non-whitelisted key."""
    unknown = set(profile) - MACHINE_PROFILE_WHITELIST
    if unknown:
        raise ValueError(
            f"Machine profile sets non-whitelisted key(s): {sorted(unknown)}. "
            f"A machine profile may only set: {sorted(MACHINE_PROFILE_WHITELIST)}. "
            "This whitelist exists so a machine profile cannot alter the "
            "experiment being measured, even by accident (fusion-redesign D-H)."
        )


def load_machine_profile(path: str | Path) -> dict[str, Any]:
    """Load and validate a machine-profile YAML file.

    Returns the raw dict of overrides to apply onto a `TrainingConfig`.
    """
    path = Path(path)
    with open(path, "r") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    validate_machine_profile(data)
    return data


def apply_machine_profile(config: "TrainingConfig", profile: dict[str, Any]) -> None:
    """Apply a validated machine profile onto `config`, in place.

    Callers that already hold a raw dict (e.g. from a test) may pass it
    directly; this function re-validates regardless, so there is exactly one
    code path that can silently accept a non-whitelisted key.
    """
    validate_machine_profile(profile)
    for key, value in profile.items():
        setattr(config, key, value)


def derive_grad_accum_steps(effective_batch: int, batch_size: int) -> int:
    """Derive the number of gradient-accumulation micro-steps.

    `grad_accum_steps` is never itself a config field — it is always
    recomputed from `effective_batch` and `batch_size`, so an operator
    cannot silently change the effective batch by editing a machine
    profile's `batch_size` alone.

    Raises:
        ValueError: if `effective_batch` is not evenly divisible by
            `batch_size` — a fractional accumulation count would mean the
            recorded `effective_batch` does not match what actually ran.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if effective_batch % batch_size != 0:
        raise ValueError(
            f"effective_batch ({effective_batch}) is not evenly divisible by "
            f"batch_size ({batch_size}). grad_accum_steps must be a whole "
            "number, or the recorded effective_batch would not match what "
            "actually ran (fusion-redesign D-H)."
        )
    return effective_batch // batch_size


def experiment_sha256(experiment_config_path: str | Path) -> str:
    """SHA-256 hex digest of the experiment YAML file's raw bytes.

    Computed over raw bytes, not parsed/re-serialized content, so any
    byte-level change — including comments or formatting — changes the
    hash. Two runs are comparable only if this matches (design.md D-H).
    """
    path = Path(experiment_config_path)
    return hashlib.sha256(path.read_bytes()).hexdigest()
