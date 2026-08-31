"""Single source of truth for FPN stride resolution (fusion-redesign D-D).

`head_strides` is the one config field every other level-count-dependent
component derives from: `FPNNeck.emit_levels`, `YOLODetectionHead`'s level
count, `YOLOv8Loss.strides`, and `decode_detections(strides=...)`. Before
this module, `[8, 16, 32]` was a literal duplicated at eight call sites
(design.md D-D/§5) — most of those duplications fail loudly on a mismatch,
but two do not:

- `assigner_level_ranges` shorter than `len(head_strides) - 1` leaves the
  coarsest pyramid level with zero positive assignments, silently.
- A stale hardcoded `strides=(8, 16, 32)` decoding a 4-level checkpoint
  misaligns predictions and anchor grids without raising.

`validate_strides` makes both classes of defect a loud `ValueError` at
config-construction or resolution time instead of a training run that
"succeeds" while silently mis-training or mis-evaluating one level.
"""

from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance, type-checking only
    from .config import TrainingConfig

# All strides the pyramid can emit, finest-first: P2, P3, P4, P5.
ALL_STRIDES: tuple[int, ...] = (4, 8, 16, 32)

# Stride -> pyramid level index (0=P2, 1=P3, 2=P4, 3=P5).
STRIDE_TO_LEVEL: dict[int, int] = {stride: i for i, stride in enumerate(ALL_STRIDES)}

DEFAULT_HEAD_STRIDES: list[int] = [4, 8, 16, 32]

# StudentModel is out of scope for fusion-redesign (D-2: kept unchanged) and
# always emits 3 fixed levels at these strides, regardless of MasterModel's
# `head_strides`. This is the single canonical definition of that constant
# in the whole repository — every other file imports it from here rather
# than re-declaring the literal, so `tests/test_stride_literals.py`'s
# source-level `[8, 16, 32]` / `(8, 16, 32)` guard has exactly one place to
# exempt instead of an ever-growing allowlist.
STUDENT_STRIDES: tuple[int, ...] = (8, 16, 32)


def validate_strides(
    strides: Sequence[int],
    level_ranges: Sequence[float] | None = None,
) -> None:
    """Raise `ValueError` if `strides` or `level_ranges` violate an invariant.

    Invariants:
    - `strides` is non-empty.
    - `strides` is a strictly increasing subsequence of `ALL_STRIDES`
      (`(4, 8, 16, 32)`); any other value, duplicate, or non-increasing
      order raises.
    - If `level_ranges` is given, `len(level_ranges) == len(strides) - 1`
      (D-D's "one genuinely silent failure": a shorter `level_ranges`
      leaves the coarsest level with zero positive assignments, silently).
    """
    if not strides:
        raise ValueError("head_strides must be non-empty.")

    unknown = [s for s in strides if s not in STRIDE_TO_LEVEL]
    if unknown:
        raise ValueError(
            f"head_strides contains unsupported stride(s) {unknown}; "
            f"every stride must be one of {ALL_STRIDES}."
        )

    if len(set(strides)) != len(strides):
        raise ValueError(f"head_strides must not contain duplicates, got {list(strides)}.")

    if list(strides) != sorted(strides):
        raise ValueError(
            f"head_strides must be strictly increasing (finest-first), got {list(strides)}."
        )

    if level_ranges is not None and len(level_ranges) != len(strides) - 1:
        raise ValueError(
            f"assigner_level_ranges must have exactly len(head_strides) - 1 "
            f"entries ({len(strides) - 1} for head_strides={list(strides)}), "
            f"got {len(level_ranges)} ({list(level_ranges)}). A shorter "
            "level_ranges silently starves the coarsest pyramid level of "
            "positive assignments (fusion-redesign design.md D-D)."
        )


def resolve_head_strides(config: "TrainingConfig") -> list[int]:
    """Return `config.head_strides` after validating it against the config's
    `assigner_level_ranges`."""
    strides = list(config.head_strides)
    validate_strides(strides, getattr(config, "assigner_level_ranges", None))
    return strides


def resolve_from_checkpoint(checkpoint: dict) -> list[int]:
    """Read and validate the stride list a checkpoint was trained with.

    `_save_checkpoint` persists `config.__dict__` (which includes
    `head_strides`) under the `"config"` key, so strides travel with every
    checkpoint at zero extra cost. Raises `KeyError` if the checkpoint has
    no recorded config/`head_strides` (e.g. an old raw state_dict) — this is
    intentionally loud rather than silently falling back to `[8, 16, 32]`.
    """
    config_dict = checkpoint.get("config")
    if not config_dict or "head_strides" not in config_dict:
        raise KeyError(
            "Checkpoint has no recorded 'config.head_strides'. Refusing to "
            "assume a default stride list — pass --override head_strides=... "
            "explicitly, or use a checkpoint saved by the current Trainer."
        )
    strides = list(config_dict["head_strides"])
    validate_strides(strides)
    return strides


def resolve_active_strides(config: "TrainingConfig") -> list[int]:
    """Return the stride list actually used by the active model.

    `StudentModel` is out of scope for this redesign (D-2: kept unchanged)
    and always uses 3 fixed levels at `[8, 16, 32]`, regardless of
    `config.head_strides` (which only applies to `MasterModel`). This
    resolver is the single place that branches on `model_type` so
    `Trainer`/decode call sites never have to.
    """
    if config.model_type == "student":
        return list(STUDENT_STRIDES)
    return resolve_head_strides(config)


def select_by_strides(
    features: Sequence,
    source_strides: Sequence[int],
    target_strides: Sequence[int],
):
    """Select elements of `features` by stride, not by position.

    `source_strides[i]` names the stride of `features[i]`. Returns the
    subset of `features` whose stride is in `target_strides`, in
    `target_strides`'s order. Raises `ValueError` if any `target_strides`
    entry is absent from `source_strides`, instead of silently truncating
    a positional `zip` — the failure mode identified in design.md §5 for
    `ProjectionLayers.forward` and the KD teacher-feature slicing it feeds.
    """
    if len(features) != len(source_strides):
        raise ValueError(
            f"features has {len(features)} entries but source_strides has "
            f"{len(source_strides)}; they must be the same length and aligned by index."
        )
    index_by_stride = {stride: i for i, stride in enumerate(source_strides)}
    missing = [s for s in target_strides if s not in index_by_stride]
    if missing:
        raise ValueError(
            f"target stride(s) {missing} not present in source_strides "
            f"{list(source_strides)}; cannot select by index."
        )
    return [features[index_by_stride[s]] for s in target_strides]
