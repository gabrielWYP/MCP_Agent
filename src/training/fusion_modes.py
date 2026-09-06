"""Single source of truth for `fusion_mode` and its checkpoint schema tag.

`MasterModel` supports two mutually exclusive multimodal fusion
architectures, selected by `TrainingConfig.fusion_mode`:

- ``"early"`` (default, fusion-redesign): RGB and NIR are stacked into one
  4-channel tensor before a single ConvNeXt stem; one backbone stream, one
  FPN (`EarlyFusionBackbone` + `FPNNeck`).
- ``"cross_attention"``: the pre-redesign two-stream architecture — two
  modality-specific stems over shared ConvNeXt stages, per-stage
  cross-attention (RGB queries NIR), and two FPNs fused per level
  (`DualConvNeXtBackbone` + `CrossModalFusion` + `DualFPN`).

Everything upstream of the shared detection head is disjoint between the
two: `backbone.stem.*`/`backbone.stages.*`/`neck.fpn.*` on the early path vs
`backbone.rgb_stem.*`/`backbone.nir_stem.*`/`backbone.shared_stages.*`/
`fusion.*`/`neck.fpn_rgb.*`/`neck.fpn_nir.*`/`neck.fusion_convs.*` on the
cross-attention path. Loading one into the other is therefore always an
error, and `arch_version` is what turns it into a single actionable sentence
instead of a 200-line missing/unexpected-key dump:

    arch_version 1 -> cross_attention   (two-stream)
    arch_version 2 -> early             (single-stream, early fusion)

`None` is accepted only on the cross-attention path: checkpoints saved
before the `arch_version` tag existed (e.g. `checkpoints/mastermodel_mango`)
predate the redesign entirely, so an untagged checkpoint is a two-stream one
by construction. It is deliberately NOT accepted for `"early"` — an untagged
checkpoint can never be an early-fusion one.
"""

from __future__ import annotations

FUSION_MODE_EARLY = "early"
FUSION_MODE_CROSS_ATTENTION = "cross_attention"

FUSION_MODES: tuple[str, ...] = (FUSION_MODE_EARLY, FUSION_MODE_CROSS_ATTENTION)

DEFAULT_FUSION_MODE: str = FUSION_MODE_EARLY

# Checkpoint state_dict schema version per fusion mode (fusion-redesign D-C).
ARCH_VERSION_BY_FUSION_MODE: dict[str, int] = {
    FUSION_MODE_CROSS_ATTENTION: 1,
    FUSION_MODE_EARLY: 2,
}

# What `checkpoint.get("arch_version")` may be for a checkpoint of each mode.
# Only the cross-attention path accepts the untagged `None` (see module
# docstring) — a checkpoint written by the current Trainer always carries an
# explicit int.
ACCEPTED_ARCH_VERSIONS_BY_FUSION_MODE: dict[str, frozenset] = {
    FUSION_MODE_CROSS_ATTENTION: frozenset({1, None}),
    FUSION_MODE_EARLY: frozenset({2}),
}


def validate_fusion_mode(fusion_mode: str) -> None:
    """Raise `ValueError` unless `fusion_mode` is one of `FUSION_MODES`.

    Loud on anything else: a typo like `"cross-attention"` must not fall
    back to the default and silently train the wrong architecture.
    """
    if fusion_mode not in FUSION_MODES:
        raise ValueError(
            f"Invalid fusion_mode {fusion_mode!r}. Must be one of "
            f"{list(FUSION_MODES)}."
        )


def arch_version_for_mode(fusion_mode: str) -> int:
    """Return the `arch_version` a checkpoint of `fusion_mode` is tagged with."""
    validate_fusion_mode(fusion_mode)
    return ARCH_VERSION_BY_FUSION_MODE[fusion_mode]


def accepted_arch_versions(fusion_mode: str) -> frozenset:
    """Return the `arch_version` values loadable on `fusion_mode`'s path."""
    validate_fusion_mode(fusion_mode)
    return ACCEPTED_ARCH_VERSIONS_BY_FUSION_MODE[fusion_mode]


def fusion_mode_for_arch_version(arch_version: int | None) -> str | None:
    """Return the fusion mode an `arch_version` belongs to, or None.

    Used only to make a mismatch error message name the mode the checkpoint
    actually came from; returns `None` for an unrecognised tag rather than
    guessing.
    """
    if arch_version is None:
        return FUSION_MODE_CROSS_ATTENTION
    for mode, version in ARCH_VERSION_BY_FUSION_MODE.items():
        if version == arch_version:
            return mode
    return None
