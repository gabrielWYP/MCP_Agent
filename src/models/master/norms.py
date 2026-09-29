"""
Normalization-layer selection for the MasterModel neck and head.

The ConvNeXt backbone uses LayerNorm and is not affected by this module.
The only BatchNorm layers downstream of the backbone live in the neck
(`DualFPN.fusion_convs`, `fusion_mode="cross_attention"` only); the
detection head and the early-fusion `FPNNeck` currently have none, so for
them `neck_head_norm` is a no-op by construction.

- `"bn"` (default): `nn.BatchNorm2d`. Keeps every existing checkpoint
  loadable.
- `"gn"`: `nn.GroupNorm`. Batch-size independent, so the statistics used
  at train and eval time are identical — relevant with a physical batch of
  4, where BatchNorm running statistics are noisy and drift between epochs.

GroupNorm group count: `resolve_gn_groups(channels, gn_groups)` returns the
largest divisor of `channels` that is <= `gn_groups` (e.g. 256 channels,
32 groups -> 32; 48 channels, 32 groups -> 24; a prime channel count falls
back to 1 group, i.e. LayerNorm over C,H,W).

The two choices have different state_dict schemas (BatchNorm carries
`running_mean`/`running_var`/`num_batches_tracked` buffers, GroupNorm does
not), so a checkpoint of one is unloadable into the other.
`check_norm_compatible` turns that into one actionable sentence.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

NORM_BATCH = "bn"
NORM_GROUP = "gn"
NECK_HEAD_NORMS: tuple[str, ...] = (NORM_BATCH, NORM_GROUP)
DEFAULT_NECK_HEAD_NORM = NORM_BATCH
DEFAULT_GN_GROUPS = 32

# Modules whose norm layers `neck_head_norm` governs.
NECK_HEAD_PREFIXES: tuple[str, ...] = ("neck.", "head.")


def validate_neck_head_norm(norm: str, gn_groups: int) -> None:
    """Raise `ValueError` on an unknown norm kind or a non-positive group count."""
    if norm not in NECK_HEAD_NORMS:
        raise ValueError(
            f"Invalid neck_head_norm '{norm}'. Must be one of: {NECK_HEAD_NORMS}"
        )
    if isinstance(gn_groups, bool) or not isinstance(gn_groups, int) or gn_groups < 1:
        raise ValueError(f"gn_groups must be a positive int, got {gn_groups!r}.")


def resolve_gn_groups(channels: int, gn_groups: int) -> int:
    """Largest divisor of `channels` that is <= `gn_groups` (at least 1)."""
    for groups in range(min(gn_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def make_norm(norm: str, channels: int, gn_groups: int = DEFAULT_GN_GROUPS) -> nn.Module:
    """Build the 2D norm layer selected by `norm` over `channels` channels."""
    validate_neck_head_norm(norm, gn_groups)
    if norm == NORM_GROUP:
        return nn.GroupNorm(resolve_gn_groups(channels, gn_groups), channels)
    return nn.BatchNorm2d(channels)


def _has_batchnorm_stats(state_dict: Mapping[str, torch.Tensor]) -> bool:
    return any(
        key.startswith(NECK_HEAD_PREFIXES) and key.endswith(".running_mean")
        for key in state_dict
    )


def check_norm_compatible(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    source: str = "checkpoint",
) -> None:
    """Raise unless `state_dict`'s neck/head norm schema matches `model`'s.

    BatchNorm is detectable by its running-statistics buffers; their
    presence in exactly one of the two state dicts means a BatchNorm
    checkpoint is being loaded into a GroupNorm model or vice versa. Models
    with no neck/head norm at all (the early-fusion path) always pass.
    """
    model_bn = _has_batchnorm_stats(model.state_dict())
    ckpt_bn = _has_batchnorm_stats(state_dict)
    if model_bn == ckpt_bn:
        return
    ckpt_kind, model_kind = (NORM_BATCH, NORM_GROUP) if ckpt_bn else (NORM_GROUP, NORM_BATCH)
    raise ValueError(
        f"{source} has neck/head norm '{ckpt_kind}' but the model was built "
        f"with neck_head_norm='{model_kind}'. BatchNorm and GroupNorm "
        "checkpoints are not interchangeable — rebuild the model with "
        f"neck_head_norm='{ckpt_kind}'."
    )
