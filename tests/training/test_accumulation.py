"""D-H: gradient accumulation must be attribution-neutral.

`batch_size=2, grad_accum_steps=4` and `batch_size=8, grad_accum_steps=1`
processing the same 8 samples must produce the same parameter update after
one effective step, and `grad_clip` must apply once per effective step, not
once per micro-batch.

See openspec/changes/fusion-redesign/design.md D-H.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.config import TrainingConfig
from src.training.loop import Trainer


# 8 fixed samples: x values and a fixed (not necessarily learnable-perfect)
# linear target function y = 2x + 1. Only used to produce a non-trivial,
# reproducible gradient — the model does not need to converge.
_XS = [0.5, -0.3, 1.2, 0.1, -0.7, 0.9, 0.2, -0.4]
_YS = [2.0 * x + 1.0 for x in _XS]

_INITIAL_WEIGHT = 0.3
_INITIAL_BIAS = -0.1


class _LinearModel(nn.Module):
    """A single linear unit with deterministic, fixed initial parameters —
    no randomness, so two independently-constructed instances start
    identical without needing to share a manual_seed call."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        with torch.no_grad():
            self.linear.weight.fill_(_INITIAL_WEIGHT)
            self.linear.bias.fill_(_INITIAL_BIAS)

    def freeze_backbone(self, freeze_stages: int) -> None:
        pass

    def forward(self, rgb, nir=None):
        return {"preds": self.linear(rgb)}


def _mse_criterion(preds: torch.Tensor, targets: dict) -> tuple[torch.Tensor, dict]:
    """Fake criterion: mean-squared error against a target smuggled through
    `targets["labels"][0]` (a whole-batch tensor, not the usual per-image
    list — acceptable here since nothing else reads this batch's fields)."""
    y = targets["labels"][0]
    loss = ((preds - y) ** 2).mean()
    return loss, {"cls_loss": 0.0, "box_loss": 0.0}


def _make_batch(xs: list[float], ys: list[float]) -> dict:
    x = torch.tensor(xs, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)
    return {
        "rgb": x,
        "nir": torch.zeros_like(x),
        "bboxes": [torch.zeros(0, 4)] * len(xs),
        "labels": [y],
    }


class _GroupedLoader:
    """Yields the fixed 8-sample dataset grouped into `group_size` chunks,
    in a fixed order, so two loaders with different group sizes still see
    the same samples in the same relative order."""

    def __init__(self, group_size: int):
        self.group_size = group_size

    def __iter__(self):
        for i in range(0, len(_XS), self.group_size):
            yield _make_batch(_XS[i : i + self.group_size], _YS[i : i + self.group_size])


def _make_trainer(batch_size: int, effective_batch: int) -> Trainer:
    config = TrainingConfig(
        model_type="student",
        batch_size=batch_size,
        effective_batch=effective_batch,
        num_workers=0,
        precision="fp32",
        grad_clip=1e6,  # effectively disabled; this test is about accumulation, not clipping
        output_dir="/tmp/test_accumulation_unused",
    )
    trainer = Trainer(
        model=_LinearModel(),
        config=config,
        train_loader=_GroupedLoader(batch_size),
        val_loader=_GroupedLoader(batch_size),
    )
    trainer.criterion = _mse_criterion
    return trainer


class TestAccumulationEquivalence:
    """3.7: batch_size=2/accum=4 vs batch_size=8/accum=1 — same effective
    step, same parameter update."""

    def test_accumulated_and_full_batch_updates_match(self):
        trainer_accum = _make_trainer(batch_size=2, effective_batch=8)
        trainer_full = _make_trainer(batch_size=8, effective_batch=8)
        assert trainer_accum.grad_accum_steps == 4
        assert trainer_full.grad_accum_steps == 1

        optimizer_accum = SGD(trainer_accum.model.parameters(), lr=0.1)
        optimizer_full = SGD(trainer_full.model.parameters(), lr=0.1)

        trainer_accum._train_epoch(optimizer_accum, epoch=1, phase=1)
        trainer_full._train_epoch(optimizer_full, epoch=1, phase=1)

        assert torch.allclose(
            trainer_accum.model.linear.weight, trainer_full.model.linear.weight, atol=1e-6
        )
        assert torch.allclose(
            trainer_accum.model.linear.bias, trainer_full.model.linear.bias, atol=1e-6
        )

    def test_accumulated_run_takes_exactly_one_optimizer_step(self):
        trainer_accum = _make_trainer(batch_size=2, effective_batch=8)
        optimizer = SGD(trainer_accum.model.parameters(), lr=0.1)

        result = trainer_accum._train_epoch(optimizer, epoch=1, phase=1)

        assert result["steps_taken"] == 1.0

    def test_full_batch_run_also_takes_exactly_one_optimizer_step(self):
        trainer_full = _make_trainer(batch_size=8, effective_batch=8)
        optimizer = SGD(trainer_full.model.parameters(), lr=0.1)

        result = trainer_full._train_epoch(optimizer, epoch=1, phase=1)

        assert result["steps_taken"] == 1.0


class TestGradClipOncePerEffectiveStep:
    """3.8: grad_clip is applied once per effective step, not per micro-batch."""

    def test_clip_called_once_per_effective_step(self, monkeypatch):
        trainer = _make_trainer(batch_size=2, effective_batch=8)
        optimizer = SGD(trainer.model.parameters(), lr=0.1)

        calls = []
        original_clip = nn.utils.clip_grad_norm_

        def counting_clip(*args, **kwargs):
            calls.append(1)
            return original_clip(*args, **kwargs)

        monkeypatch.setattr(nn.utils, "clip_grad_norm_", counting_clip)

        trainer._train_epoch(optimizer, epoch=1, phase=1)

        # 8 samples / batch_size=2 => 4 micro-batches, grad_accum_steps=4
        # => exactly 1 effective step => exactly 1 clip call, not 4.
        assert len(calls) == 1

    def test_clip_called_once_per_effective_step_two_windows(self, monkeypatch):
        """16 samples at batch_size=2/accum=4 => 2 effective steps => 2 clip calls."""
        config = TrainingConfig(
            model_type="student",
            batch_size=2,
            effective_batch=8,
            num_workers=0,
            precision="fp32",
            grad_clip=1e6,
            output_dir="/tmp/test_accumulation_unused",
        )

        class _DoubleLoader:
            def __iter__(self):
                xs = _XS + _XS  # 16 samples => 8 micro-batches of 2 => 2 effective steps
                ys = _YS + _YS
                for i in range(0, len(xs), 2):
                    yield _make_batch(xs[i : i + 2], ys[i : i + 2])

        trainer = Trainer(
            model=_LinearModel(),
            config=config,
            train_loader=_DoubleLoader(),
            val_loader=_DoubleLoader(),
        )
        trainer.criterion = _mse_criterion
        optimizer = SGD(trainer.model.parameters(), lr=0.1)

        calls = []
        original_clip = nn.utils.clip_grad_norm_

        def counting_clip(*args, **kwargs):
            calls.append(1)
            return original_clip(*args, **kwargs)

        monkeypatch.setattr(nn.utils, "clip_grad_norm_", counting_clip)

        result = trainer._train_epoch(optimizer, epoch=1, phase=1)

        assert len(calls) == 2
        assert result["steps_taken"] == 2.0
