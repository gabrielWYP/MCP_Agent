"""D-G: OOM/NaN batch-skip guards must count and raise, never skip silently.

Verified defect (openspec/changes/fusion-redesign/design.md D-G):
`Trainer._train_epoch` used to catch CUDA OOM and NaN/Inf losses, print a
one-line warning, `continue`, and floor the loss divisor at
`max(n_batches, 1)`. A run in which every batch failed therefore completed
"successfully" with `total_loss=0.0`, saved a checkpoint, and produced a
validation mAP from a model that never received a gradient.

These tests use a fake loader and a fault-injecting model/criterion; no GPU
and no real CUDA OOM are required — the OOM path is exercised by raising a
`RuntimeError` whose message contains "out of memory", which is exactly the
string match `_train_epoch` uses to route the exception.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.config import TrainingConfig
from src.training.loop import Trainer


def _fake_batch() -> dict:
    return {
        "rgb": torch.zeros(1, 3, 4, 4),
        "nir": torch.zeros(1, 1, 4, 4),
        "bboxes": [torch.zeros(0, 4)],
        "labels": [torch.zeros(0, dtype=torch.long)],
    }


class _FakeLoader:
    """Iterable yielding `n` fake batches per pass, replayable across epochs."""

    def __init__(self, n: int):
        self.n = n

    def __iter__(self):
        return (_fake_batch() for _ in range(self.n))


class _FaultModelBase(nn.Module):
    """Shared no-op `freeze_backbone` — `_train_phase` calls it unconditionally
    regardless of model_type, and these fakes are not MasterModel/StudentModel."""

    def freeze_backbone(self, freeze_stages: int) -> None:
        pass


class _OOMModel(_FaultModelBase):
    """Every forward pass raises a CUDA-OOM-shaped RuntimeError."""

    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))

    def forward(self, rgb, nir=None):
        raise RuntimeError("CUDA out of memory. Tried to allocate 999.00 GiB")


class _PartialOOMModel(_FaultModelBase):
    """Alternates: fails every other call with a CUDA-OOM RuntimeError."""

    def __init__(self, fail_every: int):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))
        self.fail_every = fail_every
        self.calls = 0

    def forward(self, rgb, nir=None):
        self.calls += 1
        if self.calls % self.fail_every == 0:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
        return {"preds": None}


class _StubModel(_FaultModelBase):
    """Trivial forward; the criterion is monkeypatched to control the loss."""

    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))

    def forward(self, rgb, nir=None):
        return {"preds": None}


def _make_trainer(model: nn.Module, n_batches: int) -> Trainer:
    """Build a real Trainer with a fault-injecting model and a fake loader.

    `val_loader` is never iterated by `_train_epoch`, so an empty fake loader
    is sufficient — these tests exercise `_train_epoch` directly and never
    call `_validate`/`evaluate`.
    """
    config = TrainingConfig(
        # "student" avoids the E6 rgb_stem-grad-norm instrumentation, which
        # assumes a real MasterModel backbone and would break these
        # fault-injecting fakes.
        model_type="student",
        batch_size=1,
        # effective_batch == batch_size => grad_accum_steps == 1, so these
        # D-G-focused tests are independent of the accumulation behavior
        # covered separately in test_accumulation.py.
        effective_batch=1,
        num_workers=0,
        precision="fp32",
        output_dir="/tmp/test_loop_guards_unused",
    )
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=_FakeLoader(n_batches),
        val_loader=_FakeLoader(0),
    )
    return trainer


def _optimizer_for(trainer: Trainer) -> AdamW:
    return AdamW(trainer.model.parameters(), lr=1e-3)


class TestAllBatchesFail:
    """1.1/1.2/1.3: an epoch where every batch fails must raise, not report loss 0.0."""

    def test_all_oom_epoch1_raises_on_zero_steps(self):
        """Every batch OOMs in epoch 1: OOM itself is tolerated on epoch 1,
        but zero optimizer steps is fatal regardless of epoch number."""
        trainer = _make_trainer(_OOMModel(), n_batches=3)
        optimizer = _optimizer_for(trainer)

        with pytest.raises(RuntimeError, match="zero optimizer"):
            trainer._train_epoch(optimizer, epoch=1, phase=1)

    def test_all_nan_raises_and_records_nan_skipped(self):
        """Every batch produces a NaN loss: nan_skipped is counted and
        surfaced in the raised error, and the epoch raises via steps_taken==0."""
        trainer = _make_trainer(_StubModel(), n_batches=4)
        trainer.criterion = lambda preds, targets: (
            torch.tensor(float("nan")),
            {"cls_loss": 0.0, "box_loss": 0.0},
        )
        optimizer = _optimizer_for(trainer)

        with pytest.raises(RuntimeError, match="nan_skipped=4") as exc_info:
            trainer._train_epoch(optimizer, epoch=1, phase=1)
        assert "zero optimizer" in str(exc_info.value)

    def test_steps_taken_zero_raises_on_empty_loader(self):
        """No batches at all is another route to steps_taken==0; must raise
        the same way regardless of cause."""
        trainer = _make_trainer(_StubModel(), n_batches=0)
        optimizer = _optimizer_for(trainer)

        with pytest.raises(RuntimeError, match="zero optimizer"):
            trainer._train_epoch(optimizer, epoch=1, phase=1)


class TestOOMEpochTolerance:
    """D-G: OOM is tolerated on epoch 1 (allocator warm-up) only."""

    def test_partial_oom_epoch1_tolerated_and_counted(self):
        """2 of 4 batches OOM in epoch 1; the epoch completes and records
        oom_skipped without raising, because steps_taken > 0."""
        model = _PartialOOMModel(fail_every=2)
        trainer = _make_trainer(model, n_batches=4)
        trainer.criterion = lambda preds, targets: (
            torch.tensor(0.5, requires_grad=True),
            {"cls_loss": 0.25, "box_loss": 0.25},
        )
        optimizer = _optimizer_for(trainer)

        result = trainer._train_epoch(optimizer, epoch=1, phase=1)

        assert result["oom_skipped"] == 2.0
        assert result["nan_skipped"] == 0.0
        assert result["steps_taken"] == 2.0

    def test_partial_oom_epoch2_raises_immediately(self):
        """The same fault pattern on epoch 2 must raise as soon as the first
        OOM is hit — the epoch-1 tolerance does not extend past epoch 1."""
        model = _PartialOOMModel(fail_every=2)
        trainer = _make_trainer(model, n_batches=4)
        trainer.criterion = lambda preds, targets: (
            torch.tensor(0.5, requires_grad=True),
            {"cls_loss": 0.25, "box_loss": 0.25},
        )
        optimizer = _optimizer_for(trainer)

        with pytest.raises(RuntimeError, match="epoch 2"):
            trainer._train_epoch(optimizer, epoch=2, phase=1)


class TestSuccessfulEpochCounters:
    """A fully healthy epoch still reports the D-G counters (wiring for
    LossHistory/TensorBoard/checkpoint downstream consumers)."""

    def test_clean_epoch_reports_zero_skips(self):
        model = _StubModel()
        trainer = _make_trainer(model, n_batches=3)
        trainer.criterion = lambda preds, targets: (
            torch.tensor(0.5, requires_grad=True),
            {"cls_loss": 0.25, "box_loss": 0.25},
        )
        optimizer = _optimizer_for(trainer)

        result = trainer._train_epoch(optimizer, epoch=1, phase=1)

        assert result["oom_skipped"] == 0.0
        assert result["nan_skipped"] == 0.0
        assert result["steps_taken"] == 3.0
        assert result["total_loss"] == pytest.approx(0.5)


class TestCountersWiredToArtifacts:
    """1.5: counters must reach LossHistory.extra_losses and the checkpoint
    dict, not just the in-memory return value of `_train_epoch`."""

    def test_train_phase_records_counters_in_history_and_checkpoint(self, tmp_path):
        model = _StubModel()
        config = TrainingConfig(
            model_type="student",
            batch_size=1,
            effective_batch=1,
            num_workers=0,
            precision="fp32",
            epochs_phase1=1,
            warmup_epochs=1,
            save_interval=1,
            output_dir=str(tmp_path),
        )
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=_FakeLoader(2),
            val_loader=_FakeLoader(0),
        )
        trainer.criterion = lambda preds, targets: (
            torch.tensor(0.5, requires_grad=True),
            {"cls_loss": 0.25, "box_loss": 0.25},
        )
        # Bypass real evaluation — this test is about counter plumbing, not mAP.
        trainer._validate = lambda epoch, phase: {"map50": 0.0, "map_50_95": 0.0}

        trainer._train_phase(phase=1, epochs=1, lr=1e-3, freeze_stages=0)

        assert trainer.loss_history.extra_losses["oom_skipped"] == [0.0]
        assert trainer.loss_history.extra_losses["nan_skipped"] == [0.0]
        assert trainer.loss_history.extra_losses["steps_taken"] == [2.0]

        checkpoint = torch.load(tmp_path / "best_model.pt", weights_only=False)
        assert checkpoint["oom_skipped"] == 0.0
        assert checkpoint["nan_skipped"] == 0.0
        assert checkpoint["steps_taken"] == 2.0
