"""Weight EMA (`src/training/ema.py`) and its Trainer integration.

Covers the warm-up ramp math, buffer copying, the once-per-optimizer-step
update contract under gradient accumulation, evaluation on the EMA weights,
the checkpoint payload, and config validation.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.config import TrainingConfig
from src.training.ema import ModelEMA
from src.training.loop import Trainer

_XS = [0.5, -0.3, 1.2, 0.1, -0.7, 0.9, 0.2, -0.4]
_YS = [2.0 * x + 1.0 for x in _XS]


class _LinearBNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.bn = nn.BatchNorm1d(1)
        with torch.no_grad():
            self.linear.weight.fill_(0.3)
            self.linear.bias.fill_(-0.1)

    def freeze_backbone(self, freeze_stages: int) -> None:
        pass

    def forward(self, rgb, nir=None):
        return {"preds": self.linear(self.bn(rgb))}


def _mse_criterion(preds, targets):
    loss = ((preds - targets["labels"][0]) ** 2).mean()
    return loss, {"cls_loss": 0.0, "box_loss": 0.0}


class _Loader:
    def __init__(self, group_size: int):
        self.group_size = group_size

    def __iter__(self):
        for i in range(0, len(_XS), self.group_size):
            x = torch.tensor(_XS[i : i + self.group_size]).unsqueeze(1)
            y = torch.tensor(_YS[i : i + self.group_size]).unsqueeze(1)
            yield {"rgb": x, "nir": torch.zeros_like(x),
                   "bboxes": [torch.zeros(0, 4)] * len(x), "labels": [y]}


def _trainer(tmp_path, use_ema: bool, batch_size: int = 2, effective_batch: int = 4) -> Trainer:
    config = TrainingConfig(
        model_type="student", batch_size=batch_size, effective_batch=effective_batch,
        num_workers=0, precision="fp32", device="cpu", grad_clip=1e6,
        output_dir=str(tmp_path), use_ema=use_ema, ema_decay=0.9, ema_tau=2.0,
    )
    trainer = Trainer(model=_LinearBNModel(), config=config,
                      train_loader=_Loader(batch_size), val_loader=[])
    trainer.criterion = _mse_criterion
    return trainer


class TestModelEMAMath:
    def test_decay_ramp(self):
        ema = ModelEMA(nn.Linear(1, 1), decay=0.999, tau=200.0)
        assert ema.current_decay() == 0.0
        for updates in (1, 200, 2000):
            ema.updates = updates
            assert ema.current_decay() == pytest.approx(0.999 * (1 - math.exp(-updates / 200.0)))
        assert ema.current_decay() == pytest.approx(0.999, abs=1e-4)

    def test_update_blends_parameters_with_ramped_decay(self):
        live = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            live.weight.fill_(0.0)
        ema = ModelEMA(live, decay=0.9, tau=2.0)
        with torch.no_grad():
            live.weight.fill_(1.0)
        ema.update(live)
        d = 0.9 * (1 - math.exp(-1 / 2.0))
        assert ema.updates == 1
        assert ema.module.weight.item() == pytest.approx(d * 0.0 + (1 - d) * 1.0)

    def test_buffers_are_copied_not_averaged(self):
        live = nn.BatchNorm1d(2)
        ema = ModelEMA(live, decay=0.999, tau=1.0)
        with torch.no_grad():
            live.running_mean.fill_(5.0)
            live.running_var.fill_(3.0)
            live.num_batches_tracked.fill_(7)
        ema.update(live)
        assert torch.equal(ema.module.running_mean, live.running_mean)
        assert torch.equal(ema.module.running_var, live.running_var)
        assert ema.module.num_batches_tracked.item() == 7

    def test_shadow_is_frozen_and_independent(self):
        live = nn.Linear(1, 1)
        ema = ModelEMA(live)
        assert all(not p.requires_grad for p in ema.module.parameters())
        assert not ema.module.training
        assert ema.module.weight.data_ptr() != live.weight.data_ptr()


class TestTrainerIntegration:
    def test_updates_once_per_optimizer_step_under_accumulation(self, tmp_path):
        # 8 samples, batch 2, effective 4 -> 4 micro-batches, 2 optimizer steps.
        trainer = _trainer(tmp_path, use_ema=True)
        assert trainer.grad_accum_steps == 2
        result = trainer._train_epoch(SGD(trainer.model.parameters(), lr=0.1), epoch=1, phase=1)
        assert result["steps_taken"] == 2.0
        assert trainer.ema.updates == 2

    def test_ema_off_by_default(self, tmp_path):
        trainer = _trainer(tmp_path, use_ema=False)
        assert trainer.ema is None
        assert trainer.eval_model is trainer.model

    def test_eval_only_trainer_builds_no_ema(self, tmp_path):
        config = TrainingConfig(model_type="student", num_workers=0, device="cpu",
                                output_dir=str(tmp_path), use_ema=True)
        trainer = Trainer(model=_LinearBNModel(), config=config, train_loader=None, val_loader=[])
        assert trainer.ema is None

    def test_evaluation_runs_on_ema_weights(self, tmp_path):
        trainer = _trainer(tmp_path, use_ema=True)
        trainer._train_epoch(SGD(trainer.model.parameters(), lr=0.1), epoch=1, phase=1)
        assert trainer.eval_model is trainer.ema.module
        assert not torch.equal(trainer.ema.module.linear.weight, trainer.model.linear.weight)

        called = []
        trainer._decode_predictions = lambda *a, **k: (
            torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long)
        )
        x = torch.zeros(1, 1)
        output = {"preds": [], "cls_preds": []}
        trainer.ema.module.forward = lambda rgb, nir=None: (called.append("ema"), output)[1]
        trainer.model.forward = lambda rgb, nir=None: (called.append("live"), output)[1]
        trainer.val_loader = [{"rgb": x, "nir": x, "bboxes": [torch.zeros(0, 4)],
                               "labels": [torch.zeros(0, dtype=torch.long)]}]
        trainer.collect_predictions()
        assert called == ["ema"]

    def test_checkpoint_stores_ema_as_eval_weights_and_live_for_resume(self, tmp_path):
        trainer = _trainer(tmp_path, use_ema=True)
        trainer._train_epoch(SGD(trainer.model.parameters(), lr=0.1), epoch=1, phase=1)
        trainer._save_checkpoint(1, 1, {}, "ckpt.pt")
        ckpt = torch.load(tmp_path / "ckpt.pt", weights_only=False)
        assert ckpt["use_ema"] is True
        assert ckpt["ema_updates"] == 2
        for key, value in trainer.ema.state_dict().items():
            assert torch.equal(ckpt["model_state_dict"][key], value)
        for key, value in trainer.model.state_dict().items():
            assert torch.equal(ckpt["live_model_state_dict"][key], value)

    def test_checkpoint_without_ema_keeps_legacy_layout(self, tmp_path):
        trainer = _trainer(tmp_path, use_ema=False)
        trainer._save_checkpoint(1, 1, {}, "ckpt.pt")
        ckpt = torch.load(tmp_path / "ckpt.pt", weights_only=False)
        assert ckpt["use_ema"] is False
        assert "live_model_state_dict" not in ckpt


class TestConfigValidation:
    def test_defaults(self):
        config = TrainingConfig()
        assert (config.use_ema, config.ema_decay, config.ema_tau) == (False, 0.999, 200.0)

    @pytest.mark.parametrize("decay", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_decay_outside_open_unit_interval(self, decay):
        with pytest.raises(ValueError, match="ema_decay"):
            TrainingConfig(ema_decay=decay)

    @pytest.mark.parametrize("tau", [0.0, -5.0])
    def test_rejects_non_positive_tau(self, tau):
        with pytest.raises(ValueError, match="ema_tau"):
            TrainingConfig(ema_tau=tau)
