"""
Exponential moving average (EMA) of model weights, YOLOv8-style.

Small datasets with few optimizer steps per epoch (~13-19 here) make the
live weights at the end of any single epoch a noisy sample; validation AP
then oscillates from one epoch to the next. An EMA of the weights averages
over the last ~1/(1 - decay) optimizer steps and is what gets validated,
selected and saved when `TrainingConfig.use_ema` is on.

Decay warm-up: `d = decay * (1 - exp(-updates / tau))`. Early on `d` is
close to 0, so the EMA tracks the live model instead of staying anchored to
the random/pretrained initialization; after a few `tau` updates `d`
approaches `decay`. `tau` must be scaled to the run's step budget — YOLOv8's
2000 assumes tens of thousands of steps; a 60-epoch run here takes ~800.

Only floating-point **parameters** are averaged. **Buffers** (BatchNorm
running statistics, `num_batches_tracked`) are copied verbatim from the
live model on every update: BatchNorm stats are already running averages
and must describe the activations the averaged weights actually produce as
closely as the live model allows.

`update()` is called once per optimizer step, never per micro-batch, so
gradient accumulation does not change the effective averaging horizon.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


class ModelEMA:
    """EMA shadow copy of `model`.

    Args:
        model: The live model. Deep-copied once; the copy is kept in eval
            mode with `requires_grad=False`.
        decay: Asymptotic EMA decay in (0, 1).
        tau: Warm-up time constant, in optimizer steps (> 0).
        updates: Number of updates already applied (for resuming).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, tau: float = 200.0, updates: int = 0):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.tau = tau
        self.updates = updates

    def current_decay(self) -> float:
        """Decay applied by the most recent (or, before any, the next) update."""
        return self.decay * (1.0 - math.exp(-self.updates / self.tau))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend `model`'s parameters into the EMA and copy its buffers."""
        self.updates += 1
        d = self.current_decay()
        live_params = dict(model.named_parameters())
        for name, ema_p in self.module.named_parameters():
            live_p = live_params[name].detach()
            if ema_p.dtype.is_floating_point:
                ema_p.mul_(d).add_(live_p.to(ema_p.dtype), alpha=1.0 - d)
            else:
                ema_p.copy_(live_p)
        live_buffers = dict(model.named_buffers())
        for name, ema_b in self.module.named_buffers():
            ema_b.copy_(live_buffers[name].detach())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.module.state_dict()
