"""
Knowledge Distillation Trainer — subclass of Trainer with frozen teacher.

Extends the base Trainer with:
    - Frozen MasterModel teacher loaded from checkpoint
    - 4 ProjectionLayers groups attached to StudentModel as submodules
    - Overridden _train_epoch with KD forward pass (teacher → project → MSE)
    - Inherited _validate (student-only), fit, checkpoint, early stopping

Teacher is stored as a plain attribute (NOT nn.Module submodule) so it is
naturally excluded from optimizer.param_groups and model.state_dict().
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .kd_config import KDConfig
from .kd_loss import KDLoss
from .loop import Trainer
from src.models.master.master_model import MasterModel
from src.models.master.distill_projections import (
    backbone_projections,
    fpn_projections,
    head_projections,
)


class KDTrainer(Trainer):
    """Knowledge distillation trainer with frozen teacher.

    Attaches 4 projection groups to the student model as registered submodules,
    loads a frozen MasterModel teacher, and overrides _train_epoch to compute
    both detection loss and per-level MSE distillation loss.

    Args:
        model: StudentModel instance (projections attached before super().__init__).
        config: KDConfig with teacher_checkpoint and KD hyperparameters.
        train_loader: Training data loader.
        val_loader: Validation data loader.
    """

    def __init__(
        self,
        model: nn.Module,
        config: KDConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        # Attach projection layers BEFORE super().__init__() so they are
        # moved to device by self.model.to(device) in Trainer.__init__.
        model.kd_proj_backbone = backbone_projections()
        model.kd_proj_fpn = fpn_projections()
        model.kd_proj_head_cls = head_projections()
        model.kd_proj_head_reg = head_projections()

        super().__init__(model, config, train_loader, val_loader)

        # Load frozen teacher — stored as plain attribute, NOT a submodule.
        # This ensures teacher params are excluded from:
        #   - optimizer.param_groups (built in _train_phase via self.model.parameters())
        #   - model.state_dict() (used in inherited _save_checkpoint)
        self.teacher = self._load_teacher(config)
        self.teacher.to(self.device)

        # KD loss
        self.kd_criterion = KDLoss(
            level_weights=config.distill_levels,
            temperature=config.kd_temperature,
        ).to(self.device)

    @staticmethod
    def _load_teacher(config: KDConfig) -> MasterModel:
        """Load a frozen MasterModel from checkpoint.

        Args:
            config: KDConfig with teacher_checkpoint path and num_classes.

        Returns:
            MasterModel in eval mode with all params frozen.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
        """
        ckpt_path = Path(config.teacher_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Teacher checkpoint not found: {ckpt_path}"
            )

        teacher = MasterModel(
            num_classes=config.num_classes,
            pretrained_backbone=False,
            backbone_variant=config.backbone_variant,
        )

        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        teacher.load_state_dict(checkpoint["model_state_dict"])

        # Freeze all teacher parameters
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        print(f"[KDTrainer] Teacher loaded from {ckpt_path} "
              f"({sum(p.numel() for p in teacher.parameters()):,} params, frozen)")

        return teacher

    def _train_epoch(self, optimizer: AdamW, epoch: int, phase: int) -> dict:
        """Run one KD training epoch.

        Teacher forward under no_grad → project features → student forward →
        det_loss + kd_weight * kd_loss → backward → grad clip → step.

        Returns:
            Dict with avg cls_loss, box_loss, kd_loss, total_loss.
        """
        self.model.train()
        self.teacher.eval()  # Ensure teacher stays in eval mode

        total_cls = 0.0
        total_box = 0.0
        total_kd = 0.0
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            rgb = batch["rgb"].to(self.device)
            nir = batch["nir"].to(self.device)
            bboxes = [b.to(self.device) for b in batch["bboxes"]]
            labels = [l.to(self.device) for l in batch["labels"]]

            optimizer.zero_grad(set_to_none=True)

            try:
                with autocast("cuda", enabled=self.config.amp):
                    # --- Teacher forward (no grad) ---
                    with torch.no_grad():
                        t_out = self.teacher(rgb, nir)
                        # Use distill_backbone_rgb[2:] for S3, S4 — student is
                        # RGB-only, so distill from RGB teacher features.
                        proj_backbone = self.model.kd_proj_backbone(
                            t_out["distill_backbone_rgb"][2:]
                        )
                        proj_fpn = self.model.kd_proj_fpn(
                            t_out["distill_fpn"]
                        )
                        proj_head_cls = self.model.kd_proj_head_cls(
                            t_out["distill_head_cls"]
                        )
                        proj_head_reg = self.model.kd_proj_head_reg(
                            t_out["distill_head_reg"]
                        )

                    # --- Student forward (RGB only) ---
                    s_out = self.model(rgb)

                    # --- Detection loss ---
                    targets = {"bboxes": bboxes, "labels": labels}
                    det_loss, det_dict = self.criterion(s_out["preds"], targets)

                    # --- KD loss ---
                    kd_teacher = {
                        "backbone": proj_backbone,
                        "fpn": proj_fpn,
                        "head_cls": proj_head_cls,
                        "head_reg": proj_head_reg,
                    }
                    kd_student = {
                        "backbone": s_out["distill_backbone"],
                        "fpn": s_out["distill_fpn"],
                        "head_cls": s_out["distill_head_cls"],
                        "head_reg": s_out["distill_head_reg"],
                    }
                    kd_loss, kd_per_level = self.kd_criterion(kd_teacher, kd_student)

                    # --- Total loss ---
                    loss = det_loss + self.config.kd_weight * kd_loss

                # NaN / Inf guard: skip batch if loss explodes
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  [NaN] Skipping batch (loss={loss.item():.2f})")
                    continue

                if self.config.amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                else:
                    loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

                if self.config.amp:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  [OOM] Skipping batch, clearing CUDA cache.")
                    torch.cuda.empty_cache()
                    continue
                raise

            total_cls += det_dict["cls_loss"]
            total_box += det_dict["box_loss"]
            total_kd += kd_loss.item()
            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

        n_batches = max(n_batches, 1)
        return {
            "cls_loss": total_cls / n_batches,
            "box_loss": total_box / n_batches,
            "kd_loss": total_kd / n_batches,
            "total_loss": total_loss / n_batches,
        }
