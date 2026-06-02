"""
Two-phase training loop for MasterModel fine-tuning.

Phase 1: Freeze backbone (stages 1-4), train fusion + neck + head.
Phase 2: Unfreeze stages 3-4, train with lower LR.

Features:
    - AMP (automatic mixed precision)
    - Gradient clipping (max_norm=10.0)
    - CosineAnnealingWarmRestarts with linear warmup
    - Early stopping on val mAP@0.5 (patience=15)
    - Best checkpoint + periodic checkpointing
    - TensorBoard logging
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .loss import YOLOv8Loss
from .metrics import compute_map, LossHistory


class Trainer:
    """Two-phase training loop for MasterModel.

    Args:
        model: MasterModel instance.
        config: TrainingConfig with all hyperparameters.
        train_loader: Training data loader.
        val_loader: Validation data loader.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Loss
        self.criterion = YOLOv8Loss(
            num_classes=config.num_classes,
            box_weight=config.box_weight,
            cls_weight=config.cls_weight,
            class_weights=config.class_weights,
        ).to(self.device)

        # AMP
        self.scaler = GradScaler("cuda", enabled=config.amp)

        # Output directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.loss_history = LossHistory()
        self.best_map50 = 0.0
        self.patience_counter = 0
        self.global_step = 0

        # TensorBoard
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))
        except ImportError:
            print("[Trainer] TensorBoard not available, skipping logging.")

    def fit(self) -> dict:
        """Run both training phases sequentially.

        Returns:
            Dict with final metrics and checkpoint paths.
        """
        print(f"\n{'='*60}")
        print(f"Training MasterModel on {self.device}")
        print(f"Phase 1: {self.config.epochs_phase1} epochs (frozen backbone)")
        print(f"Phase 2: {self.config.epochs_phase2} epochs (unfreeze stages 3-4)")
        print(f"{'='*60}\n")

        # Phase 1
        if self.config.epochs_phase1 > 0:
            self._train_phase(
                phase=1,
                epochs=self.config.epochs_phase1,
                lr=self.config.lr_phase1,
                freeze_stages=4,
            )

        # Phase 2
        if self.config.epochs_phase2 > 0:
            self._train_phase(
                phase=2,
                epochs=self.config.epochs_phase2,
                lr=self.config.lr_phase2,
                freeze_stages=2,
                unfreeze_stages=[2, 3],
            )

        if self.writer:
            self.writer.close()

        return {
            "best_map50": self.best_map50,
            "loss_history": {
                "cls_loss": self.loss_history.cls_loss,
                "box_loss": self.loss_history.box_loss,
                "total_loss": self.loss_history.total_loss,
            },
            "checkpoint": str(self.output_dir / "best_model.pt"),
        }

    def _train_phase(
        self,
        phase: int,
        epochs: int,
        lr: float,
        freeze_stages: int = 0,
        unfreeze_stages: list[int] | None = None,
    ):
        """Run training for one phase.

        Args:
            phase: Phase number (1 or 2).
            epochs: Number of epochs.
            lr: Learning rate for this phase.
            freeze_stages: Number of backbone stages to freeze.
            unfreeze_stages: Specific stages to unfreeze (Phase 2).
        """
        print(f"\n{'─'*50}")
        print(f"Phase {phase}: {epochs} epochs | LR={lr}")
        print(f"{'─'*50}")

        # Freeze/unfreeze backbone
        self.model.freeze_backbone(freeze_stages=freeze_stages)
        if unfreeze_stages is not None:
            self.model.unfreeze_backbone_stages(unfreeze_stages)

        # Count trainable params
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,}")

        # Optimizer (only trainable params)
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler: warmup + cosine annealing
        warmup_epochs = self.config.warmup_epochs
        total_epochs = epochs

        def warmup_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0

        warmup_scheduler = LambdaLR(optimizer, warmup_lambda)
        cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, total_epochs - warmup_epochs), T_mult=2
        )

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Train
            train_metrics = self._train_epoch(optimizer, epoch, phase)

            # Validate
            val_metrics = self._validate(epoch, phase)

            elapsed = time.time() - t0

            # Update scheduler
            if epoch <= warmup_epochs:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()

            # Log
            map50 = val_metrics.get("map50", 0.0)
            print(
                f"  [P{phase}] Epoch {epoch:02d}/{epochs} | "
                f"loss={train_metrics['total_loss']:.4f} "
                f"(cls={train_metrics['cls_loss']:.4f}, box={train_metrics['box_loss']:.4f}) | "
                f"mAP@0.5={map50:.4f} | {elapsed:.1f}s"
            )

            # Record loss history
            self.loss_history.update(
                cls=train_metrics["cls_loss"],
                box=train_metrics["box_loss"],
                total=train_metrics["total_loss"],
            )

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar(f"Phase{phase}/train/total_loss", train_metrics["total_loss"], epoch)
                self.writer.add_scalar(f"Phase{phase}/train/cls_loss", train_metrics["cls_loss"], epoch)
                self.writer.add_scalar(f"Phase{phase}/train/box_loss", train_metrics["box_loss"], epoch)
                self.writer.add_scalar(f"Phase{phase}/val/map50", map50, epoch)
                self.writer.add_scalar(f"Phase{phase}/val/map_50_95", val_metrics.get("map_50_95", 0.0), epoch)
                self.writer.add_scalar(f"Phase{phase}/lr", optimizer.param_groups[0]["lr"], epoch)

                # Per-class AP
                for cls_id, ap in val_metrics.get("per_class_ap_50", {}).items():
                    self.writer.add_scalar(f"Phase{phase}/val/ap50_class{cls_id}", ap, epoch)

            # Checkpoint: best mAP (use >= to save on first epoch even if mAP=0)
            if map50 >= self.best_map50:
                self.best_map50 = map50
                self.patience_counter = 0
                self._save_checkpoint(epoch, phase, val_metrics, "best_model.pt")
                print(f"    ✓ New best mAP@0.5: {map50:.4f}")
            else:
                self.patience_counter += 1

            # Periodic checkpoint
            if epoch % self.config.save_interval == 0:
                self._save_checkpoint(epoch, phase, val_metrics, f"checkpoint_epoch{epoch}.pt")

            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"  Early stopping at epoch {epoch} (patience={self.config.patience})")
                break

    def _train_epoch(self, optimizer: AdamW, epoch: int, phase: int) -> dict:
        """Run one training epoch.

        Returns:
            Dict with avg cls_loss, box_loss, total_loss.
        """
        self.model.train()
        total_cls = 0.0
        total_box = 0.0
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
                    output = self.model(rgb, nir)
                    predictions = output["preds"]

                    targets = {"bboxes": bboxes, "labels": labels}
                    loss, loss_dict = self.criterion(predictions, targets)

                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

                self.scaler.step(optimizer)
                self.scaler.update()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  [OOM] Skipping batch, clearing CUDA cache.")
                    torch.cuda.empty_cache()
                    continue
                raise

            total_cls += loss_dict["cls_loss"]
            total_box += loss_dict["box_loss"]
            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

        n_batches = max(n_batches, 1)
        return {
            "cls_loss": total_cls / n_batches,
            "box_loss": total_box / n_batches,
            "total_loss": total_loss / n_batches,
        }

    @torch.no_grad()
    def _validate(self, epoch: int, phase: int) -> dict:
        """Run validation and compute mAP metrics.

        Returns:
            Dict with map50, map_50_95, per_class_ap_50, per_class_ap_50_95.
        """
        self.model.eval()

        all_pred_boxes = []
        all_pred_scores = []
        all_pred_labels = []
        all_gt_boxes = []
        all_gt_labels = []

        for batch in self.val_loader:
            rgb = batch["rgb"].to(self.device)
            nir = batch["nir"].to(self.device)

            with autocast("cuda", enabled=self.config.amp):
                output = self.model(rgb, nir)

            # Decode predictions for mAP computation
            preds = output["preds"]  # list of (B, nc+4, H, W)
            cls_preds = output["cls_preds"]  # list of (B, nc, H, W)

            B = rgb.shape[0]
            for b in range(B):
                # Simple decoding: take max-score class per spatial location
                # For mAP, we need per-image predictions
                pred_boxes_b, pred_scores_b, pred_labels_b = self._decode_predictions(
                    preds, cls_preds, b
                )
                all_pred_boxes.append(pred_boxes_b)
                all_pred_scores.append(pred_scores_b)
                all_pred_labels.append(pred_labels_b)

                # GT
                gt_bboxes = batch["bboxes"][b]
                gt_labels_batch = batch["labels"][b]
                all_gt_boxes.append(gt_bboxes)
                all_gt_labels.append(gt_labels_batch)

        # Compute mAP
        metrics = compute_map(
            pred_boxes=all_pred_boxes,
            pred_scores=all_pred_scores,
            pred_labels=all_pred_labels,
            gt_boxes=all_gt_boxes,
            gt_labels=all_gt_labels,
            num_classes=self.config.num_classes,
        )

        return metrics

    def _decode_predictions(
        self,
        preds: list[torch.Tensor],
        cls_preds: list[torch.Tensor],
        batch_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode model predictions for a single image in the batch.

        Returns:
            pred_boxes: (P, 4) cxcywh normalized.
            pred_scores: (P,) confidence scores.
            pred_labels: (P,) class IDs.
        """
        all_boxes = []
        all_scores = []
        all_labels = []

        strides = [8, 16, 32]

        for level, (pred, cls_pred, stride) in enumerate(zip(preds, cls_preds, strides)):
            # pred: (B, nc+4, H, W), cls_pred: (B, nc, H, W)
            H, W = pred.shape[2], pred.shape[3]
            nc = self.config.num_classes

            # Extract for this batch item
            p = pred[batch_idx]  # (nc+4, H, W)
            c = cls_pred[batch_idx]  # (nc, H, W)

            # Class scores (sigmoid)
            scores = c.sigmoid()  # (nc, H, W)
            max_scores, max_labels = scores.max(dim=0)  # (H, W)

            # Bbox deltas
            reg = p[nc:]  # (4, H, W)

            # Filter by confidence threshold
            threshold = 0.25
            mask = max_scores > threshold

            if not mask.any():
                continue

            # Get positions
            ys, xs = mask.nonzero(as_tuple=True)

            # Decode bboxes
            dx = reg[0, ys, xs]
            dy = reg[1, ys, xs]
            w = reg[2, ys, xs].exp()
            h = reg[3, ys, xs].exp()

            # Anchor centers
            anchor_x = (xs.float() + 0.5) * stride
            anchor_y = (ys.float() + 0.5) * stride

            cx = anchor_x + dx
            cy = anchor_y + dy

            # Normalize to [0, 1] (assuming image_size)
            img_size = self.config.image_size
            boxes = torch.stack([
                cx / img_size,
                cy / img_size,
                w / img_size,
                h / img_size,
            ], dim=1)  # (K, 4) cxcywh normalized

            all_boxes.append(boxes)
            all_scores.append(max_scores[mask])
            all_labels.append(max_labels[mask])

        if all_boxes:
            return (
                torch.cat(all_boxes).cpu(),
                torch.cat(all_scores).cpu(),
                torch.cat(all_labels).cpu(),
            )
        else:
            return (
                torch.zeros(0, 4),
                torch.zeros(0),
                torch.zeros(0, dtype=torch.long),
            )

    def _save_checkpoint(
        self,
        epoch: int,
        phase: int,
        metrics: dict,
        filename: str,
    ):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save({
            "epoch": epoch,
            "phase": phase,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics,
            "best_map50": self.best_map50,
            "config": self.config.__dict__,
        }, path)
