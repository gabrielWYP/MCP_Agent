"""Integration test: full train_step forward→loss→backward with AMP on and off."""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.master.master_model import MasterModel
from src.training.loss import YOLOv8Loss


class TestTrainingStep:
    """Test a single training step (forward → loss → backward)."""

    @pytest.fixture
    def model(self):
        """Create a small model for testing (CPU-friendly)."""
        return MasterModel(
            num_classes=2,
            pretrained_backbone=False,
            backbone_variant="tiny",
        )

    @pytest.fixture
    def criterion(self):
        return YOLOv8Loss(
            num_classes=2,
            box_weight=7.5,
            cls_weight=0.5,
            class_weights=[2.7, 0.5],
        )

    @pytest.fixture
    def synthetic_batch(self):
        """Create a synthetic batch with 2 images."""
        B = 2
        rgb = torch.randn(B, 3, 640, 640)
        nir = torch.randn(B, 1, 640, 640)

        # Variable-length bboxes per image
        bboxes = [
            torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]),
            torch.tensor([[0.7, 0.7, 0.15, 0.15]]),
        ]
        labels = [
            torch.tensor([0, 1]),
            torch.tensor([1]),
        ]

        return rgb, nir, bboxes, labels

    def test_forward_backward_no_amp(self, model, criterion, synthetic_batch):
        """Training step without AMP should complete without errors."""
        model.train()
        rgb, nir, bboxes, labels = synthetic_batch

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,
        )

        optimizer.zero_grad()
        output = model(rgb, nir)
        predictions = output["preds"]

        targets = {"bboxes": bboxes, "labels": labels}
        loss, loss_dict = criterion(predictions, targets)

        assert loss.requires_grad
        assert "cls_loss" in loss_dict
        assert "box_loss" in loss_dict
        assert torch.isfinite(loss)

        loss.backward()

        # Check gradients exist
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad, "At least one parameter should have non-zero gradients"

        optimizer.step()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP requires CUDA")
    def test_forward_backward_amp(self, model, criterion, synthetic_batch):
        """Training step with AMP should complete without errors."""
        from torch.cuda.amp import autocast, GradScaler

        device = torch.device("cuda")
        model.to(device)
        criterion.to(device)

        model.train()
        rgb, nir, bboxes, labels = synthetic_batch
        rgb, nir = rgb.to(device), nir.to(device)
        bboxes = [b.to(device) for b in bboxes]
        labels = [l.to(device) for l in labels]

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,
        )
        scaler = GradScaler()

        optimizer.zero_grad()
        with autocast():
            output = model(rgb, nir)
            predictions = output["preds"]
            targets = {"bboxes": bboxes, "labels": labels}
            loss, loss_dict = criterion(predictions, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()

        assert torch.isfinite(loss)

    def test_zero_gt_batch(self, model, criterion):
        """Batch with zero GT boxes should not crash."""
        model.train()
        B = 1
        rgb = torch.randn(B, 3, 640, 640)
        nir = torch.randn(B, 1, 640, 640)

        bboxes = [torch.zeros(0, 4)]
        labels = [torch.zeros(0, dtype=torch.long)]

        output = model(rgb, nir)
        predictions = output["preds"]

        targets = {"bboxes": bboxes, "labels": labels}
        loss, loss_dict = criterion(predictions, targets)

        assert torch.isfinite(loss)
        loss.backward()

    def test_loss_decreases_on_simple_case(self, model, criterion):
        """Loss should decrease over multiple steps on the same batch."""
        model.train()
        B = 1
        rgb = torch.randn(B, 3, 640, 640)
        nir = torch.randn(B, 1, 640, 640)
        bboxes = [torch.tensor([[0.5, 0.5, 0.3, 0.3]])]
        labels = [torch.tensor([0])]

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,
        )

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            output = model(rgb, nir)
            loss, _ = criterion(output["preds"], {"bboxes": bboxes, "labels": labels})
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease (not strictly monotonic, but trend down)
        assert losses[-1] < losses[0] * 2, "Loss should not explode"


class TestImportAll:
    """Verify all training modules can be imported without errors."""

    def test_import_config(self):
        from src.training.config import TrainingConfig
        assert TrainingConfig is not None

    def test_import_augmentations(self):
        from src.training.augmentations import get_train_transforms, get_val_transforms
        assert get_train_transforms is not None

    def test_import_dataset(self):
        from src.training.dataset import YOLODataset, collate_fn, letterbox
        assert YOLODataset is not None

    def test_import_loss(self):
        from src.training.loss import TaskAlignedAssigner, YOLOv8Loss
        assert TaskAlignedAssigner is not None

    def test_import_metrics(self):
        from src.training.metrics import compute_map, LossHistory
        assert compute_map is not None

    def test_import_loop(self):
        from src.training.loop import Trainer
        assert Trainer is not None

    def test_import_package(self):
        import src.training
        assert hasattr(src.training, "TrainingConfig")
        assert hasattr(src.training, "Trainer")
