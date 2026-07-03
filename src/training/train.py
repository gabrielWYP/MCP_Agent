"""
CLI entry point for MasterModel training.

Usage:
    python -m src.training.train --config configs/training.yaml
    python -m src.training.train --config configs/training.yaml --override batch_size=4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.config import TrainingConfig
from src.training.dataset import YOLODataset, build_dataloader, collate_fn, build_weighted_sampler
from src.training.loop import Trainer
from src.models.master.master_model import MasterModel
from src.models.student.student_model import StudentModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MasterModel / StudentModel Training")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["master", "student"],
        default=None,
        help="Model type to train: master or student.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config values: key=value pairs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig()

    # Apply overrides
    for override in args.override:
        key, value = override.split("=")
        # Type coercion
        if hasattr(config, key):
            current = getattr(config, key)
            if isinstance(current, bool):
                value = value.lower() in ("true", "1", "yes")
            elif isinstance(current, int):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)
            setattr(config, key, value)

    # CLI --model flag overrides config model_type
    if args.model is not None:
        config.model_type = args.model

    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    print(f"Config: model_type={config.model_type}, backbone={config.backbone_variant}, image_size={config.image_size}")
    if config.model_type == "student":
        print(f"  Single-phase: {config.epochs} epochs, lr={config.lr}")
    else:
        print(f"  Phase 1: {config.epochs_phase1} epochs, lr={config.lr_phase1}")
        print(f"  Phase 2: {config.epochs_phase2} epochs, lr={config.lr_phase2}")
    print(f"  Batch size: {config.batch_size}, AMP: {config.amp}")

    # Create datasets. Do not pass explicit transforms here: YOLODataset's
    # default path keeps spatial augmentations synchronized across RGB, NIR,
    # and bounding boxes.
    train_dataset = YOLODataset(
        rgb_dir=config.rgb_dir,
        nir_dir=config.nir_dir,
        labels_dir=config.labels_dir,
        split="train",
        image_size=config.image_size,
        nir_mean=config.nir_mean,
        nir_std=config.nir_std,
        letterbox_value=config.letterbox_value,
    )

    val_dataset = YOLODataset(
        rgb_dir=config.rgb_dir,
        nir_dir=config.nir_dir,
        labels_dir=config.labels_dir,
        split="val",
        image_size=config.image_size,
        nir_mean=config.nir_mean,
        nir_std=config.nir_std,
        letterbox_value=config.letterbox_value,
    )

    print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val images")

    # DataLoaders (with graceful fallback to num_workers=0 if IPC sockets fail)
    train_sampler = build_weighted_sampler(train_dataset)
    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = build_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Create model
    if config.model_type == "student":
        model = StudentModel(num_classes=config.num_classes)
        params = model.count_parameters()
        print(f"StudentModel parameters: {params['total']:,} total")
    else:
        model = MasterModel(
            num_classes=config.num_classes,
            pretrained_backbone=True,
            backbone_variant=config.backbone_variant,
        )
        params = model.count_parameters()
        print(f"Model parameters: {params['total']:,} total, {params['backbone']:,} backbone")

    # Train
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    results = trainer.fit()

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best mAP@0.5: {results['best_map50']:.4f}")
    print(f"  Checkpoint: {results['checkpoint']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
