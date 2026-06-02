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
from src.training.dataset import YOLODataset, collate_fn, build_weighted_sampler
from src.training.augmentations import get_train_transforms, get_val_transforms
from src.training.loop import Trainer
from src.models.master.master_model import MasterModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MasterModel Training")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file.",
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

    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    print(f"Config: backbone={config.backbone_variant}, image_size={config.image_size}")
    print(f"  Phase 1: {config.epochs_phase1} epochs, lr={config.lr_phase1}")
    print(f"  Phase 2: {config.epochs_phase2} epochs, lr={config.lr_phase2}")
    print(f"  Batch size: {config.batch_size}, AMP: {config.amp}")

    # Create datasets
    train_transform = get_train_transforms(config.image_size)
    val_transform = get_val_transforms(config.image_size)

    train_dataset = YOLODataset(
        rgb_dir=config.rgb_dir,
        nir_dir=config.nir_dir,
        labels_dir=config.labels_dir,
        split="train",
        transform=train_transform,
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
        transform=val_transform,
        image_size=config.image_size,
        nir_mean=config.nir_mean,
        nir_std=config.nir_std,
        letterbox_value=config.letterbox_value,
    )

    print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val images")

    # DataLoaders
    train_sampler = build_weighted_sampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Create model
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
