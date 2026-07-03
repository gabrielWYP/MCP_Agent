"""
CLI entry point for Knowledge Distillation training.

Usage:
    python -m src.training.kd_train --config configs/kd_training.yaml
    python -m src.training.kd_train --config configs/kd_training.yaml --override kd_weight=0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.kd_config import KDConfig
from src.training.dataset import YOLODataset, build_dataloader, collate_fn, build_weighted_sampler
from src.training.kd_trainer import KDTrainer
from src.models.student.student_model import StudentModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kd_training.yaml",
        help="Path to KD YAML config file.",
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
    config = KDConfig.from_yaml(args.config)

    # Apply overrides
    for override in args.override:
        key, value = override.split("=")
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

    print(f"KD Training Config:")
    print(f"  Teacher: {config.teacher_checkpoint}")
    print(f"  Student: num_classes={config.num_classes}, backbone={config.backbone_variant}")
    print(f"  Epochs: {config.epochs}, patience: {config.patience}")
    print(f"  Batch size: {config.batch_size}, AMP: {config.amp}")
    print(f"  KD weight: {config.kd_weight}, temperature: {config.kd_temperature}")
    print(f"  Distill levels: {config.distill_levels}")

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

    # Create student model
    model = StudentModel(num_classes=config.num_classes)
    params = model.count_parameters()
    print(f"StudentModel parameters: {params['total']:,} total")

    # KD Trainer (attaches projections + loads teacher)
    trainer = KDTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    results = trainer.fit()

    print(f"\n{'='*60}")
    print(f"KD Training complete!")
    print(f"  Best mAP@0.5: {results['best_map50']:.4f}")
    print(f"  Checkpoint: {results['checkpoint']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
