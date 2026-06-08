"""
Training configuration for MasterModel fine-tuning.

All hyperparameters for the 2-phase training pipeline are defined here.
Supports loading from YAML files for experiment management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainingConfig:
    """Configuration for MasterModel 2-phase training.

    Attributes:
        rgb_dir: Path to RGB images directory.
        nir_dir: Path to NIR images directory.
        labels_dir: Path to YOLO label files root (contains train/, val/).
        output_dir: Where to save checkpoints and logs.
        backbone_variant: ConvNeXt variant ("tiny" or "small").
        num_classes: Number of detection classes.
        image_size: Input image size (square).
        letterbox_value: Padding pixel value for letterbox.
        batch_size: Training batch size.
        epochs_phase1: Epochs for Phase 1 (frozen backbone).
        epochs_phase2: Epochs for Phase 2 (partial unfreeze).
        lr_phase1: Learning rate for Phase 1.
        lr_phase2: Learning rate for Phase 2.
        weight_decay: AdamW weight decay.
        box_weight: Regression loss weight.
        cls_weight: Classification loss weight.
        class_weights: Per-class BCE weights [sano, danado].
        warmup_epochs: Linear warmup epochs per phase.
        patience: Early stopping patience (epochs without mAP improvement).
        log_interval: Steps between TensorBoard logs.
        save_interval: Epochs between periodic checkpoints.
        amp: Enable automatic mixed precision.
        grad_clip: Gradient clipping max norm.
        num_workers: DataLoader workers.
        nir_mean: NIR channel normalization mean.
        nir_std: NIR channel normalization std.
        seed: Random seed.
    """

    # Paths
    rgb_dir: str = "data/cache/mango/rgb"
    nir_dir: str = "data/cache/mango/nir"
    labels_dir: str = "data/annotations/yolo/labels"
    output_dir: str = "checkpoints/mastermodel"

    # Model
    backbone_variant: str = "tiny"
    num_classes: int = 2

    # Augmentation
    image_size: int = 640
    letterbox_value: int = 114

    # Training
    batch_size: int = 2
    epochs_phase1: int = 50
    epochs_phase2: int = 30
    lr_phase1: float = 1e-3
    lr_phase2: float = 1e-4
    weight_decay: float = 5e-4

    # Loss
    box_weight: float = 7.5
    cls_weight: float = 0.5
    class_weights: list[float] = field(default_factory=lambda: [2.7, 0.5])

    # Scheduler
    warmup_epochs: int = 3
    patience: int = 15

    # Logging
    log_interval: int = 10
    save_interval: int = 5

    # Mixed precision
    amp: bool = True

    # Gradient clipping
    grad_clip: float = 10.0

    # DataLoader
    num_workers: int = 4

    # NIR normalization stats (computed from dataset)
    nir_mean: float = 0.0569
    nir_std: float = 0.0546

    # Model type dispatch
    model_type: str = "master"  # "master" | "student"

    # Single-phase training (student)
    epochs: int = 50
    lr: float = 1e-3

    # Reproducibility
    seed: int = 42

    def __post_init__(self) -> None:
        valid_types = {"master", "student"}
        if self.model_type not in valid_types:
            raise ValueError(
                f"Invalid model_type '{self.model_type}'. "
                f"Must be one of: {valid_types}"
            )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML config file.

        Returns:
            TrainingConfig instance with values from the file.
        """
        path = Path(path)
        with open(path, "r") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                {k: v for k, v in self.__dict__.items()},
                f,
                default_flow_style=False,
            )

    @property
    def total_epochs(self) -> int:
        if self.model_type == "student":
            return self.epochs
        return self.epochs_phase1 + self.epochs_phase2
