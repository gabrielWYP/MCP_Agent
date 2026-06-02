"""
Training package for MasterModel — multimodal mango damage detection.

Provides:
    - TrainingConfig: dataclass with all hyperparameters
    - YOLODataset: RGB+NIR pair loader with YOLO labels
    - YOLOv8Loss: TAL-based detection loss
    - Trainer: 2-phase fine-tuning loop
    - compute_map: mAP evaluation metrics
"""

from .config import TrainingConfig
from .augmentations import get_train_transforms, get_val_transforms
from .dataset import YOLODataset, collate_fn, build_weighted_sampler, letterbox
from .loss import TaskAlignedAssigner, YOLOv8Loss
from .metrics import compute_map, LossHistory
from .loop import Trainer

__all__ = [
    "TrainingConfig",
    "get_train_transforms",
    "get_val_transforms",
    "YOLODataset",
    "collate_fn",
    "build_weighted_sampler",
    "letterbox",
    "TaskAlignedAssigner",
    "YOLOv8Loss",
    "compute_map",
    "LossHistory",
    "Trainer",
]
