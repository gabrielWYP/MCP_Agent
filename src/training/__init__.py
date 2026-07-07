"""
Training package for MasterModel — multimodal mango damage detection.

Provides:
    - TrainingConfig: dataclass with all hyperparameters
    - YOLODataset: RGB+NIR pair loader with YOLO labels
    - YOLOv8Loss: TAL-based detection loss
    - Trainer: 2-phase fine-tuning loop
    - compute_map: mAP evaluation metrics
"""

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

_LAZY_IMPORTS = {
    "TrainingConfig": ("src.training.config", "TrainingConfig"),
    "get_train_transforms": ("src.training.augmentations", "get_train_transforms"),
    "get_val_transforms": ("src.training.augmentations", "get_val_transforms"),
    "YOLODataset": ("src.training.dataset", "YOLODataset"),
    "collate_fn": ("src.training.dataset", "collate_fn"),
    "build_weighted_sampler": ("src.training.dataset", "build_weighted_sampler"),
    "letterbox": ("src.training.dataset", "letterbox"),
    "TaskAlignedAssigner": ("src.training.loss", "TaskAlignedAssigner"),
    "YOLOv8Loss": ("src.training.loss", "YOLOv8Loss"),
    "compute_map": ("src.training.metrics", "compute_map"),
    "LossHistory": ("src.training.metrics", "LossHistory"),
    "Trainer": ("src.training.loop", "Trainer"),
}


def __getattr__(name: str):
    """Lazily import training symbols so lightweight helpers do not import torch."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'src.training' has no attribute '{name}'")

    import importlib

    module_name, symbol_name = _LAZY_IMPORTS[name]
    module = importlib.import_module(module_name)
    symbol = getattr(module, symbol_name)
    globals()[name] = symbol
    return symbol
