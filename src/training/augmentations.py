"""
Bbox-aware augmentation pipelines for YOLO-style detection training.

Train transforms include spatial and photometric augmentations.
Val transforms only resize (letterbox is handled in the dataset).

Uses albumentations with BBoxParams(format='yolo') so bounding box
coordinates are automatically adjusted during spatial transforms.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 640) -> A.Compose:
    """Build training augmentation pipeline with bbox-aware transforms.

    Args:
        image_size: Target image size (not used here — letterbox handles resize).

    Returns:
        albumentations Compose with bbox params configured for YOLO format.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3, border_mode=0),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.3
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.2
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.1,
            min_area=1.0,
        ),
    )


def get_val_transforms(image_size: int = 640) -> A.Compose:
    """Build validation transform pipeline (no augmentation).

    Args:
        image_size: Target image size (not used here — letterbox handles resize).

    Returns:
        albumentations Compose with bbox params configured for YOLO format.
    """
    return A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.1,
            min_area=1.0,
        ),
    )
