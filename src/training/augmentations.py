"""
Bbox-aware augmentation pipelines for YOLO-style detection training.

Train transforms include aggressive spatial and photometric augmentations
for small-dataset fine-tuning. Val transforms only resize.

Uses albumentations with BBoxParams(format='yolo') so bounding box
coordinates are automatically adjusted during spatial transforms.
"""

from __future__ import annotations

import albumentations as A


def get_train_transforms(image_size: int = 640) -> A.Compose:
    """Build training augmentation pipeline with bbox-aware transforms.

    Designed for small datasets (~24 images). Applies aggressive spatial
    and photometric augmentations to maximize sample diversity.

    Args:
        image_size: Target image size (not used here — letterbox handles resize).

    Returns:
        albumentations Compose with bbox params configured for YOLO format.
    """
    return A.Compose(
        [
            # --- Spatial transforms (dimension-preserving, bbox-aware) ---
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=20, p=0.4, border_mode=0),
            A.Perspective(
                scale=(0.02, 0.05),
                keep_size=True,
                p=0.2,
            ),
            A.Transpose(p=0.1),

            # --- Photometric transforms (RGB) ---
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.4,
            ),
            A.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.05,
                p=0.3,
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
            A.ISONoise(p=0.15),

            # --- Dropout / occlusions ---
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill_value=114,
                p=0.2,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.15,
            min_area=2.0,
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
