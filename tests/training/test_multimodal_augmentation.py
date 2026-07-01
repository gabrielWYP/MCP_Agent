"""Tests for aligned RGB/NIR augmentation in YOLODataset."""

from pathlib import Path
import sys

import albumentations as A
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.dataset import YOLODataset


def test_default_transform_path_flips_rgb_nir_and_bboxes_together():
    """RGB, NIR, and boxes must share the same sampled spatial transform."""
    dataset = YOLODataset.__new__(YOLODataset)
    dataset.split = "train"
    dataset.spatial_transform = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.0,
            min_area=0.0,
        ),
        additional_targets={"nir": "image"},
    )
    dataset.rgb_photometric_transform = None
    dataset.transform = None

    nir = np.zeros((4, 4), dtype=np.uint8)
    nir[:, 0] = 10
    nir[:, 3] = 200

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:, :, 0] = nir

    bboxes = np.array([[0.25, 0.5, 0.25, 0.5]], dtype=np.float32)
    labels = np.array([1], dtype=np.int64)

    rgb_aug, nir_aug, bboxes_aug, labels_aug = dataset._apply_default_transforms(
        rgb_lb=rgb,
        nir_lb=nir,
        bboxes=bboxes,
        labels=labels,
    )

    assert np.array_equal(rgb_aug[:, :, 0], nir_aug)
    assert bboxes_aug[0, 0] == pytest.approx(0.75)
    assert bboxes_aug[0, 1] == pytest.approx(0.5)
    assert labels_aug.tolist() == [1]
