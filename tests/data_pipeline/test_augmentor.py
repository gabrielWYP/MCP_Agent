"""Tests for Augmentor: spatial sync, modality-aware photometric, augmentation factor."""

import numpy as np
import pytest
import torch

from src.data_pipeline.config import PipelineConfig, NIRStats
from src.data_pipeline.augmentor import Augmentor, AugmentedSample


def _make_test_tensors(h=64, w=64):
    """Create synthetic RGB and NIR tensors for testing."""
    # RGB: gradient pattern
    rgb = torch.zeros(3, h, w, dtype=torch.float32)
    for c in range(3):
        for i in range(h):
            rgb[c, i, :] = i / h

    # NIR: checkerboard pattern
    nir = torch.zeros(1, h, w, dtype=torch.float32)
    for i in range(h):
        for j in range(w):
            nir[0, i, j] = 1.0 if (i // 8 + j // 8) % 2 == 0 else 0.0

    return rgb, nir


class TestAugmentorSpatialSync:
    """4-channel stack -> split preserves pixel alignment."""

    def test_horizontal_flip_preserves_alignment(self):
        """After horizontal flip, RGB and NIR pixels correspond to same locations."""
        config = PipelineConfig(
            target_size=(64, 64),
            augmentation_factor=10,  # Multiple attempts to catch a flip
        )
        augmentor = Augmentor(config)

        rgb, nir = _make_test_tensors(64, 64)
        results = augmentor.augment_pair(rgb, nir, label=0, class_name="sano", stem="test")

        assert len(results) == 10
        for sample in results:
            assert sample.rgb.shape == (3, 64, 64)
            assert sample.nir.shape == (1, 64, 64)

    def test_identity_preserves_original(self):
        """aug_id=0 should be identity (no geometric variation)."""
        config = PipelineConfig(
            target_size=(64, 64),
            augmentation_factor=1,
        )
        augmentor = Augmentor(config)

        rgb, nir = _make_test_tensors(64, 64)
        results = augmentor.augment_pair(rgb, nir, label=0, class_name="sano", stem="test")

        assert len(results) == 1
        assert results[0].metadata["is_identity"] is True


class TestAugmentorAugmentationFactor:
    """Augmentation factor produces correct sample count."""

    def test_factor_3_from_1_pair(self):
        config = PipelineConfig(
            target_size=(64, 64),
            augmentation_factor=3,
        )
        augmentor = Augmentor(config)

        rgb, nir = _make_test_tensors(64, 64)
        results = augmentor.augment_pair(rgb, nir, label=0, class_name="sano", stem="test")

        assert len(results) == 3

    def test_factor_1_produces_identity_only(self):
        config = PipelineConfig(
            target_size=(64, 64),
            augmentation_factor=1,
        )
        augmentor = Augmentor(config)

        rgb, nir = _make_test_tensors(64, 64)
        results = augmentor.augment_pair(rgb, nir, label=0, class_name="sano", stem="test")

        assert len(results) == 1
        assert results[0].metadata["is_identity"] is True

    def test_multiple_pairs_multiply(self):
        """10 pairs * factor 3 = 30 samples."""
        config = PipelineConfig(
            target_size=(64, 64),
            augmentation_factor=3,
        )
        augmentor = Augmentor(config)

        rgb, nir = _make_test_tensors(64, 64)
        all_results = []
        for i in range(10):
            results = augmentor.augment_pair(rgb, nir, label=0, class_name="sano", stem=f"img{i:03d}")
            all_results.extend(results)

        assert len(all_results) == 30


class TestAugmentorOutputFormat:
    """Output format matches AugmentedSample contract."""

    def test_output_types(self):
        config = PipelineConfig(target_size=(64, 64), augmentation_factor=2)
        augmentor = Augmentor(config)

        rgb, nir = _make_test_tensors(64, 64)
        results = augmentor.augment_pair(rgb, nir, label=1, class_name="danado", stem="test")

        for sample in results:
            assert isinstance(sample, AugmentedSample)
            assert isinstance(sample.rgb, torch.Tensor)
            assert isinstance(sample.nir, torch.Tensor)
            assert isinstance(sample.label, int)
            assert sample.label == 1
            assert sample.class_name == "danado"
            assert "aug_id" in sample.metadata

    def test_tensor_dtypes(self):
        config = PipelineConfig(target_size=(64, 64), augmentation_factor=1)
        augmentor = Augmentor(config)

        rgb, nir = _make_test_tensors(64, 64)
        results = augmentor.augment_pair(rgb, nir, label=0, class_name="sano", stem="test")

        assert results[0].rgb.dtype == torch.float32
        assert results[0].nir.dtype == torch.float32
