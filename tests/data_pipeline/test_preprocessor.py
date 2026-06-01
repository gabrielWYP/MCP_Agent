"""Tests for Preprocessor: loading, homography warp, resize, normalize."""

import cv2
import numpy as np
import pytest
import torch

from src.data_pipeline.config import PipelineConfig, NIRStats
from src.data_pipeline.preprocessor import Preprocessor


def _create_test_image(path, width=100, height=80, channels=3, value=128):
    """Create a synthetic test image."""
    if channels == 3:
        img = np.full((height, width, channels), value, dtype=np.uint8)
    else:
        img = np.full((height, width), value, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


class TestPreprocessorValidPair:
    """Valid pair produces correct shapes and types."""

    def test_correct_shapes(self, tmp_path):
        rgb_path = _create_test_image(tmp_path / "rgb.jpg", 100, 80, 3)
        nir_path = _create_test_image(tmp_path / "nir.jpg", 100, 80, 1)

        config = PipelineConfig(target_size=(640, 640))
        preprocessor = Preprocessor(config)
        result = preprocessor.process_pair(str(rgb_path), str(nir_path))

        assert result is not None
        assert result["rgb"].shape == (3, 640, 640)
        assert result["nir"].shape == (1, 640, 640)
        assert result["rgb"].dtype == torch.float32
        assert result["nir"].dtype == torch.float32

    def test_configurable_target_size(self, tmp_path):
        rgb_path = _create_test_image(tmp_path / "rgb.jpg", 100, 80, 3)
        nir_path = _create_test_image(tmp_path / "nir.jpg", 100, 80, 1)

        config = PipelineConfig(target_size=(416, 416))
        preprocessor = Preprocessor(config)
        result = preprocessor.process_pair(str(rgb_path), str(nir_path))

        assert result["rgb"].shape == (3, 416, 416)
        assert result["nir"].shape == (1, 416, 416)

    def test_metadata_includes_homography_flag(self, tmp_path):
        rgb_path = _create_test_image(tmp_path / "rgb.jpg", 100, 80, 3)
        nir_path = _create_test_image(tmp_path / "nir.jpg", 100, 80, 1)

        config = PipelineConfig(target_size=(224, 224))
        preprocessor = Preprocessor(config)
        result = preprocessor.process_pair(str(rgb_path), str(nir_path))

        assert "homography_applied" in result["metadata"]
        assert result["metadata"]["homography_applied"] is False


class TestPreprocessorCorruptImage:
    """Corrupt/zero-byte images are logged and skipped."""

    def test_corrupt_nir_returns_none(self, tmp_path):
        rgb_path = _create_test_image(tmp_path / "rgb.jpg", 100, 80, 3)
        # Create zero-byte NIR file
        nir_path = tmp_path / "nir_corrupt.jpg"
        nir_path.write_bytes(b"")

        config = PipelineConfig(target_size=(224, 224))
        preprocessor = Preprocessor(config)
        result = preprocessor.process_pair(str(rgb_path), str(nir_path))

        assert result is None

    def test_corrupt_rgb_returns_none(self, tmp_path):
        rgb_path = tmp_path / "rgb_corrupt.jpg"
        rgb_path.write_bytes(b"not an image")
        nir_path = _create_test_image(tmp_path / "nir.jpg", 100, 80, 1)

        config = PipelineConfig(target_size=(224, 224))
        preprocessor = Preprocessor(config)
        result = preprocessor.process_pair(str(rgb_path), str(nir_path))

        assert result is None

    def test_nonexistent_file_returns_none(self, tmp_path):
        config = PipelineConfig(target_size=(224, 224))
        preprocessor = Preprocessor(config)
        result = preprocessor.process_pair("/nonexistent/rgb.jpg", "/nonexistent/nir.jpg")

        assert result is None


class TestPreprocessorHomography:
    """Homography warp and fallback behavior."""

    def test_missing_homography_uses_resize(self, tmp_path):
        rgb_path = _create_test_image(tmp_path / "rgb.jpg", 100, 80, 3)
        nir_path = _create_test_image(tmp_path / "nir.jpg", 100, 80, 1)

        config = PipelineConfig(
            target_size=(224, 224),
            homography_path=None,
        )
        preprocessor = Preprocessor(config)
        result = preprocessor.process_pair(str(rgb_path), str(nir_path))

        assert result is not None
        assert result["metadata"]["homography_applied"] is False

    def test_nonexistent_homography_file_uses_resize(self, tmp_path, caplog):
        rgb_path = _create_test_image(tmp_path / "rgb.jpg", 100, 80, 3)
        nir_path = _create_test_image(tmp_path / "nir.jpg", 100, 80, 1)

        config = PipelineConfig(
            target_size=(224, 224),
            homography_path="/nonexistent/homography.npy",
        )

        import logging
        with caplog.at_level(logging.WARNING):
            preprocessor = Preprocessor(config)

        assert preprocessor.H is None
        assert preprocessor.homography_applied is False

        result = preprocessor.process_pair(str(rgb_path), str(nir_path))
        assert result is not None

    def test_valid_homography_applied(self, tmp_path):
        rgb_path = _create_test_image(tmp_path / "rgb.jpg", 100, 80, 3)
        nir_path = _create_test_image(tmp_path / "nir.jpg", 100, 80, 1)

        # Create identity homography matrix
        H = np.eye(3, dtype=np.float64)
        h_path = tmp_path / "homography.npy"
        np.save(str(h_path), H)

        config = PipelineConfig(
            target_size=(224, 224),
            homography_path=str(h_path),
        )
        preprocessor = Preprocessor(config)

        assert preprocessor.H is not None
        assert preprocessor.homography_applied is True

        result = preprocessor.process_pair(str(rgb_path), str(nir_path))
        assert result is not None
        assert result["metadata"]["homography_applied"] is True
