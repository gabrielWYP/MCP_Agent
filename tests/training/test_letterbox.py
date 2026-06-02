"""Unit test: letterbox function — output shape and bbox rescaling correctness."""

import numpy as np
import pytest

from src.training.dataset import letterbox


class TestLetterbox:
    """Test letterbox resize + padding."""

    def test_square_image(self):
        """Square image should scale to target_size with no padding."""
        img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        padded, scale, pad_x, pad_y = letterbox(img, target_size=640, pad_value=114)

        assert padded.shape == (640, 640, 3)
        assert scale == pytest.approx(2.0)
        assert pad_x == 0
        assert pad_y == 0

    def test_landscape_image(self):
        """Landscape image: longer side (width) scales to 640, height padded."""
        img = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        padded, scale, pad_x, pad_y = letterbox(img, target_size=640, pad_value=114)

        assert padded.shape == (640, 640, 3)
        assert scale == pytest.approx(1.0)
        assert pad_x == 0
        # Height: 360 * 1.0 = 360, pad = (640 - 360) / 2 = 140
        assert pad_y == 140

    def test_portrait_image(self):
        """Portrait image: longer side (height) scales to 640, width padded."""
        img = np.random.randint(0, 255, (1280, 720, 3), dtype=np.uint8)
        padded, scale, pad_x, pad_y = letterbox(img, target_size=640, pad_value=114)

        assert padded.shape == (640, 640, 3)
        assert scale == pytest.approx(0.5)
        # Width: 720 * 0.5 = 360, pad = (640 - 360) / 2 = 140
        assert pad_x == 140
        assert pad_y == 0

    def test_grayscale_image(self):
        """Grayscale (single-channel) image should work."""
        img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        padded, scale, pad_x, pad_y = letterbox(img, target_size=640, pad_value=114)

        assert padded.shape == (640, 640)
        assert scale == pytest.approx(1.0)

    def test_pad_value(self):
        """Padding pixels should have the specified value."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        padded, scale, pad_x, pad_y = letterbox(img, target_size=640, pad_value=114)

        # Check a padding pixel (top-left corner should be padding)
        assert padded[0, 0, 0] == 114

    def test_bbox_rescaling(self):
        """Test that bbox rescaling produces correct coordinates.

        Original image: 1280x720, bbox at center (0.5, 0.5, 0.2, 0.2).
        After letterbox to 640: scale=0.5, pad_x=140, pad_y=0.
        Expected: cx = (0.5 * 1280 * 0.5 + 140) / 640 = (320 + 140) / 640 = 0.71875
                  cy = (0.5 * 720 * 0.5 + 0) / 640 = 180 / 640 = 0.28125
                  w = (0.2 * 1280 * 0.5) / 640 = 128 / 640 = 0.2
                  h = (0.2 * 720 * 0.5) / 640 = 72 / 640 = 0.1125
        """
        from src.training.dataset import YOLODataset

        bboxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        orig_w, orig_h = 1280, 720
        scale = 0.5
        pad_x, pad_y = 140, 0

        # Use the dataset's internal method
        dataset = YOLODataset.__new__(YOLODataset)
        dataset.image_size = 640

        result = dataset._rescale_bboxes(bboxes, orig_w, orig_h, scale, pad_x, pad_y)

        assert result[0, 0] == pytest.approx(0.71875, abs=0.01)
        assert result[0, 1] == pytest.approx(0.28125, abs=0.01)
        assert result[0, 2] == pytest.approx(0.2, abs=0.01)
        assert result[0, 3] == pytest.approx(0.1125, abs=0.01)
