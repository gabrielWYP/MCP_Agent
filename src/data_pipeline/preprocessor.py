"""
Image preprocessing: load, homography warp, resize, normalize.

Transforms raw RGB+NIR image pairs into MasterModel-ready tensors:
- RGB: (3, H, W) float32, ImageNet-normalized
- NIR: (1, H, W) float32, dataset-stats-normalized
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from src.data_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Preprocesses RGB+NIR image pairs for the MasterModel.

    Loads images, applies homography warp to NIR (or resize fallback),
    resizes to target dimensions, and normalizes per-channel.
    """

    def __init__(self, config: PipelineConfig):
        self.target_size = config.target_size  # (H, W)
        self.rgb_mean = np.array(config.rgb_mean, dtype=np.float32)
        self.rgb_std = np.array(config.rgb_std, dtype=np.float32)
        self.nir_mean = config.nir_stats.mean
        self.nir_std = config.nir_stats.std

        # Load homography matrix once
        self.H: Optional[np.ndarray] = None
        self.homography_applied = False
        if config.homography_path:
            hpath = Path(config.homography_path)
            if hpath.exists():
                try:
                    self.H = np.load(str(hpath))
                    self.homography_applied = True
                    logger.info("Loaded homography from %s", hpath)
                except Exception as e:
                    logger.warning("Failed to load homography %s: %s. Using resize fallback.", hpath, e)
            else:
                logger.warning("Homography file not found: %s. Using resize fallback.", hpath)

    def process_pair(
        self,
        rgb_path: str,
        nir_path: str,
    ) -> Optional[dict]:
        """
        Process a single RGB+NIR pair.

        Args:
            rgb_path: Path to RGB image file
            nir_path: Path to NIR image file

        Returns:
            Dict with keys: rgb (tensor 3,H,W), nir (tensor 1,H,W), metadata
            None if loading fails (corrupt image)
        """
        # Load RGB
        rgb = self._load_rgb(rgb_path)
        if rgb is None:
            logger.error("Failed to load RGB image: %s", rgb_path)
            return None

        # Load NIR
        nir = self._load_nir(nir_path)
        if nir is None:
            logger.error("Failed to load NIR image: %s", nir_path)
            return None

        # Align NIR to RGB spatial coordinates
        h, w = rgb.shape[:2]
        nir_aligned = self._align_nir(nir, (h, w))

        # Resize both to target dimensions
        target_h, target_w = self.target_size
        rgb_resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        nir_resized = cv2.resize(nir_aligned, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Normalize
        rgb_normalized = self._normalize_rgb(rgb_resized)
        nir_normalized = self._normalize_nir(nir_resized)

        # Convert to tensors (H,W,C) -> (C,H,W)
        rgb_tensor = torch.from_numpy(rgb_normalized).permute(2, 0, 1).contiguous()
        nir_tensor = torch.from_numpy(nir_normalized).unsqueeze(0).contiguous()

        metadata = {
            "source_rgb": rgb_path,
            "source_nir": nir_path,
            "original_size": (h, w),
            "target_size": self.target_size,
            "homography_applied": self.homography_applied,
        }

        return {
            "rgb": rgb_tensor,
            "nir": nir_tensor,
            "metadata": metadata,
        }

    def _load_rgb(self, path: str) -> Optional[np.ndarray]:
        """Load RGB image as (H, W, 3) uint8."""
        try:
            img = cv2.imread(path)
            if img is None:
                return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error("Error loading RGB image %s: %s", path, e)
            return None

    def _load_nir(self, path: str) -> Optional[np.ndarray]:
        """Load NIR image as (H, W) uint8 grayscale."""
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            return img
        except Exception as e:
            logger.error("Error loading NIR image %s: %s", path, e)
            return None

    def _align_nir(self, nir: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        """
        Align NIR to RGB coordinates via homography warp or resize fallback.
        """
        h, w = target_hw

        if self.H is not None:
            try:
                # H maps RGB→NIR; warpPerspective needs NIR→RGB, so invert
                warped = cv2.warpPerspective(nir, np.linalg.inv(self.H), (w, h))
                return warped
            except Exception as e:
                logger.warning("Homography warp failed: %s. Using resize fallback.", e)

        # Fallback: simple resize
        if not self.homography_applied:
            logger.warning("No homography available. Using resize fallback for NIR alignment.")
        return cv2.resize(nir, (w, h), interpolation=cv2.INTER_LINEAR)

    def _normalize_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """Normalize RGB with ImageNet stats. Input: (H,W,3) uint8 -> (H,W,3) float32."""
        img = rgb.astype(np.float32) / 255.0
        img = (img - self.rgb_mean) / self.rgb_std
        return img

    def _normalize_nir(self, nir: np.ndarray) -> np.ndarray:
        """Normalize NIR with dataset stats. Input: (H,W) uint8 -> (H,W) float32."""
        img = nir.astype(np.float32) / 255.0
        img = (img - self.nir_mean) / self.nir_std
        return img
