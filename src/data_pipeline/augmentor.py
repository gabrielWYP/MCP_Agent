"""
Spatially-synchronized RGB+NIR augmentation.

Strategy: stack RGB+NIR into 4-channel (H,W,4) -> apply spatial transforms
via albumentations -> split back to RGB(3,H,W) + NIR(1,H,W) -> apply
modality-aware photometric transforms independently.
"""

import logging
from typing import NamedTuple

import albumentations as A
import numpy as np
import torch

from src.data_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class AugmentedSample(NamedTuple):
    """A single augmented RGB+NIR sample."""
    rgb: torch.Tensor    # (3, H, W) float32
    nir: torch.Tensor    # (1, H, W) float32
    label: int
    class_name: str
    metadata: dict


class Augmentor:
    """
    Applies spatially-synchronized augmentations to RGB+NIR pairs.

    Uses 4-channel temp stack for spatial transforms (guaranteeing sub-pixel
    alignment), then splits and applies modality-aware photometric transforms.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.augmentation_factor = config.augmentation_factor
        self.spatial_intensity = config.spatial_intensity
        self.photometric_intensity = config.photometric_intensity
        self.target_size = config.target_size

        # Build transform pipelines
        self.spatial_transform = self._build_spatial_transform()
        self.rgb_photometric = self._build_rgb_photometric()
        self.nir_photometric = self._build_nir_photometric()

    def _build_spatial_transform(self) -> A.Compose:
        """Spatial transforms applied to the 4-channel stack."""
        si = self.spatial_intensity
        h, w = self.target_size

        return A.Compose([
            A.RandomResizedCrop(
                size=(h, w),
                scale=(0.7, 1.0),
                ratio=(0.75, 1.33),
                p=0.5,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05 * si,
                scale_limit=0.1 * si,
                rotate_limit=int(15 * si),
                p=0.4,
            ),
        ])

    def _build_rgb_photometric(self) -> A.Compose:
        """Photometric transforms for RGB channels only."""
        pi = self.photometric_intensity

        return A.Compose([
            A.ColorJitter(
                brightness=pi,
                contrast=pi,
                saturation=pi * 0.7,
                hue=pi * 0.15,
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(p=0.2),
        ])

    def _build_nir_photometric(self) -> A.Compose:
        """Photometric transforms for NIR channel only (intensity, no color)."""
        pi = self.photometric_intensity

        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=pi,
                contrast_limit=pi,
                p=0.5,
            ),
            A.GaussNoise(p=0.2),
        ])

    def augment_pair(
        self,
        rgb_tensor: torch.Tensor,
        nir_tensor: torch.Tensor,
        label: int,
        class_name: str,
        stem: str = "",
    ) -> list[AugmentedSample]:
        """
        Augment a single RGB+NIR pair.

        Args:
            rgb_tensor: (3, H, W) float32 normalized RGB
            nir_tensor: (1, H, W) float32 normalized NIR
            label: Class label index
            class_name: Class name string
            stem: Source filename stem

        Returns:
            List of AugmentedSample (length = augmentation_factor)
        """
        results = []

        # Convert tensors back to numpy for albumentations
        # (C,H,W) -> (H,W,C), denormalize to uint8 for albumentations
        rgb_np = self._tensor_to_numpy_rgb(rgb_tensor)
        nir_np = self._tensor_to_numpy_nir(nir_tensor)

        for aug_idx in range(self.augmentation_factor):
            if aug_idx == 0:
                # Identity transform (original)
                aug_rgb = rgb_np.copy()
                aug_nir = nir_np.copy()
            else:
                # Stack to 4-channel for spatial sync
                stacked = np.concatenate([rgb_np, nir_np], axis=-1)  # (H,W,4)

                # Apply spatial transforms
                spatial_result = self.spatial_transform(image=stacked)
                augmented_stack = spatial_result["image"]  # (H',W',4)

                # Split back
                aug_rgb = augmented_stack[:, :, :3]  # (H,W,3)
                aug_nir = augmented_stack[:, :, 3:4]  # (H,W,1)

                # Apply modality-aware photometric transforms
                rgb_result = self.rgb_photometric(image=aug_rgb)
                aug_rgb = rgb_result["image"]

                nir_result = self.nir_photometric(image=aug_nir)
                aug_nir = nir_result["image"]

            # Convert back to normalized tensors
            rgb_out = self._numpy_to_tensor_rgb(aug_rgb)
            nir_out = self._numpy_to_tensor_nir(aug_nir)

            metadata = {
                "source_stem": stem,
                "aug_id": aug_idx,
                "is_identity": aug_idx == 0,
            }

            results.append(AugmentedSample(
                rgb=rgb_out,
                nir=nir_out,
                label=label,
                class_name=class_name,
                metadata=metadata,
            ))

        return results

    def _tensor_to_numpy_rgb(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert (3,H,W) normalized tensor to (H,W,3) uint8 numpy."""
        arr = tensor.permute(1, 2, 0).numpy()  # (H,W,3)
        # Denormalize: reverse ImageNet normalization
        mean = np.array(self.config.rgb_mean, dtype=np.float32)
        std = np.array(self.config.rgb_std, dtype=np.float32)
        arr = arr * std + mean
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return arr

    def _tensor_to_numpy_nir(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert (1,H,W) normalized tensor to (H,W,1) uint8 numpy."""
        arr = tensor.squeeze(0).numpy()  # (H,W)
        # Denormalize
        arr = arr * self.config.nir_stats.std + self.config.nir_stats.mean
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return arr[:, :, np.newaxis]  # (H,W,1)

    def _numpy_to_tensor_rgb(self, arr: np.ndarray) -> torch.Tensor:
        """Convert (H,W,3) uint8 numpy to (3,H,W) normalized tensor."""
        img = arr.astype(np.float32) / 255.0
        mean = np.array(self.config.rgb_mean, dtype=np.float32)
        std = np.array(self.config.rgb_std, dtype=np.float32)
        img = (img - mean) / std
        return torch.from_numpy(img).permute(2, 0, 1).contiguous()

    def _numpy_to_tensor_nir(self, arr: np.ndarray) -> torch.Tensor:
        """Convert (H,W,1) uint8 numpy to (1,H,W) normalized tensor."""
        img = arr.squeeze(-1).astype(np.float32) / 255.0
        img = (img - self.config.nir_stats.mean) / self.config.nir_stats.std
        return torch.from_numpy(img).unsqueeze(0).contiguous()
