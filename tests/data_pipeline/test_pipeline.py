"""Integration test: full pipeline with local dummy pairs."""

import json
import cv2
import numpy as np
import pytest
import torch

from src.data_pipeline.config import PipelineConfig, NIRStats
from src.data_pipeline.pipeline import DataAugmentationPipeline


def _create_dummy_pairs(tmp_path, classes=None, pairs_per_class=2):
    """Create dummy RGB+NIR image pairs on disk."""
    if classes is None:
        classes = ["sano", "danado"]

    local_pairs = []
    for class_name in classes:
        rgb_dir = tmp_path / "cache" / class_name / "rgb"
        nir_dir = tmp_path / "cache" / class_name / "nir"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        nir_dir.mkdir(parents=True, exist_ok=True)

        for i in range(pairs_per_class):
            stem = f"img{i:03d}"

            # Create RGB image (100x80, 3 channels)
            rgb_img = np.random.randint(0, 255, (80, 100, 3), dtype=np.uint8)
            rgb_path = rgb_dir / f"{stem}.jpg"
            cv2.imwrite(str(rgb_path), rgb_img)

            # Create NIR image (100x80, grayscale)
            nir_img = np.random.randint(0, 255, (80, 100), dtype=np.uint8)
            nir_path = nir_dir / f"{stem}.jpg"
            cv2.imwrite(str(nir_path), nir_img)

            local_pairs.append({
                "class_name": class_name,
                "stem": stem,
                "rgb_path": str(rgb_path),
                "nir_path": str(nir_path),
            })

    return local_pairs


class TestPipelineEndToEnd:
    """Full pipeline with local dummy pairs."""

    def test_full_pipeline_produces_manifest(self, tmp_path):
        pairs = _create_dummy_pairs(tmp_path, classes=["sano", "danado"], pairs_per_class=2)

        config = PipelineConfig(
            target_size=(128, 128),
            augmentation_factor=3,
            output_dir=str(tmp_path / "output"),
        )

        pipeline = DataAugmentationPipeline(config)
        stats = pipeline.run(local_pairs=pairs)

        assert stats["downloaded"] == 4  # 2 classes * 2 pairs
        assert stats["processed"] == 4
        assert stats["augmented"] == 12  # 4 pairs * factor 3
        assert stats["skipped"] == 0

        # Check manifest exists
        manifest_path = tmp_path / "output" / "dataset_index.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)
        assert len(manifest) == 12

    def test_pipeline_tensor_shapes(self, tmp_path):
        pairs = _create_dummy_pairs(tmp_path, classes=["sano"], pairs_per_class=1)

        config = PipelineConfig(
            target_size=(224, 224),
            augmentation_factor=2,
            output_dir=str(tmp_path / "output"),
        )

        pipeline = DataAugmentationPipeline(config)
        pipeline.run(local_pairs=pairs)

        # Load a saved .pt file and check shapes
        output_dir = tmp_path / "output"
        pt_files = list(output_dir.glob("*.pt"))
        assert len(pt_files) == 2

        loaded = torch.load(str(pt_files[0]), weights_only=False)
        assert loaded["rgb"].shape == (3, 224, 224)
        assert loaded["nir"].shape == (1, 224, 224)
        assert loaded["rgb"].dtype == torch.float32
        assert loaded["nir"].dtype == torch.float32

    def test_pipeline_stats_dict(self, tmp_path):
        pairs = _create_dummy_pairs(tmp_path, classes=["sano"], pairs_per_class=3)

        config = PipelineConfig(
            target_size=(64, 64),
            augmentation_factor=1,
            output_dir=str(tmp_path / "output"),
        )

        pipeline = DataAugmentationPipeline(config)
        stats = pipeline.run(local_pairs=pairs)

        assert "downloaded" in stats
        assert "processed" in stats
        assert "augmented" in stats
        assert "skipped" in stats
        assert "unmatched" in stats
        assert stats["augmented"] == 3  # 3 pairs * factor 1

    def test_pipeline_handles_corrupt_pair(self, tmp_path):
        """Pipeline skips corrupt images without crashing."""
        pairs = _create_dummy_pairs(tmp_path, classes=["sano"], pairs_per_class=2)

        # Corrupt one NIR file
        corrupt_nir = pairs[0]["nir_path"]
        with open(corrupt_nir, "wb") as f:
            f.write(b"corrupt data")

        config = PipelineConfig(
            target_size=(64, 64),
            augmentation_factor=2,
            output_dir=str(tmp_path / "output"),
        )

        pipeline = DataAugmentationPipeline(config)
        stats = pipeline.run(local_pairs=pairs)

        assert stats["skipped"] == 1
        assert stats["processed"] == 1
        assert stats["augmented"] == 2  # 1 valid pair * factor 2

    def test_pipeline_empty_input(self, tmp_path):
        config = PipelineConfig(
            target_size=(64, 64),
            augmentation_factor=1,
            output_dir=str(tmp_path / "output"),
        )

        pipeline = DataAugmentationPipeline(config)
        stats = pipeline.run(local_pairs=[])

        assert stats["downloaded"] == 0
        assert stats["processed"] == 0
        assert stats["augmented"] == 0
