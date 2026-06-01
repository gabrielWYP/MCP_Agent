"""
Vector store: per-image .pt files + dataset_index.json manifest.

Each augmented sample is saved as {stem}_aug{N}.pt containing
{rgb, nir, label, class, meta}. The manifest is an idempotent
overwrite listing all output files with metadata.
"""

import json
import logging
from pathlib import Path
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)


class ManifestEntry(NamedTuple):
    """A single entry in the dataset manifest."""
    source_oci_key: str
    local_path: str
    label: int
    class_name: str
    aug_id: int
    homography_applied: bool


class VectorStore:
    """
    Writes per-image .pt vector files and maintains a dataset manifest.
    """

    def save_sample(
        self,
        rgb: torch.Tensor,
        nir: torch.Tensor,
        label: int,
        class_name: str,
        stem: str,
        aug_id: int,
        output_dir: str,
        metadata: dict | None = None,
    ) -> ManifestEntry:
        """
        Save a single augmented sample to disk.

        Args:
            rgb: (3, H, W) tensor
            nir: (1, H, W) tensor
            label: Class label
            class_name: Class name
            stem: Source filename stem
            aug_id: Augmentation index
            output_dir: Output directory
            metadata: Additional metadata dict

        Returns:
            ManifestEntry for this sample
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        filename = f"{stem}_aug{aug_id}.pt"
        filepath = out_path / filename

        data = {
            "rgb": rgb,
            "nir": nir,
            "label": label,
            "class": class_name,
            "meta": metadata or {},
        }

        torch.save(data, str(filepath))

        source_key = metadata.get("source_oci_key", "") if metadata else ""
        homography_applied = metadata.get("homography_applied", False) if metadata else False

        return ManifestEntry(
            source_oci_key=source_key,
            local_path=str(filepath),
            label=label,
            class_name=class_name,
            aug_id=aug_id,
            homography_applied=homography_applied,
        )

    def write_manifest(
        self,
        entries: list[ManifestEntry],
        output_dir: str,
    ) -> str:
        """
        Write the dataset_index.json manifest.

        Idempotent: overwrites any existing manifest.

        Args:
            entries: List of ManifestEntry objects
            output_dir: Output directory

        Returns:
            Path to the written manifest file
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        manifest_path = out_path / "dataset_index.json"

        manifest_data = [
            {
                "source_oci_key": e.source_oci_key,
                "local_path": e.local_path,
                "label": e.label,
                "class_name": e.class_name,
                "aug_id": e.aug_id,
                "homography_applied": e.homography_applied,
            }
            for e in entries
        ]

        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2)

        logger.info("Wrote manifest with %d entries to %s", len(entries), manifest_path)
        return str(manifest_path)
