"""Tests for VectorStore: save/load roundtrip and manifest."""

import json
import pytest
import torch

from src.data_pipeline.vector_store import VectorStore, ManifestEntry


class TestVectorStoreSaveLoad:
    """Save -> load roundtrip preserves tensor equality and metadata."""

    def test_roundtrip_preserves_tensors(self, tmp_path):
        store = VectorStore()

        rgb = torch.randn(3, 64, 64)
        nir = torch.randn(1, 64, 64)

        entry = store.save_sample(
            rgb=rgb,
            nir=nir,
            label=0,
            class_name="sano",
            stem="img001",
            aug_id=0,
            output_dir=str(tmp_path),
            metadata={"source_oci_key": "bucket/sano/rgb/img001.jpg", "homography_applied": True},
        )

        # Load back
        loaded = torch.load(entry.local_path, weights_only=False)
        assert torch.equal(loaded["rgb"], rgb)
        assert torch.equal(loaded["nir"], nir)
        assert loaded["label"] == 0
        assert loaded["class"] == "sano"

    def test_roundtrip_preserves_metadata(self, tmp_path):
        store = VectorStore()

        rgb = torch.randn(3, 64, 64)
        nir = torch.randn(1, 64, 64)
        meta = {"custom_key": "custom_value", "homography_applied": False}

        entry = store.save_sample(
            rgb=rgb, nir=nir, label=1, class_name="danado",
            stem="img002", aug_id=2, output_dir=str(tmp_path),
            metadata=meta,
        )

        loaded = torch.load(entry.local_path, weights_only=False)
        assert loaded["meta"]["custom_key"] == "custom_value"

    def test_filename_format(self, tmp_path):
        store = VectorStore()
        rgb = torch.randn(3, 32, 32)
        nir = torch.randn(1, 32, 32)

        entry = store.save_sample(
            rgb=rgb, nir=nir, label=0, class_name="sano",
            stem="test_img", aug_id=5, output_dir=str(tmp_path),
        )

        assert entry.local_path.endswith("test_img_aug5.pt")


class TestVectorStoreManifest:
    """Manifest writing and validation."""

    def test_manifest_written(self, tmp_path):
        store = VectorStore()
        entries = [
            ManifestEntry(
                source_oci_key=f"bucket/sano/rgb/img{i:03d}.jpg",
                local_path=str(tmp_path / f"img{i:03d}_aug0.pt"),
                label=0,
                class_name="sano",
                aug_id=0,
                homography_applied=True,
            )
            for i in range(5)
        ]

        manifest_path = store.write_manifest(entries, str(tmp_path))
        assert manifest_path.endswith("dataset_index.json")

        with open(manifest_path) as f:
            data = json.load(f)

        assert len(data) == 5
        assert data[0]["class_name"] == "sano"
        assert data[0]["homography_applied"] is True

    def test_manifest_30_entries_all_files_exist(self, tmp_path):
        store = VectorStore()
        entries = []

        for i in range(10):
            for aug_id in range(3):
                rgb = torch.randn(3, 32, 32)
                nir = torch.randn(1, 32, 32)

                entry = store.save_sample(
                    rgb=rgb, nir=nir, label=0, class_name="sano",
                    stem=f"img{i:03d}", aug_id=aug_id,
                    output_dir=str(tmp_path),
                    metadata={"source_oci_key": f"bucket/sano/rgb/img{i:03d}.jpg"},
                )
                entries.append(entry)

        manifest_path = store.write_manifest(entries, str(tmp_path))

        with open(manifest_path) as f:
            data = json.load(f)

        assert len(data) == 30

        # Verify all referenced files exist
        for entry in data:
            assert (tmp_path / entry["local_path"].split("/")[-1]).exists() or \
                   __import__("pathlib").Path(entry["local_path"]).exists()

    def test_manifest_idempotent_overwrite(self, tmp_path):
        store = VectorStore()

        # Write first manifest
        entries1 = [
            ManifestEntry("k1", "p1.pt", 0, "sano", 0, True),
        ]
        store.write_manifest(entries1, str(tmp_path))

        # Overwrite with second manifest
        entries2 = [
            ManifestEntry("k1", "p1.pt", 0, "sano", 0, True),
            ManifestEntry("k2", "p2.pt", 1, "danado", 0, False),
        ]
        store.write_manifest(entries2, str(tmp_path))

        manifest_path = tmp_path / "dataset_index.json"
        with open(manifest_path) as f:
            data = json.load(f)

        assert len(data) == 2  # Second write overwrites first
