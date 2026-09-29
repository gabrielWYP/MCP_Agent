"""Manifest-driven label resolution (label_resolution="manifest")."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.dataset import YOLODataset, resolve_label_paths


def _pair(root: Path, stem: str, label_split: str | None, content: str = "0 0.5 0.5 0.1 0.1\n") -> None:
    (root / "rgb").mkdir(parents=True, exist_ok=True)
    (root / "nir").mkdir(parents=True, exist_ok=True)
    (root / "rgb" / f"{stem}.jpg").touch()
    (root / "nir" / f"{stem.replace('_rgb', '_nir')}.jpg").touch()
    if label_split is not None:
        path = root / "labels" / label_split / f"{stem}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def _dataset(root: Path, manifest: dict, split: str) -> YOLODataset:
    path = root / "fold.json"
    path.write_text(json.dumps(manifest))
    return YOLODataset(
        rgb_dir=root / "rgb", nir_dir=root / "nir", labels_dir=root / "labels",
        split=split, manifest_path=path, label_resolution="manifest",
    )


def test_labels_resolved_across_split_subdirs(tmp_path: Path) -> None:
    _pair(tmp_path, "mango_rgb_1", "train")
    _pair(tmp_path, "mango_rgb_2", "val", "0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.05 0.05\n")
    _pair(tmp_path, "mango_rgb_3", "test")
    manifest = {"train": ["mango_rgb_3"], "val": [], "test": ["mango_rgb_1", "mango_rgb_2"]}

    test_ds = _dataset(tmp_path, manifest, "test")
    assert [p["label_path"].parent.name for p in test_ds.pairs] == ["train", "val"]
    assert test_ds.get_class_counts() == {0: 2, 1: 1}
    report = test_ds.gt_count_report()
    assert report["loaded"] == report["on_disk"] and report["label_file_count"] == 2

    train_ds = _dataset(tmp_path, manifest, "train")
    assert [p["rgb_path"].stem for p in train_ds.pairs] == ["mango_rgb_3"]


def test_missing_label_raises(tmp_path: Path) -> None:
    _pair(tmp_path, "mango_rgb_1", None)
    with pytest.raises(FileNotFoundError, match="mango_rgb_1"):
        _dataset(tmp_path, {"train": [], "val": [], "test": ["mango_rgb_1"]}, "test")


def test_ambiguous_label_raises(tmp_path: Path) -> None:
    _pair(tmp_path, "mango_rgb_1", "train")
    _pair(tmp_path, "mango_rgb_1", "val")
    with pytest.raises(ValueError, match="Ambiguous"):
        resolve_label_paths(tmp_path / "labels", ["mango_rgb_1"])
    with pytest.raises(ValueError, match="Ambiguous"):
        _dataset(tmp_path, {"train": [], "val": [], "test": ["mango_rgb_1"]}, "test")


def test_missing_image_raises(tmp_path: Path) -> None:
    _pair(tmp_path, "mango_rgb_1", "train")
    (tmp_path / "nir" / "mango_nir_1.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="RGB/NIR pair"):
        _dataset(tmp_path, {"train": [], "val": [], "test": ["mango_rgb_1"]}, "test")


def test_non_disjoint_manifest_raises(tmp_path: Path) -> None:
    _pair(tmp_path, "mango_rgb_1", "train")
    with pytest.raises(ValueError, match="not disjoint"):
        _dataset(tmp_path, {"train": ["mango_rgb_1"], "val": [], "test": ["mango_rgb_1"]}, "test")


def test_manifest_mode_requires_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a manifest_path"):
        YOLODataset(rgb_dir=tmp_path, nir_dir=tmp_path, labels_dir=tmp_path, label_resolution="manifest")


def test_split_dir_mode_still_rejects_fold_manifest(tmp_path: Path) -> None:
    """Default behaviour is unchanged: a regrouped manifest trips the leakage guard."""
    _pair(tmp_path, "mango_rgb_1", "train")
    path = tmp_path / "fold.json"
    path.write_text(json.dumps({"train": [], "val": [], "test": ["mango_rgb_1"]}))
    with pytest.raises(ValueError, match="diverges from"):
        YOLODataset(rgb_dir=tmp_path / "rgb", nir_dir=tmp_path / "nir",
                    labels_dir=tmp_path / "labels", split="train", manifest_path=path)
