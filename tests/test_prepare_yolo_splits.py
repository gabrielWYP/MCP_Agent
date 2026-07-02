"""Tests for YOLO split preparation."""

from pathlib import Path

from scripts.prepare_yolo_splits import (
    SplitConfig,
    assign_splits,
    discover_paired_rgb_stems,
    write_empty_label_files,
)


def test_discover_paired_rgb_stems_only_keeps_complete_pairs(tmp_path: Path) -> None:
    rgb_dir = tmp_path / "rgb"
    nir_dir = tmp_path / "nir"
    rgb_dir.mkdir()
    nir_dir.mkdir()

    (rgb_dir / "mango_rgb_001.jpg").touch()
    (nir_dir / "mango_nir_001.jpg").touch()
    (rgb_dir / "mango_rgb_002.jpg").touch()

    assert discover_paired_rgb_stems(rgb_dir, nir_dir) == ["mango_rgb_001"]


def test_assign_splits_is_deterministic() -> None:
    stems = [f"mango_rgb_{idx:03d}" for idx in range(10)]
    config = SplitConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=7)

    first = assign_splits(stems, config)
    second = assign_splits(stems, config)

    assert first == second
    assert len(first["train"]) == 6
    assert len(first["val"]) == 2
    assert len(first["test"]) == 2
    assert sorted(first["train"] + first["val"] + first["test"]) == stems


def test_write_empty_label_files_creates_split_dirs(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    split_map = {
        "train": ["mango_rgb_001"],
        "val": ["mango_rgb_002"],
        "test": ["mango_rgb_003"],
    }

    write_empty_label_files(split_map, labels_dir)

    assert (labels_dir / "train" / "mango_rgb_001.txt").exists()
    assert (labels_dir / "val" / "mango_rgb_002.txt").exists()
    assert (labels_dir / "test" / "mango_rgb_003.txt").exists()
