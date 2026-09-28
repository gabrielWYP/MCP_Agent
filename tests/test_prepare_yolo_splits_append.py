"""Tests for the append mode of YOLO split preparation."""

import json
import sys
from pathlib import Path

import pytest

from scripts import prepare_yolo_splits
from scripts.prepare_yolo_splits import (
    SPLITS,
    SplitConfig,
    append_to_split_map,
    write_new_label_files,
)

EXISTING = {
    "train": ["mango_rgb_100", "mango_rgb_101", "mango_rgb_102"],
    "val": ["mango_rgb_103"],
    "test": ["mango_rgb_104"],
}


def _new_stems(count: int, start: int = 1000) -> list[str]:
    return [f"mango_rgb_{idx}" for idx in range(start, start + count)]


def test_append_keeps_existing_splits_and_merges_sorted() -> None:
    new_stems = _new_stems(10)

    merged, added = append_to_split_map(EXISTING, new_stems, SplitConfig())

    assert set(merged) == set(SPLITS)
    assert set(added) == set(SPLITS)
    for split in SPLITS:
        assert set(EXISTING[split]) <= set(merged[split])
        assert merged[split] == sorted(EXISTING[split] + added[split])
        assert added[split] == sorted(added[split])
    all_added = [stem for split in SPLITS for stem in added[split]]
    assert sorted(all_added) == sorted(new_stems)


def test_append_does_not_mutate_existing_manifest() -> None:
    existing = {split: list(stems) for split, stems in EXISTING.items()}

    append_to_split_map(existing, _new_stems(5), SplitConfig())

    assert existing == EXISTING


def test_append_fifty_stems_default_ratios_is_40_5_5_and_deterministic() -> None:
    new_stems = _new_stems(50)

    first_merged, first_added = append_to_split_map(EXISTING, new_stems, SplitConfig())
    second_merged, second_added = append_to_split_map(
        EXISTING, list(reversed(new_stems)), SplitConfig()
    )

    assert [len(first_added[split]) for split in SPLITS] == [40, 5, 5]
    assert first_added == second_added
    assert first_merged == second_merged


def test_append_different_seed_changes_assignment() -> None:
    new_stems = _new_stems(50)

    _, seed_a = append_to_split_map(EXISTING, new_stems, SplitConfig(seed=1))
    _, seed_b = append_to_split_map(EXISTING, new_stems, SplitConfig(seed=2))

    assert seed_a != seed_b


def test_append_overlap_with_existing_raises() -> None:
    with pytest.raises(ValueError, match="mango_rgb_103"):
        append_to_split_map(EXISTING, ["mango_rgb_103", "mango_rgb_999"], SplitConfig())


def test_append_empty_new_stems_raises() -> None:
    with pytest.raises(ValueError, match="No new stems"):
        append_to_split_map(EXISTING, [], SplitConfig())


def test_write_new_label_files_creates_only_added(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    added = {"train": ["mango_rgb_1", "mango_rgb_2"], "val": ["mango_rgb_3"], "test": []}

    write_new_label_files(added, staging)

    created = sorted(str(path.relative_to(staging)) for path in staging.rglob("*.txt"))
    assert created == [
        "train/mango_rgb_1.txt",
        "train/mango_rgb_2.txt",
        "val/mango_rgb_3.txt",
    ]
    assert all(path.read_text() == "" for path in staging.rglob("*.txt"))


def test_write_new_label_files_refuses_to_overwrite(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    (staging / "val").mkdir(parents=True)
    existing = staging / "val" / "mango_rgb_3.txt"
    existing.write_text("1 0.5 0.5 0.1 0.1\n")
    added = {"train": ["mango_rgb_1"], "val": ["mango_rgb_3"], "test": []}

    with pytest.raises(FileExistsError, match="mango_rgb_3"):
        write_new_label_files(added, staging)

    assert existing.read_text() == "1 0.5 0.5 0.1 0.1\n"
    assert not (staging / "train" / "mango_rgb_1.txt").exists()


def _make_pair(rgb_dir: Path, nir_dir: Path, image_id: int) -> None:
    (rgb_dir / f"mango_rgb_{image_id}.jpg").write_bytes(b"\xff\xd8fake")
    (nir_dir / f"mango_nir_{image_id}.jpg").write_bytes(b"\xff\xd8fake")


def _write_export(path: Path, image_ids: list[int]) -> None:
    tasks = [
        {"file_upload": f"abcd{idx:04d}-mango_nir_{image_id}.jpg", "annotations": [{"result": []}]}
        for idx, image_id in enumerate(image_ids)
    ]
    path.write_text(json.dumps(tasks))


def _setup_cli_fixture(tmp_path: Path) -> dict[str, Path]:
    rgb_dir = tmp_path / "rgb"
    nir_dir = tmp_path / "nir"
    rgb_dir.mkdir()
    nir_dir.mkdir()
    for image_id in range(100, 105):
        _make_pair(rgb_dir, nir_dir, image_id)
    new_ids = list(range(1780238700, 1780238720))
    for image_id in new_ids:
        _make_pair(rgb_dir, nir_dir, image_id)

    manifest = tmp_path / "splits.json"
    manifest.write_text(json.dumps(EXISTING))

    labels_dir = tmp_path / "labels"
    (labels_dir / "val").mkdir(parents=True)
    sentinel = labels_dir / "val" / "mango_rgb_103.txt"
    sentinel.write_text("1 0.5 0.5 0.2 0.2\n")

    export = tmp_path / "export.json"
    # The last id has no RGB/NIR pair on disk and must be excluded.
    _write_export(export, new_ids + [1780239999])

    return {
        "rgb": rgb_dir,
        "nir": nir_dir,
        "manifest": manifest,
        "labels": labels_dir,
        "sentinel": sentinel,
        "export": export,
        "staging": tmp_path / "staging",
        "tasks": tmp_path / "tasks.json",
    }


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["prepare_yolo_splits.py", *argv])
    return prepare_yolo_splits.main()


def _append_argv(paths: dict[str, Path]) -> list[str]:
    return [
        "--rgb-dir", str(paths["rgb"]),
        "--nir-dir", str(paths["nir"]),
        "--labels-dir", str(paths["labels"]),
        "--manifest", str(paths["manifest"]),
        "--label-studio-tasks", str(paths["tasks"]),
        "--append-export", str(paths["export"]),
        "--append-labels-dir", str(paths["staging"]),
    ]


def test_cli_append_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _setup_cli_fixture(tmp_path)

    assert _run_main(monkeypatch, _append_argv(paths)) == 0

    merged = json.loads(paths["manifest"].read_text())
    expected_new = {f"mango_rgb_{image_id}" for image_id in range(1780238700, 1780238720)}
    for split in SPLITS:
        assert set(EXISTING[split]) <= set(merged[split])
        assert merged[split] == sorted(merged[split])
    all_stems = [stem for split in SPLITS for stem in merged[split]]
    assert len(all_stems) == len(set(all_stems))
    assert set(all_stems) == expected_new | {s for stems in EXISTING.values() for s in stems}
    assert "mango_rgb_1780239999" not in all_stems
    assert [len(merged[split]) - len(EXISTING[split]) for split in SPLITS] == [16, 2, 2]

    staged = {
        (path.parent.name, path.stem) for path in paths["staging"].rglob("*.txt")
    }
    expected_staged = {
        (split, stem)
        for split in SPLITS
        for stem in merged[split]
        if stem in expected_new
    }
    assert staged == expected_staged

    assert paths["sentinel"].read_text() == "1 0.5 0.5 0.2 0.2\n"
    assert sorted(p.name for p in paths["labels"].rglob("*")) == ["mango_rgb_103.txt", "val"]
    assert not paths["tasks"].exists()


def test_cli_append_rerun_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _setup_cli_fixture(tmp_path)
    assert _run_main(monkeypatch, _append_argv(paths)) == 0
    after_first = paths["manifest"].read_text()

    assert _run_main(monkeypatch, _append_argv(paths)) == 1
    assert paths["manifest"].read_text() == after_first


def test_cli_append_missing_manifest_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _setup_cli_fixture(tmp_path)
    paths["manifest"].unlink()

    assert _run_main(monkeypatch, _append_argv(paths)) == 1
    assert not paths["staging"].exists()


def test_cli_append_rejects_reconcile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _setup_cli_fixture(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        _run_main(monkeypatch, [*_append_argv(paths), "--reconcile"])
    assert excinfo.value.code == 2


def test_cli_append_requires_staging_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _setup_cli_fixture(tmp_path)
    argv = _append_argv(paths)[:-2]

    with pytest.raises(SystemExit) as excinfo:
        _run_main(monkeypatch, argv)
    assert excinfo.value.code == 2
