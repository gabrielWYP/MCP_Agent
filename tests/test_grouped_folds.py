"""Tests for scripts/make_grouped_folds.py and scripts/run_kfold.py."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts.make_grouped_folds import (
    group_by_local_day,
    make_grouped_folds,
    merge_small_groups,
)
from scripts.run_kfold import build_train_command, discover_folds

LIMA = ZoneInfo("America/Lima")


def _stem(day: str, hour: int, minute: int) -> str:
    ts = int(datetime.fromisoformat(f"{day}T{hour:02d}:{minute:02d}:00").replace(tzinfo=LIMA).timestamp())
    return f"mango_rgb_{ts}"


# day -> number of images; 05-24 has a single image that must merge into 05-31.
DAY_SIZES = {
    "2026-05-24": 1, "2026-05-31": 12, "2026-06-05": 20, "2026-06-06": 7,
    "2026-06-20": 10, "2026-06-21": 11, "2026-06-29": 21, "2026-06-30": 14,
}


def _write_manifest(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    stem_day: dict[str, str] = {}
    for day, n in DAY_SIZES.items():
        for i in range(n):
            stem_day[_stem(day, 10, i)] = day
    stems = sorted(stem_day)
    manifest = {"train": stems[::3], "val": stems[1::3], "test": stems[2::3]}
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(manifest))
    return path, stem_day


def _group_of(meta: dict) -> dict[str, str]:
    return {stem: gid for gid, g in meta["groups"].items() for stem in g["stems"]}


def test_local_day_uses_timezone_not_utc() -> None:
    # 22:30 in Lima on 06-05 is already 06-06 in UTC.
    late = _stem("2026-06-05", 22, 30)
    early = _stem("2026-06-05", 8, 0)
    groups = group_by_local_day([late, early], "America/Lima")
    assert [g.group_id for g in groups] == ["2026-06-05"]
    assert len(group_by_local_day([late, early], "UTC")) == 2


def test_small_group_merges_into_chronologically_nearest() -> None:
    stems = [_stem("2026-05-24", 10, 0)] + [_stem("2026-05-31", 10, i) for i in range(6)] + [
        _stem("2026-06-20", 10, i) for i in range(6)
    ]
    groups = merge_small_groups(group_by_local_day(stems, "America/Lima"), min_group_size=5)
    assert [(g.group_id, g.size) for g in groups] == [("2026-05-24+2026-05-31", 7), ("2026-06-20", 6)]


def test_folds_are_group_disjoint_and_cover_each_stem_once(tmp_path: Path) -> None:
    manifest, stem_day = _write_manifest(tmp_path)
    out = tmp_path / "folds"
    meta = make_grouped_folds(manifest, out, k=5, min_group_size=5, labels_dir=None)

    assert "2026-05-24+2026-05-31" in meta["groups"]
    group_of = _group_of(meta)
    test_counts: dict[str, int] = {}
    for k in range(5):
        fold = json.loads((out / f"fold_{k}.json").read_text())
        assert set(fold) == {"train", "val", "test"}
        for split in fold:
            assert fold[split] == sorted(fold[split])
        split_groups = {s: {group_of[x] for x in fold[s]} for s in fold}
        assert not split_groups["train"] & split_groups["val"]
        assert not split_groups["train"] & split_groups["test"]
        assert not split_groups["val"] & split_groups["test"]
        assert set(fold["train"]) | set(fold["val"]) | set(fold["test"]) == set(stem_day)
        # val is the next fold's test set.
        nxt = json.loads((out / f"fold_{(k + 1) % 5}.json").read_text())
        assert fold["val"] == nxt["test"]
        for stem in fold["test"]:
            test_counts[stem] = test_counts.get(stem, 0) + 1
    assert test_counts == {stem: 1 for stem in stem_day}


def test_fold_generation_is_byte_deterministic(tmp_path: Path) -> None:
    manifest, _ = _write_manifest(tmp_path)
    make_grouped_folds(manifest, tmp_path / "a", k=5, labels_dir=None)
    make_grouped_folds(manifest, tmp_path / "b", k=5, labels_dir=None)
    files = sorted(p.name for p in (tmp_path / "a").iterdir())
    assert files == [f"fold_{k}.json" for k in range(5)] + ["folds_meta.json"]
    for name in files:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_k_greater_than_groups_fails(tmp_path: Path) -> None:
    manifest, _ = _write_manifest(tmp_path)  # 7 groups after merging
    with pytest.raises(ValueError, match="only 7 group"):
        make_grouped_folds(manifest, tmp_path / "folds", k=8, labels_dir=None)


def test_damage_balancing_is_optional_and_used_when_labels_exist(tmp_path: Path) -> None:
    manifest, stem_day = _write_manifest(tmp_path)
    labels = tmp_path / "labels"
    # Missing labels: must not fail, just disable balancing.
    meta = make_grouped_folds(manifest, tmp_path / "f0", k=5, labels_dir=labels)
    assert meta["damage_balancing"] is False

    for i, stem in enumerate(sorted(stem_day)):
        path = labels / ("train", "val", "test")[i % 3] / f"{stem}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("0 0.5 0.5 0.4 0.4\n" + "1 0.5 0.5 0.1 0.1\n" * (i % 2))
    meta = make_grouped_folds(manifest, tmp_path / "f1", k=5, labels_dir=labels)
    assert meta["damage_balancing"] is True
    assert sum(f["damage_boxes"] for f in meta["folds"].values()) == len(stem_day) // 2


def test_run_kfold_command_and_fold_discovery(tmp_path: Path) -> None:
    for k in (0, 1, 2):
        (tmp_path / f"fold_{k}.json").write_text("{}")
    (tmp_path / "folds_meta.json").write_text("{}")
    assert discover_folds(tmp_path) == [0, 1, 2]

    cmd = build_train_command(
        Path("configs/experiment/twostream.yaml"), Path("configs/machines/rtx3080.yaml"),
        tmp_path / "fold_1.json", Path("out/fold1/seed7"), 7, ["epochs=1"],
    )
    overrides = cmd[cmd.index("--override") + 1:]
    assert f"split_manifest={tmp_path / 'fold_1.json'}" in overrides
    assert "label_resolution=manifest" in overrides
    assert "output_dir=out/fold1/seed7" in overrides
    assert "seed=7" in overrides and overrides[-1] == "epochs=1"
