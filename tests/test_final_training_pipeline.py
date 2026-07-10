"""Contract tests for versioned training artifact helpers."""

from __future__ import annotations

from pathlib import Path

from src.training.run_artifacts import (
    build_versioned_stage_plan,
    final_stage_name,
    finalize_stage,
    prepare_atomic_run,
    publish_atomic_run,
    release_atomic_run,
)


def test_final_stage_name_includes_timestamp_and_map() -> None:
    name = final_stage_name("20260704T000000Z", 0.81234)

    assert name == "best_model_20260704T000000Z_mAP0p8123"


def test_dry_run_finalize_returns_planned_stage_result(tmp_path: Path) -> None:
    result = finalize_stage(
        tmp_dir=tmp_path / ".partial" / "maestro",
        final_parent=tmp_path / "maestro",
        run_id="20260704T000000Z",
        command=["python3", "-m", "src.training.train"],
        stage_key="maestro",
        stage_label="Maestro",
        dry_run=True,
    )

    assert result.stage_key == "maestro"
    assert result.best_map50 == 0.0
    assert result.output_dir.endswith("maestro/best_model_20260704T000000Z_mAP0p0000")
    assert result.checkpoint.endswith("best_model.pt")


def test_build_versioned_stage_plan_reuses_timestamp(tmp_path: Path) -> None:
    plan = build_versioned_stage_plan(
        output_root=tmp_path,
        stage_key="destilado",
        run_id="20260704T000000Z",
    )

    assert plan.run_id == "20260704T000000Z"
    assert plan.stage_label == "Destilado"
    assert plan.tmp_dir == tmp_path / "20260704T000000Z" / ".partial" / "destilado"
    assert plan.final_parent == tmp_path / "20260704T000000Z" / "destilado"


def test_atomic_run_is_hidden_until_published(tmp_path: Path) -> None:
    plan = prepare_atomic_run(tmp_path, "20260704T000000Z")
    try:
        (plan.staging_dir / "run_summary.json").write_text("{}", encoding="utf-8")

        assert plan.staging_dir.exists()
        assert not plan.run_dir.exists()

        publish_atomic_run(plan)

        assert plan.run_dir.exists()
        assert (plan.run_dir / "run_summary.json").exists()
        assert not plan.staging_dir.exists()
    finally:
        release_atomic_run(plan)


def test_atomic_run_rejects_concurrent_run_id(tmp_path: Path) -> None:
    first = prepare_atomic_run(tmp_path, "20260704T000000Z")
    try:
        import pytest

        with pytest.raises(FileExistsError, match="already active"):
            prepare_atomic_run(tmp_path, "20260704T000000Z")
    finally:
        release_atomic_run(first)
