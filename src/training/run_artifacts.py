"""Utilities for timestamped, atomic training run artifacts."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


STAGE_LABELS: dict[str, str] = {
    "maestro": "Maestro",
    "estudiante": "Estudiante",
    "destilado": "Destilado",
}


@dataclass(frozen=True)
class StageResult:
    """Metadata for one completed training stage."""

    stage_key: str
    stage_label: str
    output_dir: str
    checkpoint: str
    best_map50: float
    command: list[str]


@dataclass(frozen=True)
class VersionedStagePlan:
    """Resolved paths for one atomic versioned stage run."""

    run_id: str
    run_dir: Path
    stage_key: str
    stage_label: str
    tmp_dir: Path
    final_parent: Path


def utc_timestamp() -> str:
    """Return a filesystem-safe UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def final_stage_name(run_id: str, best_map50: float) -> str:
    """Build the final directory name for one stage."""
    metric = f"{best_map50:.4f}".replace(".", "p")
    return f"best_model_{run_id}_mAP{metric}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a deterministic JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_best_map50(checkpoint_path: Path) -> float:
    """Read the best mAP@0.5 from a saved checkpoint."""
    import torch

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing stage checkpoint: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    metric = checkpoint.get("best_map50")
    if metric is None:
        metric = checkpoint.get("metrics", {}).get("map50")
    if metric is None:
        raise ValueError(f"Checkpoint does not contain best mAP@0.5: {checkpoint_path}")
    return float(metric)


def build_versioned_stage_plan(
    *,
    output_root: str | Path,
    stage_key: str,
    run_id: str | None = None,
) -> VersionedStagePlan:
    """Resolve temporary and final paths for one versioned stage.

    Args:
        output_root: Root directory for timestamped runs.
        stage_key: One of ``maestro``, ``estudiante`` or ``destilado``.
        run_id: Optional existing timestamp to reuse across independent runs.

    Returns:
        VersionedStagePlan with temporary and final parent directories.
    """
    if stage_key not in STAGE_LABELS:
        valid = ", ".join(sorted(STAGE_LABELS))
        raise ValueError(f"Invalid stage_key '{stage_key}'. Expected one of: {valid}")

    resolved_run_id = run_id or utc_timestamp()
    run_dir = Path(output_root) / resolved_run_id
    return VersionedStagePlan(
        run_id=resolved_run_id,
        run_dir=run_dir,
        stage_key=stage_key,
        stage_label=STAGE_LABELS[stage_key],
        tmp_dir=run_dir / ".partial" / stage_key,
        final_parent=run_dir / stage_key,
    )


def finalize_stage(
    *,
    tmp_dir: Path,
    final_parent: Path,
    run_id: str,
    command: list[str],
    stage_key: str,
    stage_label: str,
    dry_run: bool = False,
) -> StageResult:
    """Move a successful temporary stage directory into its final location."""
    if dry_run:
        planned_dir = final_parent / final_stage_name(run_id, 0.0)
        return StageResult(
            stage_key=stage_key,
            stage_label=stage_label,
            output_dir=str(planned_dir),
            checkpoint=str(planned_dir / "best_model.pt"),
            best_map50=0.0,
            command=command,
        )

    checkpoint_path = tmp_dir / "best_model.pt"
    best_map50 = load_best_map50(checkpoint_path)
    final_dir = final_parent / final_stage_name(run_id, best_map50)

    if final_parent.exists() and any(final_parent.iterdir()):
        raise FileExistsError(f"Stage directory already contains artifacts: {final_parent}")
    if final_dir.exists():
        raise FileExistsError(f"Final stage directory already exists: {final_dir}")

    final_parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_dir), str(final_dir))

    result = StageResult(
        stage_key=stage_key,
        stage_label=stage_label,
        output_dir=str(final_dir),
        checkpoint=str(final_dir / "best_model.pt"),
        best_map50=best_map50,
        command=command,
    )
    write_json(final_dir / "stage_summary.json", asdict(result))
    return result


def load_stage_result(path: Path) -> StageResult:
    """Load one stage summary JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return StageResult(**data)


def collect_stage_results(run_dir: Path) -> list[StageResult]:
    """Collect stage summaries already written under a run directory."""
    results: list[StageResult] = []
    for stage_key in STAGE_LABELS:
        for summary_path in sorted((run_dir / stage_key).glob("*/stage_summary.json")):
            results.append(load_stage_result(summary_path))
    return results


def generate_run_summary_image(run_dir: Path, results: list[StageResult]) -> None:
    """Generate a compact PNG summary for immediate visual inspection."""
    if not results:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable; skipping final summary PNG")
        return

    labels = [result.stage_label for result in results]
    scores = [result.best_map50 for result in results]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, scores, color=["#4C78A8", "#F58518", "#54A24B"][: len(results)])
    ax.set_ylim(0.0, max(1.0, max(scores) * 1.15))
    ax.set_ylabel("Best mAP@0.5")
    ax.set_title("Training run summary")
    ax.grid(axis="y", alpha=0.25)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(run_dir / "final_metrics_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_run_summary(run_dir: Path, run_id: str, results: list[StageResult]) -> None:
    """Write root-level JSON and PNG summaries for a timestamped run."""
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": [asdict(result) for result in results],
    }
    write_json(run_dir / "run_summary.json", summary)
    generate_run_summary_image(run_dir, results)
