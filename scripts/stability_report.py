"""
Epoch-to-epoch stability report over one or more `metrics_history.csv` files.

Quantifies the validation-AP oscillation a run shows in its last N epochs —
the symptom weight EMA / GroupNorm are meant to remove — so runs can be
compared with one table instead of by eyeballing curves.

Per run (last N epochs, default 30; fewer if the run is shorter):
    - epochs:            total epochs recorded
    - mango AP50 mean / std (population std)
    - mango mean |Δ|:    mean absolute epoch-to-epoch change of mango AP50
    - mango ≥0.9 frac:   fraction of epochs with mango AP50 >= 0.9
    - damage AP50 max / mean
    - damage mean |Δ|

Usage:
    python scripts/stability_report.py \\
        baseline=checkpoints/a/metrics_history.csv \\
        ema=checkpoints/b/metrics_history.csv \\
        --last 30 --out reports/stability.md

A bare path (no `label=`) is labelled by its parent directory name.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

MANGO_COLUMN = "ap50_class_0"
DAMAGE_COLUMN = "ap50_class_1"
MANGO_STABLE_THRESHOLD = 0.9


@dataclass(frozen=True)
class StabilityStats:
    label: str
    epochs: int
    window: int
    mango_mean: float
    mango_std: float
    mango_mean_abs_delta: float
    mango_frac_ge_threshold: float
    damage_max: float
    damage_mean: float
    damage_mean_abs_delta: float


def _mean_abs_delta(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.fmean(abs(b - a) for a, b in zip(values, values[1:]))


def read_history(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no epoch rows.")
    for column in (MANGO_COLUMN, DAMAGE_COLUMN):
        if column not in rows[0]:
            raise ValueError(f"{path} has no '{column}' column.")
    return rows


def compute_stability(rows: list[dict[str, str]], label: str, last: int = 30) -> StabilityStats:
    """Stability statistics over the last `last` epoch rows."""
    if last < 1:
        raise ValueError(f"last must be >= 1, got {last}.")
    window_rows = rows[-last:]
    mango = [float(r[MANGO_COLUMN]) for r in window_rows]
    damage = [float(r[DAMAGE_COLUMN]) for r in window_rows]
    return StabilityStats(
        label=label,
        epochs=len(rows),
        window=len(window_rows),
        mango_mean=statistics.fmean(mango),
        mango_std=statistics.pstdev(mango),
        mango_mean_abs_delta=_mean_abs_delta(mango),
        mango_frac_ge_threshold=sum(v >= MANGO_STABLE_THRESHOLD for v in mango) / len(mango),
        damage_max=max(damage),
        damage_mean=statistics.fmean(damage),
        damage_mean_abs_delta=_mean_abs_delta(damage),
    )


def render_markdown(stats: list[StabilityStats], last: int) -> str:
    header = (
        f"| run | epochs | window | mango AP50 mean | mango AP50 std | mango mean \\|Δ\\| "
        f"| mango ≥{MANGO_STABLE_THRESHOLD} frac | damage AP50 max | damage AP50 mean "
        f"| damage mean \\|Δ\\| |"
    )
    lines = [
        f"## Stability over the last {last} epochs",
        "",
        header,
        "|" + "---|" * 10,
    ]
    for s in stats:
        lines.append(
            f"| {s.label} | {s.epochs} | {s.window} | {s.mango_mean:.4f} | {s.mango_std:.4f} "
            f"| {s.mango_mean_abs_delta:.4f} | {s.mango_frac_ge_threshold:.2f} "
            f"| {s.damage_max:.4f} | {s.damage_mean:.4f} | {s.damage_mean_abs_delta:.4f} |"
        )
    return "\n".join(lines) + "\n"


def parse_run(spec: str) -> tuple[str, Path]:
    """`label=path` or a bare path (labelled by its parent directory)."""
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label, Path(path)
    path = Path(spec)
    return path.parent.name or path.stem, path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("runs", nargs="+", help="metrics_history.csv paths, optionally as label=path.")
    parser.add_argument("--last", type=int, default=30, help="Epoch window (default: 30).")
    parser.add_argument("--out", type=Path, default=None, help="Optional markdown output path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stats = [
        compute_stability(read_history(path), label, last=args.last)
        for label, path in map(parse_run, args.runs)
    ]
    report = render_markdown(stats, args.last)
    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
