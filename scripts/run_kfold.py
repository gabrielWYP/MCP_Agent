#!/usr/bin/env python3
"""Train one experiment config on every grouped fold (and seed).

Each (fold, seed) run is an ordinary `python -m src.training.train` invocation
of the given experiment + machine profile, with only these overrides:

    split_manifest=<folds-dir>/fold_{k}.json
    label_resolution=manifest       # labels resolved across split subdirs
    output_dir=<output-root>/fold{k}/seed{s}
    seed=<s>

plus any `--override` passed through verbatim (e.g. `epochs=1` for a smoke
run). The experiment file itself is untouched, so `experiment_sha256` stays
comparable across folds.

Usage:
    python scripts/run_kfold.py --config configs/experiment/twostream.yaml \\
        --machine configs/machines/rtx3080.yaml --folds-dir data/annotations/yolo/folds \\
        --seeds 42 1337 --resume-existing

    # Evaluate afterwards with scripts/aggregate_kfold.py (same --config/--output-root).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def discover_folds(folds_dir: Path) -> list[int]:
    """Return the sorted fold indices with a `fold_{k}.json` in `folds_dir`."""
    indices = []
    for path in folds_dir.glob("fold_*.json"):
        suffix = path.stem.removeprefix("fold_")
        if suffix.isdigit():
            indices.append(int(suffix))
    if not indices:
        raise FileNotFoundError(f"No fold_*.json manifests in {folds_dir}")
    return sorted(indices)


def default_output_root(config_path: Path) -> Path:
    """`<experiment output_dir>_kfold`, e.g. checkpoints/twostream_kfold."""
    data = yaml.safe_load(config_path.read_text()) or {}
    base = data.get("output_dir") or f"checkpoints/{config_path.stem}"
    return Path(f"{base.rstrip('/')}_kfold")


def run_dir(output_root: Path, fold: int, seed: int) -> Path:
    return output_root / f"fold{fold}" / f"seed{seed}"


def build_train_command(
    config: Path,
    machine: Path | None,
    fold_manifest: Path,
    output_dir: Path,
    seed: int,
    extra_overrides: list[str],
    model: str | None = None,
) -> list[str]:
    """Return the argv for one fold/seed training run."""
    cmd = [sys.executable, "-m", "src.training.train", "--config", str(config)]
    if machine is not None:
        cmd += ["--machine", str(machine)]
    if model is not None:
        cmd += ["--model", model]
    cmd += [
        "--override",
        f"split_manifest={fold_manifest}",
        "label_resolution=manifest",
        f"output_dir={output_dir}",
        f"seed={seed}",
        *extra_overrides,
    ]
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an experiment on grouped k-fold manifests.")
    parser.add_argument("--config", required=True, type=Path, help="Experiment YAML.")
    parser.add_argument("--machine", type=Path, default=None, help="Machine-profile YAML.")
    parser.add_argument("--folds-dir", type=Path, default=Path("data/annotations/yolo/folds"))
    parser.add_argument("--folds", type=int, nargs="*", default=None,
                        help="Fold indices to run (default: every fold in --folds-dir).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Runs go to <output-root>/fold{k}/seed{s} "
                             "(default: <experiment output_dir>_kfold).")
    parser.add_argument("--model", choices=["master", "student"], default=None)
    parser.add_argument("--override", nargs="*", default=[],
                        help="Extra key=value overrides passed to every run (e.g. epochs=1).")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    parser.add_argument("--resume-existing", action="store_true",
                        help="Skip (fold, seed) runs whose best_model.pt already exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    folds = args.folds if args.folds else discover_folds(args.folds_dir)
    output_root = args.output_root or default_output_root(args.config)
    logger.info("Experiment %s -> %s (folds=%s seeds=%s)", args.config, output_root, folds, args.seeds)

    status: dict[str, str] = {}
    for fold in folds:
        fold_manifest = args.folds_dir / f"fold_{fold}.json"
        if not fold_manifest.exists():
            raise FileNotFoundError(f"Missing fold manifest: {fold_manifest}")
        for seed in args.seeds:
            key = f"fold{fold}/seed{seed}"
            out = run_dir(output_root, fold, seed)
            if args.resume_existing and (out / "best_model.pt").exists():
                logger.info("[%s] best_model.pt exists, skipping.", key)
                status[key] = "skipped"
                continue
            cmd = build_train_command(
                args.config, args.machine, fold_manifest, out, seed, args.override, args.model,
            )
            logger.info("[%s] %s", key, " ".join(cmd))
            if args.dry_run:
                status[key] = "dry-run"
                continue
            result = subprocess.run(cmd, cwd=PROJECT_ROOT)
            status[key] = "ok" if result.returncode == 0 else f"failed (exit {result.returncode})"
            if result.returncode != 0:
                logger.error("[%s] training failed with exit code %d", key, result.returncode)

    logger.info("Summary: %s", json.dumps(status, indent=2))
    return 1 if any(v.startswith("failed") for v in status.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
