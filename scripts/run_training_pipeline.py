#!/usr/bin/env python3
"""Orchestrate the current RGB/NIR YOLO training pipeline.

This replaces the retired precomputed-tensor pipeline. The intended flow is:

1. Prepare deterministic split label files.
2. Run Florence-2 on RGB images to write class 0 mango boxes.
3. Convert Label Studio NIR damage boxes to RGB YOLO class 1 boxes.
4. Train with on-the-fly RGB/NIR augmentation in ``src.training``.

Label Studio remains a human-in-the-loop step. If the export JSON is missing,
the pipeline stops after split preparation and optional Florence-2 annotation.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_command(command: list[str], dry_run: bool) -> None:
    """Run a subprocess command with structured logging."""
    logger.info("Running: %s", " ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run the current training pipeline.")
    parser.add_argument("--rgb-dir", default="data/cache/mango/rgb")
    parser.add_argument("--nir-dir", default="data/cache/mango/nir")
    parser.add_argument("--labels-dir", default="data/annotations/yolo/labels")
    parser.add_argument("--split-manifest", default="data/annotations/yolo/splits.json")
    parser.add_argument("--label-studio-export", default="data/label_studio_nir/export.json")
    parser.add_argument("--homography", default="notebooks/matriz_homografia_aruco.npy")
    parser.add_argument("--training-config", default="configs/training_mango.yaml")
    parser.add_argument("--florence-model", default="microsoft/Florence-2-large")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-florence", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the orchestrated pipeline."""
    args = parse_args()
    python = sys.executable

    if not args.skip_prepare:
        run_command(
            [
                python,
                "scripts/prepare_yolo_splits.py",
                "--rgb-dir",
                args.rgb_dir,
                "--nir-dir",
                args.nir_dir,
                "--labels-dir",
                args.labels_dir,
                "--manifest",
                args.split_manifest,
                "--seed",
                str(args.seed),
                "--train-ratio",
                str(args.train_ratio),
                "--val-ratio",
                str(args.val_ratio),
                "--test-ratio",
                str(args.test_ratio),
            ],
            args.dry_run,
        )

    if not args.skip_florence:
        command = [
            python,
            "scripts/annotate_mango_florence.py",
            "--rgb-dir",
            args.rgb_dir,
            "--labels-dir",
            args.labels_dir,
            "--model",
            args.florence_model,
        ]
        if args.device:
            command.extend(["--device", args.device])
        run_command(command, args.dry_run)

    export_path = Path(args.label_studio_export)
    if export_path.exists() and not args.skip_convert:
        run_command(
            [
                python,
                "scripts/convert_nir_labels.py",
                "--export",
                args.label_studio_export,
                "--homography",
                args.homography,
                "--output-dir",
                args.labels_dir,
                "--splits",
                args.labels_dir,
                "--rgb-dir",
                args.rgb_dir,
                "--nir-dir",
                args.nir_dir,
            ],
            args.dry_run,
        )
    elif not args.skip_convert:
        logger.warning(
            "Label Studio export not found at %s. Stop here, export NIR damage "
            "annotations, then rerun with --skip-prepare --skip-florence.",
            export_path,
        )
        return 0

    if not args.skip_train:
        run_command(
            [
                python,
                "-m",
                "src.training.train",
                "--config",
                args.training_config,
            ],
            args.dry_run,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
