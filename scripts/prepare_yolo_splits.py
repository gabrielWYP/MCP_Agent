#!/usr/bin/env python3
"""Prepare deterministic YOLO split label files for paired RGB/NIR data.

The current training pipeline needs a stable train/val/test assignment before
running Florence-2 and before converting Label Studio NIR boxes. Both downstream
steps write labels into split-specific directories, so this script creates the
split map and empty YOLO files up front.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for deterministic split generation."""

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42

    def validate(self) -> None:
        """Validate split ratios."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")
        if min(self.train_ratio, self.val_ratio, self.test_ratio) < 0:
            raise ValueError("Split ratios must be non-negative.")


def discover_paired_rgb_stems(rgb_dir: Path, nir_dir: Path) -> list[str]:
    """Return sorted RGB stems that have a matching NIR image."""
    stems: list[str] = []
    for rgb_path in sorted(rgb_dir.glob("*.jpg")):
        nir_name = rgb_path.name.replace("_rgb", "_nir")
        if (nir_dir / nir_name).exists():
            stems.append(rgb_path.stem)
        else:
            logger.warning("Skipping %s: missing NIR pair %s", rgb_path.name, nir_name)
    return stems


def extract_label_studio_image_name(task: dict) -> str:
    """Extract the original image filename from a Label Studio task."""
    for candidate in (task.get("file_upload", ""), task.get("data", {}).get("image", "")):
        if not candidate:
            continue
        name = Path(candidate).name
        return name.split("-", 1)[1] if "-" in name else name
    return ""


def load_reviewed_rgb_stems(export_path: Path) -> set[str]:
    """Load RGB stems represented in a Label Studio export.

    The export is treated as the source of truth for images that were reviewed
    by a human. Tasks with an empty ``result`` still count as reviewed negatives.
    """
    with export_path.open() as file:
        tasks = json.load(file)

    stems: set[str] = set()
    for task in tasks:
        image_name = extract_label_studio_image_name(task)
        if not image_name:
            continue
        stems.add(Path(image_name).stem.replace("_nir", "_rgb"))
    return stems


def assign_splits(stems: list[str], config: SplitConfig) -> dict[str, list[str]]:
    """Assign stems to train/val/test deterministically."""
    config.validate()
    shuffled = list(stems)
    rng = random.Random(config.seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * config.train_ratio)
    val_count = int(total * config.val_ratio)

    split_map = {
        "train": sorted(shuffled[:train_count]),
        "val": sorted(shuffled[train_count: train_count + val_count]),
        "test": sorted(shuffled[train_count + val_count:]),
    }
    return split_map


def write_empty_label_files(split_map: dict[str, list[str]], labels_dir: Path) -> None:
    """Create empty YOLO label files for every split assignment."""
    for split, stems in split_map.items():
        split_dir = labels_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            label_path = split_dir / f"{stem}.txt"
            label_path.touch(exist_ok=True)


def write_split_manifest(split_map: dict[str, list[str]], output_path: Path) -> None:
    """Persist the split assignment for reproducibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split_map, indent=2, sort_keys=True) + "\n")


def write_label_studio_tasks(split_map: dict[str, list[str]], output_path: Path) -> None:
    """Write a simple Label Studio task list for NIR-only damage labeling."""
    tasks = []
    for split in SPLITS:
        for rgb_stem in split_map.get(split, []):
            nir_name = f"{rgb_stem.replace('_rgb', '_nir')}.jpg"
            tasks.append(
                {
                    "data": {"image": f"/data/local-files/?d=mango/nir/{nir_name}"},
                    "meta": {"split": split, "rgb_stem": rgb_stem},
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare YOLO split label files.")
    parser.add_argument("--rgb-dir", default="data/cache/mango/rgb")
    parser.add_argument("--nir-dir", default="data/cache/mango/nir")
    parser.add_argument("--labels-dir", default="data/annotations/yolo/labels")
    parser.add_argument("--manifest", default="data/annotations/yolo/splits.json")
    parser.add_argument("--label-studio-tasks", default="data/annotations/label_studio_nir/tasks.json")
    parser.add_argument(
        "--reviewed-export",
        default=None,
        help="Optional Label Studio export JSON used to restrict splits to reviewed images.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    """Run split preparation."""
    args = parse_args()
    rgb_dir = Path(args.rgb_dir)
    nir_dir = Path(args.nir_dir)
    labels_dir = Path(args.labels_dir)

    stems = discover_paired_rgb_stems(rgb_dir, nir_dir)
    if args.reviewed_export:
        reviewed_stems = load_reviewed_rgb_stems(Path(args.reviewed_export))
        before_count = len(stems)
        stems = [stem for stem in stems if stem in reviewed_stems]
        logger.info(
            "Restricted split candidates using %s: %d -> %d images",
            args.reviewed_export,
            before_count,
            len(stems),
        )

    if not stems:
        logger.error("No paired RGB/NIR images found under %s and %s", rgb_dir, nir_dir)
        return 1

    split_config = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    split_map = assign_splits(stems, split_config)

    write_empty_label_files(split_map, labels_dir)
    write_split_manifest(split_map, Path(args.manifest))
    write_label_studio_tasks(split_map, Path(args.label_studio_tasks))

    logger.info(
        "Prepared splits: train=%d val=%d test=%d total=%d",
        len(split_map["train"]),
        len(split_map["val"]),
        len(split_map["test"]),
        len(stems),
    )
    logger.info("Labels root: %s", labels_dir)
    logger.info("Split manifest: %s", args.manifest)
    logger.info("Label Studio task template: %s", args.label_studio_tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
