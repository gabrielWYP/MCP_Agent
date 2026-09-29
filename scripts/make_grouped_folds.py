#!/usr/bin/env python3
"""Build group-disjoint K-fold split manifests (grouped by capture day).

The canonical `splits.json` assigns images to train/val/test at random, one
image at a time. The same physical mango is photographed several times per
capture session (rotations, other angles) and may reappear on other days, so
a per-image split leaks near-duplicates across train/val/test, and a 20-image
test split makes damage AP extremely noisy.

This script regroups the canonical manifest's labelled stems by the LOCAL
calendar day of their capture timestamp (`mango_rgb_<unix_ts>`), merges tiny
days into the chronologically nearest day, and distributes whole groups over
K folds with a deterministic greedy balancer. For fold k:

    test  = fold k
    val   = fold (k + 1) mod K
    train = every other fold

All three are group-disjoint. Each fold manifest has exactly the
`splits.json` schema, so training/evaluation only need
`--override split_manifest=<fold json> label_resolution=manifest` (labels
stay where they are; `YOLODataset` resolves each stem across the split
subdirectories of `labels_dir`).

Usage:
    python scripts/make_grouped_folds.py
    python scripts/make_grouped_folds.py --k 5 --min-group-size 5 --tz America/Lima

Outputs (deterministic: same inputs -> byte-identical files):
    <output-dir>/fold_{k}.json
    <output-dir>/folds_meta.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.dataset import MANIFEST_SPLITS, build_label_index  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DAMAGE_CLASS_ID = 1
_TIMESTAMP_RE = re.compile(r"_(\d{9,})$")


@dataclass
class CaptureGroup:
    """A set of stems that must never be split across train/val/test."""

    group_id: str
    days: list[str]
    stems: list[str]
    timestamps: list[int]
    damage_boxes: int = 0

    @property
    def size(self) -> int:
        return len(self.stems)

    @property
    def start(self) -> int:
        return min(self.timestamps)

    @property
    def end(self) -> int:
        return max(self.timestamps)


@dataclass
class Fold:
    """One fold: an ordered collection of whole groups."""

    index: int
    groups: list[CaptureGroup] = field(default_factory=list)

    @property
    def size(self) -> int:
        return sum(g.size for g in self.groups)

    @property
    def damage_boxes(self) -> int:
        return sum(g.damage_boxes for g in self.groups)

    @property
    def stems(self) -> list[str]:
        return sorted(stem for g in self.groups for stem in g.stems)


def stem_timestamp(stem: str) -> int:
    """Parse the trailing unix timestamp of a `mango_rgb_<unix_ts>` stem."""
    match = _TIMESTAMP_RE.search(stem)
    if match is None:
        raise ValueError(f"Cannot parse a capture timestamp from stem '{stem}'.")
    return int(match.group(1))


def load_manifest_stems(manifest_path: Path) -> list[str]:
    """Return the sorted union of the manifest's train/val/test stems."""
    manifest = json.loads(manifest_path.read_text())
    stems: set[str] = set()
    for split in MANIFEST_SPLITS:
        stems.update(manifest.get(split, []))
    return sorted(stems)


def group_by_local_day(stems: list[str], tz_name: str) -> list[CaptureGroup]:
    """Group stems by the local calendar day of their capture timestamp.

    Returns groups in chronological order.
    """
    tz = ZoneInfo(tz_name)
    by_day: dict[str, list[str]] = {}
    for stem in stems:
        day = datetime.fromtimestamp(stem_timestamp(stem), tz=timezone.utc).astimezone(tz)
        by_day.setdefault(day.date().isoformat(), []).append(stem)
    groups = [
        CaptureGroup(
            group_id=day,
            days=[day],
            stems=sorted(day_stems),
            timestamps=sorted(stem_timestamp(s) for s in day_stems),
        )
        for day, day_stems in by_day.items()
    ]
    groups.sort(key=lambda g: (g.start, g.group_id))
    return groups


def _merge(a: CaptureGroup, b: CaptureGroup) -> CaptureGroup:
    first, second = (a, b) if (a.start, a.group_id) <= (b.start, b.group_id) else (b, a)
    days = sorted(set(first.days) | set(second.days))
    return CaptureGroup(
        group_id="+".join(days),
        days=days,
        stems=sorted(first.stems + second.stems),
        timestamps=sorted(first.timestamps + second.timestamps),
        damage_boxes=first.damage_boxes + second.damage_boxes,
    )


def merge_small_groups(groups: list[CaptureGroup], min_group_size: int) -> list[CaptureGroup]:
    """Merge every group smaller than `min_group_size` into its nearest neighbour.

    "Nearest" is the chronological neighbour (previous or next group) with the
    smallest time gap between the two groups' closest captures; ties go to
    the earlier neighbour. The smallest offending group (earliest on ties) is
    merged first, repeating until none is left or a single group remains.
    """
    groups = sorted(groups, key=lambda g: (g.start, g.group_id))
    while len(groups) > 1:
        small = [i for i, g in enumerate(groups) if g.size < min_group_size]
        if not small:
            break
        i = min(small, key=lambda idx: (groups[idx].size, groups[idx].start))
        candidates = []
        if i > 0:
            candidates.append((groups[i].start - groups[i - 1].end, 0, i - 1))
        if i < len(groups) - 1:
            candidates.append((groups[i + 1].start - groups[i].end, 1, i + 1))
        _, _, j = min(candidates)
        logger.info(
            "Merging group %s (%d images) into nearest group %s (%d images)",
            groups[i].group_id, groups[i].size, groups[j].group_id, groups[j].size,
        )
        merged = _merge(groups[i], groups[j])
        lo, hi = sorted((i, j))
        groups = groups[:lo] + [merged] + groups[hi + 1:]
    return groups


def count_damage_boxes(labels_dir: Path | None, stems: list[str]) -> dict[str, int] | None:
    """Count class-1 boxes per stem, or None if the labels are not usable.

    Labels are optional: any missing or ambiguous label file disables
    damage balancing (with a warning) instead of failing fold generation.
    """
    if labels_dir is None:
        return None
    index = build_label_index(labels_dir)
    unusable = [s for s in stems if len(index.get(s, [])) != 1]
    if unusable:
        logger.warning(
            "Damage-box balancing disabled: %d stem(s) have missing/ambiguous labels under %s "
            "(e.g. %s).", len(unusable), labels_dir, unusable[:3],
        )
        return None
    counts: dict[str, int] = {}
    for stem in stems:
        n = 0
        for line in index[stem][0].read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5 and int(float(parts[0])) == DAMAGE_CLASS_ID:
                n += 1
        counts[stem] = n
    return counts


def assign_groups_to_folds(groups: list[CaptureGroup], k: int) -> list[Fold]:
    """Greedy, deterministic balancing of whole groups over `k` folds.

    Groups are placed largest-first (ties: more damage boxes, then group id)
    onto the fold with the fewest images (ties: fewest damage boxes, then
    lowest fold index).
    """
    if k < 2:
        raise ValueError(f"K must be >= 2, got {k}.")
    if k > len(groups):
        raise ValueError(
            f"K={k} folds requested but only {len(groups)} group(s) exist after merging; "
            "every fold needs at least one whole group. Lower --k or --min-group-size."
        )
    folds = [Fold(index=i) for i in range(k)]
    for group in sorted(groups, key=lambda g: (-g.size, -g.damage_boxes, g.group_id)):
        target = min(folds, key=lambda f: (f.size, f.damage_boxes, f.index))
        target.groups.append(group)
    return folds


def build_fold_manifests(folds: list[Fold]) -> list[dict[str, list[str]]]:
    """Return one splits.json-schema manifest per fold (test=k, val=k+1 mod K)."""
    k = len(folds)
    manifests = []
    for i in range(k):
        val_idx = (i + 1) % k
        train = sorted(s for f in folds if f.index not in (i, val_idx) for s in f.stems)
        manifests.append({"train": train, "val": folds[val_idx].stems, "test": folds[i].stems})
    return manifests


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def make_grouped_folds(
    manifest_path: Path,
    output_dir: Path,
    k: int = 5,
    min_group_size: int = 5,
    tz_name: str = "America/Lima",
    labels_dir: Path | None = None,
) -> dict:
    """Generate and write fold manifests plus `folds_meta.json`. Returns the meta dict."""
    manifest_bytes = manifest_path.read_bytes()
    stems = load_manifest_stems(manifest_path)
    if not stems:
        raise ValueError(f"Manifest {manifest_path} lists no stems.")

    groups = group_by_local_day(stems, tz_name)
    damage = count_damage_boxes(labels_dir, stems)
    if damage is not None:
        for g in groups:
            g.damage_boxes = sum(damage[s] for s in g.stems)
    logger.info("Local-day groups (%s): %s", tz_name, [(g.group_id, g.size) for g in groups])

    groups = merge_small_groups(groups, min_group_size)
    for g in groups:
        logger.info("Group %-24s %3d images, %3d damage boxes", g.group_id, g.size, g.damage_boxes)

    folds = assign_groups_to_folds(groups, k)
    manifests = build_fold_manifests(folds)

    output_dir.mkdir(parents=True, exist_ok=True)
    for i, manifest in enumerate(manifests):
        (output_dir / f"fold_{i}.json").write_text(_dump(manifest))
        logger.info(
            "fold_%d: train=%d val=%d test=%d (test damage boxes=%d, groups=%s)",
            i, len(manifest["train"]), len(manifest["val"]), len(manifest["test"]),
            folds[i].damage_boxes, [g.group_id for g in folds[i].groups],
        )

    meta = {
        "tz": tz_name,
        "min_group_size": min_group_size,
        "k": k,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "damage_balancing": damage is not None,
        "num_images": len(stems),
        "groups": {
            g.group_id: {
                "days": g.days,
                "size": g.size,
                "damage_boxes": g.damage_boxes if damage is not None else None,
                "stems": g.stems,
            }
            for g in groups
        },
        "folds": {
            str(f.index): {
                "groups": sorted(g.group_id for g in f.groups),
                "size": f.size,
                "damage_boxes": f.damage_boxes if damage is not None else None,
            }
            for f in folds
        },
    }
    (output_dir / "folds_meta.json").write_text(_dump(meta))
    logger.info("Wrote %d fold manifests + folds_meta.json to %s", k, output_dir)
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build capture-day grouped K-fold manifests.")
    parser.add_argument("--manifest", default="data/annotations/yolo/splits.json",
                        help="Canonical split manifest; its train+val+test union is the image set.")
    parser.add_argument("--output-dir", default="data/annotations/yolo/folds")
    parser.add_argument("--k", type=int, default=5, help="Number of folds (default: 5).")
    parser.add_argument("--min-group-size", type=int, default=5,
                        help="Groups smaller than this merge into the nearest day (default: 5).")
    parser.add_argument("--tz", default="America/Lima",
                        help="Timezone defining the capture calendar day (default: America/Lima).")
    parser.add_argument("--labels-dir", default="data/annotations/yolo/labels",
                        help="Labels root used only for the damage-box tie-breaker (optional).")
    parser.add_argument("--no-label-balance", action="store_true",
                        help="Ignore labels entirely (no damage-box tie-breaking).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels_dir = None if args.no_label_balance else Path(args.labels_dir)
    make_grouped_folds(
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
        k=args.k,
        min_group_size=args.min_group_size,
        tz_name=args.tz,
        labels_dir=labels_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
