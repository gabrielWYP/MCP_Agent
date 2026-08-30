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
from dataclasses import dataclass, field
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
    """Create empty YOLO label files for every split assignment.

    Also prunes any stray ``.txt`` file already present under a split
    directory whose stem is not part of that split's target assignment.
    Without this, re-running this script with a new shuffle leaves behind
    files from the previous assignment (``label_path.touch(exist_ok=True)``
    never removed anything), and ``YOLODataset._load_pairs`` — which reads
    directory contents rather than the manifest — ends up training/evaluating
    on the union of every historical split. See ``reconcile_splits`` for a
    dry-run-capable cleanup of an *existing* manifest/label-directory pair.
    """
    for split, stems in split_map.items():
        split_dir = labels_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        target_stems = set(stems)
        for existing in split_dir.glob("*.txt"):
            if existing.stem not in target_stems:
                existing.unlink()

        for stem in stems:
            label_path = split_dir / f"{stem}.txt"
            label_path.touch(exist_ok=True)


@dataclass(frozen=True)
class ReconcileReport:
    """Result of reconciling on-disk label directories against a split manifest.

    Attributes:
        strays: split -> stems present in that split's directory whose
            manifest-correct split is elsewhere, and that are safe to delete
            (a byte-identical copy already exists at the correct location).
        conflicts: split -> stems that ALSO belong elsewhere per the manifest,
            but where deleting is unsafe (content differs from the correct
            location, or the correct location has no file at all). Fixing
            this requires human review of MUST NOT be silently discarded.
        unmanifested: stems present on disk in some split directory that do
            not appear in the manifest at all. Always reported, never deleted.
        applied: True once the strays have actually been removed from disk.
    """

    strays: dict[str, list[str]] = field(default_factory=dict)
    conflicts: dict[str, list[str]] = field(default_factory=dict)
    unmanifested: list[str] = field(default_factory=list)
    applied: bool = False

    @property
    def has_conflicts(self) -> bool:
        return any(self.conflicts.values())

    @property
    def stray_count(self) -> int:
        return sum(len(stems) for stems in self.strays.values())


def _label_stems(split_dir: Path) -> dict[str, Path]:
    """Map label-file stem -> path for every ``.txt`` file in a directory."""
    if not split_dir.exists():
        return {}
    return {path.stem: path for path in split_dir.glob("*.txt")}


def plan_reconciliation(labels_dir: Path, manifest: dict[str, list[str]]) -> ReconcileReport:
    """Compute what reconciling ``labels_dir`` against ``manifest`` would do.

    Read-only: never touches the filesystem. A stem is a "stray" in split
    ``s`` when the manifest assigns it to a different split ``t`` AND the
    file at ``t`` already exists with byte-identical content (D2 safety
    precondition) — deleting the copy in ``s`` loses nothing. If content
    differs, or no file exists at the correct location, the stem is a
    "conflict" instead: reconciliation must refuse to guess and must not
    delete anything for that stem.
    """
    stem_to_split: dict[str, str] = {}
    for split, stems in manifest.items():
        for stem in stems:
            stem_to_split[stem] = split

    disk_by_split = {split: _label_stems(labels_dir / split) for split in manifest}

    strays: dict[str, list[str]] = {split: [] for split in manifest}
    conflicts: dict[str, list[str]] = {split: [] for split in manifest}
    unmanifested: set[str] = set()

    for split, files_by_stem in disk_by_split.items():
        for stem, path in files_by_stem.items():
            correct_split = stem_to_split.get(stem)
            if correct_split is None:
                unmanifested.add(stem)
                continue
            if correct_split == split:
                continue  # already in the right place

            correct_path = disk_by_split.get(correct_split, {}).get(stem)
            if correct_path is not None and correct_path.read_bytes() == path.read_bytes():
                strays[split].append(stem)
            else:
                conflicts[split].append(stem)

    for split in strays:
        strays[split].sort()
    for split in conflicts:
        conflicts[split].sort()

    return ReconcileReport(
        strays=strays,
        conflicts=conflicts,
        unmanifested=sorted(unmanifested),
        applied=False,
    )


def apply_reconciliation(labels_dir: Path, report: ReconcileReport) -> ReconcileReport:
    """Delete the stray files identified by ``plan_reconciliation``.

    Refuses to run (raises ``ValueError``) if the report carries any
    conflicts — reconciliation is all-or-nothing so a partially-applied
    cleanup can never be mistaken for a clean one.
    """
    if report.has_conflicts:
        raise ValueError(
            "Refusing to apply reconciliation: unresolved conflicts present. "
            "Resolve them manually first (see report.conflicts)."
        )
    for split, stems in report.strays.items():
        for stem in stems:
            (labels_dir / split / f"{stem}.txt").unlink(missing_ok=True)
    return ReconcileReport(
        strays=report.strays,
        conflicts=report.conflicts,
        unmanifested=report.unmanifested,
        applied=True,
    )


def reconcile_splits(
    labels_dir: Path,
    manifest: dict[str, list[str]],
    dry_run: bool = True,
) -> ReconcileReport:
    """Reconcile ``labels_dir`` against ``manifest``.

    Dry-run by default (D2): always safe to call, never deletes anything.
    Pass ``dry_run=False`` to actually remove strays — this still refuses to
    delete anything if any conflicts are present, and never touches
    ``unmanifested`` stems.

    This function's fail-closed behavior on conflicts is unchanged and is
    the DEFAULT path. See ``reconcile_splits_resolving_conflicts`` for the
    opt-in Q11 resolution strategy — it does not weaken this function.
    """
    report = plan_reconciliation(labels_dir, manifest)
    if dry_run or report.has_conflicts:
        return report
    return apply_reconciliation(labels_dir, report)


DAMAGE_CLASS_ID = 1


def _parse_yolo_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file into ``(class_id, cx, cy, w, h)`` tuples."""
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = (float(value) for value in parts[1:5])
        boxes.append((cls_id, cx, cy, w, h))
    return boxes


def _class_stats(path: Path, class_id: int) -> tuple[int, float]:
    """Return ``(box_count, total_area)`` for one class in a label file.

    Area is ``sum(w * h)`` over that class's boxes, in normalized YOLO units.
    """
    matching = [box for box in _parse_yolo_boxes(path) if box[0] == class_id]
    count = len(matching)
    area = sum(w * h for _, _, _, w, h in matching)
    return count, area


@dataclass(frozen=True)
class ConflictResolutionDecision:
    """Auditable record of one Q11 conflict-resolution decision.

    Attributes:
        stem: The conflicting stem.
        kept_split: Split directory the WINNING copy's content physically
            came from (may differ from ``manifest_split`` — the rule ranks
            by damage annotation quality, not by manifest placement).
        kept_damage_count / kept_damage_area: Class-1 stats of the winner.
        discarded_split: Split directory the LOSING copy's content came from.
        discarded_damage_count / discarded_damage_area: Class-1 stats of the loser.
        manifest_split: The split the manifest actually assigns this stem to
            (the destination directory after resolution, regardless of winner).
        rule: Which rule decided it — ``max_damage_count``, ``tiebreak_area``,
            or ``tiebreak_manifest_copy``.
    """

    stem: str
    kept_split: str
    kept_damage_count: int
    kept_damage_area: float
    discarded_split: str
    discarded_damage_count: int
    discarded_damage_area: float
    manifest_split: str
    rule: str


def resolve_conflict(
    stem: str,
    split_a: str,
    path_a: Path,
    split_b: str,
    path_b: Path,
    manifest_split: str,
    damage_class_id: int = DAMAGE_CLASS_ID,
) -> ConflictResolutionDecision:
    """Decide which of two conflicting copies of ``stem`` is authoritative (Q11).

    Rule: keep the copy with MORE class-1 (damage) boxes, regardless of which
    split physically holds it — damage does not vanish from a photograph, so
    a pass that found fewer regions than another pass over the same image is
    the incomplete one, not evidence of a "healthier" image.

    Tiebreak, in order, for equal damage counts:
        1. Larger total class-1 area (more damage captured).
        2. The manifest-correct copy (``manifest_split``).
    """
    count_a, area_a = _class_stats(path_a, damage_class_id)
    count_b, area_b = _class_stats(path_b, damage_class_id)

    if count_a != count_b:
        rule = "max_damage_count"
        a_wins = count_a > count_b
    elif area_a != area_b:
        rule = "tiebreak_area"
        a_wins = area_a > area_b
    else:
        rule = "tiebreak_manifest_copy"
        a_wins = split_a == manifest_split

    if a_wins:
        kept_split, kept_count, kept_area = split_a, count_a, area_a
        disc_split, disc_count, disc_area = split_b, count_b, area_b
    else:
        kept_split, kept_count, kept_area = split_b, count_b, area_b
        disc_split, disc_count, disc_area = split_a, count_a, area_a

    return ConflictResolutionDecision(
        stem=stem,
        kept_split=kept_split,
        kept_damage_count=kept_count,
        kept_damage_area=kept_area,
        discarded_split=disc_split,
        discarded_damage_count=disc_count,
        discarded_damage_area=disc_area,
        manifest_split=manifest_split,
        rule=rule,
    )


def plan_conflict_resolution(
    labels_dir: Path,
    manifest: dict[str, list[str]],
    report: ReconcileReport,
    damage_class_id: int = DAMAGE_CLASS_ID,
) -> list[ConflictResolutionDecision]:
    """Compute a Q11 resolution decision for every conflicting stem in ``report``.

    Read-only: never touches the filesystem. Does not mutate ``report``.
    """
    stem_to_split: dict[str, str] = {}
    for split, stems in manifest.items():
        for stem in stems:
            stem_to_split[stem] = split

    decisions: list[ConflictResolutionDecision] = []
    for wrong_split, stems in report.conflicts.items():
        for stem in stems:
            correct_split = stem_to_split[stem]
            wrong_path = labels_dir / wrong_split / f"{stem}.txt"
            correct_path = labels_dir / correct_split / f"{stem}.txt"
            decisions.append(
                resolve_conflict(
                    stem, wrong_split, wrong_path, correct_split, correct_path,
                    manifest_split=correct_split, damage_class_id=damage_class_id,
                )
            )

    decisions.sort(key=lambda decision: decision.stem)
    return decisions


def apply_conflict_resolution(
    labels_dir: Path,
    decisions: list[ConflictResolutionDecision],
) -> None:
    """Apply Q11 resolution decisions: destructive.

    For each decision, the winning copy's content ends up at the
    manifest-correct location (overwriting an inferior copy there if the
    winner physically lived elsewhere), and every other physical copy of
    that stem is deleted. Caller must gate this behind an explicit opt-in
    (this module never calls it implicitly) and should default to dry-run.
    """
    for decision in decisions:
        manifest_path = labels_dir / decision.manifest_split / f"{decision.stem}.txt"
        if decision.kept_split == decision.manifest_split:
            # Winner is already at the correct location; just remove the loser.
            discarded_path = labels_dir / decision.discarded_split / f"{decision.stem}.txt"
            discarded_path.unlink(missing_ok=True)
        else:
            # Winner lives in the wrong split; promote its content into the
            # manifest-correct location, then remove the old wrong-split file.
            kept_path = labels_dir / decision.kept_split / f"{decision.stem}.txt"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_bytes(kept_path.read_bytes())
            kept_path.unlink(missing_ok=True)


def write_conflict_resolution_log(
    decisions: list[ConflictResolutionDecision],
    unmanifested: list[str],
    output_path: Path,
) -> None:
    """Append an auditable Q11 decision log to ``output_path`` (Markdown).

    Every resolution decision — stem, which copy was kept, which was
    discarded, the damage counts/areas on each, and which rule decided it —
    is recorded so the choice can be audited later, independent of dry-run.
    """
    lines = [
        "## Conflict Resolution Log (Q11)",
        "",
        "Rule: keep the copy with more class-1 (damage) boxes, regardless of which "
        "split physically holds it. Tiebreak, in order: (1) larger total class-1 "
        "area, (2) the manifest-correct copy.",
        "",
        "| Stem | Kept split | Kept damage (count, area) | Discarded split | Discarded damage (count, area) | Rule |",
        "|---|---|---|---|---|---|",
    ]
    for decision in decisions:
        lines.append(
            f"| {decision.stem} | {decision.kept_split} | "
            f"{decision.kept_damage_count}, {decision.kept_damage_area:.6f} | "
            f"{decision.discarded_split} | "
            f"{decision.discarded_damage_count}, {decision.discarded_damage_area:.6f} | "
            f"{decision.rule} |"
        )
    lines.append("")
    lines.append(
        f"Unmanifested stem(s) — reported, never auto-deleted regardless of "
        f"conflict resolution: {unmanifested if unmanifested else 'none'}"
    )
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def reconcile_splits_resolving_conflicts(
    labels_dir: Path,
    manifest: dict[str, list[str]],
    dry_run: bool = True,
    damage_class_id: int = DAMAGE_CLASS_ID,
) -> tuple[ReconcileReport, list[ConflictResolutionDecision]]:
    """Reconcile ``labels_dir`` against ``manifest``, resolving conflicts (Q11).

    OPT-IN ONLY — the default reconciliation path (``reconcile_splits``) stays
    fail-closed. Callers must explicitly choose this function (the CLI gates
    it behind ``--resolve-conflicts``) to apply the maintainer-approved
    resolution rule instead of refusing on conflicting content.

    Dry-run remains the default here too: pass ``dry_run=False`` to actually
    write/delete anything. The single unmanifested stem is still only ever
    reported, never auto-deleted, regardless of this flag.

    Returns the ORIGINAL plan (every stray and conflict found, unmodified)
    alongside the resolution decision computed for each conflicting stem, so
    every decision is inspectable even in dry-run mode.
    """
    report = plan_reconciliation(labels_dir, manifest)
    decisions = plan_conflict_resolution(labels_dir, manifest, report, damage_class_id)

    if dry_run:
        return report, decisions

    if decisions:
        apply_conflict_resolution(labels_dir, decisions)

    # Safe (byte-identical) strays are independent of conflicts and can
    # always be removed once conflicts are resolved above.
    stray_only_report = ReconcileReport(
        strays=report.strays,
        conflicts={split: [] for split in report.conflicts},
        unmanifested=report.unmanifested,
        applied=False,
    )
    applied_report = apply_reconciliation(labels_dir, stray_only_report)
    return applied_report, decisions


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
    parser.add_argument(
        "--reconcile",
        action="store_true",
        help=(
            "Reconcile the on-disk label directories against --manifest instead of "
            "generating a new split. Removes stray label files left over from an "
            "earlier shuffle (see D2 in openspec/changes/damage-map-audit/design.md)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With --reconcile: report what would change without deleting anything "
            "(default: True). Pass --no-dry-run to actually delete strays."
        ),
    )
    parser.add_argument(
        "--resolve-conflicts",
        action="store_true",
        help=(
            "OPT-IN (default off): instead of refusing on annotation conflicts, apply "
            "the Q11 resolution rule — keep the copy with more class-1 (damage) boxes, "
            "tiebreak by area then the manifest-correct copy. Does not weaken the "
            "default fail-closed behavior; the unmanifested stem is still only ever "
            "reported. See openspec/changes/damage-map-audit/proposal.md Round 5 Q11."
        ),
    )
    parser.add_argument(
        "--damage-class-id",
        type=int,
        default=DAMAGE_CLASS_ID,
        help="Class ID treated as 'damage' for --resolve-conflicts (default: 1).",
    )
    parser.add_argument(
        "--decision-log",
        default="reports/damage-map-audit/w0-reconcile.md",
        help=(
            "With --resolve-conflicts: path to append the auditable decision log to "
            "(default: reports/damage-map-audit/w0-reconcile.md)."
        ),
    )
    return parser.parse_args()


def _run_reconcile(args: argparse.Namespace) -> int:
    """Handle ``--reconcile`` mode. Returns a process exit code."""
    labels_dir = Path(args.labels_dir)
    manifest_path = Path(args.manifest)

    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return 1

    manifest = json.loads(manifest_path.read_text())

    decisions: list[ConflictResolutionDecision] = []
    if args.resolve_conflicts:
        report, decisions = reconcile_splits_resolving_conflicts(
            labels_dir, manifest, dry_run=args.dry_run, damage_class_id=args.damage_class_id,
        )
        for decision in decisions:
            logger.info(
                "Resolved '%s': kept %s copy (damage=%d, area=%.6f) over %s copy "
                "(damage=%d, area=%.6f) — rule=%s",
                decision.stem, decision.kept_split, decision.kept_damage_count,
                decision.kept_damage_area, decision.discarded_split,
                decision.discarded_damage_count, decision.discarded_damage_area, decision.rule,
            )
        write_conflict_resolution_log(decisions, report.unmanifested, Path(args.decision_log))
    else:
        report = reconcile_splits(labels_dir, manifest, dry_run=args.dry_run)

    if report.unmanifested:
        logger.warning(
            "%d stem(s) present on disk but absent from the manifest entirely "
            "(never deleted, needs manual review): %s",
            len(report.unmanifested),
            report.unmanifested,
        )

    if report.has_conflicts and not args.resolve_conflicts:
        for split, stems in report.conflicts.items():
            for stem in stems:
                logger.error(
                    "Conflict: '%s' is misplaced in split '%s' but its manifest-correct "
                    "location is missing or byte-different — refusing to delete. "
                    "Pass --resolve-conflicts to apply the Q11 resolution rule instead.",
                    stem,
                    split,
                )
        logger.error("Reconciliation aborted: %d conflicting stem(s), nothing deleted.",
                      sum(len(v) for v in report.conflicts.values()))
        return 1

    verb = "Would remove" if args.dry_run else "Removed"
    for split, stems in report.strays.items():
        if stems:
            logger.info("%s %d stray label file(s) from split '%s': %s", verb, len(stems), split, stems)

    if args.dry_run:
        logger.info(
            "Dry run: %d stray file(s) would be removed%s, nothing was deleted. "
            "Re-run with --no-dry-run to apply.",
            report.stray_count,
            f" and {len(decisions)} conflict(s) would be resolved" if args.resolve_conflicts else "",
        )
    else:
        logger.info(
            "Reconciliation complete: removed %d stray label file(s)%s.",
            report.stray_count,
            f", resolved {len(decisions)} conflict(s)" if args.resolve_conflicts else "",
        )

    return 0


def main() -> int:
    """Run split preparation."""
    args = parse_args()
    rgb_dir = Path(args.rgb_dir)
    nir_dir = Path(args.nir_dir)
    labels_dir = Path(args.labels_dir)

    if args.reconcile:
        return _run_reconcile(args)

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
