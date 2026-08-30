# W0 — Split Reconciliation Report

Change: `damage-map-audit` · Phase 0, task 0.5 · Initial run: 2026-08-29 · Resolved: 2026-08-29
(Round 5, Q11 conflict resolution)

## Result: RESOLVED — split reconciled to 148/18/20, zero cross-split leakage

## History

### Initial finding (pre-Round-5): BLOCKED

The reconciliation tool (`scripts/prepare_yolo_splits.py::reconcile_splits`) ran and found that
**automatic reconciliation could not safely resolve this dataset** without a resolution policy:

| Category | Count |
|---|---|
| Manifest total (148 train / 18 val / 20 test) | 186 |
| Safe strays (byte-identical duplicate at the correct location) | 0 |
| Conflicts (misplaced copy's content differs from the correct location's copy) | 21 |
| Unmanifested (present on disk, absent from the manifest entirely) | 1 |

Every one of the 21 misplaced stems had real, non-identical annotation content at both its
current (wrong) location and its manifest-correct location — the class-0 mango box was
byte-identical between copies in the 4 tied cases, but the class-1 (damage) boxes differed in
count or position. Deleting either copy risked destroying genuine annotation data, so
`reconcile_splits` correctly refused (fail-closed, D2) and made no changes.

### Round 5 — Q11: maintainer resolution policy (RESOLVED 2026-08-29)

**Rule: keep the copy with the most class-1 (damage) boxes, regardless of which split holds it.**
Damage does not vanish from a photograph, so between a pass that found N damage regions and a
pass that found fewer, the one with fewer is the incomplete annotation.

**Tiebreak for equal damage counts, in order:**
1. Larger total class-1 area (`sum(w*h)` over damage boxes) — more damage captured.
2. If still tied, the manifest-correct copy.

Implemented as an OPT-IN destructive path (`--resolve-conflicts`, dry-run by default, same as the
base `--reconcile`) — `reconcile_splits_resolving_conflicts()` in `scripts/prepare_yolo_splits.py`.
The default `reconcile_splits()` / `--reconcile` (without `--resolve-conflicts`) is UNCHANGED and
still fails closed on any conflict. The unmanifested stem is still only ever reported, never
auto-deleted, by both paths.

### Outcome of the 21 conflicts

17 resolved by `max_damage_count` (the copy with strictly more damage boxes won, in 6 of those 17
cases the winning copy was NOT the manifest-correct one — content quality decided the winner, and
its content was then promoted into the manifest-correct location). The remaining 4 resolved by
`tiebreak_area` (equal damage count, larger total area won) — these are exactly the 4 equal-count
conflicts flagged in Round 5 (e.g. `mango_rgb_1780238853`, where a damage box sits at y=0.435 in
one copy and y=0.358 in the other — a different location, not jitter). Zero conflicts reached the
second tiebreak level (`tiebreak_manifest_copy`); no exact area tie occurred in practice.

## Conflict Resolution Log (Q11)

Rule: keep the copy with more class-1 (damage) boxes, regardless of which split physically holds
it. Tiebreak, in order: (1) larger total class-1 area, (2) the manifest-correct copy.

| Stem | Kept split | Kept damage (count, area) | Discarded split | Discarded damage (count, area) | Rule |
|---|---|---|---|---|---|
| mango_rgb_1780237660 | train | 1, 0.006123 | val | 0, 0.000000 | max_damage_count |
| mango_rgb_1780238785 | test | 2, 0.005603 | train | 1, 0.008456 | max_damage_count |
| mango_rgb_1780238853 | train | 2, 0.005681 | test | 2, 0.004169 | tiebreak_area |
| mango_rgb_1780238898 | train | 1, 0.003126 | test | 1, 0.002878 | tiebreak_area |
| mango_rgb_1780241873 | test | 1, 0.008190 | train | 1, 0.004137 | tiebreak_area |
| mango_rgb_1780241925 | test | 1, 0.008219 | train | 1, 0.006738 | tiebreak_area |
| mango_rgb_1780676138 | train | 4, 0.007347 | test | 2, 0.009414 | max_damage_count |
| mango_rgb_1780676273 | train | 3, 0.006580 | test | 2, 0.007980 | max_damage_count |
| mango_rgb_1780681072 | train | 1, 0.000792 | test | 0, 0.000000 | max_damage_count |
| mango_rgb_1780681297 | train | 2, 0.006663 | val | 1, 0.009587 | max_damage_count |
| mango_rgb_1780681321 | train | 3, 0.008147 | val | 2, 0.007834 | max_damage_count |
| mango_rgb_1780683326 | val | 1, 0.003031 | train | 0, 0.000000 | max_damage_count |
| mango_rgb_1780683618 | val | 2, 0.006608 | train | 0, 0.000000 | max_damage_count |
| mango_rgb_1780683932 | val | 3, 0.010189 | train | 0, 0.000000 | max_damage_count |
| mango_rgb_1780684450 | val | 8, 0.103080 | train | 0, 0.000000 | max_damage_count |
| mango_rgb_1780684735 | val | 3, 0.014394 | test | 2, 0.013221 | max_damage_count |
| mango_rgb_1780685184 | val | 1, 0.005901 | test | 0, 0.000000 | max_damage_count |
| mango_rgb_1780685426 | val | 12, 0.105899 | train | 0, 0.000000 | max_damage_count |
| mango_rgb_1780685537 | test | 1, 0.004270 | train | 0, 0.000000 | max_damage_count |
| mango_rgb_1780685562 | test | 1, 0.003851 | train | 0, 0.000000 | max_damage_count |
| mango_rgb_1780685619 | test | 1, 0.009612 | train | 0, 0.000000 | max_damage_count |

For each row, "kept" content is now at the manifest-correct location (overwritten there if the
winner physically lived in the wrong split); "discarded" content was deleted from wherever it
physically was. Nothing outside these 21 stems and the pre-existing (zero) safe strays was
touched.

## Command run for real (destructive, applied)

```
./.venv/bin/python scripts/prepare_yolo_splits.py --reconcile --resolve-conflicts --no-dry-run
```

Output: `Reconciliation complete: removed 0 stray label file(s), resolved 21 conflict(s).`

## Verified final state (2026-08-29)

```
train: 148 files   (manifest: 148)
val:    18 files   (manifest: 18)
test:   21 files   (manifest: 20, +1 unmanifested — see below)

train ∩ val  = {} (0)
train ∩ test = {} (0)
val   ∩ test = {} (0)
```

**Split sizes match the manifest exactly (148/18/20) and all pairwise intersections are empty.**
The leakage defect this phase exists to fix (`train ∩ val = 8`, `train ∩ test = 11`,
`val ∩ test = 2` stems, pre-fix) is closed.

`YOLODataset`'s manifest guard (D1) confirms this directly: loading `split="train"` and
`split="val"` with `manifest_path=data/annotations/yolo/splits.json` now succeeds cleanly
(148 and 18 pairs respectively). Loading `split="test"` still raises — see below.

## Residual: 1 unmanifested stem — still open, by design

`mango_rgb_1780685735` — present in `labels/test/` on disk, absent from `splits.json` under any
split. Per the binding Round 5 requirement ("The single stem absent from the manifest is still
REPORTED, never auto-deleted... do not let the new strategy swallow it"), this file was NOT
touched by `--resolve-conflicts` and is not part of the 21 resolved conflicts above. It is a
SEPARATE, still-open item: the maintainer must decide whether to add it to a split in
`splits.json` or delete it. Until that decision is made, `YOLODataset(split="test",
manifest_path=...)` will correctly raise (fail-closed, D1) rather than silently include or drop
it — evaluate_checkpoint.py --split test will not run until this is resolved (or manifest_path is
explicitly set to None to bypass the guard, which is not recommended).

## Status

- **Task 0.5 is COMPLETE.** The resulting split is 148/18/20 with zero cross-split leakage,
  matching the manifest exactly aside from the disclosed, still-open unmanifested stem.
- Every AP figure computed against the PRE-reconciliation label directories remains invalid, per
  the original leakage finding. Phase 4 (E0 onward) may now proceed using this reconciled split.
- The one remaining follow-up (the unmanifested stem in `test/`) is unrelated to Q11 and requires
  a separate maintainer decision before `--split test` evaluation can run cleanly with the
  manifest guard enabled.
