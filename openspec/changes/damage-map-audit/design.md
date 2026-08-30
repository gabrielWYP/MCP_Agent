# Design: Damage-class detection audit — assigner + eval-metric fixes

Change id: `damage-map-audit` · Store: hybrid · Inputs: `proposal.md` (authoritative, incl. the four
binding Maintainer Decisions), `explore.md` (background).

---

## Prerequisite P1 — the 27-vs-58 GT discrepancy: RESOLVED (not a metrics bug)

**`metrics.py` is exonerated.** In `_compute_operating_point` (`src/training/metrics.py:219-247`),
`tp` increments exactly once per element added to `matched`, and `fn += len(targets) - len(matched)`
(line 240). Therefore `tp + fn == Σ len(targets)` identically. No size filter, no `max_det`, no
truncation, no remap. The 27 is a faithful report: **only 27 class-1 GT boxes ever reached
`compute_map`.**

**The real defect is split drift.** Two independent measurements pin it:

| Source | mango GT | damage GT |
|---|---|---|
| `labels/val/` directory on disk today (25 files) | 25 | 58 |
| `splits.json` `"val"` list (18 stems) | 18 | 28 |
| Run `20260710T204144308622Z` estudiante, every epoch (`tp+fn`) | **18** | **27** |

The mango column matches `splits.json` **exactly** (18 files × 1 mango box). The run evaluated the
18-image manifest split; the directory has since grown to 25.

Mechanism: `scripts/prepare_yolo_splits.py:105-112` (`write_empty_label_files`) only calls
`label_path.touch(exist_ok=True)` — it **never removes** files from a previous assignment. Because
`YOLODataset._load_pairs` (`src/training/dataset.py:206-210`) keys off *directory contents*, not the
manifest, the effective split is the union of every historical shuffle. Verified consequences today:

- `labels/val/` holds 7 stems that `splits.json` assigns elsewhere — 5 to train
  (`1780683326, 1780683618, 1780683932, 1780684450, 1780685426`) and 2 to test
  (`1780684735, 1780685184`). Those 7 carry exactly the 30 extra damage boxes.
- `labels/train/` holds 3 manifest-val stems (`1780237660, 1780681297, 1780681321`) plus test stems.

**Both directions leak.** Running Step A/B against the current directory would validate on training
images. This does **not** block the plan, but it inserts a mandatory **W0** before A0/A/B.

Residual: manifest-val expects 28 damage, the run reported 27 — a **one-box** delta, attributable to
the val bbox filter (`augmentations.py:153`, `min_visibility=0.1 / min_area=1.0`) acting on boxes
whose components were clipped independently at `dataset.py:393`. W0-c makes this self-reporting.

## Prerequisite P2 — training cost: MEASURED

Method: stages run sequentially per `run_summary.json`; each stage's TensorBoard file embeds its
creation unix time, so consecutive stage starts bound each stage's wall clock. Epochs = CSV rows.

| Run | estudiante epochs | wall span | s/epoch | maestro | s/epoch |
|---|---|---|---|---|---|
| `20260705T004211Z` | 46 | 317 s | 6.9 | 63 / 560 s | 8.9 |
| `20260705T215557Z` | 50 | 330 s | 6.6 | 80 / 792 s | 9.9 |
| `20260710T204144308622Z` | 33 | 220 s | 6.7 | 74 / 677 s | 9.1 |

**Step B costs 4–6 minutes**, not hours: `epochs: 50` capped, patience 15 → 33–50 epochs at
≈6.7 s/epoch = 220–335 s. The 84-epoch worst case is ≈9.5 min. Caveat: measured on the cloud GPU
instance (`ip-10-192-*`, `/home/zeus/miniconda3/envs/cloudspace`). Local WSL2 CPU is unmeasured and
expected 1–2 orders slower — **Step B must run on the GPU instance.** The Q3 authorisation is
comfortably cheap; repeating Step B across 3 seeds costs ~20 min and is recommended.

---

## Architecture Decisions

| # | Decision | Chosen | Rejected | Rationale |
|---|---|---|---|---|
| D1 | Split integrity | Fail-fast guard in `YOLODataset` + `prepare_yolo_splits.py --reconcile` | Make the dataset read `splits.json` directly | Silent auto-correction hides that runs used a different split; a loud error forces one explicit reconciliation commit |
| D2 | Reconcile safety | Dry-run default; delete a stray only when the manifest-correct split already holds a byte-identical file; otherwise report and exit non-zero | Unconditional delete | Label files carry annotation content; a destructive op must fail closed |
| D3 | Decode sync (E4) | Extract `src/training/decode.py::decode_detections`; `Trainer._decode_predictions` and `visualize_damage_predictions.decode_predictions` become thin wrappers | Patch both call sites | They already drifted — the script uses unclamped `reg.exp()` (line 341) vs. the clamped `_decode_bboxes` (`loss.py:400`). One function is the only durable fix |
| D4 | NMS | `torchvision.ops.batched_nms` (present: `torchvision/ops/boxes.py:51`) | Per-class `nms` loop; local impl | One vectorized call gives per-class semantics via coordinate offsetting |
| D5 | E3 decode | Emit one candidate per `(class, anchor)` above threshold | Keep `scores.max(dim=0)` (`loop.py:463`) | Binding decision Q1; 94% nesting makes argmax structurally erase damage |
| D6 | Center sampling (A1) | Mask the alignment matrix **before** `topk`: anchor valid if inside the GT box **or** within `center_radius × stride_of_that_anchor` of the GT center; `center_radius = 2.5` (stride units) | Post-`topk` filter (status quo, `loss.py:132-143`) | Filtering after `topk` wastes the budget on invalid anchors and can empty a GT; masking first makes `topk` return only usable candidates |
| D7 | Fallback (A2) | Runs **after** the spatial mask and top-k; falls back to the nearest **anchor** centre from `anchors` | Status quo at `loss.py:106-115` | Two bugs: it runs before the filter that erases it, and it ranks by *predicted* box centres (`pred_bboxes`, lines 110-111) while the filter tests *anchor* centres (line 126) |
| D8 | Per-level policy (A3) | FCOS-style size bins as a second pre-`topk` mask: `max(w,h) < 64 → P3/s8`, `< 128 → P4/s16`, else `P5/s32`; open-ended, no GT can be orphaned | Leave all levels eligible | Damage median 30.2 px never belongs on stride 32; binning also removes most mango/damage anchor competition — the explore doc's winner-take-all mechanism |
| D9 | Rollout of W1 | Ships behind config, default **off** (`assigner_center_radius: 0.0` = legacy strict containment) | Ship enabled | Lets A0 measure pre-fix counts on the shipped build; flipping one key is the whole Step B delta and the whole rollback |
| D10 | Eval entrypoint | New `scripts/evaluate_checkpoint.py`; extract `Trainer._validate` body into public `Trainer.evaluate()` | Reuse `train.py` with `epochs=0` | No eval-only path exists today; `_validate` needs no `train_loader` |

## Data Flow

```
labels/{split}/ ──┐
                  ├─► YOLODataset ──► collate_fn ──► val_loader ──► model
splits.json ──────┘   [W0 guard: dir == manifest, else raise]              │
                                                                          ▼
                                              src/training/decode.py :: decode_detections
                                              per-class threshold (E3) → batched_nms (E1)
                                                     │                         │
                              Trainer.evaluate() ◄───┘                         └──► visualize_damage_predictions.py
                                     │
                          compute_map(score_threshold=cfg.conf_threshold)

Training: YOLOv8Loss ─► _generate_anchors → (anchors, anchor_strides)
          ─► TaskAlignedAssigner: [level-size mask] ∧ [center-sampling mask] → alignment.topk
             → >0 filter → nearest-anchor fallback (guarantees ≥1) → conflict resolution
             → optional stats sink → assigner_stats.csv
```

## File Changes

| File | Action | What |
|---|---|---|
| `scripts/prepare_yolo_splits.py` | Modify | `--reconcile` / `--dry-run`; prune strays per D2 |
| `src/training/dataset.py` | Modify | `_load_pairs` manifest guard (D1); GT-count report (W0-c) |
| `src/training/decode.py` | **Create** | `decode_detections(...)` — shared, model-free (D3/D4/D5) |
| `src/training/loop.py` | Modify | `_decode_predictions` → wrapper; public `evaluate()`; pass `conf_threshold` to `compute_map` |
| `src/training/metrics.py` | Modify | `compute_map(..., score_threshold)` — drop the hardcoded `0.25` at line 315 |
| `src/training/config.py` | Modify | `class_weights` default `[2.7, 0.5]` → `[0.5, 1.5]` (line 76); new fields below |
| `src/training/loss.py` | Modify | `_generate_anchors` returns per-anchor strides; assigner D6/D7/D8/D9 + `collect_stats` |
| `scripts/evaluate_checkpoint.py` | **Create** | Eval-only + A0 instrumentation entrypoint |
| `scripts/visualize_damage_predictions.py` | Modify | Delegate to `decode_detections` |
| `configs/*.yaml` | Modify | New threshold/assigner keys |
| `openspec/config.yaml` | Modify | Delete stale DualFPN note (line 11); refresh runner to `./.venv/bin/python -m pytest`, 93 tests |
| `openspec/changes/pipeline-orchestration-docs/explore.md` | Modify | Delete the `[1.27, 0.83]` claim |
| `tests/test_decode.py`, `tests/test_assigner.py`, `tests/test_split_integrity.py` | **Create** | See protocol |

## Interfaces

```python
# src/training/decode.py
def decode_detections(
    preds: list[Tensor], cls_preds: list[Tensor], *, batch_idx: int,
    num_classes: int, image_size: int, strides: Sequence[int] = (8, 16, 32),
    conf_threshold: float = 0.25, nms_iou_threshold: float = 0.5,
    nms_enabled: bool = True, per_class_candidates: bool = True,
    max_detections: int = 300, normalize: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:  # boxes cxcywh, scores, labels
```

New `TrainingConfig` fields: `conf_threshold=0.25`, `nms_iou_threshold=0.5`, `nms_enabled=True`,
`decode_per_class=True`, `max_detections=300`, `assigner_center_radius=0.0`,
`assigner_level_ranges=[64,128]`, `assigner_collect_stats=False`.

## Evaluation Protocol (validation mechanics)

Prerequisite: **PR1 merged and `splits.json` reconciled.** The `final_runs/*/**.pt` checkpoints are
**not in the repo** (only CSV/PNG/JSON/tfevents survive). Use `checkpoints/student/best_model.pt`, or
have the maintainer restore the estudiante `best_model.pt` from the GPU instance for continuity with
the recorded 0.041.

```bash
# Step A — NMS-corrected baseline (minutes, no training). Run twice from one checkpoint.
./.venv/bin/python scripts/evaluate_checkpoint.py --config configs/training_student.yaml \
  --model student --checkpoint checkpoints/student/best_model.pt --split val \
  --override nms_enabled=false decode_per_class=false --output-dir reports/damage-map-audit/A-pre
./.venv/bin/python scripts/evaluate_checkpoint.py --config configs/training_student.yaml \
  --model student --checkpoint checkpoints/student/best_model.pt --split val \
  --override nms_enabled=true decode_per_class=true nms_iou_threshold=0.5 \
  --output-dir reports/damage-map-audit/A

# Step A0 — kill switch (minutes, no training, legacy assignment)
./.venv/bin/python scripts/evaluate_checkpoint.py --config configs/training_student.yaml \
  --model student --checkpoint checkpoints/student/best_model.pt --split train \
  --assigner-stats --override assigner_center_radius=0.0 \
  --output-dir reports/damage-map-audit/A0-pre       # → assigner_stats.csv

# Step B — only if A0 confirms. 4–6 min on the GPU instance.
./.venv/bin/python -m src.training.train --config configs/training_student.yaml --model student \
  --override output_dir=checkpoints/exp/damage-map-audit-B assigner_center_radius=2.5
# then re-run A0 with assigner_center_radius=2.5 → median positives/damage GT MUST be ≥ 3.0
```

Pre-registered bars are unchanged from the proposal (A0 confirm <1.0 vs ≥5.0; refute ≥3.0;
B vs A: ≥2× and ≥+0.10; guardrail `AP50_c0(B) ≥ 0.9 × AP50_c0(A)`). Per Q4, an A0 refutation ships
PR1+PR2 and leaves `assigner_center_radius` at 0.0.

## Testing Strategy

| Layer | Test | Approach |
|---|---|---|
| Unit | Sub-stride GT gets ≥1 positive | Synthetic 4 px GT at 640 px; assert `fg_mask.sum() ≥ 1` |
| Unit | Level binning | 30 px GT → only stride-8 anchors; 200 px GT → only stride-32 |
| Unit | Legacy default | `assigner_center_radius=0.0` reproduces current assignment byte-for-byte |
| Unit | Per-class decode (E3) | Anchor with both classes above threshold emits 2 candidates |
| Unit | NMS | 3 overlapping same-class boxes → 1; overlapping cross-class → 2 |
| Unit | Decode parity | `decode_detections` output identical for the trainer and script wrappers |
| Unit | Split guard | Directory ≠ manifest → raises; reconcile dry-run reports and mutates nothing |
| Integration | GT-count report | Loaded GT counts equal a direct label-file count for the reconciled split |
| Regression | `./.venv/bin/python -m pytest` | 93 existing + new stay green (~80 s) |

## Threat Matrix

N/A — no routing, shell, subprocess, VCS/PR automation, executable-file classification, or
process-integration boundary. One destructive-filesystem boundary exists
(`prepare_yolo_splits.py --reconcile`); it fails closed per D2 (dry-run default, byte-identity
precondition, non-zero exit on conflict) and is covered by the split-guard tests.

## Delivery — budget exceeded, chain required

`review_budget_lines: 800`. Authored estimate with E3 and the new W0: W0 ≈155, W2+decode ≈330,
eval entrypoint ≈195, W1 ≈340, W3 ≈25, docs ≈60 → **≈1105 lines**. Chain:

| PR | Scope | ~lines | Unblocks |
|---|---|---|---|
| PR1 | W0 split integrity + W3 hygiene | 180 | A trustworthy val split |
| PR2 | W2 decode/NMS/E3 + `evaluate_checkpoint.py` | 525 | Step A baseline |
| PR3 | W1 assigner + instrumentation, default off (D9) | 340 | Step A0, then Step B |
| PR4 | Validation report (docs only) | 60 | — |

`Decision needed before apply: Yes` · `Chained PRs recommended: Yes` · `400-line budget risk: High`

## Migration / Rollout

No data migration. PR1 changes the val set (25 → 18 images) — every AP figure recorded before PR1 is
superseded and must be marked so. All behaviour is config-gated (D9), so rollback is a config edit;
code revert is per-PR and independent.

## Open Questions

- [ ] Restore the estudiante `best_model.pt` from the GPU instance, or accept
      `checkpoints/student/best_model.pt` as the Step A subject? (Blocks Step A, not the code.)
- [ ] `labels/train/` also contains manifest-val stems — reconciliation shrinks the train set too.
      Confirm no re-annotation is expected before PR1 prunes.
