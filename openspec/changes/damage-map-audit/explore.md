# Explore: Damage-class (class 1) catastrophic mAP audit

Status: complete (exploration only — no fixes implemented)
Change id: `damage-map-audit`
Artifact store: hybrid (this file + Engram observation `sdd/damage-map-audit/explore`, id 378)

## Problem

Class 1 (damage) scores an AP50 of 0.04–0.09 in every training run ever recorded, while
class 0 (mango) reaches up to 0.98 in the same runs. The gap is systematic, not run-specific.

| Run | mAP50 | AP50 class_0 (mango) | AP50 class_1 (damage) |
|---|---|---|---|
| 20260705T004211Z maestro | 0.216 | 0.382 | 0.051 |
| 20260705T004211Z estudiante | 0.394 | 0.747 | 0.041 |
| 20260705T004211Z destilado | 0.488 | 0.912 | 0.065 |
| 20260705T215557Z estudiante | 0.335 | 0.579 | 0.090 |
| 20260705T215557Z destilado | 0.520 | 0.977 | 0.062 |
| latest `.partial` run | 0.256 | 0.474 | 0.038 |

## Executive summary

The most probable root cause is a **structural mismatch between the dataset's label semantics
and a winner-take-all detection architecture**.

Class 0 (mango) is a whole-fruit box. Class 1 (damage) is a sub-region box that lives *inside*
that mango box. The two classes occupy the same pixels by construction. But the architecture
can only ever emit **one class and one box per anchor**, at three independent layers. Wherever
damage and mango compete for the same anchor, one of them is erased.

This is not a hyperparameter problem. `class_weights` already weights damage 3x higher and has
not moved the disparity, because loss weighting cannot fix an anchor that was never assigned.

## Measured evidence

### Box nesting is total (verified, all splits)

Measured directly over every label file in `data/annotations/yolo/labels/`:

```
images containing both classes: 163
damage boxes:                   335
  fully inside a mango box:     315  (94.0%)
  center inside, not contained:  20  (6.0%)
  outside any mango box:          0  (0.0%)
```

Every single damage box overlaps a mango box. None is independent.

### Instance counts (verified) — damage is NOT the rare class

```
split   files   class_0 (mango)   class_1 (damage)
train    159          159               244
val       25           25                58
test      24           24                33
```

Damage instances **outnumber** mango instances in every split. This disproves the
"class 1 is instance-rare" framing that motivated the class-imbalance hypothesis.

### The three winner-take-all layers (verified in code)

1. **Assignment** — `src/training/loss.py:148-152`. `TaskAlignedAssigner` resolves conflicts by
   keeping only the GT with the highest alignment score per anchor:
   `if target_scores[b, idx] < score or target_classes[b, idx] == -1:` then overwrite.
   One anchor, one GT, one class.
2. **Regression head** — `src/models/master/head.py:65`.
   `self.reg_pred = nn.Conv2d(in_channels, 4, kernel_size=1)` emits 4 values per anchor,
   not `4 * num_classes`. The box is class-agnostic: an anchor cannot describe both the
   mango box and the damage box it contains.
3. **Decode** — `src/training/loop.py:462-463`. `max_scores, max_labels = scores.max(dim=0)`
   keeps one winning class per anchor and discards the other class's score at that location.

### Supporting signal

In `checkpoints/final_runs/20260705T215557Z/destilado/.../metrics_history.csv`, per-class AP
oscillates in near mutual exclusion epoch to epoch — ten consecutive epochs with class_1 AP at
exactly 0.0 while class_0 swings 0.0 to 0.92, and an epoch with class_0 at 0.0 while class_1 is
0.038. A smoothly-low curve would indicate genuine task difficulty; this alternation is the
signature of representation collapse under competition. (Reported by the explore agent;
treated here as supporting evidence, not as the primary proof.)

## Ranked candidate root causes

### 1. Nested taxonomy vs. winner-take-all architecture — MODEL defect, high confidence

Evidence above. Reproduces identically in teacher, student and distilled student, so it is not
specific to any one architecture.

Cheapest confirming experiment: instrument `TaskAlignedAssigner` to log positive-anchor counts
per class during a forward pass over val data using an existing checkpoint. No retraining needed.
If class 1 receives near-zero positive anchors, the mechanism is confirmed.

### 2. Damage boxes are very small — MODEL defect, medium-high

Damage boxes measure as little as ~5-16 px on a 640 px input, at or below a single stride-8 cell.
Compounds cause 1: a small box wins fewer anchors in any alignment-score contest.

### 3. No NMS in the eval decode — METRIC defect, high confidence, does NOT explain the gap

`src/training/loop.py:433-515` takes every anchor above a hardcoded 0.25 threshold across
strides 8/16/32 (8400 anchors at 640) with zero suppression. The latest run shows 725 false
positives for 18 true positives on class 0, and 1102 for 16 on class 1.

This makes every absolute AP number untrustworthy and must be fixed. But it depresses both
classes roughly equally, so it cannot explain a 10-20x *disparity* between them.

Cheapest experiment: re-run `compute_map` over an existing checkpoint's saved predictions with
`torchvision.ops.nms` applied first. No retraining.

### 4. KD transfers the teacher's damage-blindness — MODEL defect, medium

`kd_trainer.py` combines `det_loss + kd_weight(1.0) * kd_loss` against a teacher whose own damage
AP is near zero. The distilled student's class_1 AP is not meaningfully better than the plain
student's despite much higher overall mAP.

Cheapest experiment: one short run with `kd_weight=0`.

### 5. Teacher underperforms its own student — MODEL defect, explains a SEPARATE anomaly

`MasterModel.freeze_backbone()` (`src/models/master/master_model.py:170-198`) unconditionally
freezes `rgb_stem`. The production maestro run's own CSV contains only `phase=1` rows: Phase 2
backbone unfreeze never ran.

This plausibly explains why maestro (0.216) underperforms estudiante (0.394) overall. It does
not explain the class gap, which is equally present in the fully-unfrozen student.

### 6. Class imbalance — DOWNGRADED, disproved as stated

Damage outnumbers mango in every split (see counts above). The rare-class framing is wrong.

## Metric defects vs. model defects

The distinction matters because the fixes are unrelated:

- **Metric defects** (AP is measured wrongly): missing NMS (cause 3), and the unresolved
  GT-count discrepancy below. Fixing these changes the *numbers* without changing the model.
- **Model defects** (the model genuinely fails): the winner-take-all mismatch (1), small boxes (2),
  KD transfer (4), frozen backbone (5). Fixing these changes what the model *learns*.

Fix the metric first. Until NMS exists, no measurement can be trusted to evaluate any model fix.

## Verified vs. inferred

| Claim | Status |
|---|---|
| 100% of damage boxes overlap a mango box; 94% fully contained | Verified — measured over all label files |
| Damage outnumbers mango in all three splits | Verified — measured |
| TAL keeps one GT per anchor | Verified — `loss.py:148-152` |
| Regression head is class-agnostic (4 outputs, not 4*nc) | Verified — `head.py:65` |
| Decode keeps one class per anchor | Verified — `loop.py:462-463` |
| No NMS anywhere in the eval path | Verified — `loop.py:433-515` |
| `class_weights: [0.5, 1.5]` in all three configs | Verified — `configs/*.yaml` |
| Phase 2 unfreeze never ran in the production maestro run | Reported by explore agent, not independently re-verified |
| Per-class AP oscillation indicates representation collapse | Inferred from the metrics CSV, not instrumented |
| Class 1 receives near-zero positive anchors | **Hypothesis** — the central claim, not yet measured |

## Open questions for the maintainer

1. **GT count discrepancy.** The latest run reports `tp_class_1 + fn_class_1 = 27` for validation,
   but the val labels contain 58 damage instances. Where do the other 31 go? If the metric is not
   seeing all ground truth, that is a second metric defect independent of NMS.
2. **`class_weights` has three different values in the repo.** `configs/*.yaml` set `[0.5, 1.5]`;
   the dataclass default at `src/training/config.py:76` is `[2.7, 0.5]` — inverted, favouring mango;
   and the sibling `pipeline-orchestration-docs/explore.md` claims `[1.27, 0.83]`. Which is intended?
3. **Is Phase 2 backbone unfreeze intentionally skipped** for the production maestro run?
4. **Should the taxonomy itself change?** Given that damage is always a sub-region of a mango,
   is single-label detection the right formulation at all, versus damage-only detection,
   segmentation, or a two-stage crop-then-classify design?

## Stale documentation found

- `openspec/config.yaml` records a DualFPN "4 levels vs 3 expected" mismatch as a known broken
  test. `neck.py` already drops P2 and returns exactly 3 levels; the note appears stale.
- The `class_weights` disagreement in open question 2.

## Recommended scope for the proposal

1. Fix the NMS metric defect first — cheap, orthogonal, and a precondition for trusting any
   later measurement.
2. Resolve the GT-count discrepancy (27 vs 58) — possible second metric defect.
3. Add non-destructive per-class positive-anchor instrumentation to confirm or refute the
   central hypothesis before committing to any architecture change.
4. Treat the nested-class winner-take-all mismatch as the primary model workstream, pending 3.
5. Treat the frozen-backbone / skipped Phase 2 issue as a separate, smaller workstream.
6. Reconcile the documentation discrepancies as low-effort hygiene.

Steps 1-3 are cheap and evidence-producing. Step 4 is where the real cost lives and should not
start until step 3 has confirmed the mechanism.
