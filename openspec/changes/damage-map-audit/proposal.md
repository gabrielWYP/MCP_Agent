# Proposal: Damage-class (class 1) detection audit — assigner + eval-metric fixes

Change id: `damage-map-audit` · Store: hybrid · Supersedes the explore doc's cause #1 framing.

## Intent

Class 1 (damage) scores AP50 0.04–0.09 in every recorded run while class 0 (mango) reaches 0.98.
Two independent, localized defects are the leading explanation:

1. **Assigner (model defect, hypothesis under test).** `TaskAlignedAssigner.__call__`
   (`src/training/loss.py:100-152`) applies a strict anchor-center-inside-GT filter at
   lines 132-143 *after* the "keep at least 1 anchor per GT" fallback at lines 106-115.
   The fallback is therefore defeated: `pos_idx = pos_idx[inside]` followed by
   `if len(pos_idx) == 0: continue` can drop a GT entirely. Anchor centers are spaced
   `stride` apart, so a GT narrower than the stride may contain zero anchor centers.
   Measured at 640px: mango min short side 97px (0% below stride 32); damage median
   30.2px, min 4.5px, **54.6% below stride 32**, 12.8% below stride 16, 4.5% below stride 8.
   Mango is always assignable at all levels; damage frequently is not.
2. **Evaluation (metric defect, verified).** `Trainer._decode_predictions`
   (`src/training/loop.py:433-515`) emits every anchor above a hardcoded `threshold = 0.25`
   across 8400 anchors with **no NMS** (latest run: 1102 FP vs 16 TP on class 1). Every
   absolute AP in the repo is untrustworthy, so no model fix can currently be validated.

**Status of the central claim:** the assigner mechanism is verified in code and the size
disparity is measured, but "damage GTs receive near-zero positive anchors" has **not been
instrumented**. This proposal treats it as the hypothesis under test, with an explicit
kill-switch (Step A0) before any training cost is spent.

Not in this change: taxonomy redesign, multi-label heads, segmentation, two-stage crop-classify.

## Scope

### In Scope — W1 Assigner (`src/training/loss.py`)
- A1: replace the strict `inside` containment filter with YOLOv8-style center sampling —
  accept anchors whose center is within a configurable radius (in stride units) of the GT center,
  so sub-stride GTs still receive positives.
- A2: move the "keep at least 1 anchor per GT" fallback to run **after** the spatial filter so the
  guarantee actually holds; drop no GT silently.
- A3: per-level assignment review — stride 32 cannot serve 54.6% of damage boxes yet consumes
  top-k budget. Either restrict candidate levels by GT size or document why not. Decision belongs
  to sdd-design.
- A4: non-destructive instrumentation counting positive anchors **per class and per FPN level**,
  off by default, emitted to the run directory.

### In Scope — W2 Evaluation (`src/training/loop.py`)
- E1: apply NMS (`torchvision.ops.nms`, per class) in the eval decode path. Precondition for W1.
- E2: make the 0.25 confidence threshold and the NMS IoU threshold configurable
  (`src/training/config.py`), defaults preserving current behaviour where sensible.
- E3: `scores.max(dim=0)` at `loop.py:463` emits only the argmax class per anchor. With 94% of
  damage boxes fully inside a mango box, this structurally suppresses damage candidates at shared
  anchors. Proposed: emit per-class candidates above threshold instead of argmax-only.
  **Flagged — confirm before bundling** (see Open Questions Q1).
- E4: keep `scripts/visualize_damage_predictions.py` (open PR #10) consistent with the trainer
  decode, or it will misreport once NMS lands.

### In Scope — W3 Hygiene (low cost)
- H1: reconcile `class_weights` — three conflicting values: `configs/*.yaml` `[0.5, 1.5]`,
  dataclass default `src/training/config.py:76` `[2.7, 0.5]` (inverted, favours mango),
  `pipeline-orchestration-docs/explore.md` `[1.27, 0.83]`. Requires maintainer intent (Q2).
- H2: remove the stale "DualFPN 4 levels vs 3 expected" broken-test note in `openspec/config.yaml`;
  `neck.py` already returns 3 levels.

### Out of Scope — deferred, flagged for maintainer decision, NOT silently bundled
- `MasterModel.freeze_backbone()` (`src/models/master/master_model.py:170-198`) unconditionally
  freezes `rgb_stem`; the production maestro CSV has only `phase=1` rows, so Phase 2 unfreeze never
  ran. Plausibly explains teacher 0.216 < student 0.394 — a **separate anomaly** from the class gap.
  Its own change, its own PR.
- Unresolved possible second metric defect: latest run reports `tp_class_1 + fn_class_1 = 27` on val,
  but val labels contain 58 damage instances. Independent of NMS. **Must be investigated during
  sdd-design**; if it proves to be a GT-loading bug it invalidates Step A's baseline and this change
  blocks on it.
- `kd_weight=0` ablation; taxonomy/architecture redesign; retraining the maestro.

## Capabilities

### New Capabilities
- None.

### Modified Capabilities
- `yolo-loss`: "Task-Aligned Assignment" requirement changes — center-sampling tolerance replaces
  strict containment; every GT MUST receive at least one positive anchor; per-class positive-anchor
  instrumentation.
- `training-loop`: prediction decode MUST apply per-class NMS; confidence and NMS IoU thresholds
  MUST be configurable, not hardcoded.
- `training-metrics`: AP is computed over NMS-suppressed predictions; per-class AP semantics and the
  27-vs-58 GT-count question.

## Approach

Fix the metric first (W2), then the model (W1), and attribute them **separately**. A single run
changing both proves nothing about either.

## Validation training plan (mandatory)

All AP figures are val split, at the same image size and seed as the compared run.

| Step | Action | Cost | Purpose |
|---|---|---|---|
| **A0** | Run A4 instrumentation on an **existing** checkpoint, forward pass over train split, unmodified assigner. Record median/mean positive anchors per GT per class per level. | minutes, no training | **Kill switch.** Direct evidence for the hypothesis. |
| **A** | Re-evaluate the **same existing checkpoint** with E1+E2 (NMS) applied. No retraining. | minutes | The only trustworthy baseline. Isolates the metric defect. |
| **B** | Retrain with W1 (A1+A2+A3) applied, evaluate with NMS. Re-run A0 instrumentation post-fix. | one full run, 46–84 epochs | Isolates the assigner fix, measured against Step A. |

**Model selection.** A0/A/B run on **estudiante** (`configs/training_student.yaml`) — it is the
fastest to retrain, is fully unfrozen (so the frozen-backbone confound is absent), and already
shows the class gap (0.747 vs 0.041). Maestro and destilado are **not** retrained in this change;
if B confirms, a follow-up change re-runs the full pipeline.

Commands (venv interpreter — `python` is NOT on PATH):
```
./.venv/bin/python -m src.training.train --config configs/training_student.yaml \
    --override <instrumentation/eval flags to be named in sdd-design>
```
Exact flag names, the eval-only entrypoint for Step A, and the checkpoint path are sdd-design's job.

**Duration estimate: uncertain.** Recent runs ran 46–84 epochs; wall-clock per epoch is not recorded
in this analysis. sdd-design MUST extract actual per-epoch time from an existing run log and restate
the estimate before Step B is scheduled.

### Falsifiable criteria — fixed BEFORE any run

**Mechanism (A0, deterministic, decides whether Step B happens at all):**
- CONFIRM if median positive anchors per **damage** GT < 1.0 while per **mango** GT ≥ 5.0.
- **REFUTE if median positive anchors per damage GT ≥ 3.0 pre-fix.** Then the assigner is not
  starving damage, the primary hypothesis is **wrong**, Step B MUST NOT run, and this proposal
  must be rewritten around the surviving causes (E3 decode suppression, small-object capacity,
  the 27-vs-58 GT discrepancy). Say so plainly; do not rationalize.
- Post-fix re-run MUST show median ≥ 3.0 per damage GT, or A1/A2 did not do what they claim.

**Outcome (B vs A, on class-1 AP50):**
- CONFIRM: `AP50_class1(B) ≥ 2 × AP50_class1(A)` **and** `≥ +0.10` absolute.
- INCONCLUSIVE: between +0.02 and the confirm bar — a real but insufficient effect; a second cause
  dominates.
- **REFUTE: `|AP50_class1(B) − AP50_class1(A)| ≤ 0.02`.** The assigner fix is not the lever. Record
  it as a negative result; keep W2 (it is correct regardless) and reopen the cause ranking.
- Guardrail: `AP50_class0(B) ≥ 0.9 × AP50_class0(A)`. A damage gain bought by wrecking mango is a
  regression, not a fix.
- Class-1 AP50 from **pre-NMS** runs (0.041 etc.) is NOT a valid comparison target for B. Step A is.

## Affected Areas

| Area | Impact | Description |
|---|---|---|
| `src/training/loss.py` | Modified | Center sampling, fallback ordering, per-level policy, instrumentation |
| `src/training/loop.py` | Modified | Per-class NMS, configurable thresholds, decode semantics (E3) |
| `src/training/config.py` | Modified | New threshold fields; `class_weights` default reconciliation |
| `configs/*.yaml` | Modified | Threshold + class_weights values |
| `scripts/visualize_damage_predictions.py` | Modified | Keep decode consistent with trainer (PR #10) |
| `openspec/config.yaml` | Modified | Remove stale broken-test note |
| tests | New | Assigner sub-stride GT coverage; NMS decode; threshold config |

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Hypothesis is wrong; class-1 AP stays flat | Medium | Step A0 kill switch before spending a run; refutation criterion fixed in advance |
| Center sampling floods positives, hurts class 0 | Medium | Radius is configurable; guardrail `AP50_class0(B) ≥ 0.9 × A` |
| Assigner change invalidates all existing checkpoints | High (accepted) | Step A's NMS-corrected baseline is the only legitimate comparison point |
| 27-vs-58 GT discrepancy corrupts Step A's baseline | Medium | Must be resolved in sdd-design; blocks the plan if it is a GT-loading bug |
| Change exceeds the 800-line review budget | Medium | See Delivery below |
| PR #10 visualizer diverges from the fixed decode | High | E4 is in scope; sequence after #9/#10 merge |
| Training run cost is not yet quantified | High | sdd-design must restate duration from real logs before Step B |

## Delivery

`review_budget_lines: 800`. Rough authored estimate: W1 ~140, W2 ~120, W3 ~40, tests ~250,
docs ~40 → **~590 lines**, within budget *only if* the deferred frozen-backbone work stays out.
If sdd-tasks forecasts above 800, split as a chained PR series:
**PR1** W2+W3 (eval/NMS/hygiene, independently valuable, unblocks measurement) →
**PR2** W1 (assigner + instrumentation) → **PR3** validation run report.
Do not silently exceed the budget.

## Rollback Plan

- W2 and W1 are independent commits on separate PRs; revert either without touching the other.
- Behaviour is gated by config: NMS on/off, confidence and IoU thresholds, center-sampling radius,
  and instrumentation are all config-driven. Reverting to the previous config restores prior
  behaviour without a code revert.
- Existing checkpoints are untouched. Step A is read-only re-evaluation.
- If Step B regresses class 0 past the guardrail, revert PR2, keep PR1, publish the negative result.

## Dependencies

- Open PRs #9 (hygiene) and #10 (visualizer) merged or explicitly sequenced before E4.
- `torchvision` NMS available in `.venv` (verify in sdd-design).
- Test suite: `./.venv/bin/python -m pytest` — 93 tests, ~80s, currently green. `nbformat` absent.
- Maintainer answers to Q1–Q3 below.

## Maintainer Decisions (question round — RESOLVED 2026-08-29)

All four questions were answered by the maintainer. These decisions are binding on
`sdd-spec`, `sdd-design`, `sdd-tasks`, and `sdd-apply`. Do not re-ask them.

1. **Q1 — decode semantics (E3): INCLUDE in this change.**
   The decode must emit per-class candidates above threshold instead of only the argmax class.
   Rationale accepted: with 94% of damage boxes fully inside a mango box, an argmax-per-anchor
   decode erases damage candidates at shared anchors. Repairing the assigner while leaving the
   argmax decode in place would be incoherent — damage could win the anchor during training and
   still be discarded at evaluation. W2 already modifies this code path, so marginal cost is low.
   E3 moves from "open question" to **In Scope, W2**.

2. **Q2 — `class_weights`: `[0.5, 1.5]` from `configs/*.yaml` is authoritative.**
   **Decision stands, but the original rationale was wrong and is corrected here.**

   Corrected finding: `[2.7, 0.5]` is NOT an arbitrary inverted default. It is derived from
   `openspec/specs/yolo-loss/spec.md:35-42`, which states that mango (sano) is the underrepresented
   class at a 1:5.9 ratio and SHALL therefore carry the higher weight. That reasoning is sound
   inverse-frequency weighting; only its counts have aged. Measured over the current dataset:
   mango 208, damage 335, ratio 1:1.61 — not 1:5.38. Inverse frequency on current data yields
   `[1.305, 0.810]`, which is essentially the `[1.27, 0.83]` the sibling doc reports. So the
   sibling doc's value is the correctly recomputed one, and `[0.5, 1.5]` is the only one of the
   three with no derivation at all — it weights the class that is NOT rare more heavily, most
   likely following the class-imbalance theory this audit disproved.

   **Why `[0.5, 1.5]` is nonetheless the right choice for this change:** every existing checkpoint
   was trained under it. Step A re-evaluates such a checkpoint and Step B retrains. Changing the
   weights would introduce a second variable and destroy attribution — a class-1 AP change could
   no longer be assigned to the assigner fix rather than the reweighting. Holding the weights
   constant is an experimental-design requirement, not an endorsement of the value.

   Actions, revised:
   - Align the dataclass default at `src/training/config.py:76` to `[0.5, 1.5]` so the repo has one
     source of truth for the duration of this experiment.
   - Do NOT silently delete the `[1.27, 0.83]` claim in the sibling doc; it is arithmetically
     correct for current data. Annotate it instead.
   - Update the stale counts and the resulting weights in `openspec/specs/yolo-loss/spec.md`
     (currently 24 sano / 129 danado) to reflect the measured 208 / 335, and record that the
     prescribed weights are deliberately NOT in force during this experiment.
   - **Revisiting `class_weights` on inverse-frequency grounds is a follow-up AFTER the validation
     completes, never during it.** Changing them mid-experiment corrupts Step B.

3. **Q3 — validation budget: full run authorised for Step B.**
   46–84 epochs, matching recent runs. Class-1 AP is noisy at this data volume and a shortened run
   carries a high risk of an inconclusive result that would force a repeat. Step A0 remains the
   gate: this spend only happens if the hypothesis survives the kill switch.

4. **Q4 — refutation policy: proceed with W2+W3, stop W1.**
   If A0 refutes the hypothesis, NMS and the hygiene work still ship, because the eval decode is
   incorrect regardless of whether the assigner hypothesis holds — without it no measurement in
   this project is trustworthy. The cause ranking is reopened with the A0 counts as new evidence.
   This confirms the proposal's stated default.

## Original Open Questions (superseded by the decisions above)

1. **Q1 — decode semantics (E3).** Should the decode emit per-class candidates above threshold
   instead of only the argmax class? It is cheap and touches code W2 already changes, but it widens
   scope beyond "add NMS". Include in this change, or defer?
2. **Q2 — `class_weights` intent.** Which of `[0.5, 1.5]` (configs), `[2.7, 0.5]` (dataclass
   default, inverted), `[1.27, 0.83]` (sibling doc) is authoritative? Answer changes H1 and the
   Step B run config.
3. **Q3 — validation budget.** Is one estudiante retrain (46–84 epochs) an acceptable spend for
   Step B, or should a shortened run with a proportionally adjusted criterion be used instead?
4. **Q4 — refutation policy.** If A0 refutes the hypothesis, should this change stop and return to
   explore, or proceed with W2+W3 alone as standalone metric-correctness work?
   (Proposal's default: proceed with W2+W3, stop W1.)

## Success Criteria

- [ ] A0 instrumentation runs and produces per-class, per-level positive-anchor counts on an
      existing checkpoint with no retraining.
- [ ] The A0 result is recorded and interpreted against the pre-registered confirm/refute bars,
      including publishing a refutation if that is the outcome.
- [ ] Eval decode applies per-class NMS; confidence and NMS IoU thresholds are configurable.
- [ ] Every GT, including sub-stride GTs, receives at least one positive anchor — covered by a
      regression test with a synthetic 4px-wide GT at 640px.
- [ ] Step A NMS-corrected baseline is published as the new reference; prior AP figures are marked
      superseded.
- [ ] Step B completes and its class-1 AP50 is compared against Step A only, with the class-0
      guardrail checked.
- [ ] `./.venv/bin/python -m pytest` remains green (93 tests + new ones).
- [ ] `class_weights` has exactly one authoritative value in the repo.

## Maintainer Decisions — Round 2 (RESOLVED 2026-08-29, post-design)

Triggered by the design phase exceeding the review budget and by the split-leakage discovery.
Binding on `sdd-tasks` and `sdd-apply`.

### Context: verified data leakage supersedes prior priorities

Direct measurement of label directory contents (not manifest comparison — literal set intersection):

```
train ∩ val  =  8 stems   (32% of the 25-image val set is also in train)
train ∩ test = 11 stems   (46% of the test set is also in train)
val   ∩ test =  2 stems
```

Cause: `scripts/prepare_yolo_splits.py:105-112` calls `label_path.touch(exist_ok=True)` and never
prunes files from a previous shuffle, while `YOLODataset._load_pairs` (`src/training/dataset.py:206-210`)
keys off directory contents rather than the manifest. The effective split is the union of every
historical assignment. Disk holds 187 unique stems against the manifest's 186.

**Every AP figure recorded in this project is therefore invalid**, including class 0's 0.98, which is
inflated by memorization. The class-1 figure of 0.041 was also measured on a contaminated split, so
the true magnitude of the problem this change exists to solve is currently unknown.

W0 (split reconciliation) is now the first work item, ahead of A0, NMS, and the assigner.

### Decisions

5. **Q5 — delivery: SINGLE PR with `size:exception`.**
   `delivery_strategy` moves from `ask-on-risk` to `exception-ok`. The design estimates ~1105
   authored lines against the 800-line budget. The orchestrator recommended a 4-PR chain and
   recorded the reviewability concern; the maintainer accepted the exception explicitly. Proceed
   as one PR. Do not silently re-split.

6. **Q6 — Step A subject checkpoint: `checkpoints/student/best_model.pt`.**
   The `final_runs/**/best_model.pt` artifacts are absent from the repo (`.pt` is gitignored) and
   remain on the cloud GPU instance. The local student checkpoint is used instead. Its baseline AP
   will differ from the published run, which is acceptable because the baseline is recomputed after
   split reconciliation regardless — the published figures are invalid.

7. **Q7 — split repair: reconcile against `data/annotations/yolo/splits.json`.**
   Honor the manifest's 148/18/20 assignment and remove orphaned files left by earlier shuffles.
   Preserves original intent and is auditable against a file that already existed. Reconciliation
   is destructive (it deletes label files) and MUST fail closed: dry-run by default, byte-identity
   precondition before any delete, non-zero exit on conflict. The one disk stem absent from the
   manifest must be reported, not silently deleted.

### Carried forward from the design phase

- Step B costs 4–6 minutes on the GPU instance (measured 6.6–6.9 s/epoch from tfevents timestamps
  against CSV epoch counts), not hours. At that price Step B SHOULD run multiple seeds; a single
  seed cannot separate signal from noise on a class this small. Local WSL2 CPU is unmeasured and
  expected 1–2 orders of magnitude slower.
- Second assigner defect, additional to the fallback-ordering bug: the fallback ranks candidates by
  predicted box centers (`src/training/loss.py:110-111`, `pred_bboxes`, which move during training)
  while the containment filter tests fixed anchor grid centers (`src/training/loss.py:126`,
  `anchors[pos_idx]`). Two different coordinate sources; the local variable is misleadingly named
  `anchor_cx` while holding predicted-box coordinates.
- P1 resolved: the metric is NOT defective. `_compute_operating_point` (`src/training/metrics.py:219-247`)
  is algebraically exact — `fn += len(targets) - len(matched)` makes `tp + fn == ΣGT` by construction.
  The 27-vs-58 gap was split drift, now explained by the leakage above.

## Maintainer Decisions — Round 3 (RESOLVED 2026-08-29, post-tasks)

### Correction: local GPU training IS available

The earlier claim that Step B requires the cloud GPU instance was WRONG and is retracted.
CUDA is reachable from WSL2 and verified working:

```
nvidia-smi:  NVIDIA GeForce GTX 1660 SUPER, 6144 MiB, driver 610.53
torch:       2.9.1+cu128, torch.cuda.is_available() == True
device:      GTX 1660 SUPER, 6.4 GB, compute capability 7.5
```

All training steps run locally. Timing will differ from the cloud instance's measured
6.6-6.9 s/epoch — this GPU is slower — so the design's 4-6 minute estimate must be re-measured
locally on the first real run rather than assumed.

**VRAM contingency.** `configs/training_student.yaml` uses `batch_size: 8` at `image_size: 640`
with `amp: false`, against ~5.6 GB free. A nano student should fit. If it OOMs, enable `amp`
(fp16 is supported on cc7.5) or drop to `batch_size: 4` — but that choice MUST be fixed BEFORE
the Step A baseline and held constant through every subsequent experiment, or attribution breaks.

### 8. Q8 — methodology: the cycle must be COMPLETE

Binding maintainer directive: this change must follow a full hypothesis-then-test loop —
hypothesis, experiment that can confirm or refute it, next hypothesis — rather than testing one
hypothesis and shelving the rest as "deferred". A deferred hypothesis is an untested one.

The audit surfaced six hypotheses. All six are IN SCOPE for this change:

| | Hypothesis | Prior status |
|---|---|---|
| H0 | Split leakage contaminates all measurement | VERIFIED — a defect, no longer a hypothesis |
| H1 | The assigner starves small damage boxes of positive anchors | had a test (A0) |
| H2 | Missing NMS depresses AP | had a test (Step A) |
| H3 | Argmax decode suppresses damage at shared anchors | no isolated test |
| H4 | KD transfers the teacher's damage-blindness | deferred, untested |
| H5 | The frozen backbone sinks the teacher | deferred, untested |

### The experiment ladder — replaces the flat A0/A/B plan

Ordering principle: leakage first, then every inference-only test, then training runs only for
hypotheses that survive. **E0 through E3 require no training at all** — four hypotheses are
confirmed or killed before any GPU time is spent. E0 comes first because reconciling the splits
may move the real numbers enough to invalidate later hypotheses for free.

| # | Experiment | Tests | Training |
|---|---|---|---|
| E0 | Reconcile splits, re-measure the existing checkpoint | H0 | no |
| E1 | Apply NMS to the same checkpoint's predictions | H2 | no |
| E2 | Instrument positive anchors per class and level | H1 | no |
| E3 | Per-class decode versus argmax on the same checkpoint | H3 | no |
| E4 | Retrain with the assigner fix, 3 seeds | H1 causally | yes |
| E5 | Retrain with `kd_weight=0` | H4 | yes |
| E6 | Retrain the teacher with Phase 2 unfreeze | H5 | yes |

Every rung MUST carry a pre-registered confirm bar AND a refute bar, fixed before the run, in the
style already established for A0. An experiment with no failure condition discards nothing.
Each rung's outcome must be published even when it is negative.

## Maintainer Decisions — Round 4 (RESOLVED 2026-08-29, thresholds)

`sdd-tasks` correctly declined to invent numeric bars for E1, E3, E5 and E6 and flagged them for
maintainer input. Three were then DERIVED rather than guessed, and one was restructured. A derived
bar is not an invented one; an invented bar makes refutation theatre.

9. **Q9 — thresholds: approved as derived below.** Written into `tasks.md` and marked
   `APPROVED Round 4`.

   - **E1 (NMS)** — derived from what NMS mechanically does: it removes duplicate boxes for the same
     object, keeping the highest scoring one, so it must raise precision sharply while barely moving
     recall. CONFIRM `precision_c1` rises >= 5x AND `recall_c1` falls <= 0.05. REFUTE `precision_c1`
     rises < 1.5x. Compared within the rung (`nms_enabled` true vs false, same checkpoint, same
     split) so E0's split change cannot invalidate it. Guard: a recall drop above 0.05 means the NMS
     IoU threshold is too aggressive — retune before reading the result.

   - **E3 (per-class decode)** — restructured rather than guessed. New task 4.7a measures the
     ceiling first: count anchors where BOTH class scores exceed threshold, since those are the only
     places argmax can be erasing a damage candidate. That measured number becomes the bar.
     CONFIRM `recall_c1` rises >= 50% of the ceiling; REFUTE <= 10%. A negligible ceiling closes E3
     without running it, and that is a legitimate recorded outcome.

   - **E5 (KD ablation)** — reuses E4's format: CONFIRM `>= 2x` and `>= +0.10`; REFUTE `|delta| <= 0.02`.
     Prior evidence already weighs weakly AGAINST H4 and must be stated in the report: distilled
     scored 0.062 / 0.065 on damage versus the plain student's 0.041 / 0.090 — distilled wins one run
     and loses the other. A refutation here is the expected outcome, not a surprise.

   - **E6 (teacher unfreeze)** — principled, derived from the anomaly's own definition. The anomaly
     IS "the teacher scores below its own student", so it is resolved exactly when that stops being
     true. CONFIRM maestro `mAP50 >= 0.394` on the reconciled split. REFUTE unchanged within noise
     **only when `rgb_stem` grad-norm logging proves the unfreeze actually fired** — an unfreeze that
     silently did not fire is a failed run, not a refuted hypothesis.

10. **Q10 — E6 production fix ships in THIS PR.**
    E6 requires repairing the maestro training schedule so Phase 2 unfreeze fires
    (`src/models/master/master_model.py:170-198`; the production CSV holds only `phase=1` rows).
    That is production code, not validation. It ships together with E6: without the fix E6 cannot
    run, and without E6 the fix is unverified. Consistent with the accepted `size:exception`.

## Maintainer Decisions — Round 5 (RESOLVED 2026-08-29, annotation conflicts)

### New finding: the label copies disagree, and "healthy" is indistinguishable from "unlabelled"

Split reconciliation surfaced a defect larger than the leakage itself. Of the 21 stems present in
more than one split directory, **zero have byte-identical copies — all 21 conflict in content**:

```
mango_rgb_1780685426:  train(damage=0)  vs  val(damage=12)
mango_rgb_1780684450:  train(damage=0)  vs  val(damage=8)
mango_rgb_1780683932:  train(damage=0)  vs  val(damage=3)
```

Eight train copies carry ZERO damage boxes while their twin carries up to 12. Those images contain
visible damage that was presented to the model as background.

Verified in `src/training/loss.py:481`: `target_cls = torch.zeros_like(pred_cls)` and the focal BCE
runs over ALL anchors, so absence of a class-1 box is active negative supervision across all 8400
anchors. The model does receive "there is no damage here" automatically — which is precisely why an
unlabelled-damage image is indistinguishable from a genuinely healthy one. Both produce an
all-zeros class-1 target. The YOLO label format cannot express "verified absent" separately from
"not yet annotated".

Scale indicator (not proof): 37/159 train files (23%) carry no damage box at all, against 2/25
(8%) in val. Only the 21 duplicated stems are verifiable; the rest have no twin to compare against.

### 11. Q11 — conflict resolution: keep the copy with the most class-1 boxes.

Maintainer decision, applied without per-image visual inspection.

Rationale, and a correction: the orchestrator initially objected that "6 of 21 have more damage in
train, contradicting the rule". **That objection was wrong and is withdrawn.** The rule selects the
copy with more damage boxes regardless of which split holds it, so train winning in 6 cases is
consistent with it, not a counterexample. The underlying logic is sound: damage does not vanish
from a photograph, so a pass that found 12 regions and a pass that found 0 identify which one is
incomplete.

**Tiebreak for the 4 equal-count conflicts** (orchestrator judgment, disclosed): in all four, the
class-0 mango box is byte-identical between copies while the class-1 boxes differ in coordinates —
these are two runs of the same automated damage pipeline, not one annotated and one not. In one
case (`mango_rgb_1780238853`) a damage box sits at y=0.435 in one copy and y=0.358 in the other,
which is a different location, not jitter. Tiebreak order:
1. larger total class-1 area (more damage captured, consistent with the chosen principle);
2. if still tied, the manifest-correct copy.

The single stem absent from the manifest is still REPORTED, never auto-deleted.

### 12. Q12 — H6 runs first, before E1.

`H6: damage is systematically under-annotated in the training split` is now a stronger candidate
than H1. It takes the first rung, because if supervision is corrupted then every training rung
(E4-E6) measures against bad labels and its outcome means nothing either way.

H6's test is data-only, no training: after the 21 conflicts are resolved, recompute the
damage-free-file rate per split. CONFIRM if train's rate remains materially above val's; REFUTE if
reconciliation closes the gap. A persistent gap means the 37 unverifiable train files are suspect
and visual inspection returns to the table.
