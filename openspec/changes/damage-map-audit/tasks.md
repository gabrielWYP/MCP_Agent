# Tasks: Damage-class detection audit — assigner + eval-metric fixes

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~1165 (W0 155, W2+decode 330, eval entrypoint 195, W1 340, W3 25, docs 60, +~60 for the Round 3 E5/E6 ladder additions: maestro Phase-2 unfreeze fix + requires_grad/grad-norm logging, kd_weight-ablation wiring) |
| 400-line budget risk | High |
| Chained PRs recommended | No — maintainer accepted `size:exception` (Round 2, Q5); Round 3's added E4–E6 ladder rungs are validation execution, not source diff, so they do not change this decision |
| Suggested split | Single PR, internally sequenced as 4 work units |
| Delivery strategy | exception-ok |
| Chain strategy | size-exception |

Decision needed before apply: No
Chained PRs recommended: No
Chain strategy: size-exception
400-line budget risk: High

### Suggested Work Units (internal sequencing within the one PR)

| Unit | Goal | Likely PR | Focused test command | Runtime harness | Rollback boundary |
|------|------|-----------|----------------------|-----------------|-------------------|
| 1 | W0 split reconciliation | Single PR (exception) | `./.venv/bin/python -m pytest tests/test_split_integrity.py` | `./.venv/bin/python scripts/prepare_yolo_splits.py --reconcile --dry-run` | Revert reconcile commit; `splits.json` and label dirs restored from git |
| 2 | W2 decode/NMS/E3 + eval entrypoint | Single PR (exception) | `./.venv/bin/python -m pytest tests/test_decode.py` | `./.venv/bin/python scripts/evaluate_checkpoint.py --config configs/training_student.yaml --checkpoint checkpoints/student/best_model.pt --split val` | Set `nms_enabled=false decode_per_class=false` in config, revert `decode.py` |
| 3 | W1 assigner + instrumentation | Single PR (exception) | `./.venv/bin/python -m pytest tests/test_assigner.py` | `evaluate_checkpoint.py --assigner-stats --override assigner_center_radius=0.0` | Set `assigner_center_radius=0.0` (D9 legacy default) |
| 4 | W3 hygiene + validation report | Single PR (exception) | `./.venv/bin/python -m pytest` (full suite) | N/A — docs/config only | Revert doc/config edits independently |
| 5 | H6, E0–E6 experiment ladder (validation execution) | Single PR (exception) | `./.venv/bin/python -m pytest tests/test_decode.py tests/test_assigner.py tests/test_split_integrity.py` | `evaluate_checkpoint.py` (E0–E3, no training), label-file recount (H6, no training), `train.py --override ...` (E4–E6, local GPU) | Config-only levers (`assigner_center_radius`, `kd_weight`, unfreeze schedule); no code revert needed to stop the ladder |

E5/E6 add one small code surface beyond the four units above: the maestro Phase-2 unfreeze fix (E6, `src/models/master/master_model.py`). All GPU training in this change runs locally (GTX 1660 SUPER, CUDA 12.8, cc7.5, verified via `nvidia-smi` and `torch.cuda.is_available()`) — the earlier cloud-instance requirement is retracted.

## Phase 0: Split Integrity (W0) — MUST land and run before any measurement task

- [x] 0.1 Fix `write_empty_label_files` in `scripts/prepare_yolo_splits.py:105-112` to prune stray label files not in the target split before writing (per `data/annotations/yolo/splits.json`).
- [x] 0.2 Add `--reconcile`/`--dry-run` (default) to `prepare_yolo_splits.py`: dry-run reports every disk stem absent from the manifest across train/val/test, deletes nothing; real run requires byte-identity precondition (D2) and exits non-zero on conflict.
- [x] 0.3 Add a manifest guard to `YOLODataset._load_pairs` (`src/training/dataset.py:206-210`): raise if a split's directory contents diverge from `splits.json` (D1); add a GT-count report method (W0-c).
- [x] 0.4 Create `tests/test_split_integrity.py`: reproduce the current leakage (train∩val 8, train∩test 11, val∩test 2 stems) on a fixture; assert dry-run mutates nothing and reconcile removes only manifest-orphaned stems, reporting the disk stem absent from the manifest instead of deleting it silently.
- [x] 0.5 Run real reconciliation against `data/annotations/yolo/splits.json`; confirm the resulting split is 148/18/20; record removed stems in `reports/damage-map-audit/w0-reconcile.md`.
  **RESOLVED (2026-08-29, Round 5 Q11):** the maintainer resolved the 21 real content conflicts found under 0.5 with an explicit rule — keep the copy with more class-1 (damage) boxes regardless of split, tiebreak by area then the manifest-correct copy. Implemented as an opt-in `--resolve-conflicts` flag on `prepare_yolo_splits.py` (`reconcile_splits_resolving_conflicts`, `resolve_conflict`, `apply_conflict_resolution`), dry-run by default, every decision logged. Ran for real: `./.venv/bin/python scripts/prepare_yolo_splits.py --reconcile --resolve-conflicts --no-dry-run` → 21/21 conflicts resolved (17 max_damage_count, 4 tiebreak_area, 0 tiebreak_manifest_copy). Verified: train=148, val=18, test=20+1 unmanifested; train∩val=train∩test=val∩test=∅. Full decision log and the 1 still-open unmanifested stem (`mango_rgb_1780685735`, reported not deleted, separate follow-up) in `reports/damage-map-audit/w0-reconcile.md`.

## Phase 1: Shared Decode Module (W2)

- [x] 1.1 Create `src/training/decode.py::decode_detections(...)` per the design interface: per-class threshold emission (E3/D5), `torchvision.ops.batched_nms` per-class suppression (E1/D4), configurable `conf_threshold`/`nms_iou_threshold`/`nms_enabled`/`per_class_candidates`/`max_detections`.
- [x] 1.2 Create `tests/test_decode.py`: same-class duplicates collapse via NMS; cross-class overlapping boxes both survive; 0/1/2-candidate emission cases; legacy-threshold default (0.25) preserved.
- [x] 1.3 Add `conf_threshold=0.25`, `nms_iou_threshold=0.5`, `nms_enabled=True`, `decode_per_class=True`, `max_detections=300` to `TrainingConfig` (`src/training/config.py`); wire equivalents into `configs/*.yaml`.
- [x] 1.4 Refactor `Trainer._decode_predictions` (`src/training/loop.py:433-515`) into a thin wrapper over `decode_detections`; extract `Trainer._validate` body into a public `Trainer.evaluate()` (D10).
- [x] 1.5 Update `src/training/metrics.py::compute_map` to accept `score_threshold`, drop the hardcoded `0.25` at line 315; add the GT-count reconciliation check (`tp+fn == ΣGT`, log any documented exclusion) per the training-metrics spec.
- [x] 1.6 Refactor `scripts/visualize_damage_predictions.py` to delegate to `decode_detections`, removing its divergent unclamped `reg.exp()` (line 341); add a parity test asserting trainer and script decode identical output on identical raw tensors. Coordinate with open PR #10 before merging to avoid a conflicting decode rewrite.
- [x] 1.7 Create `scripts/evaluate_checkpoint.py`: eval-only entrypoint calling `Trainer.evaluate()`, with an `--assigner-stats` flag wiring to the A4 instrumentation added in Phase 2.
- [x] 1.8 Verify `torchvision.ops.batched_nms` is importable in `.venv` before relying on it in 1.1/1.7.

## Phase 2: Assigner Fix (W1) — ships behind `assigner_center_radius=0.0` default (D9)

- [x] 2.1 `src/training/loss.py::_generate_anchors`: return per-anchor stride alongside centers.
- [x] 2.2 Implement center-sampling admissibility (D6) BEFORE `topk`: anchor valid if inside the GT box OR within `assigner_center_radius × stride` of the GT center; default radius 2.5 via new `TrainingConfig.assigner_center_radius` (0.0 = legacy strict containment).
- [x] 2.3 Move the "keep ≥1 anchor per GT" fallback to run AFTER the spatial mask/top-k (D7/A2); fix its coordinate source to rank by `anchors` centers instead of `pred_bboxes` (current `loss.py:110-111`), matching the filter tested at line 126.
- [x] 2.4 Add a per-level size-bin mask (D8/A3), pre-`topk`, open-ended: `max(w,h)<64→P3`, `<128→P4`, else `P5`; add `assigner_level_ranges=[64,128]` config field.
- [x] 2.5 Add non-destructive `collect_stats` instrumentation (A4): per-class, per-level positive-anchor counts, off by default (`assigner_collect_stats=False`), written to `assigner_stats.csv` in the run directory; must not alter `target_classes/bboxes/scores/fg_mask`.
- [x] 2.6 Create `tests/test_assigner.py`: sub-stride 4px GT gets ≥1 positive; level binning (30px GT→stride-8 only, 200px GT→stride-32 only); `assigner_center_radius=0.0` reproduces current assignment byte-for-byte; instrumentation on/off yields identical `fg_mask`; fallback executes after the spatial filter and never empties a GT.

## Phase 3: Hygiene (W3)

- [x] 3.1 Align `TrainingConfig.class_weights` default (`src/training/config.py:76`) from `[2.7, 0.5]` to `[0.5, 1.5]`; update any test asserting the old default.
- [x] 3.2 Update `openspec/specs/yolo-loss/spec.md`: replace stale 24 sano/129 danado counts with measured 208/335; state `[0.5,1.5]` is deliberately in force for this experiment, not the inverse-frequency `[1.27,0.83]`.
- [x] 3.3 Annotate (do not delete) the `[1.27, 0.83]` claim in `openspec/changes/pipeline-orchestration-docs/explore.md`: arithmetically correct for current data, not in force during this experiment.
- [x] 3.4 Remove the stale "DualFPN 4 levels vs 3 expected" note from `openspec/config.yaml`; refresh the test-runner note to `./.venv/bin/python -m pytest`, 93+ tests.

## Phase 4: Experiment Ladder H6, E0–E6 (Round 3 Q8, Round 5 Q12) — replaces the flat A0/A/B plan

All seven audit hypotheses (H0–H6) get their own rung. Order: leakage (E0) first, then the
supervision-quality check (H6, promoted to run before any other inference-only rung by Round 5
Q12 — if the training labels themselves are corrupted, every training rung (E4–E6) measures
against bad supervision and its outcome means nothing either way), then the remaining
inference-only rungs (E1–E3, no training, all on `checkpoints/student/best_model.pt` post
Phase-0 reconciliation), then training rungs (E4–E6) only for hypotheses that survive.
**Correction (Round 3): all training runs locally** — GTX 1660 SUPER, 6 GB, CUDA 12.8, cc7.5,
verified via `nvidia-smi` and `torch.cuda.is_available()`. The earlier cloud-instance requirement
and its "4–6 min" timing are retracted; do not carry that figure forward as fact.

- [ ] 4.0 **Pin the VRAM/precision lever before any rung runs**: `configs/training_student.yaml` is `batch_size=8, image_size=640, amp=false` against ~5.6 GB free — a nano student should fit. If the first real GPU run OOMs, switch to `amp=true` (fp16 supported on cc7.5) or `batch_size=4`. Whichever is chosen MUST be fixed before the E0/E1 baseline measurements and held constant through every later rung (E0–E5), or attribution breaks.
- [ ] 4.1 **E0 — reconcile & re-measure (H0, no training)**: run `evaluate_checkpoint.py` on the Phase-0-reconciled val split with the legacy (no-NMS) decode; report mango/damage AP50 as the clean-split reference point, replacing the contaminated 0.98/0.041 figures.
  CONFIRM: train∩val=0, train∩test=0, val∩test=0, split sizes match manifest 148/18/20. REFUTE: N/A — H0 is already a verified defect (Round 2), not an open hypothesis; this rung validates the fix rather than testing the claim.
- [ ] 4.2 Publish E0 result to `reports/damage-map-audit/e0-reconcile.md`.
- [ ] 4.3 **H6 — damage under-annotation check (Round 5 Q12; data-only, no training; runs FIRST, before E1)**: `TaskAlignedAssigner.__call__` supervises every anchor at every training step — `src/training/loss.py`'s `target_cls = torch.zeros_like(pred_cls)` means the absence of a class-1 box is active negative supervision across all 8400 anchors, so an unlabelled-damage image is indistinguishable from a genuinely healthy one in the loss. Pre-reconciliation damage-free-file rates: train 37/159 (23%), val 2/25 (8%), test 6/24 (25%). After the Phase-0 conflict resolution (Q11) lands for real, recompute the damage-free-file rate per split (count label files with zero class-1 boxes / total label files, per split) and compare train's rate against val's.
  **CONFIRM: train's damage-free rate remains materially above val's after reconciliation** — the (up to) 37 unverifiable train files are suspect and visual inspection returns to the table.
  **REFUTE: reconciliation closes the gap** — the asymmetry was an artifact of the leaked duplicates, not a real annotation-completeness defect.
  This rung tests supervision integrity itself, independent of decode (E1/E3) or assigner (E2) behavior — it must run before any training rung (E4–E6) is scheduled, per Q12: training against corrupted supervision makes every later training rung's outcome meaningless either way.
- [ ] 4.4 Publish H6 result to `reports/damage-map-audit/h6-annotation-completeness.md`.
- [ ] 4.5 **E1 — apply NMS to the same checkpoint (H2, no training)**: on the reconciled split, compare `nms_enabled=true` vs `nms_enabled=false` (both `decode_per_class=false`); re-measure the FP/TP count baseline (was 1102 FP/16 TP on the contaminated split) and class-1 AP50.
  Bar (APPROVED Round 4, derived from NMS mechanics — not maintainer-set, not invented): NMS removes duplicate boxes for the same object, keeping the highest-scoring one. It must therefore raise precision sharply while leaving recall almost untouched; a recall collapse means it is deleting genuine detections, not duplicates. Both figures are compared WITHIN this rung (`nms_enabled=true` vs `false`, same checkpoint, same reconciled split), so E0's split change cannot invalidate the bar.
  **CONFIRM: `precision_c1` rises >= 5x AND `recall_c1` falls by <= 0.05 absolute.**
  **REFUTE: `precision_c1` rises < 1.5x** — the FP flood is not duplicate-driven and H2 does not explain the depressed AP.
  Guard: if `recall_c1` falls by more than 0.05, the NMS IoU threshold is too aggressive; retune it before reading the result rather than recording a refutation.
- [ ] 4.6 Publish E1 result to `reports/damage-map-audit/e1-nms.md`; this NMS-corrected figure becomes reference "A" for the E4/E5 outcome checks.
- [ ] 4.7 **E2 — assigner instrumentation (H1, no training)** [formerly A0]: run `evaluate_checkpoint.py --split train --assigner-stats --override assigner_center_radius=0.0` on the reconciled split; record median positive anchors per GT, per class, per level.
  CONFIRM: median positive anchors per damage GT <1.0 AND per mango GT ≥5.0. **REFUTE: median damage ≥3.0** pre-fix.
- [ ] 4.8 **E2 gate**: CONFIRM → unlocks E4 (4.12). REFUTE → publish the refutation in `reports/damage-map-audit/e2-refute.md`, do NOT run E4; the H1-causal test closes, but E5/E6 are unaffected since they test independent hypotheses. INCONCLUSIVE → hold E4 pending an explicit maintainer decision.
- [ ] 4.9 **E3 — per-class decode vs argmax (H3, no training)**: same checkpoint/split, compare `decode_per_class=true` vs `false` under the NMS settings fixed in E1; compare damage recall/AP50.
  Bar (APPROVED Round 4 — measure the ceiling first, then set the bar from it): do NOT guess a threshold. Task 4.9a below measures the maximum recovery per-class decode could possibly achieve; that measured number becomes this rung's bar.
  **CONFIRM: `recall_c1` rises by >= 50% of the measured ceiling from 4.9a.**
  **REFUTE: `recall_c1` rises by <= 10% of that ceiling.**
  If 4.9a shows the ceiling is negligible, E3 is closed as inapplicable WITHOUT running the comparison, and that is recorded as a legitimate outcome — argmax was not suppressing damage in practice.
- [ ] 4.9a **E3 ceiling measurement (prerequisite for 4.9's bar)**: on the same checkpoint and reconciled split, count anchors where BOTH class scores exceed the confidence threshold — these are the only locations where argmax can be erasing a damage candidate. Report the count, and the damage recall that would result if every one of them were recovered. That figure is the maximum E3 can achieve and becomes 4.9's bar. A negligible count closes E3 without running it.
- [ ] 4.10 Publish E3 result to `reports/damage-map-audit/e3-decode.md`.
- [ ] 4.11 **Local timing measurement**: on whichever of E4/E5/E6 runs first, measure actual per-epoch wall clock on the local GPU and restate the duration estimate for the remaining training rungs before they are scheduled.
- [ ] 4.12 **E4 — retrain with assigner fix, 3 seeds (H1 causal; gated on 4.8 CONFIRM only)**: local GPU, `assigner_center_radius=2.5`, VRAM lever pinned at 4.0; re-run E2 instrumentation post-fix (expect median ≥3.0/damage GT).
  CONFIRM: `AP50_c1(E4) ≥ 2×AP50_c1(E1)` and `≥+0.10` absolute, AND guardrail `AP50_c0(E4) ≥ 0.9×AP50_c0(E1)`. **REFUTE: `|Δ| ≤ 0.02`** — record as negative, keep Phase 1 work, reopen the cause ranking. Else INCONCLUSIVE.
- [ ] 4.13 Publish E4 result (or refutation) to `reports/damage-map-audit/e4-assigner.md`.
- [ ] 4.14 **E5 — retrain with `kd_weight=0` (H4; runs unconditionally, independent of E2/E4)**: local GPU, student config, `kd_weight=0`, `assigner_center_radius` held at E4's concluded value (0.0 if REFUTE/not run, 2.5 if CONFIRM) so only `kd_weight` varies against the matching baseline.
  Bar (APPROVED Round 4 — reuses E4's format): **CONFIRM `AP50_c1 >= 2x baseline` and `>= +0.10` absolute; REFUTE `|delta| <= 0.02`.** Baseline = E4's checkpoint if E4 ran, else E1.
  Prior evidence already weighs weakly AGAINST H4 and must be stated in the report: distilled scored 0.062 and 0.065 on damage while the plain student scored 0.041 and 0.090 — the distilled model wins one run and loses the other. If E5 refutes, that is the expected outcome, not a surprise.
- [ ] 4.15 Publish E5 result to `reports/damage-map-audit/e5-kd-ablation.md`.
- [ ] 4.16 **E6 — retrain the maestro with Phase-2 unfreeze (H5; runs unconditionally; separate anomaly, not the class-1 gap)**: uses `configs/training_mango.yaml` (the repo's actual master/teacher config — corrected from an earlier draft's `configs/training_teacher.yaml`, which does not exist; see the decision note below), not the student config, and runs longer than E4/E5. Fix the maestro training schedule so Phase 2 unfreeze actually fires (`MasterModel.freeze_backbone()`, `src/models/master/master_model.py:170-198` — production CSV currently has only `phase=1` rows), and add `rgb_stem` `requires_grad`/grad-norm logging to verify the transition fired. Tests teacher 0.216 vs its own student 0.394, not the damage-class gap.
  Bar (APPROVED Round 4, principled — derived from the anomaly's own definition): the anomaly IS "the teacher scores below its own student". It is resolved exactly when that stops being true.
  **CONFIRM: maestro `mAP50 >= 0.394`** (its student's figure), recomputed on the reconciled split.
  **REFUTE: maestro mAP50 unchanged within noise despite `rgb_stem` grad-norm logging proving the unfreeze fired.** A refutation is only valid if the unfreeze is verified to have happened; an unfreeze that silently did not fire is a failed run, not a refuted hypothesis.
  Scope note (APPROVED Round 4): the production fix to the maestro schedule ships in THIS PR together with E6. The fix and the experiment that validates it travel together — without the fix E6 cannot run, and without E6 the fix is unverified.
  **Config decision (2026-08-29, this batch):** `configs/training_teacher.yaml` referenced by an earlier task draft does not exist in the repo. Decided to correct the task reference to the config that already serves this role, `configs/training_mango.yaml` (backbone_variant/epochs_phase1/epochs_phase2/output_dir=checkpoints/mastermodel_mango — i.e. the master/teacher training config), rather than create a second, largely-duplicate config file. A separate `training_teacher.yaml` is not required by the design; `training_mango.yaml` already is the maestro config, and adding a near-duplicate would reintroduce exactly the config-drift risk Phase 3 (W3 hygiene) was fixing elsewhere in this change.
  **Readiness note (2026-08-29, from Phase 0-3/5 apply batch):** the code-level fix and instrumentation this task needs already ship in this PR: `MasterModel.unfreeze_backbone_stages(..., unfreeze_rgb_stem=True)` (wired from `Trainer._train_phase`'s Phase 2 call) and `Trainer._rgb_stem_grad_norm()` + TensorBoard logging (`Phase2/rgb_stem_requires_grad`, `Phase2/rgb_stem_grad_norm`).
- [ ] 4.17 Publish E6 result to `reports/damage-map-audit/e6-teacher-unfreeze.md`.
- [ ] 4.18 Publish `reports/damage-map-audit/validation-report.md` summarizing all eight rungs (H6, E0–E6): bar, outcome (confirmed/refuted/inconclusive), and evidence for each — supersedes the old flat A0/A/B report.

## Phase 5: Regression

- [x] 5.1 Run `./.venv/bin/python -m pytest` full suite; confirm the 93 existing tests plus every test added in Phases 0–2 pass.
- [x] 5.2 Confirm `class_weights` has exactly one authoritative value across `configs/*.yaml`, `src/training/config.py`, and spec docs (final success-criteria check).
