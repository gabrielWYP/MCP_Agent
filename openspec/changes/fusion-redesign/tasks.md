# Tasks: Early-fusion redesign — 4-channel RGB+NIR backbone

Branch point: `9e43aa6` on `main` (`damage-map-audit` merged, PRs #10/#11). PR 1 branches from
`9e43aa6`; PR 2 branches from PR 1's merge commit — it is a hard prerequisite, not a convenience.

**Deviation from the 530-word skill budget** is deliberate, for the same reason `design.md` states
one: two chained PRs spanning 9 workstreams, a blocking-prerequisite dependency, TDD pairing per
behavior task, and a 9-hypothesis validation ladder with pre-registered bars cannot be represented
faithfully in checklist form at that length without silently dropping ordering or bar information.

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ≈3,480 (band 3,200-4,200): PR 1 ≈1,105 / PR 2 ≈2,375 |
| 400-line budget risk | High |
| Chained PRs recommended | Yes |
| Suggested split | PR 1 (trainer correctness + hardware portability) → PR 2 (architecture), stacked |
| Delivery strategy | exception-ok (Round 3: both PRs exceed the 800-line `review_budget_lines`; the split is for reviewability/sequencing, not budget compliance — exception already granted) |
| Chain strategy | stacked-to-main |

Decision needed before apply: No
Chained PRs recommended: Yes
Chain strategy: stacked-to-main
400-line budget risk: High

### Suggested Work Units

| Unit | Goal | PR | Focused test command | Runtime harness | Rollback boundary |
|---|---|---|---|---|---|
| 1 | D-G: OOM/NaN counters raise, `max(n,1)` removed | PR1 | `./.venv/bin/python -m pytest tests/training/test_loop_guards.py -v` | N/A — fault-injected fake loader, no GPU | `loop.py` guard block only |
| 2 | W9a precision enum (`fp32\|fp16\|bf16`) | PR1 | `pytest tests/training/test_precision.py` | `train.py --override precision=fp32` smoke, CPU | `precision.py` + `config.py` field |
| 3 | W9b/c machine profiles, `experiment_sha256`, accumulation, device | PR1 | `pytest tests/training/test_machine.py tests/training/test_accumulation.py` | CPU accumulation-equivalence run: batch2/accum4 vs batch8/accum1, same seed | `machine.py`, `configs/machines/*`, `configs/experiment/*` |
| 4 | `train.py` override list coercion | PR1 | `pytest tests/training/test_train_overrides.py` | N/A — parser unit test | override parser diff only |
| 5 | W1 `EarlyFusionBackbone` + stem inflation | PR2 | `pytest src/models/master/test_arch.py -k backbone` | `scripts/stem_init_audit.py --init inflation` (H-A) | `backbone.py` |
| 6 | W2 `FPNNeck` + `strides.py` (D-D) | PR2 | `pytest tests/training/test_strides.py src/models/master/test_arch.py -k neck` | N/A — CPU shape test | `neck.py`, `strides.py` |
| 7 | W3/W4 delete fusion, rewire `MasterModel` | PR2 | `pytest src/models/master/test_arch.py` | `scripts/measure_vram.py --variant new_4lvl` (V0) | `git revert` restores `fusion.py` + `DualFPN` |
| 8 | W5 head dedup, W7 KD rename + projection fix | PR2 | `pytest tests/models/student/test_head.py src/models/master/test_arch.py -k head` | N/A — `torch.allclose` regression | `head.py`, `distill_projections.py` |
| 9 | Eval-path stride fix + repo-guard test | PR2 | `pytest tests/test_stride_literals.py` | `scripts/evaluate_checkpoint.py --device cpu` smoke on a v2 checkpoint | `evaluate_checkpoint.py`, `visualize_damage_predictions.py` |
| 10 | Validation ladder + report (V0…H-F, H-BF16 registration) | PR2 (post-merge) | N/A — training runs, not unit tests | full rung commands, design §6 | `reports/fusion-redesign/`; no code revert needed |

---

## Phase 1 — PR1: D-G loop guards (land first; overlaps `damage-map-audit`'s guards)

- [x] 1.1 RED: `tests/training/test_loop_guards.py` — fake loader where every batch OOMs; assert epoch raises after epoch 1.
- [x] 1.2 RED: same file — fake loader where every batch is NaN; assert `nan_skipped` recorded and epoch raises.
- [x] 1.3 RED: assert `steps_taken == 0` raises regardless of skip cause.
- [x] 1.4 GREEN: `src/training/loop.py:364-366,391-396,404` — add `oom_skipped`/`nan_skipped`/`steps_taken` counters; epoch-1 tolerates OOM skips, epoch ≥2 raises on any; replace `max(n_batches,1)` with an explicit raise on 0.
- [x] 1.5 GREEN: wire counters into `LossHistory.extra_losses`, TensorBoard, and the checkpoint dict.
- [x] 1.6 Run `./.venv/bin/python -m pytest tests/training/test_loop_guards.py tests/test_training_loop.py tests/test_final_training_pipeline.py -v`; must be green before Phase 2.

### Addition to Phase 1 (maintainer decision, added after Phase 1-5 were first completed)

`src/training/kd_trainer.py` had the identical D-G defect (`KDTrainer._train_epoch` NaN/OOM caught-and-`continue`d before the counter incremented, `n_batches = max(n_batches, 1)`), correctly flagged as an issue in the first apply pass rather than fixed as unassigned work. The maintainer approved extending PR1 to cover it, since this PR exists specifically so training results can be trusted, and the sibling trainer had zero test coverage.

- [x] 1.7 GREEN: `src/training/kd_trainer.py` `_train_epoch` — mirror the D-G treatment from `loop.py`: `oom_skipped`/`nan_skipped`/`steps_taken` counters, OOM tolerated epoch 1 only (raises immediately from epoch 2), `steps_taken == 0` raises unconditionally, `max(n_batches, 1)` removed. No gradient accumulation added (out of scope; `KDTrainer` has none and none was requested) — `steps_taken` here is one optimizer step per successful micro-batch, stated explicitly as the one structural divergence from `loop.py`.
- [x] 1.8 RED/GREEN: `tests/training/test_kd_trainer_guards.py` (new, `KDTrainer` had zero prior coverage) — built a real `KDTrainer` (real `StudentModel` + real untrained `MasterModel` teacher checkpoint, since `distill_projections.py`'s projection factories hardcode channel counts a fake model would have to reimplement) with a fake batch loader. Covers: all-OOM epoch 1 raises (zero steps), all-NaN epoch raises and names `nan_skipped` in the message, OOM not tolerated from epoch 2, two partial-failure scenarios (NaN and OOM) complete with correct counters, and a clean epoch surfaces all three counters in the returned metrics.
- [x] 1.9 Run `./.venv/bin/python -m pytest tests/training/test_kd_trainer_guards.py -v` and the full suite; must stay green.

## Phase 2 — PR1: W9a precision enum

- [x] 2.1 RED: `tests/training/test_precision.py` — `precision` accepts only `fp32|fp16|bf16`, default `fp32`; `amp` key rejected/absent.
- [x] 2.2 RED: bf16 on a monkeypatched non-bf16 device raises and names the card.
- [x] 2.3 GREEN: create `src/training/precision.py` (`autocast_ctx`, `make_scaler`); `config.py:116` `amp: bool` → `precision: str = "fp32"` with `__post_init__` validation.
- [x] 2.4 GREEN: wire `loop.py:23,75,353,466`; scope `backbone.py:160-168`'s forced-fp32 guard to `precision == "fp16"` only.
- [x] 2.5 Mechanical: `configs/training_mango.yaml:42` and `configs/training_student.yaml` `amp:` → `precision:`. Also `configs/kd_training.yaml:19` (not explicitly named in this task, but `KDConfig.from_yaml` would raise `TypeError` on a stale `amp:` key now that the field is removed — fixed for correctness). `kd_trainer.py`/`kd_train.py`'s own `.amp` usages and `KDConfig.amp` field also migrated to `precision` for the same reason (amp is removed, not deprecated, system-wide).
- [x] 2.6 Run `pytest tests/training/test_precision.py -v`.

## Phase 3 — PR1: W9b/c machine profiles, accumulation, device

- [x] 3.1 RED: `tests/training/test_machine.py` — a machine profile setting a non-whitelisted key raises.
- [x] 3.2 RED: `effective_batch % batch_size != 0` raises; `grad_accum_steps` is derived, never read from file.
- [x] 3.3 RED: `experiment_sha256` stable across machine profiles, changes on a one-byte experiment-file edit.
- [x] 3.4 GREEN: create `src/training/machine.py` (whitelist `{device,batch_size,num_workers,pin_memory,precision}`, `experiment_sha256`, derived `grad_accum_steps`).
- [x] 3.5 GREEN: `run_artifacts.py:72-83,224` — record `experiment_sha256`, `effective_batch`, `precision`, `device` in `stage_summary.json` and the checkpoint dict. (`StageResult` gained 4 optional fields; `Trainer._save_checkpoint` now writes `experiment_sha256` alongside `config.__dict__`, which already carries `effective_batch`/`precision`/`device` as declared config fields.)
- [x] 3.6 GREEN: `loop.py:58` — device from `config.device` (default `"auto"`), not hardcoded; `train.py:176,186`, `kd_train.py:147,157` — `pin_memory` from config.
- [x] 3.7 RED: `tests/training/test_accumulation.py` — `batch_size=2,accum=4` vs `batch_size=8,accum=1` produce `torch.allclose` param updates after one effective step, fixed seed, CPU.
- [x] 3.8 RED: `grad_clip` applied once per effective step, not per micro-batch (call counter on `clip_grad_norm_`).
- [x] 3.9 GREEN: `loop.py:344-402` — scale loss by `1/accum`, `backward()` every micro-batch, `clip_grad_norm_`+`optimizer.step()`+`zero_grad()` only every `accum` steps (flush remainder at epoch end); move `grad_clip` (`:383`) inside the accumulation boundary.
- [x] 3.10 GREEN: create `configs/machines/{gtx1660s,rtx3080,rtx_a5000,cpu}.yaml`; `train.py` gains `--machine`.
- [x] 3.11 Run `pytest tests/training/test_machine.py tests/training/test_accumulation.py -v`.

## Phase 4 — PR1: `--override` list coercion

- [x] 4.1 RED: `tests/training/test_train_overrides.py` — `--override head_strides=[4,8,16,32]` yields `list[int]`, not the literal string. (`head_strides` does not exist as a config field until PR2; tested against the same generic, field-name-agnostic coercion path using existing list fields `class_weights`/`assigner_level_ranges`.)
- [x] 4.2 GREEN: `train.py:88-99` — port `evaluate_checkpoint.py:103-106`'s list branch (`json.loads` when value starts with `[`, else comma-split floats).
- [x] 4.3 Run `pytest tests/training/test_train_overrides.py -v`.

## Phase 5 — PR1: integration and close-out

- [x] 5.1 Full suite: `./.venv/bin/python -m pytest` — 134 pre-existing + new PR1 tests green. Result: 184/184 passed (134 baseline + 50 new PR1 tests) in ~57s.
- [x] 5.2 Update `openspec/specs/training-loop` delta for D-G, accumulation, precision, portability requirements. **Deviation**: these requirements were NOT already drafted in specs/ as the task text assumed — the existing `spec.md` covered only the PR2 end-to-end schedule (D-F). Added 4 new `### Requirement:` sections (Fatal Batch-Skip Guards, Gradient Accumulation, Explicit Precision Selection, Machine-Profile/Experiment Split) with GIVEN/WHEN/THEN scenarios matching shipped PR1 code. Also updated the engram `sdd/fusion-redesign/spec` artifact to match.
- [ ] 5.3 Open PR 1 against `main`; merge before starting Phase 6. **Not performed by sdd-apply** — PR creation/merge is a delivery-workflow action outside this executor's role; the branch `feat/trainer-correctness-portability` is ready for the orchestrator/maintainer to open the PR.

## Phase 6 — PR2: W1 backbone + stem inflation

- [x] 6.1 RED: `torch.allclose` — `W_new[:, :3] == W_imagenet*0.75`, `W_new[:, 3] == mean(W_imagenet,dim=1)*0.75`, bias/LN copied verbatim, against a freshly loaded `convnext_tiny`. (`tests/models/master/test_backbone.py::TestStemInflation`)
- [x] 6.2 GREEN: `src/models/master/backbone.py` — `EarlyFusionBackbone(pretrained, variant, in_channels=4)` replacing `DualConvNeXtBackbone`; `_load_pretrained_stem()` implementing D-B. `in_channels=3` (verbatim copy) also supported for the H-D control arm.
- [x] 6.3 RED (H-B): NIR-only input perturbation — count params with non-zero grad; bar ≥27M. (`test_backbone.py::TestNIRCapacity`)
- [x] 6.4 GREEN: verify H-B bar against the built backbone. **Measured: 26,618,208 params reachable at a 64px probe (below the 27M bar — receptive field too small to reach every stage-4 channel), 27,820,128 (the entire backbone) at a 640px probe.** Test uses 640px + `eval()` (stochastic depth in `train()` mode makes the exact reachable-parameter count non-deterministic per forward pass).
- [x] 6.5 Zero-cost rung H-A: `scripts/stem_init_audit.py --init {inflation,zero,copy}`; CONFIRM = inflation within 2× reference across-image std at every stage and closer than naive copy; REFUTE = fall back to zero-init, record why. **Run for real** on 16 real images from `data/cache/mango/rgb` (synthetic Gaussian noise batches made the across-image std band uninformatively narrow — every noise image has nearly identical statistics — so the script prefers real data when available). **Result: CONFIRM.** Inflation stayed within the 2× band at all 4 stages; summed |deviation| from the RGB-only reference was 0.0037 for inflation vs 0.1054 for naive copy (≈28× closer) and 0.1417 for zero-init.

## Phase 7 — PR2: W2 neck + strides resolver (D-D)

- [x] 7.1 RED: `FPNNeck` emits exactly the levels named by `strides`, finest-first, 256ch, for `[4,8,16,32]` and `[8,16,32]`. (`tests/models/master/test_neck.py`)
- [x] 7.2 GREEN: `src/models/master/neck.py` — delete `DualFPN` (`:106-173`); add `FPNNeck` wrapping unchanged `SingleFPN`; rewrite module docstring.
- [x] 7.3 RED: `_generate_anchors` at `[4,8,16,32]`, 640² → 34,000 anchors, per-level `[25600,6400,1600,400]`. (`tests/training/test_tal.py::TestGenerateAnchors::test_four_level_anchor_count_at_640`) `_generate_anchors` itself needed no code change, as design.md predicted.
- [x] 7.4 RED: `_level_admissibility` with `[32,64,128]` maps a 30px box to level 0. (`tests/training/test_tal.py::TestLevelAdmissibility`)
- [x] 7.5 RED: config invariant — `len(assigner_level_ranges) != len(head_strides)-1` raises. (`tests/training/test_strides.py::TestConfigInvariantEnforced`) Enforced in `TrainingConfig.__post_init__` only for `model_type="master"` — the student is out of scope (D-2) and always uses its own fixed 3-level strides regardless of these two fields.
- [x] 7.6 GREEN: create `src/training/strides.py` (`STRIDE_TO_LEVEL`, `resolve_head_strides`, `resolve_from_checkpoint`, `validate_strides`); wire into `config.py` `__post_init__`. Also added `resolve_active_strides` (branches on `model_type`), `select_by_strides` (KD's stride-indexed slicing, §5), `STUDENT_STRIDES`/`ALL_STRIDES`/`DEFAULT_HEAD_STRIDES` constants — the single canonical source the repo-guard test (10.1) exempts.
- [x] 7.7 GREEN: remove defaults on `YOLOv8Loss(strides=...)` (now keyword-only, required) and `decode_detections(strides=...)` (now required) so a forgotten argument is a `TypeError`. Also made `YOLODetectionHead(strides=...)` keyword-only/required for the same reason (not one of the original 8 sites, but the same class of literal-default hazard). **Consequence**: `tests/training/test_training_step.py`'s `criterion` fixture and `src/models/master/test_arch.py`/`sanity_check.py` (already being rewritten under 8.5) needed an explicit `strides=` argument added.
- [x] 7.8 Zero-cost rung H-C: static assert no `adaptive_avg_pool2d` between backbone and head; finest emitted cell ≤8px. (`tests/models/master/test_master_model.py::TestNoPoolingBetweenBackboneAndHead`)

## Phase 8 — PR2: W3/W4 fusion deletion + `MasterModel` rewire

- [x] 8.1 RED: `MasterModel(head_strides=[4,8,16,32])` fwd+bwd on CPU at 128²; output dict has exactly 7 keys; every emitted level receives a non-`None` grad (D3 regression test). (`tests/models/master/test_master_model.py`)
- [x] 8.2 GREEN: delete `src/models/master/fusion.py` in full (`StageAttentionFusion`, `CrossModalFusion`).
- [x] 8.3 GREEN: `src/models/master/master_model.py` — rewire `forward`; rename `distill_backbone_rgb`→`distill_backbone`, drop `distill_backbone_fused`; `freeze_backbone`/`unfreeze_backbone_stages` lose `rgb_stem`/`nir_stem` asymmetry (renamed `unfreeze_rgb_stem`→`unfreeze_stem`). **Found and fixed, in-scope consequence**: `StudentModel.unfreeze_backbone_stages`'s no-op stub parameter was also renamed `unfreeze_rgb_stem`→`unfreeze_stem` — its own docstring states it exists so `Trainer` can call it "polymorphically on both MasterModel and StudentModel"; leaving it stale would silently break that polymorphism the first time a schedule change reached the student path with the new keyword.
- [x] 8.4 GREEN: `src/models/master/__init__.py` — remove re-exports of the three deleted classes; export `EarlyFusionBackbone`/`FPNNeck`/`SingleFPN` instead.
- [x] 8.5 GREEN: `src/models/master/test_arch.py`, `sanity_check.py`, `README.md` — remove dual-stream imports/instantiation and references; rewritten for the single-stream architecture with shape/key assertions.
- [x] 8.6 GREEN: delete `scripts/visualize_attention.py`, `scripts/attention_comparison.py`.
- [x] 8.7 RED: rewrite `tests/models/master/test_freeze_policy.py` against `stem`/`stages` names (no `rgb_stem`/`nir_stem`).
- [x] 8.8 RED: synthetic v1 checkpoint dict raises the `arch_version` error before `load_state_dict`. (`tests/test_checkpoint_arch_version.py`)
- [x] 8.9 GREEN: `_save_checkpoint` writes `arch_version: 2` (`CHECKPOINT_ARCH_VERSION` in `loop.py`); `evaluate_checkpoint.py::_load_model`, `visualize_damage_predictions.py::load_model` check it first and also resolve `head_strides` from the checkpoint via `resolve_from_checkpoint`.
- [x] 8.10 Zero-cost rung V0: `scripts/measure_vram.py --variants new_3lvl new_4lvl`, batches 1/2/4; CONFIRM 4-level peak ≤6.0GB. **Run for real on the maintainer's GTX 1660 SUPER.** `master_v1`/`new_3lvl_preW5` are not separately measurable — the pre-redesign dual-stream code and the pre-W5 duplicate head-stem computation were both deleted outright (not toggles) rather than gated behind a flag; `master_v1` was already measured in `proposal.md` Round 2. **Measured (fp32, one fwd+bwd+step, real GPU):**

  | Config | batch 1 | batch 2 | batch 4 |
  |---|---:|---:|---:|
  | `new_3lvl` (`[8,16,32]`) | 1.124 GB | 2.102 GB | — |
  | `new_4lvl` (target, `[4,8,16,32]`) | 1.420 GB | 2.655 GB | 5.135 GB |

  **CONFIRM.** 2.655 GB at the configured `batch_size=2` — well under the 6.0 GB bar, and within 2% of design.md §4's hand-computed projection (≈2.6 GB). The batch-4 measurement (5.135 GB) also matches the design's ≈5.1 GB projection closely. No D-7 fallback rung is needed.

## Phase 9 — PR2: W5 head dedup + W7 KD/projections

- [x] 9.1 RED: post-fix head output `torch.allclose` with pre-fix output; `cls_stem`/`reg_stem` called exactly once per level (forward-hook counter). (`tests/models/master/test_head.py::TestStemComputedOncePerLevel`)
- [x] 9.2 GREEN: `head.py` `DecoupledHead.forward` returns `(cls_pred, reg_pred, cls_feat, reg_feat)`; `YOLODetectionHead.forward` reuses them instead of recomputing.
- [x] 9.3 GREEN: `kd_trainer.py` — teacher key `distill_backbone` (was `distill_backbone_rgb`).
- [x] 9.4 RED: `ProjectionLayers.forward` raises on a length mismatch instead of silently truncating the `zip`. **Finding, stated honestly**: verified against the current codebase, `forward` already asserted `len(teacher_features) == self.num_levels` **before** the `zip` — it was already loud (`AssertionError`), not silently truncating, contrary to design.md §5's premise (likely written against an earlier code revision). Strengthened the assert with a message identifying both counts and added `tests/models/master/test_distill_projections.py` locking in the pass/fail paths, including the exact 4-vs-3 scenario design.md describes.
- [x] 9.5 GREEN: `distill_projections.py` — explicit length-assertion message; `kd_trainer.py` slices the teacher pyramid/head features to the student's strides by stride value (`select_by_strides`), not by position, so the correct (non-crashing) path is also correct. `KDConfig.head_strides` (inherited from `TrainingConfig`) now describes the teacher's architecture; `_load_teacher` passes it to `MasterModel(head_strides=...)`. Added `tests/training/test_kd_level_slicing.py` proving the projections receive the P3/P4/P5-shaped features, not P2/P3/P4.

## Phase 10 — PR2: evaluation-path fixes and repo guard

- [x] 10.1 RED: repo-guard test scans `src/` and `scripts/` for `[8, 16, 32]` / `(8, 16, 32)` literals and fails on a match. (`tests/test_stride_literals.py`) Implemented via `ast` (List/Tuple literal nodes evaluating to `(8,16,32)`), not a text-line scan — a text scan would also flag every comment/docstring explaining this exact guard, including in this file and every fusion-redesign docstring citing the old literal, making the guard unmaintainable. `src/training/strides.py` is the one exempted file (the canonical `STUDENT_STRIDES` definition).
- [x] 10.2 GREEN: delete the 8 literal sites (`loss.py`, `loop.py` ×2, `decode.py`, `master_model.py`, `evaluate_checkpoint.py`, `visualize_damage_predictions.py` ×2) plus 4 more the guard caught during implementation (`head.py`'s own `strides or [8,16,32]` default, `kd_trainer.py`'s and `visualize_damage_predictions.py`'s local `STUDENT_STRIDES` re-declarations, `scripts/measure_vram.py`'s variant table) — all now source strides from config, `resolve_from_checkpoint`, or import the one canonical `STUDENT_STRIDES`.
- [x] 10.3 Run `pytest tests/test_stride_literals.py -v` — green (see full-suite run below).
- [x] 10.4 Config/data: create `configs/experiment/fusion.yaml` (byte-identical across machines per D-H whitelist); update `configs/training_mango.yaml` (`batch_size` 8→2, `head_strides: [8,16,32]` + `schedule: two_phase` added explicitly — required once `TrainingConfig`'s defaults changed to the 4-level/end-to-end architecture, otherwise this file would either fail the new `assigner_level_ranges` invariant or silently switch schedules).
- [x] 10.5 GREEN: `src/training/loop.py` `end_to_end` schedule (D-F) — two-group AdamW (`backbone.stages.*` at `lr*backbone_lr_mult`; `stem`+`neck`+`head` at `lr`) via `Trainer._discriminative_param_groups`; generalized `_rgb_stem_grad_norm` into `_module_grad_norms()` logging every backbone stage, the stem, the neck, and every head level to `LossHistory.extra_losses`/TensorBoard (`grad_norm_*` keys). `schedule="two_phase"` preserved and selectable (rollback path); `TrainingConfig.total_epochs` and `Trainer.fit()` branch on it. (`tests/training/test_end_to_end_schedule.py`)

## Phase 11 — PR2: integration and close-out

- [x] 11.1 Full suite: `./.venv/bin/python -m pytest` — **267/267 passed in ~65.5s** (190 baseline + 77 new tests, across 11 new/rewritten test files plus additions to `tests/training/test_tal.py` and one line changed in `tests/training/test_training_step.py`). `tests/test_final_training_pipeline.py` needed no change (does not reference `MasterModel`/strides). `tests/test_training_loop.py` needed no change — its `total_epochs`/`model_type` tests happen to hold under the new default by inspection, not luck: `epochs_phase1=30+epochs_phase2=20=50` coincides with the default `epochs=50` the new `schedule="end_to_end"` branch reads instead.
- [x] 11.2 Confirm `openspec/changes/fusion-redesign/specs/{multimodal-early-fusion,testing-teacher-arch,training-loop,kd-training}` (delta specs — merging into `openspec/specs/` is `sdd-archive`'s job per the OpenSpec convention, not `sdd-apply`'s) wording matches shipped code. Reviewed all four against the final implementation; no drift found requiring an edit — the specs were written precisely enough that implementation followed them directly.
- [ ] 11.3 Open PR 2 against `main`, based on PR 1's merge commit; merge before Phase 12 training rungs run. **Not performed by sdd-apply** — PR creation/merge is a delivery-workflow action outside this executor's role; branch `feat/early-fusion-architecture` (cut from PR1's merge commit `6ce6697`) is ready for the orchestrator/maintainer to open the PR.

## Phase 12 — Training rungs and validation report (post-merge; not gated on further code review)

- [ ] 12.1 V1: `--override seed={42,1337,2024}` on `configs/experiment/fusion.yaml` + a machine profile; publish measured s/epoch from the first seed before scheduling H-D/H-E; record `σ_d`, `σ_L5`, mean damage AP50.
- [ ] 12.2 Confirm mango guardrail (AP50 ≥0.774) held on V1; if not, treat as a failed run per H-F, not a valid result, and stop before H-D/H-E.
- [ ] 12.3 H-D: `--override in_channels=3 seed={42,1337,2024}`; compute Δ vs V1's reused 3 seeds. Bar: CONFIRM ≥2·σ_d, REFUTE ≤1·σ_d.
- [ ] 12.4 H-E: `--override head_strides=[8,16,32] assigner_level_ranges=[64,128] seed={42,1337,2024}`; Δ vs V1. Bar: CONFIRM ≥2·σ_d, REFUTE ≤1·σ_d.
- [ ] 12.5 H-F: read `oom_skipped`, `nan_skipped`, `steps_taken`, per-module grad norms from every run above; CONFIRM = every module measurably off init; REFUTE (any bit-exact `max|γ−1|==0`) marks that run failed, not a refuted hypothesis, and blocks reading other rungs from it.
- [ ] 12.6 H-BF16: registration only in this change (Q8 resolved — deferred, gates nothing). If Ampere hardware becomes available: `--machine configs/machines/rtx_a5000.yaml --override precision=bf16 epochs=5 seed=42`; CONFIRM = 5 epochs, `nan_skipped==0`, epoch-5 loss within 1·σ_L5 of fp32; REFUTE = any non-finite loss.

## Phase 13 — Publish negative results (mandatory regardless of outcome)

- [ ] 13.1 Publish V0 measured peaks for all three configs and the fallback step taken, even if the projection held and no fallback was needed.
- [ ] 13.2 Publish H-A result (confirmed init or the zero-init fallback) with per-stage mean/std.
- [ ] 13.3 Publish H-B count and H-C static-assertion outcome.
- [ ] 13.4 Publish H-D result against its pre-registered bar verbatim, **including a REFUTE**: state plainly that crop-level complementarity did not transfer to detection at this data scale if that is the outcome, and do not retract the architecture change on that basis (D1–D4 still hold).
- [ ] 13.5 Publish H-E result verbatim; if REFUTE, record that P2 is retired to `head_strides:[8,16,32]` as the recommended default going forward, reclaiming its params/activations.
- [ ] 13.6 Publish H-F pass/fail; a failed run is not reported as a valid hypothesis result anywhere else in the document.
- [ ] 13.7 Record H-BF16 deferral explicitly (no run performed, bars fixed in advance) if Ampere access did not materialize.
- [ ] 13.8 State the Q1 limitation verbatim: this change can prove the redesign is **better**, not that it is **sufficient** — do not let a relative win be written up as fitness for use.
- [ ] 13.9 Record the comparability rule: two runs are comparable only if `experiment_sha256` **and** `effective_batch` match; note the resolved `effective_batch: 8` (Q7).
- [ ] 13.10 Write `reports/fusion-redesign/validation-report.md` collecting 13.1–13.9; cross-link `reports/damage-map-audit/` as the prior baseline, never as a superseded comparison target.
