# Proposal: Early-fusion redesign — 4-channel RGB+NIR backbone

Change id: `fusion-redesign` · Store: hybrid · `delivery_strategy: ask-on-risk` · `review_budget_lines: 800`
Depends on and follows `damage-map-audit`. Does not supersede it.

## Intent

Damage detection is pinned at **AP50 0.0643** while the same data supports **AUC 0.9171** on
RGB+NIR crops (`reports/damage-map-audit/steps-1-2-results.md`). The probe that reached 0.9171 was
plain early fusion: two channels stacked, three conv/BN/ReLU blocks, global average pool, linear.
No attention. The production teacher spends **9,400,320 parameters** on `StageAttentionFusion`
(verified: 4 stages × 12C² for C ∈ {96,192,384,768}) and reaches 0.0643.

Four verified defects explain the gap, and each is structural rather than a tuning error:

| # | Defect | Evidence |
|---|---|---|
| D1 | No NIR trunk — NIR-specific capacity is `nir_stem` = 1,824 params against `shared_stages` = 27,813,696 ImageNet RGB params | `MasterBackbone` child param counts |
| D2 | Phase 1 runs `freeze_stages=4` (`src/training/loop.py:125-131`); the evaluated checkpoint (epoch 68) is inside Phase 1; Phase 2 ran one epoch | run CSV, loop code |
| D3 | `DualFPN` builds 4 fusion stages but 3 `fusion_convs` and zips from `[1:]` (`neck.py:165`); stride-4 fusion has no gradient path | `fused_features[0].grad is None`; stage-1 LayerNorms bit-exact at init after 68 epochs (`max|gamma-1| = 0.000e+00`) |
| D4 | `max_tokens_side=20` pools every stage to 20×20 → one token = 32×32 px, against a 30 px median lesion | `fusion.py:92-97`, lesion size distribution |

`adaptive_avg_pool2d` is specifically the wrong operator here: damage-vs-healthy NIR mean intensity
has Cohen's d = 0.20, so the usable cue is textural. Average pooling preserves regional mean and
destroys texture — it discards exactly the carrier of the signal.

**Direction (maintainer decision, binding): simple early fusion.** Stack RGB+NIR as a 4-channel
input to a single backbone. Remove the two-branch cross-attention design.

## Scope

### In Scope

**W1 — Backbone (`src/models/master/backbone.py`)**
- Replace `DualConvNeXtBackbone` (two stems + one shared trunk, two forward passes) with a
  single-stream `EarlyFusionBackbone`: `Conv2d(4, 96, k=4, s=4)` stem + the same ConvNeXt stages,
  one forward pass, emitting `[S1..S4]` at `[96, 192, 384, 768]`.
- Stem inflation from ImageNet weights (decision D-1 below), with a deterministic init test.

**W2 — Neck (`src/models/master/neck.py`)**
- Delete `DualFPN` (two `SingleFPN` + 3 `fusion_convs`). Keep `SingleFPN` **unchanged** — it already
  builds `[P2, P3, P4, P5]`.
- New thin `FPNNeck` wrapping `SingleFPN` with a configurable emitted-level list. Default emits
  4 levels including P2 (decision D-3).

**W3 — Fusion removal**
- Delete `src/models/master/fusion.py` (`StageAttentionFusion`, `CrossModalFusion`) in full.

**W4 — Model wiring (`src/models/master/master_model.py`)**
- Rewire `forward` for the single stream. Output dict goes from 8 keys to 7:
  `distill_backbone_fused` is removed; `distill_backbone_rgb` is renamed `distill_backbone`
  (its channels `[384, 768]` are unchanged, so `backbone_projections()` needs no edit).
- `freeze_backbone` / `unfreeze_backbone_stages` lose the `rgb_stem`/`nir_stem` asymmetry.

**W5 — Head and strides (`src/models/master/head.py`, `configs/training_mango.yaml`, `src/training/config.py`)**
- `YOLODetectionHead` already generalises over `len(strides)`; add the stride-4 level via config
  (`head_strides: [4, 8, 16, 32]`), and widen `assigner_level_ranges` to `[32, 64, 128]`.
- Remove the duplicate stem computation at `head.py:155` vs `head.py:164-165`
  (`cls_stem`/`reg_stem` are each run twice per level — once for the prediction, once for the
  distillation feature). This doubles head activation memory and is a precondition for P2 fitting
  in 6.4 GB.

**W6 — Training schedule (`src/training/loop.py`, `configs/training_mango.yaml`)**
- Replace the Phase-1-frozen schedule for this model with end-to-end training at a discriminative
  LR (decision D-4), plus per-module grad-norm logging so D2 cannot recur silently.

**W7 — KD contract (`src/training/kd_trainer.py:151`)**
- Update the teacher feature key to `distill_backbone`. No projection shapes change.

**W8 — Specs, tests, validation report.**

### Out of Scope

- Retraining or redesigning the RGB-only student, and any KD run. The teacher must be measured
  first; the KD implication is recorded below (decision D-5), not solved here.
- Backbone variant change (`small`, 50M) — does not fit the 6.4 GB budget alongside P2.
- Anything already owned by `damage-map-audit`: split reconciliation, NMS, per-class decode,
  assigner center sampling, `class_weights`. This change consumes them; it does not touch them.
- Re-testing the four refuted hypotheses (assigner starvation, missing NMS, argmax decode
  suppression, label under-annotation). Each was refuted against a pre-registered bar.
- NIR acquisition/quantisation (77 of 256 levels) — an acquisition issue with no software fix.

## Capabilities

### New Capabilities
- `multimodal-early-fusion`: single-stream 4-channel RGB+NIR backbone; pretrained-stem channel
  inflation; configurable FPN emitted levels including stride 4; the deterministic
  NIR-capacity / gradient-reachability / no-pooling invariants.

### Modified Capabilities
- `testing-teacher-arch`: `DualConvNeXtBackbone` two-stream requirement replaced by the
  single-stream 4-channel requirement; the `CrossModalFusion` requirement is **removed**;
  "`DualFPN` SHALL output exactly 3 pyramid levels" becomes configurable levels with P2 emitted
  by default; `MasterModel` forward-dict contract 8 keys → 7.
- `training-loop`: "Two-Phase Fine-Tuning" no longer applies to this model; end-to-end training
  with discriminative LR and mandatory per-module grad-norm logging.
- `kd-training`: teacher backbone feature key rename; the RGB-only student's measured ceiling
  (0.7774 AUC vs 0.9171) is recorded as a spec-level constraint on KD expectations.

## Approach and design decisions

Decided here, with rationale. These are not open questions.

### D-1 — Pretrained stem: mean-preserving inflation

`W_new[:, :3] = W_imagenet * 3/4` and `W_new[:, 3] = mean(W_imagenet, dim=1) * 3/4`; bias and the
stem LayerNorm copied verbatim.

- **Not zero-init.** The measured cue is textural (Cohen's d = 0.20 on mean intensity is weak), so
  the NIR channel should start with ImageNet edge/texture priors rather than learn them from
  148 images.
- **Not naive copy.** Adding a 4th contributing channel raises the stem pre-activation by ≈4/3,
  perturbing the statistics the pretrained LayerNorm and stage 1 expect. The 3/4 factor is
  mean-preserving inflation.
- The assumption behind the rescale — that all four channels arrive at comparable scale — holds:
  RGB uses ImageNet normalisation and NIR uses `nir_mean=0.0569, nir_std=0.0546`.
- The existing `_load_pretrained_nir_stem` already does mean-over-RGB for the 1-channel stem, so
  this reuses proven code plus one scalar.

### D-2 — What survives, verbatim

| Component | Fate |
|---|---|
| `SingleFPN` | **Kept unchanged** — already emits P2..P5 |
| `YOLODetectionHead`, `DecoupledHead` | Kept; already generalise over `len(strides)`. One change: remove the duplicate stem computation |
| `TaskAlignedAssigner`, `YOLOv8Loss`, `_generate_anchors` | Kept unchanged; already generalise over stride lists |
| `Trainer`, decode, metrics, dataset | Kept; `configs` gain `head_strides` |
| `ProjectionLayers` / `fpn_projections` / `backbone_projections` | Kept; channels unchanged |
| `DualConvNeXtBackbone` | Replaced |
| `DualFPN`, `fusion_convs`, `fpn_nir` | Deleted (≈3.1M params) |
| `StageAttentionFusion`, `CrossModalFusion` | Deleted (9,400,320 params) |

### D-3 — P2: reconnect it, and prove it earns its cost

Reconnect. Three reasons, and one that is close to free:

1. Damage median short side is 30.2 px, p25 22 px. At stride 8 a median lesion spans ≈3.8 cells;
   at stride 4, ≈7.6. With `assigner_level_ranges` at `[32, 64, 128]`, the median damage box moves
   to the stride-4 level — this changes the assigned level for roughly half of damage GTs.
2. D3 is precisely the defect of computing the stride-4 level and discarding it. Leaving it
   disconnected reproduces the defect in a new architecture.
3. **`SingleFPN` already computes P2 today, twice** (once per branch), and both are discarded.
   Collapsing two FPNs to one and *keeping* P2 is cheaper in the neck than the status quo.

The real cost is the 4th `DecoupledHead` at 160×160: ≈2.4M parameters, and activations of
256×160×160 per stem. Removing the duplicate stem computation (W5) halves that. P2 ships behind
`head_strides`, so it is ablatable, and hypothesis H-E measures it rather than assuming it.

### D-4 — Training schedule: end-to-end, discriminative LR

`freeze_stages=0` from epoch 0; pretrained stages at 0.1× the neck/head LR.

- D2 showed the two-phase schedule was never escaped in practice.
- A 4-channel input is out of ImageNet distribution by construction. Freezing the stages that must
  adapt to it is the same category of error as D1.
- Overfitting risk on 148 images is real and is mitigated by the discriminative LR, the existing
  augmentation, and `patience: 20` — and it is *observable* in the existing train/val logging,
  where a frozen trunk's failure to learn was not.
- H-F below makes this falsifiable rather than assumed.

### D-5 — Knowledge distillation: bounded by measurement, deferred by decision

The student is RGB-only. The crop probe measures RGB alone at **0.7774** AUC against **0.9171** for
the pair. The student therefore has a materially lower ceiling than the teacher **by construction**,
not by a training defect. Consequences, recorded now:

- A KD run that fails to close the teacher-student gap is the **expected** outcome, not a bug. Any
  future KD rung must be measured against a plain RGB-only student baseline, never against the
  teacher.
- The teacher's backbone features are now genuinely multimodal (fused at the stem) rather than an
  RGB stream, so `distill_backbone` carries strictly more NIR information than `distill_backbone_rgb`
  did. That makes KD a cross-modal hallucination task: the student is asked to infer NIR-correlated
  structure from RGB. That is a legitimate setting, and 0.7774 bounds it.
- No KD training runs in this change. The contract is kept working (W7) and the bar is pre-registered
  (H-G) so the follow-up cannot start without one.

### D-6 — Parameter budget: mostly do not reinvest it

Freed: 9.40M (fusion) + ≈3.1M (`fpn_nir` + `fusion_convs`). Reinvested: ≈2.4M (the P2 head).
Net reduction ≈10M.

The rest is deliberately **not** reinvested. The probe that reached 0.9171 had three conv blocks
against a 27.8M-parameter trunk that reached 0.0643 — the measured defect is information flow, not
capacity, and 148 training images do not support more parameters. The freed budget is banked as
VRAM and wall-clock headroom, which is what buys the multi-seed runs the validation plan needs to
mean anything at this data volume.

### D-7 — Expected memory change (direction only; must be measured)

| Component | Change |
|---|---|
| Backbone | 2 streams → 1 (both stem and every stage ran twice) — roughly halved |
| Cross-attention | Removed entirely (4 stages × 400×400×8-head attention matrices, fp32-forced) |
| Neck | 2 FPNs → 1 — roughly halved |
| Head | +1 level at 160×160 — the only increase; partly offset by removing the duplicate stem compute |

Expected net: **cheaper than current at 3 levels, at or near current peak at 4 levels.** This is a
prediction, not a measurement, and it is falsifiable: task V0 measures
`torch.cuda.max_memory_allocated()` for one training step under (a) current master, (b) new 3-level,
(c) new 4-level, before any long run is scheduled.

**Fixed OOM fallback ladder** (order fixed in advance; `amp` is NOT on it — it is disabled because
`YOLOv8Loss` NaNs under fp16 and the backbone forces autocast off):
1. remove the duplicate head stem compute (already in W5);
2. P2 head at 128 channels;
3. `batch_size` 8 → 4.

Whichever step is taken **must be fixed before the baseline and held constant across every rung**,
or attribution breaks.

## Validation — complete hypothesis-then-test cycle

Ground rules, inherited and binding:

- Every hypothesis gets an experiment that can confirm **or** refute it. No hypothesis is deferred.
- CONFIRM and REFUTE bars are registered **before** any run.
- Bars are **derived** from mechanism or from a measurement. Where no bar is derivable, this
  document says so and flags it (Q1) rather than inventing a plausible number.
- Every baseline is measured on the reconciled split (148/18/21) with NMS enabled. The superseded
  published figures are never a comparison target.
- Negative results are published.

**Noise band, measured not assumed.** Rung V1 runs the baseline at **3 seeds** and records the
between-seed std of damage AP50, `σ_d`. Every subsequent outcome bar is expressed in units of `σ_d`.
No numeric AP delta is written into this document ahead of that measurement, because none is
derivable ahead of it.

### Zero-cost rungs (no training)

| # | Hypothesis | Experiment | CONFIRM | REFUTE |
|---|---|---|---|---|
| V0 | The redesign fits 6.4 GB at 4 levels | `max_memory_allocated()` on one training step, 3 configs | 4-level peak ≤ 6.0 GB | > 6.0 GB → apply the D-7 ladder in order, then re-measure |
| H-A | Mean-preserving inflation preserves ImageNet feature statistics better than zero-init or naive copy | Instantiate all three; compare per-stage activation mean/std on real batches against the RGB-only pretrained reference | inflation is within 2× the reference's own across-image std at every stage, and closer than naive copy | inflation is outside that band → fall back to zero-init and record why |
| H-B | Early fusion gives NIR real trainable capacity | Gradient audit: count parameters receiving non-zero grad from a NIR-only perturbation | ≥ 27M (vs the measured 1,824 today) | < 1M → the stem is not propagating NIR; the design is wrong |
| H-C | Lesion-scale structure now reaches the loss | Static assertions on the assembled model | no `adaptive_avg_pool2d` between backbone and head; finest emitted level's cell ≤ 8 px (vs 32 px today) | any pooling survives, or finest cell > 8 px |

H-B and H-C look near-tautological. They are registered anyway, because D1 and D3 were exactly
unverified structural assumptions that held for 68 epochs.

### Training rungs

All rungs: reconciled split, NMS on, per-class decode, `class_weights` and `batch_size` held
constant, **3 seeds each**, damage AP50 on val as the primary readout.

| # | Hypothesis | Experiment | Bar |
|---|---|---|---|
| V1 | baseline | New architecture, 4 levels, end-to-end, 3 seeds | Establishes `σ_d` and the reference mean. Not a hypothesis test |
| H-D | **The measured RGB+NIR complementarity survives the move to detection** | Identical model, 4-channel vs 3-channel RGB-only (NIR channel removed, not zeroed), 3 seeds each | CONFIRM: damage AP50(4ch) − AP50(3ch) ≥ 2·`σ_d`. REFUTE: \|Δ\| ≤ 1·`σ_d`. Derivation: the probe separates 0.9171 from 0.7774 with seed stds 0.0044 and 0.0333 — a gap far outside its own noise. A refutation here says the complementarity does **not** transfer to detection at this data scale, which would be the single most important negative result this project can produce, and it must be published as such |
| H-E | The stride-4 level improves damage detection | `head_strides` `[4,8,16,32]` vs `[8,16,32]`, `assigner_level_ranges` adjusted to match, 3 seeds each | CONFIRM ≥ 2·`σ_d`; REFUTE \|Δ\| ≤ 1·`σ_d`. Derivation: median damage max-side crosses the 32 px bin boundary, so P2 changes the assigned level for roughly half of damage GTs. A refutation retires P2 and reclaims its ≈2.4M params and its activations |
| H-F | End-to-end training actually adapts the trunk | Per-module grad-norm log across the full run; end-of-run init-distance check | CONFIRM: every emitted level's neck and head parameters, and every backbone stage, are measurably off init. **REFUTE: any module bit-exactly at init (`max\|γ−1\| == 0.000e+00`) — the exact signature that exposed D3.** A REFUTE here is a failed run, not a refuted hypothesis, and blocks reading any other rung |
| H-G | KD is worth running on the redesigned teacher | **Not run in this change.** Bar registered now so the follow-up cannot start without one | CONFIRM: distilled RGB student damage AP50 ≥ plain RGB student + 2·`σ_s` (`σ_s` measured the same way). Prior evidence weighs weakly against it: distilled scored 0.062/0.065 vs the plain student's 0.041/0.090 |

**Guardrail on every training rung:** mango AP50 ≥ 0.9 × the same rung's own baseline
(0.9 × 0.8595 = 0.774 against the current master). A damage gain bought by wrecking mango is a
regression. Mango at 0.8595 is also the evidence that the surrounding machinery — localisation,
background rejection, NMS, assigner — supports high AP for a class the model can see, so a damage
failure after this redesign cannot be attributed back to that machinery.

**On the target.** 0.9171 AUC is classification **given the correct box**. Detection additionally
requires localisation and rejection of background across ~34,000 anchors at 4 levels. 0.9171 is a
ceiling, not a predicted AP, and no bar in this document assumes AP can approach AUC.

### Cost

Per-epoch wall clock on the local GTX 1660 SUPER is **not measured**. Recent student runs were
46–84 epochs. Task V1 must measure and publish s/epoch on its first run and restate the total
schedule before the remaining rungs are scheduled. Do not assume the cloud instance's 6.6–6.9 s/epoch.

## Affected Areas

| Area | Impact | Description |
|---|---|---|
| `src/models/master/backbone.py` | Modified | Single-stream 4-channel `EarlyFusionBackbone`; inflation init |
| `src/models/master/fusion.py` | **Removed** | Whole file (≈240 lines) |
| `src/models/master/neck.py` | Modified | `DualFPN` deleted; `SingleFPN` untouched; new `FPNNeck` with configurable levels |
| `src/models/master/master_model.py` | Modified | Rewiring; 7-key output contract; freeze helpers |
| `src/models/master/head.py` | Modified | Remove duplicate stem compute |
| `src/training/loop.py` | Modified | End-to-end schedule; discriminative LR; grad-norm logging |
| `src/training/config.py` | Modified | `head_strides`, LR-multiplier field |
| `configs/training_mango.yaml` | Modified | `head_strides`, `assigner_level_ranges` |
| `src/training/kd_trainer.py` | Modified | Teacher key rename (1 line) |
| `src/models/master/test_arch.py`, `tests/` | Modified/New | Fusion tests removed; inflation, gradient-audit, P2 and level-config tests added |
| `openspec/specs/testing-teacher-arch`, `training-loop`, `kd-training` | Modified | Delta specs |
| `scripts/evaluate_checkpoint.py:149`, `scripts/visualize_damage_predictions.py:73` | Modified | Hardcoded `[8, 16, 32]` must read the checkpoint's strides |

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| H-D refutes — early fusion does not transfer the crop-level complementarity to detection | Medium | 3-seed design and a derived bar make the refutation readable; publish it. The architecture is still strictly better than the status quo on D1–D4 |
| P2 does not fit in 6.4 GB at batch 8 | Medium | V0 measures before any long run; D-7 ladder is fixed in advance and held constant |
| 148 training images + end-to-end training overfits | Medium | Discriminative LR, existing augmentation, `patience: 20`, train/val divergence is logged |
| Existing checkpoints become unloadable | High (accepted) | Architecture change is total; every checkpoint under the old design is superseded. V1 establishes the new reference |
| Conflicts with `damage-map-audit` uncommitted work and PRs #9/#10 | High | Hard sequencing: this change starts only after `damage-map-audit` merges. Overlap is limited to `configs/training_mango.yaml`, `src/training/config.py`, `tests/` |
| Exceeds the 800-line review budget | High | See Delivery. `delivery_strategy` is `ask-on-risk`; the previous `exception-ok` acceptance was scoped to `damage-map-audit` only and does **not** carry over |
| Hardcoded `[8, 16, 32]` in scripts silently misreads a 4-level checkpoint | High | In scope; covered by a test |

## Delivery

Rough authored estimate (additions + deletions): W1 ≈220, W2 ≈180, W3 ≈240 (deletion), W4 ≈120,
W5 ≈90, W6 ≈130, W7 ≈5, tests ≈350, specs ≈150 → **≈1,485 lines**, well over 800.

Recommended chained series (each slice has a clear start, finish, verification and rollback):

| PR | Content | Est. |
|---|---|---|
| 1 | `EarlyFusionBackbone` + inflation + H-A/H-B tests. Additive; nothing rewired | ≈330 |
| 2 | `FPNNeck` + configurable head levels + `master_model` rewire + delete `fusion.py` + spec deltas + test updates | ≈700 |
| 3 | Training schedule, config fields, KD key rename, script stride fixes | ≈250 |
| 4 | Validation report — V0/H-A..H-F results including negatives | docs |

Do not silently exceed the budget as a single PR. `sdd-tasks` must emit the guard forecast.

## Rollback Plan

- Slices 1–3 are independent commits on separate PRs; revert any one without touching the others.
- P2 and the level count are config-driven (`head_strides`): revert to `[8, 16, 32]` without a code
  revert.
- The LR multiplier and freeze policy are config-driven: the previous two-phase schedule is
  restorable by config.
- Slice 2 is the irreversible one — it deletes `fusion.py` and `DualFPN`. `git revert` of that PR
  restores both, and no checkpoint depends on the new files until V1 runs.
- If H-D refutes, keep the architecture (it fixes D1–D4 regardless), publish the negative result,
  and reopen the direction question rather than patching bars after the fact.

## Dependencies

- `damage-map-audit` merged: reconciled splits (148/18/21), per-class NMS decode, configurable
  thresholds, assigner center sampling. Every baseline here depends on them.
- PRs #9 and #10 merged or explicitly sequenced.
- `./.venv/bin/python -m pytest` — 134 tests, ~60s, currently green. `python` is NOT on PATH;
  `nbformat` and `sklearn` are NOT installed.
- Splits must keep reading `data/annotations/yolo/splits.json`. Nothing may reintroduce a
  directory-listing code path.
- Local GPU: GTX 1660 SUPER, 6.4 GB, cc7.5, torch 2.9.1+cu128.

## Success Criteria

- [ ] `StageAttentionFusion` and `CrossModalFusion` no longer exist in the repository.
- [ ] H-B passes: NIR-reachable trainable parameters ≥ 27M, up from the measured 1,824.
- [ ] H-C passes: no `adaptive_avg_pool2d` in the backbone→head path; finest emitted cell ≤ 8 px.
- [ ] V0 published: measured peak VRAM for the 3-level and 4-level configs, and the fallback taken.
- [ ] H-A resolved and the chosen stem init recorded with its measurement.
- [ ] V1 published: 3-seed baseline on the reconciled split with NMS, with `σ_d` and measured s/epoch.
- [ ] H-D resolved against its pre-registered bar and published, **including if it refutes**.
- [ ] H-E resolved against its pre-registered bar and published.
- [ ] H-F passes: no module bit-exactly at init after training.
- [ ] Mango guardrail held on every training rung (≥ 0.774 AP50).
- [ ] H-G bar recorded in the `kd-training` spec; no KD run in this change.
- [ ] `./.venv/bin/python -m pytest` green.
- [ ] Q1 answered and recorded before the validation report is written.

## Proposal question round — for the maintainer

The five architecture questions in the brief are **decided** above (D-1 … D-6) and are not returned
as a menu. These four are genuinely undecidable from the evidence and are flagged rather than guessed.

1. **Q1 — What absolute damage AP50 counts as project success?** No bar is derivable. 0.9171 is a
   classification ceiling on pre-localised crops with size-matched in-mango negatives; converting it
   to an AP requires a localisation model and a background-rejection model this project has not
   measured. Every bar above is therefore relative (multiples of the measured `σ_d`). If the project
   has an external requirement — a customer threshold, a paper claim, a comparison target — it must
   be stated by the maintainer, not inferred here. Without it, this change can prove the redesign is
   *better* but cannot state that it is *sufficient*.
2. **Q2 — Delivery.** `delivery_strategy` is `ask-on-risk` and the estimate is ≈1,485 lines. Accept
   the 4-PR chain above, or grant a fresh `size:exception` for this change specifically?
3. **Q3 — Sequencing.** Confirm this change starts only after `damage-map-audit` and PRs #9/#10
   merge. If it must start in parallel, name the branch point.
4. **Q4 — H-D scope.** The RGB-only control in H-D doubles the training cost of that rung
   (6 runs instead of 3). It is the only experiment that tests the multimodal premise in detection
   rather than on crops. Confirm the spend, or state a cheaper control.

## Maintainer Decisions — Round 1 (RESOLVED 2026-08-29)

Binding on `sdd-spec`, `sdd-design`, `sdd-tasks` and `sdd-apply`. Do not re-ask.

1. **Q1 — success criterion: demonstrated relative improvement.**
   Bars stay relative, expressed in multiples of the measured between-seed standard deviation, as
   the proposal already frames them. No absolute AP50 target is set. The contribution this change
   claims is that the redesign is measurably better than the baseline under a rigorous method, not
   that it reaches a production threshold.

   State this limitation plainly in the validation report: the change can prove the redesign is
   **better**, not that it is **sufficient**. Do not let a relative win be written up as if it
   established fitness for use.

2. **Q2 — delivery: SINGLE PR with a NEW `size:exception` scoped to this change.**
   `delivery_strategy` becomes `exception-ok` for `fusion-redesign` only. Estimated ~1,485 authored
   lines against the 800 budget. The orchestrator recommended a 4-PR chain and noted that the
   previous single-PR exception ended at roughly twice its estimate; the maintainer accepted the
   exception again. Proceed as one PR and do not silently re-split.

3. **Q3 — sequencing: starts only after the pending work merges.**
   This change depends on the reconciled splits, NMS, and per-class decode, all of which currently
   live uncommitted in the `damage-map-audit` working tree, plus open PRs #9 and #10.
   `sdd-spec`, `sdd-design` and `sdd-tasks` may proceed now — they do not touch code — but
   `sdd-apply` MUST NOT begin until those land. State the branch point explicitly in tasks.

4. **Q4 — H-D authorised at full spend: 6 training runs, 3 seeds per arm.**
   The 4-channel versus RGB-only detection control is the only experiment that tests the multimodal
   premise in detection rather than on crops. A single-seed control could not separate a narrow
   result from noise. If H-D refutes, publish it — the crop-level complementarity failing to
   transfer at this data scale is the most valuable negative result available here.

### Verification note on two proposal claims

Both load-bearing code findings were independently checked by the orchestrator:

- **The head recomputes its stems.** `src/models/master/head.py:155` calls `head(feat)`, which runs
  `cls_stem` and `reg_stem`, and lines 164-165 then call `head.cls_stem(feat)` and
  `head.reg_stem(feat)` again for the distillation features. Confirmed: the same computation runs
  twice at every pyramid level. Removing it (W5) stands.
- **Fusion parameter count corrected.** Measured by instantiation, `StageAttentionFusion` totals
  **9,421,920** parameters (96: 112,032 / 192: 445,248 / 384: 1,775,232 / 768: 7,089,408), not the
  9,400,320 the 4C²+8C² formula predicts — the formula omits biases and LayerNorm parameters. The
  magnitude of the claim is unaffected.

## Maintainer Decisions — Round 2 (RESOLVED 2026-08-29, hardware portability)

### Measured local training cost — supersedes the cloud-derived estimate

Measured on the maintainer's GTX 1660 SUPER (6.44 GB), master model, fp32, synthetic batches:

| batch | peak VRAM | step time | throughput |
|---|---|---|---|
| 1 | 2.32 GB | 430 ms | 2.32 img/s |
| 2 | 4.42 GB | 888 ms | 2.25 img/s |
| 4 | 8.64 GB | — | exceeds VRAM; the WSL driver spills to host RAM |
| 8 | OOM at 13.54 GB | — | — |

Scaling is roughly 2.1 GB per sample plus 0.2 GB base. `configs/training_mango.yaml` currently sets
`batch_size: 8`, which does not fit.

**Master epoch (148 images): ~65 s. Eighty epochs: ~87 minutes.** The "4-6 minutes" figure in the
earlier design came from the cloud instance and from the far smaller student; it does not apply and
must not be carried forward.

Throughput is essentially flat between batch 1 and 2 (2.32 vs 2.25 img/s), so the card is already
saturated. **Dropping to batch 2 costs no wall-clock time** — it only removes the OOM risk.

### 5. Q5 — batch size: 2 locally.

Accepted with the measured time cost.

### 6. Q6 — hardware must be configurable so training can move to other machines.

The maintainer has access to other GPUs. Estimated, from specification ratios rather than
measurement — label these as estimates wherever they appear:

| GPU | VRAM | FP32 TFLOPS | batch that fits | 80 epochs (estimated) |
|---|---|---|---|---|
| GTX 1660 SUPER | 6.4 | 5.0 | 2 | **87 min (measured)** |
| RTX 3080 | 10 | ~29.8 | 4 | ~20-30 min |
| RTX A5000 | 24 | ~27.8 | 8 | ~15-25 min |

Theoretical ratio is 5.5-6x; realised throughput on this kind of workload is typically 50-70% of
that, so 3-5x is the honest band.

**What must become configurable, and why:**

1. **Gradient accumulation — the load-bearing item.** Training at batch 2 locally and batch 8 on an
   A5000 are different optimisations, so their results are not comparable. H-D compares two arms
   against pre-registered bars; running arms on different hardware without a fixed effective batch
   would destroy attribution. Accumulation keeps the effective batch constant and makes the hardware
   genuinely interchangeable. Physical batch becomes a memory knob; effective batch becomes the
   experimental constant, and must be recorded in every run's artifacts.
2. **Device selection in the training loop.** `src/training/loop.py:58` hardcodes
   `torch.device("cuda" if torch.cuda.is_available() else "cpu")`. `scripts/evaluate_checkpoint.py`
   already exposes `--device`; the trainer does not.
3. **Precision as an enum, not a bool.** `src/training/config.py:116` defaults `amp: bool = True`,
   so any config omitting the key silently enables fp16. Replace with an explicit
   `fp32 | fp16 | bf16` selection defaulting to `fp32`.
4. **Per-machine profiles** for `batch_size`, `num_workers`, `pin_memory` and precision, so moving
   machines is a profile switch rather than an edit of the experiment config. The experiment config
   must stay identical across machines or the comparison is void.

### New hypothesis H-BF16 — bf16 may lift the amp restriction on Ampere

`configs/training_mango.yaml:42` disables amp with the comment "FP16 causes NaN in YOLOv8Loss". That
is almost certainly fp16 exponent overflow. bf16 carries the same exponent range as fp32 and would
not overflow the same way. The A5000 and the 3080 are Ampere and support bf16; the GTX 1660 SUPER is
Turing and does not.

If it holds, bf16 would give a further 1.5-2x on those machines on top of the raw throughput gain.

**This is a hypothesis with a test, not an assumption.** CONFIRM: five epochs under bf16 complete
with finite loss throughout and a final loss within one between-seed standard deviation of the fp32
run. REFUTE: any NaN or non-finite loss. It runs only on Ampere hardware and is not a prerequisite
for anything else.

## Maintainer Decisions — Round 3 (RESOLVED 2026-08-29, post-design)

### 7. Q2 REVISED — split at the natural seam. Two PRs, not one.

The single-PR `size:exception` was granted when the estimate was ~1,485 authored lines. The design
now estimates ~3,480 (band 3,200-4,200) — 2.3x the approved figure, a material change rather than
an adjustment. `delivery_strategy` returns to a chained split:

- **PR 1 — trainer correctness and hardware portability (~1,105 lines).** W9 (machine-profile split,
  gradient accumulation, device selection, precision enum) plus D-G (OOM and NaN guards that count
  and raise instead of silently skipping).
- **PR 2 — the architecture (~2,375 lines).** Early fusion, P2 reconnection, head deduplication.

This is not a re-split on size grounds. PR 1 is a **blocking prerequisite**: until the OOM and NaN
guards are fixed, no training result from PR 2 can be trusted, so it has to land first regardless.
The seam already existed; the split just makes it a review boundary.

Both PRs still exceed the 800-line budget. That remains covered by the granted exception; the split
is for reviewability and sequencing, not budget compliance.

### 8. Q7 — `effective_batch: 8`, fixed before V1 and immutable afterwards.

Chosen for continuity with the previously published runs, which had `batch_size: 8` configured, and
because it divides evenly across all three target machines: 4 accumulation steps on the GTX 1660
SUPER at physical batch 2, 2 on the RTX 3080 at 4, 1 on the A5000 at 8.

Every run artifact must record `effective_batch` alongside `experiment_sha256`. Two runs are
comparable only when both match.

### 9. Q8 — H-BF16 deferred with its bars registered.

No Ampere access in this window. The hypothesis gates nothing, so it waits. Its CONFIRM and REFUTE
bars stay fixed in advance so the result cannot be rationalised whenever it does run.

**Its prerequisite is now understood and is not optional.** `src/training/loop.py:364-366` skips
NaN and Inf batches through the same `continue` as the OOM path, before `n_batches` is incremented,
and `:404` floors the divisor at 1. Under bf16 that would drop exactly the batches the hypothesis is
about and report the epoch mean over the survivors — **converting a REFUTE into a CONFIRM**. The
guard makes the failure invisible in the reported metrics.

D-G is therefore a precondition for H-BF16 being testable at all, not merely a robustness fix.

### Verified: the memory alarm is resolved

The design's hand-built activation model predicted 1.93 GB per sample against the 2.10 measured on
the GTX 1660 SUPER — within 9%. Recalibrated, the target configuration (4 pyramid levels, with the
head's duplicate stem computation removed) projects **1.22 GB per sample**, so at physical batch 2
it lands at roughly 2.6 GB against 6.44 GB available. The earlier projection that the design would
not fit was correct for the current master and does not hold for the redesign.

Gradient accumulation also dissolved the design's Q5 tension: with `effective_batch` held constant,
physical batch size stops being an experimental variable and becomes a memory knob, so reducing it
is attribution-neutral. Combined with the measured saturation (2.32 vs 2.25 img/s between batch 1
and 2), batch 2 costs approximately zero wall clock.
