# fusion-redesign — validation report

Discharges `openspec/changes/fusion-redesign/tasks.md` Phase 13 (13.1–13.10).
Prior baseline: [`reports/damage-map-audit/`](../damage-map-audit/) — cited here as the
baseline this change was measured against, not as a superseded target.

Date: 2026-09-06. Every number below is reproducible from the artefact named beside it.

---

## Summary

The redesign closed both defects it was built to close. Damage detection still
collapsed. On a clean, leakage-free evaluation the early-fusion master detects
**zero** damage instances on both val and test, while the two-stream architecture
it replaced detects damage on both.

The aggregate `mAP@0.5` went **up** (0.4567 → 0.5019 on val) while the class this
project exists to detect went to zero. That divergence is the central finding.

---

## 13.5 H-E — head_strides `[8,16,32]` control

**Status: RUN.** Seed 42 only; seeds 1337 and 2024 not run.

Config: `--override head_strides=[8,16,32] assigner_level_ranges=[64,128] seed=42`,
`checkpoints/fusion_redesign/he_seed42/`, 100/100 epochs, 34,095,218 params.

| metric | V1 (4 levels) | H-E (3 levels) |
|---|---|---|
| damage AP50, best epoch | 0.0036 | **0.0000** |
| epochs with damage TP > 0 | 14 / 62 | **0 / 100** |
| damage TP, maximum | 15 | **0** |
| damage FP, median per epoch | 699 | 1,104 |

Evaluated against clean labels, val (18 images, 27 damage instances):

| | damage AP50 | recall | precision | tp | fp |
|---|---|---|---|---|---|
| V1, 4 levels | 0.0038 | 0.5185 | 0.0066 | 14 | 2,117 |
| H-E, 3 levels | 0.0000 | 0.0000 | 0.0000 | 0 | 1,133 |

**Verdict against the pre-registered bar: NOT COMPUTABLE.** The bar was
`CONFIRM ≥ 2·σ_d` / `REFUTE ≤ 1·σ_d`, and `σ_d` requires the three V1 seeds of task
12.1. Seed 1337 stopped at epoch 57 and seed 2024 never started, so no `σ_d` exists.
No CONFIRM or REFUTE may be claimed.

**Task 13.5 instructed that a REFUTE retires P2 and makes `[8,16,32]` the recommended
default. That instruction must NOT be executed, and the reason is on the record here.**
The pre-registration anticipated that P2 was neutral or harmful. The measurement shows
the opposite: removing P2 took damage recall from 0.5185 to exactly 0.0000, and left
the model with zero true positives across 100 epochs. Anchor coverage measured over the
real 148-image train split explains it — under `[32,64,128]` 43 damage GT are assigned
to P2 (stride 4); moving them to P3 (stride 8) costs three quarters of their candidate
anchors, and the share of damage GT with fewer than `topk=13` anchors rises from 32% to
41%. **P2 is retained. `head_strides: [4,8,16,32]` remains the default.**

**H-E is also confounded as a test of background swamping.** It changes two things at
once: total anchors fall 34,000 → 8,400 (the intended manipulation) and anchor
starvation worsens 32% → 41% (an unintended side effect). A null result is therefore
uninterpretable. The one directional signal available argues *against* the background
hypothesis: cutting background supervision fourfold made damage false positives go **up**
(median 699 → 1,104), the opposite of what that hypothesis predicts.

## 13.4 H-D — RGB-only control (`in_channels=3`)

**Status: NOT RUN.** No result, no verdict. Recorded as an outstanding obligation.

Note for whoever runs it: the 4-channel early-fusion design removed the ablation handle
that `reports/damage-map-audit/h7-fusion-ignores-nir.md` used, so H-D is the only
remaining way to measure whether NIR contributes anything in this architecture.

## 13.1 V0 — measured peaks and fallback step

**Status: NOT RUN.** No V0 memory-projection measurements were taken. No fallback was
recorded because no measurement was made, which is not the same as a projection holding.

## 13.2 H-A — initialisation, per-stage mean/std

**Status: NOT RUN.** Per-stage init statistics were never collected.

Adjacent finding, recorded because it was observed while reading the code and is not
covered by any other task: `reg_pred.bias` is initialised to zero
(`src/models/master/head.py::_init_weights` zeroes every `Conv2d` bias, then sets only
`cls_pred.bias` to −4.595). At initialisation every level therefore predicts a 1×1 px
box against level median targets of 33 / 64 / 169 px. The comment calls this
"consistent with YOLOv8", but YOLOv8 regresses stride-normalised DFL, not raw `exp` in
absolute pixels — the init was copied from a scheme this code does not implement.

## 13.3 H-B / H-C — count and static assertion

**Status: NOT RUN.**

## 13.6 H-F — run validity, per-module gradient norms

**Status: PARTIALLY DISCHARGED.**

`checkpoints/fusion_redesign/v1_seed42/metrics_history.csv` carries
`grad_norm_backbone_stem`, `grad_norm_backbone_stage0..3`, `grad_norm_neck` and
`grad_norm_head_level0..3`. All are non-zero throughout; at epoch 62, stem 2.39,
stage0 4.25, stage3 32.9, neck 10.0, head_level3 117.5. `oom_skipped = 0` and
`nan_skipped = 0` for both V1 seed 42 and H-E.

**Both defects the redesign targeted are therefore closed**: the backbone trains from
epoch 0 (no bit-exact `max|γ−1| == 0` anywhere), and the stride-4 level is connected and
receiving gradient. The redesign did what it set out to do. Damage detection collapsed
anyway.

Unread signal, recorded for a future analyst: `grad_norm_head_level3` (P5, the mango
level) runs roughly 30–90× larger than `grad_norm_head_level2`. The loss is dominated
by the mango level.

## 13.7 H-BF16

**Status: DEFERRED, no run performed.** No Ampere hardware was available; all runs used
fp32 on a GTX 1660 SUPER (Turing, no bf16). Bars were fixed in advance and remain unused.

## 13.8 Q1 limitation — verbatim

> This change can prove the redesign is **better**, not that it is **sufficient**.

Stated here as required. In this instance the limitation binds in the opposite
direction from the one anticipated: the redesign is not better on the damage class, and
no reading of these results should present a relative outcome as fitness for use. The
best damage AP50 measured anywhere in this project is **0.1282** on an 18-image
validation split — not a deployable detector.

## 13.9 Comparability rule

> Two runs are comparable only if `experiment_sha256` **and** `effective_batch` match.

Resolved `effective_batch: 8` (Q7) held across every run reported here.

**Defect found in the rule's implementation.** `experiment_sha256` hashes the raw bytes
of the experiment YAML, so `--override` values do not enter it. V1 and H-E both report
`experiment_sha256 = 94a3f29236096afa8d56487c0f8f5bad3015f8e6c72b9dfd3fe380673fa98225`
while measuring different pyramids — the hash asserts comparability between two runs
that are not comparable. Any comparability table built on this hash alone is unsound.
Fix before publishing a ladder table: fold the resolved override set into the hash, or
record the overrides alongside it.

## 13.10 Collection

This document. Cross-links `reports/damage-map-audit/` as the prior baseline.

---

## Outstanding obligations

| Task | State |
|---|---|
| 12.1 V1 at seeds {42, 1337, 2024} | seed 42 complete; 1337 stopped at epoch 57; 2024 never started |
| 12.3 H-D RGB-only control | not run |
| 12.4 H-E at 3 seeds | seed 42 only |
| 12.5 H-F full read | partially discharged above |
| 13.1–13.3 | not run |

No `σ_d` exists. Until the three seeds are complete, every damage comparison in this
change is a single-seed observation, and the measured noise floor is wide enough to
cover most of the differences reported: the same two-stream checkpoint scores damage
AP50 **0.1282 on val and 0.0417 on test**, a threefold spread on 18- and 20-image
splits.
