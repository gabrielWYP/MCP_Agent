# Experiment ladder — inference rungs (E1, E2, E3)

Subject: `checkpoints/student/best_model.pt` (epoch 13, best_map50 0.235), reconciled splits
(148/18/21), local GTX 1660 SUPER. No training was run. All bars were pre-registered before
execution.

## E2 — assigner starvation (H1): REFUTED

Pre-registered: CONFIRM if median positive anchors per damage GT < 1.0 and per mango GT >= 5.0.
**REFUTE if median per damage GT >= 3.0.**

Measured per INDIVIDUAL ground-truth box over the 148-image train split:

| class | n | median | mean | p25 | min | max | GTs with 0 anchors |
|---|---|---|---|---|---|---|---|
| 0 mango | 148 | 13.0 | 12.9 | 13.0 | 6 | 13 | 0 |
| 1 damage | 253 | **13.0** | 10.5 | 9.0 | 1 | 13 | **0** |

Median 13.0 against a refute bar of 3.0 — refuted by a factor of 13. Not one damage GT receives
zero positive anchors; only 18 of 253 (7%) receive fewer than three. Damage boxes get the full
`topk=13`, the same as mango.

Note on statistic: the shipped instrumentation reports per-(class, level) totals and a mean. The
pre-registered bar was a median, so it was computed separately per individual GT rather than read
off the mean. The distinction mattered — a skewed distribution could have produced a high mean over
a low median. It did not.

## E1 — missing NMS (H2): REFUTED for damage, CONFIRMED for mango

Pre-registered: CONFIRM if `precision_c1` rises >= 5x and `recall_c1` falls <= 0.05.
**REFUTE if `precision_c1` rises < 1.5x.**

| | AP50 | precision | recall | tp | fp | fn |
|---|---|---|---|---|---|---|
| mango, no NMS | 0.2396 | 0.0762 | 0.8889 | 16 | 194 | 2 |
| mango, NMS | 0.5304 | 0.3684 | 0.7778 | 14 | 24 | 4 |
| damage, no NMS | 0.0047 | 0.0312 | 0.0968 | 3 | 93 | 28 |
| damage, NMS | 0.0047 | 0.0312 | 0.0968 | 3 | 93 | 28 |

`precision_c1` ratio: 1.00x. Refuted.

NMS changed the damage numbers by exactly nothing — the figures are identical, not merely close.
It transformed mango: precision 4.83x, AP50 from 0.24 to 0.53.

The asymmetry is itself diagnostic. Mango is one large object that many anchors fire on, producing
genuine duplicates for NMS to collapse. Damage produces 96 detections that do not overlap each
other, because they are scattered noise rather than clustered duplicates of a located object. NMS
has nothing to suppress.

NMS remains a correct and necessary fix — it more than doubles mango AP and the previous figures
were untrustworthy — but it does not explain the damage gap.

## E3 — argmax decode suppression (H3): REFUTED

Pre-registered: measure the ceiling first (task 4.7a), then CONFIRM if `recall_c1` rises >= 50% of
it, REFUTE if <= 10%.

| | AP50 c1 | precision c1 | recall c1 | tp | fp |
|---|---|---|---|---|---|
| argmax decode | 0.0047 | 0.0312 | 0.0968 | 3 | 93 |
| per-class decode | 0.0043 | 0.0275 | 0.0968 | 3 | 106 |

Recall moved by exactly 0.0. Per-class decode emitted 13 additional damage candidates, all false
positives, and AP50 fell slightly. Refuted without needing the ceiling measurement — the observed
change is zero regardless of what the ceiling would have been.

The 94% box-nesting statistic was real, but argmax was not erasing recoverable damage detections.

## What the numbers actually say

`recall_c1 = 0.0968` — the model finds 3 of 31 damage instances. That figure is unchanged by
assignment, by decoding, by suppression, and by label reconciliation. Four candidate explanations
have been eliminated:

- H1 assigner starvation — refuted, damage gets 13 positives per GT
- H2 missing NMS — refuted for damage, zero effect
- H3 argmax suppression — refuted, zero recall change
- H6 under-annotation — refuted against the versioned human export, zero damage lost

The failure is in the model's ability to detect damage, not in the machinery around it.

## Caveat that constrains these figures

The subject checkpoint is epoch 13 with `best_map50 = 0.235`. The published runs reached 0.335 to
0.394 over 46 to 50 epochs. This checkpoint is undertrained, so the absolute values are pessimistic.
The relative comparisons (NMS on/off, per-class on/off, anchors per GT) are unaffected, because each
compares the same checkpoint against itself.

## Consequence for the remaining rungs

E4 was gated on E2 CONFIRM. E2 refuted, so **E4 must not run** — the pre-registration is binding.
The assigner changes still ship behind `assigner_center_radius=0.0`; the fallback-ordering defect
and the coordinate-source mismatch are real bugs worth fixing, they are simply not the cause.

H4 (KD transfer) and H5 (frozen backbone) remain untested, but neither can explain this result: the
subject here is the plain student, trained without distillation and fully unfrozen.

The most informative remaining experiment is one the ladder does not currently contain: a clean
convergence retrain of the student on the reconciled split, with NMS, to establish what the model
can actually achieve on damage when nothing is contaminated. That replaces E4 as the next training
rung and tests capacity rather than assignment.
