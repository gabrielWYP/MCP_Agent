# Delta for kd-training

## MODIFIED Requirements

### Requirement: KD Forward Pass

During each training step, KDTrainer MUST run teacher forward with `torch.no_grad()` using
the stacked 4-channel (RGB+NIR) input, reading the teacher's `distill_backbone` feature key
(renamed from `distill_backbone_rgb`; channels `[384, 768]` unchanged), project teacher
features to student channel dimensions via the 4 projection groups, run student forward
with RGB only, and compute MSE loss per distillation level between projected teacher
features and student features.
(Previously: teacher forward took separate `rgb`/`nir` arguments and read the
`distill_backbone_rgb` key from an 8-key output dict)

#### Scenario: KD forward computes per-level MSE
- GIVEN a batch of `(rgb, nir, bboxes, class_labels)` and both teacher and student models
- WHEN KDTrainer executes the training step
- THEN teacher forward SHALL run under `torch.no_grad()` on the stacked 4-channel input,
  reading `distill_backbone` from the teacher's 7-key output dict
- AND each projection group SHALL map teacher channels to student channels
- AND MSE SHALL be computed per level between projected teacher and student features
- AND no gradient SHALL flow back through teacher parameters

#### Scenario: Channel dimension assertion at forward time
- GIVEN a projection group configured for teacher channels `[384, 768]` to student channels
  `[128, 256]`
- WHEN teacher outputs a feature level with an unexpected channel count
- THEN an assertion error SHALL be raised identifying the mismatched dimension

## ADDED Requirements

### Requirement: RGB-Only Student Ceiling Is a Spec-Level Constraint

Any KD evaluation MUST compare the distilled student against a plain RGB-only student
baseline trained without distillation, never against the multimodal teacher's own
performance. The measured RGB-only classification ceiling (0.7774 AUC on damage crops,
against 0.9171 AUC for RGB+NIR) MUST be recorded as the expected upper-bound context for KD
results: a KD run that fails to close the teacher-student gap is an expected outcome under
this ceiling, not evidence of a training defect. No KD training run is part of this change;
this requirement fixes the bar the future run must be measured against.

#### Scenario: KD results reported against the RGB-only baseline, not the teacher
- GIVEN a completed KD run
- WHEN results are published
- THEN the comparison baseline is the plain RGB-only student (same architecture, no
  distillation), not the multimodal teacher's AP50 or AUC

#### Scenario: KD pass bar requires margin over the RGB-only baseline
- GIVEN KD training has completed and between-seed std `sigma_s` is measured the same way
  as the baseline's `sigma_d`
- WHEN evaluating whether the distilled student is worth adopting
- THEN CONFIRM requires distilled student damage AP50 >= plain RGB-only student damage AP50
  + 2 * `sigma_s`; failing to meet this bar is a valid, publishable negative result
