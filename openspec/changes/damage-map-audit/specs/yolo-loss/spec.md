# Delta for yolo-loss

## MODIFIED Requirements

### Requirement: Task-Aligned Assignment

The loss module MUST implement a Task-Aligned Assigner that dynamically matches predictions to ground truth. For each GT box, alignment metric SHALL be `sqrt(cls_score × iou_score)`. The top-k predictions per GT SHALL be assigned as positive samples; all others are negative. Spatial admissibility of a candidate anchor MUST use a center-sampling tolerance (a configurable radius, in stride units, around the GT center) instead of strict anchor-center-inside-GT containment, so that a GT box narrower than an FPN level's stride can still admit anchor centers at that level. The "keep at least 1 anchor per GT" fallback MUST be evaluated AFTER the spatial admissibility filter runs, so the filter cannot silently erase the fallback's anchors; no GT SHALL be dropped without at least one positive anchor. Candidate-level selection per GT SHOULD account for GT size relative to each level's stride, so that top-k budget at a stride a GT cannot be reliably centered on does not starve finer-stride levels of positive anchors for that GT.
(Previously: spatial admissibility was strict anchor-center-inside-GT containment applied AFTER the "keep at least 1" fallback, which let the containment filter erase the fallback's anchors and drop GTs whose short side is smaller than the level's stride.)

#### Scenario: Standard TAL matching

- GIVEN 3 GT boxes and 8400 total predictions across P3/P4/P5
- WHEN the assigner runs with top_k=10
- THEN each GT SHALL be matched to up to 10 predictions with highest alignment scores
- AND unmatched predictions SHALL be assigned as negative samples (background)

#### Scenario: Image with no GT objects

- GIVEN a batch image with 0 GT boxes
- WHEN TAL assigner processes it
- THEN all 8400 predictions SHALL be negative samples

#### Scenario: Multiple GT boxes near same cell

- GIVEN two GT boxes with high overlap at the same grid cell
- WHEN the assigner resolves matches
- THEN each GT SHALL independently select its top-k positives (no mutual exclusion at assignment stage)

#### Scenario: Sub-stride GT still receives a positive anchor

- GIVEN a synthetic GT box 4px wide at 640px input, narrower than any anchor stride (8/16/32)
- WHEN the assigner runs center-sampling admissibility and the fallback
- THEN the GT SHALL be assigned at least one positive anchor
- AND the assignment SHALL NOT be dropped by the spatial filter

#### Scenario: Fallback guarantee holds after spatial filtering

- GIVEN a GT whose only top-k candidate anchors fail center-sampling admissibility
- WHEN the fallback ("keep at least 1 anchor") and the spatial filter both run
- THEN the fallback SHALL execute after the spatial filter so its anchors are never subsequently discarded
- AND the GT SHALL end with `len(pos_idx) >= 1`

### Requirement: Classification Loss

The loss module MUST compute BCEWithLogitsLoss on class predictions for all assigned cells. Class weights MUST be sourced from one authoritative value, `[0.5, 1.5]` (mango=class 0, damage=class 1), matching `configs/*.yaml` and the `TrainingConfig` dataclass default. Damage (class 1) SHALL carry the higher weight.
(Previously: the requirement stated weight sano≈2.7 > danado≈0.5, matching an inverted dataclass default that diverged from the value every recorded run actually used.)

#### Scenario: Authoritative class weights applied

- GIVEN `TrainingConfig` constructed without an explicit `class_weights` override
- WHEN the loss module reads `class_weights`
- THEN it SHALL default to `[0.5, 1.5]`
- AND damage (class 1) weight SHALL exceed mango (class 0) weight

#### Scenario: Negative sample classification

- GIVEN predictions assigned as negative (background)
- WHEN classification loss is computed
- THEN those cells SHALL contribute BCE loss with target class logits = 0 for all classes

## ADDED Requirements

### Requirement: Per-Class, Per-Level Positive-Anchor Instrumentation

The assigner MUST support optional, non-destructive instrumentation that counts positive anchors partitioned by ground-truth class and by FPN level. It MUST default to off, MUST NOT alter target classes, target boxes, target scores, or gradients when enabled, and MUST emit its output to the run directory.

#### Scenario: Instrumentation disabled by default

- GIVEN default `TrainingConfig`
- WHEN the assigner runs
- THEN no instrumentation output SHALL be produced
- AND assignment results SHALL be unchanged from the non-instrumented path

#### Scenario: Instrumentation records per-class, per-level counts

- GIVEN instrumentation enabled and a batch containing both mango and damage GTs
- WHEN assignment completes
- THEN the run directory SHALL contain positive-anchor counts partitioned by class and by level (P3/P4/P5)
- AND `target_classes`, `target_bboxes`, `target_scores`, and `fg_mask` SHALL be identical to a run with instrumentation disabled

#### Scenario: Instrumentation runs on an existing checkpoint with no retraining

- GIVEN an existing checkpoint and a forward-only pass over the train split with instrumentation enabled
- WHEN the pass completes
- THEN per-class, per-level positive-anchor counts SHALL be available without any optimizer step or weight update
