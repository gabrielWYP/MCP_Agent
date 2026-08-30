# yolo-loss Specification

## Purpose

YOLOv8-style loss module for MasterModel. Implements Task-Aligned Assigner (TAL) for dynamic prediction-ground-truth matching, class-weighted BCE for classification, and CIoU regression loss. Operates on the model's multi-level output format.

Class counts were re-measured during the damage-map-audit change: mango 208,
damage 335 (1:1.61 ratio) — not the previously assumed 24/129 (1:5.9). See the
Classification Loss requirement below for why `class_weights` is nonetheless
held at `[0.5, 1.5]` rather than the inverse-frequency value for the duration
of that change's experiment ladder.

## Requirements

### Requirement: Task-Aligned Assignment

The loss module MUST implement a Task-Aligned Assigner that dynamically matches predictions to ground truth. For each GT box, alignment metric SHALL be `sqrt(cls_score × iou_score)`. The top-k predictions per GT SHALL be assigned as positive samples; all others are negative.

#### Scenario: Standard TAL matching

- GIVEN 3 GT boxes and 8400 total predictions across P3/P4/P5
- WHEN the assigner runs with top_k=10
- THEN each GT SHALL be matched to up to 10 predictions with highest alignment scores
- AND unmatched predictions SHALL be assigned as negative samples (background)

#### Scenario: Image with no GT objects

- GIVEN a batch image with 0 GT boxes
- WHEN TAL assigner processes it
- THEN all 8400 predictions SHALL be negative samples
- AND classification loss SHALL penalize all cells as background

#### Scenario: Multiple GT boxes near same cell

- GIVEN two GT boxes with high overlap at the same grid cell
- WHEN the assigner resolves matches
- THEN each GT SHALL independently select its top-k positives (no mutual exclusion at assignment stage)

### Requirement: Classification Loss

The loss module MUST compute BCEWithLogitsLoss on class predictions for all assigned cells. Class weights MUST be sourced from one authoritative value, `[0.5, 1.5]` (mango=class 0, damage=class 1), matching `configs/*.yaml` and the `TrainingConfig` dataclass default. Damage (class 1) SHALL carry the higher weight.

(`[0.5, 1.5]` is held constant for the duration of the damage-map-audit experiment
ladder as an experimental-design requirement — every existing checkpoint was
trained under it, and changing it mid-experiment would introduce a second
variable and destroy attribution of any class-1 AP change to the assigner fix.
It is deliberately NOT the inverse-frequency value: measured over the current
dataset, mango has 208 instances and damage has 335 — a 1:1.61 ratio, not the
previously assumed 1:5.9 — so inverse-frequency weighting on current data
would be `[1.305, 0.810]` (≈ the `[1.27, 0.83]` recorded in
`openspec/changes/pipeline-orchestration-docs/explore.md`). Revisiting
`class_weights` on inverse-frequency grounds is a valid follow-up, but only
AFTER the current validation completes — never during it. See
`openspec/changes/damage-map-audit/proposal.md` Q2 for the full rationale.)

#### Scenario: Authoritative class weights applied

- GIVEN `TrainingConfig` constructed without an explicit `class_weights` override
- WHEN the loss module reads `class_weights`
- THEN it SHALL default to `[0.5, 1.5]`
- AND damage (class 1) weight SHALL exceed mango (class 0) weight

#### Scenario: Negative sample classification

- GIVEN predictions assigned as negative (background)
- WHEN classification loss is computed
- THEN those cells SHALL contribute BCE loss with target class logits = 0 for all classes

### Requirement: Regression Loss

The loss module MUST compute CIoU (Complete IoU) loss on bounding box predictions for positive samples only, using `torchvision.ops.complete_box_iou_loss`.

#### Scenario: CIoU loss on positive predictions

- GIVEN 15 positive predictions assigned across 3 GT boxes
- WHEN regression loss is computed
- THEN CIoU loss SHALL be calculated between predicted and GT bboxes for those 15 positives
- AND negative samples SHALL contribute zero regression loss

#### Scenario: Multi-level prediction aggregation

- GIVEN model outputs: P3 (B,6,80,80), P4 (B,6,40,40), P5 (B,6,20,20)
- WHEN loss is computed
- THEN predictions from ALL levels SHALL be flattened and pooled
- AND total regression loss SHALL be the sum across all levels

### Requirement: Loss Aggregation

Total loss MUST be computed as: `total = box_weight × reg_loss + cls_weight × cls_loss`. Default weights: `box_weight=7.5`, `cls_weight=0.5`. Loss SHALL be averaged over the batch dimension.

#### Scenario: Standard loss computation

- GIVEN a batch of 4 images with varying numbers of GT boxes
- WHEN total loss is computed
- THEN total SHALL equal `7.5 * mean_ciou + 0.5 * mean_bce`
- AND the loss SHALL be a scalar ready for `loss.backward()`
