# yolo-loss Specification

## Purpose

YOLOv8-style loss module for MasterModel. Implements Task-Aligned Assigner (TAL) for dynamic prediction-ground-truth matching, class-weighted BCE for classification (addressing 1:5.9 class imbalance), and CIoU regression loss. Operates on the model's multi-level output format.

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

The loss module MUST compute BCEWithLogitsLoss on class predictions for all assigned cells. Class weights SHALL be applied to address the 1:5.9 sano-to-danado imbalance. The weight for class sano (0) SHALL be higher than danado (1) to compensate for underrepresentation.

#### Scenario: Weighted BCE with class imbalance

- GIVEN ground truth with 24 sano and 129 danado boxes across the dataset
- WHEN classification loss is computed
- THEN sano (class 0) SHALL have weight ≈ 2.7 (inverse frequency ratio, normalized)
- AND danado (class 1) SHALL have weight ≈ 0.5
- AND BCE loss SHALL be weighted by these per-class factors

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