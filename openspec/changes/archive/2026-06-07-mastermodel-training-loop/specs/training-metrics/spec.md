# training-metrics Specification

## Purpose

Metrics computation module for detection training evaluation. Computes mAP@0.5, mAP@0.5:0.95, per-class AP, and tracks loss components across epochs. Feeds results into the training loop for early stopping decisions and TensorBoard logging.

## Requirements

### Requirement: mAP@0.5 Computation

The module MUST compute mean Average Precision at IoU threshold 0.5. For each class, it SHALL compute AP from the precision-recall curve using all-point interpolation, then average across classes.

#### Scenario: Standard mAP@0.5 on val split

- GIVEN 4 validation images with predictions and ground truth
- WHEN mAP@0.5 is computed
- THEN for each class (sano, danado), AP SHALL be calculated from the PR curve at IoU=0.5
- AND the final metric SHALL be the mean of per-class APs

#### Scenario: Class with zero detections

- GIVEN the sano class has 4 GT boxes but 0 true positives in predictions
- WHEN mAP@0.5 is computed
- THEN sano AP SHALL be 0.0
- AND the mean SHALL include this 0.0 (not exclude the class)

### Requirement: mAP@0.5:0.95 Computation

The module MUST compute COCO-style mAP averaged across 10 IoU thresholds: 0.50, 0.55, 0.60, ..., 0.95. For each threshold, AP is computed per class, then mAP is averaged across thresholds and classes.

#### Scenario: COCO-style mAP aggregation

- GIVEN predictions and ground truth for the val split
- WHEN mAP@0.5:0.95 is computed
- THEN AP SHALL be computed at each of 10 IoU thresholds for each class
- AND the final metric SHALL be the mean of all (threshold × class) AP values

#### Scenario: Trade-off between strict and loose thresholds

- GIVEN model predictions with moderate localization accuracy
- WHEN mAP is computed at IoU=0.50 vs IoU=0.95
- THEN mAP@0.5 SHALL be >= mAP@0.95 (per-class, monotonically non-increasing with threshold)

### Requirement: Per-Class AP

The module MUST report AP for each individual class (sano=0, danado=1) at both IoU thresholds (0.5 and 0.5:0.95). This enables monitoring class imbalance effects during training.

#### Scenario: Per-class AP output

- GIVEN validation completes for an epoch
- WHEN metrics are computed
- THEN the output SHALL include: `{sano_ap_50, danado_ap_50, sano_ap_50_95, danado_ap_50_95}`

#### Scenario: Severe class imbalance reflected in per-class AP

- GIVEN sano has 4 GT boxes and danado has 34 in val split
- WHEN per-class AP is computed
- THEN sano AP SHALL reflect its fewer examples (higher variance)
- AND danado AP SHALL reflect its larger sample (potentially more stable)

### Requirement: Loss Tracking

The module MUST record per-epoch loss components: cls_loss, reg_loss, total_loss. It SHALL expose these as a history dictionary for logging and visualization.

#### Scenario: Loss history recorded each epoch

- GIVEN epoch 5 completes with cls_loss=0.42, reg_loss=1.23
- WHEN loss tracking updates
- THEN `history["cls_loss"][5] = 0.42` and `history["reg_loss"][5] = 1.23`
- AND `history["total_loss"][5]` SHALL equal the weighted sum

#### Scenario: Loss history accessible for plotting

- GIVEN 10 epochs of training have completed
- WHEN loss curves are requested
- THEN the module SHALL return lists of length 10 for cls_loss, reg_loss, and total_loss