# training-metrics Specification

## Purpose

Metrics computation module for detection training evaluation. Computes mAP@0.5, mAP@0.5:0.95, per-class AP, tracks loss components across epochs, and generates training_curves.png from LossHistory data after training completes. Feeds results into the training loop for early stopping decisions and TensorBoard logging.

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

The module MUST record per-epoch loss components: cls_loss, reg_loss, total_loss. It SHALL expose these as a history dictionary for logging and visualization. The history API SHALL remain unchanged; training_curves.png reads from this existing API.
(Previously: Loss tracking was logging-only; now it also feeds PNG generation)

#### Scenario: Loss history recorded each epoch

- GIVEN epoch 5 completes with cls_loss=0.42, reg_loss=1.23
- WHEN loss tracking updates
- THEN `history["cls_loss"][5] = 0.42` and `history["reg_loss"][5] = 1.23`
- AND `history["total_loss"][5]` SHALL equal the weighted sum

#### Scenario: Loss history accessible for plotting and PNG

- GIVEN 10 epochs of training have completed
- WHEN loss curves are requested (via TensorBoard or PNG generation)
- THEN the module SHALL return lists of length 10 for cls_loss, reg_loss, and total_loss

### R1: Training Curves PNG Generation

After training completes (normal completion or early stop), the metrics module MUST generate a training_curves.png file from LossHistory data. The PNG SHALL contain 3 subplots arranged vertically: cls_loss, box_loss, and total_loss, each plotted over epoch indices.

#### Scenario: Training curves generated after successful training

- GIVEN training completes normally (all epochs or early stop)
- WHEN generate_training_curves is called with LossHistory and output_dir
- THEN training_curves.png SHALL be written to output_dir
- AND the PNG SHALL contain 3 subplots: cls_loss, box_loss, total_loss vs epoch

#### Scenario: Training curves with partial epochs (early stop)

- GIVEN training early-stopped at epoch 23 out of max_epochs=50
- WHEN generate_training_curves is called
- THEN the PNG SHALL plot 23 data points (one per completed epoch)
- AND axes SHALL be labeled "Epoch" (x) and "Loss" (y)

#### Scenario: Empty loss history

- GIVEN LossHistory with zero recorded epochs
- WHEN generate_training_curves is called
- THEN the method SHALL log a warning "No loss data to plot"
- AND SHALL NOT write a PNG file

### R2: Headless Server Graceful Degradation

On environments without a display (headless servers, CI), training_curves generation MUST NOT crash the training loop. The module SHALL log a warning and continue if matplotlib cannot render.

#### Scenario: Headless server without display

- GIVEN training runs on a headless server with DISPLAY unset
- WHEN generate_training_curves is called
- AND matplotlib raises RuntimeError or ValueError related to display
- THEN a warning SHALL be logged: "Could not generate training_curves.png: {error}"
- AND training SHALL continue without failure

#### Scenario: matplotlib import failure

- GIVEN matplotlib is not installed or fails to import
- WHEN generate_training_curves is called
- THEN the module SHALL log a warning indicating matplotlib is unavailable
- AND training SHALL continue without failure
