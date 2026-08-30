# Delta for training-metrics

## MODIFIED Requirements

### Requirement: mAP@0.5 Computation

The module MUST compute mean Average Precision at IoU threshold 0.5 over detections produced by an NMS-suppressed, per-class decode. For each class, it SHALL compute AP from the precision-recall curve using all-point interpolation, then average across classes. AP figures computed from pre-NMS decode output MUST be labeled superseded and MUST NOT be used as the comparison baseline for evaluating a model change.
(Previously: mAP@0.5 computation was agnostic to whether input predictions had been NMS-suppressed.)

#### Scenario: Standard mAP@0.5 on val split

- GIVEN 4 validation images with NMS-suppressed predictions and ground truth
- WHEN mAP@0.5 is computed
- THEN for each class (mango, damage), AP SHALL be calculated from the PR curve at IoU=0.5
- AND the final metric SHALL be the mean of per-class APs

#### Scenario: Class with zero detections

- GIVEN the mango class has GT boxes but 0 true positives in predictions
- WHEN mAP@0.5 is computed
- THEN mango AP SHALL be 0.0
- AND the mean SHALL include this 0.0 (not exclude the class)

#### Scenario: Pre-NMS baseline invalidated

- GIVEN an existing checkpoint's saved predictions were generated before per-class NMS existed
- WHEN mAP@0.5 is computed on those predictions
- THEN the resulting figure SHALL be reported as superseded
- AND MUST NOT be used as the comparison baseline for a subsequent model change

#### Scenario: NMS-corrected baseline established

- GIVEN an existing checkpoint re-evaluated with per-class NMS applied and no retraining
- WHEN mAP@0.5 is computed
- THEN the result SHALL become the new reference baseline for future comparisons

## ADDED Requirements

### Requirement: GT-Count Accounting Integrity

For each class, the sum of true positives and false negatives computed on a split MUST equal the number of ground-truth instances of that class in the split, unless a documented exclusion reason is recorded alongside the metric output.

#### Scenario: TP+FN reconciles with GT instance count

- GIVEN a validation split with 58 damage ground-truth instances
- WHEN per-class TP/FN counts are computed for damage
- THEN `tp_damage + fn_damage` SHALL equal 58, or any discrepancy SHALL be logged with an explicit reason
