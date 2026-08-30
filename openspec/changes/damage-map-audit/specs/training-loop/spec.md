# Delta for training-loop

## ADDED Requirements

### Requirement: Per-Class Non-Maximum Suppression in Decode

`Trainer._decode_predictions` MUST apply non-maximum suppression (NMS) per class before predictions are returned for evaluation or visualization. Suppression MUST be class-independent: a high-confidence box of one class MUST NOT suppress a spatially overlapping box of a different class.

#### Scenario: Duplicate same-class boxes suppressed

- GIVEN multiple overlapping anchors predict the same class above the confidence threshold
- WHEN decode runs
- THEN NMS SHALL suppress duplicates per class using the configured IoU threshold
- AND only the highest-confidence box per overlapping cluster SHALL be kept

#### Scenario: Overlapping different-class boxes both survive

- GIVEN a damage box fully contained inside a mango box, both above threshold
- WHEN per-class NMS runs
- THEN the mango detection SHALL NOT suppress the damage detection, or vice versa

### Requirement: Configurable Confidence and NMS Thresholds

The confidence threshold and the NMS IoU threshold used by the decode path MUST be configurable via `TrainingConfig`, not hardcoded in `loop.py`.

#### Scenario: Defaults preserve prior confidence behavior

- GIVEN `TrainingConfig` without an explicit threshold override
- WHEN decode runs
- THEN the confidence threshold SHALL default to 0.25

#### Scenario: Confidence threshold override respected

- GIVEN `TrainingConfig` sets a confidence threshold of 0.4
- WHEN decode runs
- THEN only anchors with a qualifying per-class score >= 0.4 SHALL become candidates

#### Scenario: NMS IoU threshold override respected

- GIVEN `TrainingConfig` sets an NMS IoU threshold different from the default
- WHEN decode runs
- THEN NMS SHALL use the configured IoU threshold instead of a hardcoded value

### Requirement: Per-Class Candidate Emission

For each anchor, the decode path MUST emit one candidate detection per class whose score exceeds the confidence threshold, instead of only the argmax class.

#### Scenario: Anchor emits multiple class candidates

- GIVEN an anchor with mango score 0.6 and damage score 0.3, both above the confidence threshold
- WHEN decode runs
- THEN two candidate detections SHALL be emitted for that anchor location, one per qualifying class, each with its own class score

#### Scenario: Anchor emits a single candidate when only one class qualifies

- GIVEN an anchor where only the mango score exceeds the confidence threshold
- WHEN decode runs
- THEN exactly one candidate SHALL be emitted for that anchor

#### Scenario: Anchor emits no candidate when no class qualifies

- GIVEN an anchor where no per-class score exceeds the confidence threshold
- WHEN decode runs
- THEN zero candidates SHALL be emitted for that anchor

### Requirement: Decode Consistency Across Consumers

Any code path that decodes raw model outputs into detections for reporting (including `scripts/visualize_damage_predictions.py`) MUST reuse the same decode semantics (thresholds, per-class candidate emission, per-class NMS) as `Trainer._decode_predictions`, rather than a duplicated implementation.

#### Scenario: Visualizer matches trainer decode output

- GIVEN identical raw model outputs and identical `TrainingConfig` thresholds
- WHEN `Trainer._decode_predictions` and the visualization script both decode them
- THEN both SHALL produce the same set of detections (same suppression and thresholding behavior)
