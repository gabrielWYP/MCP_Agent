# Delta for training-metrics

## Overview

Extends training-metrics to auto-generate training_curves.png from LossHistory after training completes. Uses matplotlib to plot 3 subplots (cls_loss, box_loss, total_loss) over epochs.

## ADDED Requirements

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

## MODIFIED Requirements

### Requirement: Loss Tracking

The module MUST record per-epoch loss components: cls_loss, reg_loss, total_loss. It SHALL expose these as a history dictionary for logging and visualization. The history API SHALL remain unchanged; training_curves.png reads from this existing API.
(Previously: Loss tracking was logging-only; now it also feeds PNG generation)

#### Scenario: Loss history recorded each epoch

- GIVEN epoch 5 completes with cls_loss=0.42, reg_loss=1.23
- WHEN loss tracking updates
- THEN history["cls_loss"][5] = 0.42 and history["reg_loss"][5] = 1.23
- AND history["total_loss"][5] SHALL equal the weighted sum

#### Scenario: Loss history accessible for plotting and PNG

- GIVEN 10 epochs of training have completed
- WHEN loss curves are requested (via TensorBoard or PNG generation)
- THEN the module SHALL return lists of length 10 for cls_loss, reg_loss, and total_loss

## Dependencies

- training-loop: Triggers PNG generation at end of training; provides LossHistory data

## Non-functional

- PNG generation SHOULD complete in <1 second for 100 epochs
- PNG file size MUST be <500KB
- MUST NOT block training pipeline if matplotlib is unavailable