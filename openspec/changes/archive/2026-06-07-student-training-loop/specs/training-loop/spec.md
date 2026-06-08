# Delta for training-loop

## Overview

Extends the training loop to support both MasterModel (2-phase, dual-input) and StudentModel (single-phase, RGB-only) via model_type dispatch. Student training uses cosine+warmup LR with the full model trainable from epoch 1.

## ADDED Requirements

### R1: Model-Type Dispatch

The Trainer MUST accept a `model_type` field in TrainingConfig with values `"master"` or `"student"`. All training behavior SHALL branch on this field. The default MUST be `"master"` for backward compatibility.

#### Scenario: Student model type configured

- GIVEN TrainingConfig with model_type="student"
- WHEN Trainer initializes
- THEN Trainer SHALL use single-phase training with cosine+warmup LR
- AND forward calls SHALL pass only `model(rgb)` (no NIR)

#### Scenario: Master model type configured (default)

- GIVEN TrainingConfig with model_type="master" or model_type unspecified
- WHEN Trainer initializes
- THEN Trainer SHALL use two-phase training (existing behavior)
- AND forward calls SHALL pass `model(rgb, nir)`

#### Scenario: Invalid model_type rejected

- GIVEN TrainingConfig with model_type="ensemble"
- WHEN Trainer initializes
- THEN Trainer SHALL raise a ValueError with message indicating valid values are "master" or "student"

### R2: Single-Phase Student Training

When model_type="student", the Trainer SHALL train the full model from epoch 1 with no freeze/unfreeze phases. The learning rate schedule MUST use cosine annealing with linear warmup.

#### Scenario: Student training loop runs single-phase

- GIVEN TrainingConfig with model_type="student" and epochs=50
- WHEN training starts
- THEN all StudentModel parameters SHALL have requires_grad=True from epoch 1
- AND cosine+warmup scheduler SHALL be applied
- AND no freeze/unfreeze calls SHALL be made

#### Scenario: Student cosine+warmup LR schedule

- GIVEN TrainingConfig with model_type="student", lr=1e-3, warmup_epochs=5
- WHEN epoch < warmup_epochs
- THEN LR SHALL increase linearly from 0 to lr
- WHEN epoch >= warmup_epochs
- THEN LR SHALL decay following cosine annealing

### R3: Forward-Call Branching

The Trainer MUST branch the forward call based on model_type. Master models receive (rgb, nir); student models receive (rgb) only.

#### Scenario: Student forward pass uses RGB only

- GIVEN model_type="student" and a batch containing (rgb, nir, bboxes, class_labels)
- WHEN the training step executes the forward pass
- THEN `model(rgb)` SHALL be called with only the RGB tensor
- AND the NIR tensor SHALL NOT be passed to the model

#### Scenario: Master forward pass unchanged

- GIVEN model_type="master" and a batch containing (rgb, nir, bboxes, class_labels)
- WHEN the training step executes the forward pass
- THEN `model(rgb, nir)` SHALL be called with both RGB and NIR tensors
- AND behavior SHALL be identical to existing two-phase training

## MODIFIED Requirements

### Requirement: Two-Phase Fine-Tuning

The trainer SHALL execute 2-phase fine-tuning ONLY when model_type="master". Phase 1 MUST freeze the entire backbone (stems + stages 1-4) and train only fusion, neck, and head modules (~18.8M params) at lr=1e-3. Phase 2 MUST unfreeze backbone stages 3-4 (stems remain frozen) and train all unfrozen parameters at lr=1e-4. When model_type="student", two-phase logic SHALL NOT execute.
(Previously: Two-phase training was unconditional for all models)

#### Scenario: Phase 1 freezes backbone and trains head

- GIVEN MasterModel with model_type="master" and 68.3M total parameters
- WHEN training phase 1 starts
- THEN backbone stems and stages 1-4 SHALL have requires_grad=False
- AND fusion, neck, and head SHALL have requires_grad=True
- AND optimizer SHALL use lr=1e-3 for trainable parameters only

#### Scenario: Phase 2 unfreeze stages 3-4

- GIVEN Phase 1 completed at least one epoch
- WHEN training transitions to Phase 2
- THEN backbone stages 3-4 SHALL have requires_grad=True
- AND stems SHALL remain frozen (requires_grad=False)
- AND optimizer SHALL use lr=1e-4

#### Scenario: Student training skips two-phase entirely

- GIVEN model_type="student"
- WHEN training begins
- THEN no freeze/unfreeze phases SHALL execute
- AND all model parameters SHALL be trainable from epoch 1

#### Scenario: ConvNeXt-Tiny backbone variant

- GIVEN training config specifies backbone_variant="tiny"
- WHEN MasterModel is instantiated
- THEN the ConvNeXt-Tiny backbone (28M params) SHALL replace ConvNeXt-Small (50M params)

### Requirement: Validation and Checkpointing

After each epoch, the trainer MUST run validation on the val split, compute mAP metrics, and save a checkpoint if val mAP@0.5 improves. Periodic checkpoints SHALL be saved every N epochs (configurable, default 10). Early stopping MUST use patience=15 for both master and student models.
(Previously: Early stopping patience was default 15 but not explicitly scoped to both model types)

#### Scenario: Best checkpoint saved on mAP improvement

- GIVEN epoch validation completes with mAP@0.5 > previous best
- WHEN the improvement exceeds 0.001
- THEN a checkpoint SHALL be saved to checkpoints/best.pt
- AND checkpoint SHALL contain: model_state, optimizer_state, scheduler_state, epoch, val_metrics, config

#### Scenario: Early stopping triggers for master

- GIVEN model_type="master" and validation mAP@0.5 has not improved for patience=15 epochs
- WHEN early stopping check runs
- THEN training SHALL stop
- AND best checkpoint SHALL be restored

#### Scenario: Early stopping triggers for student

- GIVEN model_type="student" and validation mAP@0.5 has not improved for patience=15 epochs
- WHEN early stopping check runs
- THEN training SHALL stop
- AND best checkpoint SHALL be restored

#### Scenario: Periodic checkpoint

- GIVEN checkpoint_every=10 in config
- WHEN epoch is a multiple of 10
- THEN a checkpoint SHALL be saved to checkpoints/epoch_{N}.pt

## Dependencies

- yolo-nano-student: StudentModel freeze/unfreeze stubs for API compatibility
- training-metrics: LossHistory feeds training_curves.png generation

## Non-functional

- Model-type branching MUST NOT add measurable overhead to master training loop (<1% wall time)
- Student training with ~3M params SHOULD complete one epoch in <60s on RTX 3080 (10GB)
- Config validation MUST reject unknown model_type values at startup