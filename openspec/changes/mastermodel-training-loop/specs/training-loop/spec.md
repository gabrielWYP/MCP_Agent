# training-loop Specification

## Purpose

2-phase fine-tuning trainer for the multimodal MasterModel. Manages the full training lifecycle: data loading, forward/backward passes, gradient optimization, validation, checkpointing, early stopping, and logging.

## Requirements

### Requirement: Two-Phase Fine-Tuning

The trainer SHALL execute training in two sequential phases. Phase 1 MUST freeze the entire backbone (stems + stages 1-4) and train only fusion, neck, and head modules (~18.8M params) at lr=1e-3. Phase 2 MUST unfreeze backbone stages 3-4 (stems remain frozen) and train all unfrozen parameters at lr=1e-4.

#### Scenario: Phase 1 freezes backbone and trains head

- GIVEN MasterModel with 68.3M total parameters
- WHEN training phase 1 starts
- THEN backbone stems and stages 1-4 SHALL have `requires_grad=False`
- AND fusion, neck, and head SHALL have `requires_grad=True`
- AND optimizer SHALL use lr=1e-3 for trainable parameters only

#### Scenario: Phase 2 unfreeze stages 3-4

- GIVEN Phase 1 completed at least one epoch
- WHEN training transitions to Phase 2
- THEN backbone stages 3-4 SHALL have `requires_grad=True`
- AND stems SHALL remain frozen (`requires_grad=False`)
- AND optimizer SHALL use lr=1e-4

#### Scenario: ConvNeXt-Tiny backbone variant

- GIVEN training config specifies `backbone_variant="tiny"`
- WHEN MasterModel is instantiated
- THEN the ConvNeXt-Tiny backbone (28M params) SHALL replace ConvNeXt-Small (50M params)

### Requirement: Training Step Execution

Each training step MUST perform: load batch → forward pass → loss computation → backward pass → gradient clipping → optimizer step. Mixed precision (AMP) SHALL be used when `config.amp=True`.

#### Scenario: Standard training step with AMP

- GIVEN a batch of (rgb, nir, bboxes, class_labels) tensors
- WHEN the training step runs with AMP enabled
- THEN forward pass SHALL use `torch.cuda.amp.autocast`
- AND gradients SHALL be scaled before backward pass
- AND `clip_grad_norm_` SHALL clip at `config.grad_clip` (default 10.0)

#### Scenario: OOM fallback

- GIVEN RTX 3080 (10GB VRAM) with batch_size=4 and image_size=640
- WHEN training step encounters CUDA OOM
- THEN the step SHALL be skipped and a warning logged
- AND training SHALL continue with the next batch

### Requirement: Validation and Checkpointing

After each epoch, the trainer MUST run validation on the val split, compute mAP metrics, and save a checkpoint if val mAP@0.5 improves. Periodic checkpoints SHALL be saved every N epochs (configurable, default 10).

#### Scenario: Best checkpoint saved on mAP improvement

- GIVEN epoch validation completes with mAP@0.5 > previous best
- WHEN the improvement exceeds 0.001
- THEN a checkpoint SHALL be saved to `checkpoints/best.pt`
- AND checkpoint SHALL contain: model_state, optimizer_state, scheduler_state, epoch, val_metrics, config

#### Scenario: Early stopping triggers

- GIVEN validation mAP@0.5 has not improved for `patience` epochs (default 15)
- WHEN early stopping check runs
- THEN training SHALL stop
- AND best checkpoint SHALL be restored

#### Scenario: Periodic checkpoint

- GIVEN `checkpoint_every=10` in config
- WHEN epoch is a multiple of 10
- THEN a checkpoint SHALL be saved to `checkpoints/epoch_{N}.pt`

### Requirement: Logging

The trainer MUST log to TensorBoard: loss components (cls_loss, reg_loss, total_loss) per step; mAP@0.5, mAP@0.5:0.95 per epoch; learning rate per step; per-class AP per epoch.

#### Scenario: Loss curves logged during training

- GIVEN a training step completes
- WHEN logging is enabled
- THEN cls_loss, reg_loss, and total_loss SHALL be logged as TensorBoard scalars