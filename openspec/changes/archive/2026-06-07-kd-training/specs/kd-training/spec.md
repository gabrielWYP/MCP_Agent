# Delta for kd-training

## ADDED Requirements

### R1: Teacher Loading and Freezing

KDTrainer MUST load a MasterModel from a checkpoint path specified in KDConfig. Upon loading, it SHALL verify `num_classes` matches between checkpoint and config, set the teacher to `eval()` mode, and freeze all teacher parameters (`requires_grad=False`). The teacher MUST NOT be included in the optimizer, scheduler, or checkpoint state.

#### Scenario: Teacher loaded from valid checkpoint

- GIVEN a KDConfig with `teacher_checkpoint="checkpoints/mastermodel_mango/best_model.pt"` and `num_classes=2`
- WHEN KDTrainer initializes
- THEN MasterModel SHALL be instantiated from config fields and checkpoint weights loaded
- AND teacher SHALL be in eval mode with all `requires_grad=False`
- AND teacher SHALL NOT appear in `optimizer.param_groups`

#### Scenario: Teacher checkpoint missing

- GIVEN a KDConfig with `teacher_checkpoint="nonexistent.pt"`
- WHEN KDTrainer initializes
- THEN KDTrainer SHALL raise `FileNotFoundError` with a message indicating the checkpoint path does not exist

#### Scenario: Teacher num_classes mismatch

- GIVEN a checkpoint trained with `num_classes=5` but KDConfig specifies `num_classes=2`
- WHEN KDTrainer loads the checkpoint
- THEN assertion SHALL fail at forward time with a message indicating the channel dimension mismatch

### R2: Projection Attachment as Student Submodules

KDTrainer MUST instantiate 4 `ProjectionLayers` groups (backbone, fpn, head_cls, head_reg) and attach them to the StudentModel as registered submodules (`self.model.kd_proj_backbone`, `self.model.kd_proj_fpn`, `self.model.kd_proj_head_cls`, `self.model.kd_proj_head_reg`). These projections SHALL be trainable from epoch 1 and included in `model.parameters()`, `model.state_dict()`, and `model.train()` propagation.

#### Scenario: Projections registered as submodules

- GIVEN KDTrainer with StudentModel and 4 projection groups
- WHEN projections are attached via standard attribute assignment
- THEN each projection group SHALL appear in `model.state_dict()` with key prefix `kd_proj_{level}`
- AND each projection group SHALL appear in `model.parameters()`
- AND calling `model.train()` SHALL set all projections to training mode

### R3: KD Forward Pass

During each training step, KDTrainer MUST run teacher forward with `torch.no_grad()` using both RGB and NIR inputs, project teacher features to student channel dimensions via the 4 projection groups, run student forward with RGB only, and compute MSE loss per distillation level between projected teacher features and student features.

#### Scenario: KD forward computes per-level MSE

- GIVEN a batch of `(rgb, nir, bboxes, class_labels)` and both teacher and student models
- WHEN KDTrainer executes the training step
- THEN teacher forward SHALL run under `torch.no_grad()` producing features at 4 levels
- AND each projection group SHALL map teacher channels to student channels
- AND MSE SHALL be computed per level between projected teacher and student features
- AND no gradient SHALL flow back through teacher parameters

#### Scenario: Channel dimension assertion at forward time

- GIVEN a projection group configured for teacher channels `[384, 768]` to student channels `[128, 256]`
- WHEN teacher outputs a feature level with an unexpected channel count
- THEN an assertion error SHALL be raised identifying the mismatched dimension

### R4: Loss Composition

The total training loss MUST be: `total = det_loss + kd_weight × Σ(per_level_weight × MSE_level)` where `det_loss` is the standard YOLOv8Loss (cls + reg) and `kd_weight` scales the distillation contribution. Per-level weights from KDConfig defaults: `backbone=1.0`, `head_reg=0.5`, `head_cls=0.3`, `fpn=0.1`.

#### Scenario: Combined loss computed correctly

- GIVEN det_loss=2.5, kd_weight=1.0, backbone MSE=0.04, fpn MSE=0.01, head_cls MSE=0.02, head_reg MSE=0.03
- WHEN total loss is computed
- THEN kd_loss = 1.0×0.04 + 0.1×0.01 + 0.3×0.02 + 0.5×0.03 = 0.062
- AND total = 2.5 + 1.0 × 0.062 = 2.562

#### Scenario: kd_weight set to zero disables distillation

- GIVEN KDConfig with kd_weight=0.0
- WHEN total loss is computed
- THEN total loss SHALL equal det_loss only
- AND teacher forward SHALL still execute (but contributes zero gradient)

### R5: Checkpoint Format

Checkpoints saved during KD training MUST contain `model_state_dict` (student weights + projection weights) and `config`. Teacher weights MUST NOT be included in the checkpoint. The checkpoint structure SHALL be compatible with the existing Trainer `_save_checkpoint` method.

#### Scenario: Checkpoint excludes teacher

- GIVEN a KDTrainer that has loaded a teacher model with ~25M params and a student+projections with ~3.2M+0.1M params
- WHEN a checkpoint is saved
- THEN the checkpoint SHALL contain student and projection state dicts
- AND the checkpoint SHALL NOT contain any teacher model state dict entries

### R6: Student-Only Validation

Validation during KD training MUST use student-only forward pass (RGB input only), producing mAP@0.5 and mAP@0.5:0.95 metrics. The teacher MUST NOT participate in validation inference.

#### Scenario: Validation runs student-only

- GIVEN KDTrainer at epoch end with teacher and student models loaded
- WHEN `_validate()` is called
- THEN only student model SHALL receive RGB inputs
- AND teacher model SHALL NOT be used during validation
- AND validation mAP computation SHALL be identical to standard student training

### R7: KDConfig Dataclass

KDConfig MUST extend TrainingConfig with KD-specific fields: `teacher_checkpoint` (str, required), `kd_weight` (float, default 1.0), `kd_temperature` (float, default 1.0), and `distill_levels` dict with per-level weights (defaults: backbone=1.0, fpn=0.1, head_cls=0.3, head_reg=0.5). Training parameters: `epochs=80`, `patience=30`, `amp=false`, `batch_size=2`, `model_type="student"`.

#### Scenario: KDConfig defaults applied

- GIVEN a YAML config with only `teacher_checkpoint` specified
- WHEN KDConfig is instantiated
- THEN kd_weight SHALL default to 1.0, patience to 30, amp to false, batch_size to 2
- AND model_type SHALL be "student"

## Dependencies

- **training-loop**: Inherits Trainer lifecycle (fit, validate, checkpoint, early stopping, logging). KDTrainer overrides `__init__` and `_train_epoch` only.
- **training-metrics**: Validation mAP computation used unchanged (student-only).
- **yolo-nano-student**: StudentModel provides 7-key output dict including 4 distillation feature keys. Projections attach to student as submodules via freeze stubs.

## Non-functional

| Property | Requirement |
|----------|-------------|
| GPU Memory | Teacher (~25M) + Student (~3.2M) + Projections (~0.1M) ≈ 28.3M params; batch_size=2 at 640×640 ≈ 2-3 GB VRAM |
| Numerical Stability | amp=false by default; KD loss adds gradient paths through projections but no FP16 risk initially |
| Training Duration | 80 epochs with patience=30; expect ~2-4 hours on RTX 3080 |
| Inference Cost | Zero additional inference cost — teacher excluded at validation and deployment |