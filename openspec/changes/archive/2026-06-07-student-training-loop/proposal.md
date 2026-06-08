# Proposal: Student Training Loop

## Intent

Add a StudentModel-only (no knowledge distillation) training loop. The current Trainer is hardwired for MasterModel's dual-input API (rgb, nir). StudentModel is RGB-only, ~3M params (CSPDarknetNano), and needs single-phase training — no freeze/unfreeze stages required. This produces a standalone baseline before implementing KD.

## Scope

### In Scope
- Trainer forward-call branching: `model(rgb, nir)` vs `model(rgb)`
- `freeze_backbone()` / `unfreeze_backbone_stages()` stubs on StudentModel (API compat, no-op for single-phase)
- `TrainingConfig.model_type` field + `--model {master,student}` CLI flag
- `training_curves.png` auto-generation from LossHistory via matplotlib
- `configs/training_student.yaml` (student hyperparams, no NIR fields)
- Early stopping with patience=15

### Out of Scope
- Knowledge distillation loss / teacher-student comparison
- ONNX export or inference script
- NIR skip in dataset (accept overhead — ~50 imgs, negligible)
- Two-phase training for student
- Hyperparameter tuning beyond student-appropriate defaults
- mAP overlay on training curves (deferred)

## Capabilities

### New Capabilities
- None

### Modified Capabilities
- `training-loop`: Add model-type dispatch (student single-phase vs master two-phase), branch forward call
- `training-metrics`: Add automatic `training_curves.png` generation from LossHistory data
- `yolo-nano-student`: Add `freeze_backbone()` and `unfreeze_backbone_stages()` method stubs

## Approach

Minimal-viable: add `model_type` config field, branch Trainer's `_train_epoch` and `_validate` forward calls on model type, add freeze/unfreeze stubs to StudentModel (no-op for single-phase but required for Trainer API), and generate loss curve PNG after training completes. NIR stays in batch dict (ignored by student) — dataset unchanged. Single-phase: full model trainable from epoch 1, cosine+warmup LR schedule, patience=15 early stopping.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/training/loop.py` | Modified | Branch forward call for model type |
| `src/training/train.py` | Modified | Add `--model` flag, student instantiation |
| `src/training/config.py` | Modified | Add `model_type` field, conditional NIR fields |
| `src/models/student/student_model.py` | Modified | Add freeze/unfreeze method stubs |
| `src/training/metrics.py` | Modified | Add training_curves.png generation |
| `configs/training_student.yaml` | New | Student-specific training config |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Freeze stubs misused in future KD loop | Low | Document no-op behavior clearly in docstrings |
| Student hyperparams need tuning from defaults | Medium | Use conservative LR 1e-3, early stopping patience 15 |
| training_curves.png fails on headless server | Low | Guard with try/except, log warning, continue training |
| Small dataset (~50 imgs) → overfitting | High | Heavy augmentation already in place; early stopping; accept limited generalization as baseline |

## Rollback Plan

All changes additive or branching — no existing behavior removed. Revert: `git revert` removes student branch logic; master training loop completely unaffected.

## Dependencies

- StudentModel (`yolo-nano-student` spec) already implemented
- YOLODataset, YOLOv8Loss, augmentations compatible as-is
- matplotlib for training_curves.png generation

## Success Criteria

- [ ] Training loop completes full epochs without errors (student model, single-phase, RGB-only)
- [ ] Loss decreases monotonically on train set
- [ ] `training_curves.png` generated with cls_loss, reg_loss, total_loss curves
- [ ] No regression to existing master training loop