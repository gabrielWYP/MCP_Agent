# Proposal: Cross-Modal Knowledge Distillation Training (CMD-KD)

## Intent

Transfer learned representations from a pre-trained MasterModel (RGB+NIR, mAP@0.5=0.3003) to a YOLOv8-Nano StudentModel (RGB-only, baseline mAP=0.2354) via feature-level knowledge distillation. The teacher's cross-modal features enrich the distillation signal for the RGB-only student, closing the mAP gap without adding inference cost.

## Scope

### In Scope
- KDTrainer(Trainer) subclass with 4-level distillation (backbone, FPN, head_cls, head_reg)
- Per-level KD weights: backbone=1.0, head_reg=0.5, head_cls=0.3, fpn=0.1 (YAML-tunable)
- KDConfig(TrainingConfig) dataclass with KD hyperparameters
- KDLoss(nn.Module): MSE on projected teacher vs student features
- CLI entry point via `python -m src.training.kd_train`
- 80-epoch training with patience=30, amp=false, batch_size=2

### Out of Scope
- Ablation studies of individual distillation levels
- Formal baseline-vs-KD mAP comparison experiments
- Hyperparameter auto-tuning
- Modifications to existing Trainer, StudentModel, YOLOv8Loss, or YOLODataset
- Teacher checkpoint management (retraining, versioning)

## Capabilities

### New Capabilities
- `kd-training`: Knowledge distillation training pipeline — KDTrainer, KDConfig, KDLoss, CLI entry point, and YAML config

### Modified Capabilities
- None — existing modules used without modification

## Approach

Subclass pattern: `KDTrainer(Trainer)` overrides `__init__` (load frozen teacher, attach 1×1 projection submodules to student) and `_train_epoch` (teacher forward no_grad → project → MSE per level → weighted KD loss + detection loss). All other Trainer machinery (fit, validate, checkpoint, early stopping, logging) inherited.

Projections attached via `self.model.kd_proj_* = ProjectionLayers(...)` — auto-registered in `parameters()`, `state_dict()`, `train()`. Teacher loaded once, frozen eval-only, excluded from optimizer/checkpoints.

Loss: `total = det_loss + kd_weight × Σ(kd_w_level × MSE(proj_teacher_f, student_f))`.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/training/kd_config.py` | New | KDConfig dataclass |
| `src/training/kd_loss.py` | New | KDLoss module |
| `src/training/kd_trainer.py` | New | KDTrainer subclass |
| `src/training/kd_train.py` | New | CLI entry point |
| `configs/kd_training.yaml` | New | KD hyperparameters |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| NaN during KD training (FP16 + extra gradient paths) | Medium | Start with amp=false; same NaN risk as master training |
| GPU OOM (teacher+student+projections ~28M params) | Medium | batch_size=2; monitor VRAM; increase if stable |
| KD weight imbalance harms detection accuracy | Low | Per-level weights; start backbone-heavy; tune via validation mAP |

## Rollback Plan

Delete 5 new files. Zero existing files modified — rollback is clean removal of `src/training/kd_*.py` and `configs/kd_training.yaml`. No database migrations, no config changes, no side effects.

## Dependencies

- Pre-trained MasterModel checkpoint: `checkpoints/mastermodel_mango/best_model.pt`
- Existing `distill_projections.py` — 3 projection presets (used as-is)
- Existing `TrainingConfig`, `Trainer`, `YOLOv8Loss`, `StudentModel` (all inherited/composed, not modified)

## Success Criteria

- [ ] KD training runs 80 epochs without NaN or OOM errors
- [ ] KD loss (MSE component) decreases over training
- [ ] Validation mAP@0.5 falls between student baseline (0.2354) and teacher (0.3003)
- [ ] Checkpoints contain student + projection state dicts (teacher excluded)
- [ ] Zero modifications to existing source files