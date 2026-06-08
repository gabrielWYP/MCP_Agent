# Exploration: Knowledge Distillation Training (kd-training)

## Summary

Implement Knowledge Distillation training where a pre-trained MasterModel (ConvNeXt-Tiny + cross-modal fusion, mAP@0.5 ≈ 0.30) acts as teacher for a YOLOv8-Nano StudentModel (baseline mAP 0.2354). The distillation transfers knowledge at three architectural levels: backbone, FPN neck, and detection head — using 1x1 convolution projections to align channel dimensions between teacher and student.

The implementation follows a strict subclass approach: `KDTrainer(Trainer)` overrides only `__init__` (teacher loading + projection attachment) and `_train_epoch` (KD forward pass). All other Trainer machinery — phase loop, validation, checkpointing, logging, early stopping — is inherited unchanged.

## Current State

### What works today
- **Teacher (MasterModel)**: Trained checkpoint at `checkpoints/mastermodel_mango/best_model.pt` (epoch 68, mAP@0.5=0.3003). Produces 7-key output dict including 5 distillation feature keys.
- **Student (StudentModel)**: Working forward pipeline, produces 7-key output dict with matching distill feature keys. Baseline trained to mAP 0.2354.
- **ProjectionLayers**: 3 presets implemented in `src/models/master/distill_projections.py` — `backbone_projections()`, `fpn_projections()`, `head_projections()`.
- **Trainer**: Production training loop (494 lines) with AMP, gradient clipping, cosine warmup scheduling, TensorBoard, early stopping. Dispatches master/student via `config.model_type`.
- **YOLOv8Loss**: Task-Aligned Assigner + Focal BCE + CIoU loss. Model-agnostic.
- **YOLODataset**: RGB+NIR pair loading with letterbox and augmentation.

### Teacher → Student distillation feature mapping

| KD Level | Teacher Key | Teacher Channels | Projection Preset | Student Key | Student Channels | Match |
|----------|-------------|-----------------|-------------------|-------------|-----------------|-------|
| Backbone | `distill_backbone_rgb[2:]` (S3,S4) | [384, 768] | `backbone_projections()` | `distill_backbone` | [128, 256] | ✅ |
| FPN | `distill_fpn` (P3,P4,P5) | [256, 256, 256] | `fpn_projections()` | `distill_fpn` | [128, 256, 256] | ✅ |
| Head CLS | `distill_head_cls` | [256, 256, 256] | `head_projections()` | `distill_head_cls` | [64, 128, 256] | ✅ |
| Head REG | `distill_head_reg` | [256, 256, 256] | `head_projections()` | `distill_head_reg` | [64, 128, 256] | ✅ |

## Files to Create

| # | File | Purpose | Est. Lines |
|---|------|---------|------------|
| 1 | `src/training/kd_config.py` | `KDConfig(TrainingConfig)` dataclass: adds `kd_weight`, `kd_temperature`, `teacher_checkpoint`, `distill_levels` | ~35 |
| 2 | `src/training/kd_loss.py` | `KDLoss(nn.Module)`: MSE-based feature-level distillation loss across 4 projection groups | ~50 |
| 3 | `src/training/kd_trainer.py` | `KDTrainer(Trainer)`: overrides `__init__` (teacher load + projection attach) and `_train_epoch` (KD forward) | ~180 |
| 4 | `src/training/kd_train.py` | CLI entry point: `python -m src.training.kd_train --config configs/kd_training.yaml` | ~130 |
| 5 | `configs/kd_training.yaml` | YAML config with KD hyperparameters | ~60 |
| **Total** | | | **~455** |

## Files to Modify

**0 files.** The subclass approach isolates all KD logic in new files.

## Files That DON'T Need Changes

- `src/training/loop.py` — `Trainer` base class (inherited)
- `src/training/loss.py` — `YOLOv8Loss` (composed with KDLoss, not modified)
- `src/training/config.py` — `TrainingConfig` (inherited via `KDConfig`)
- `src/training/dataset.py` — `YOLODataset` (same data pipeline)
- `src/training/train.py` — Standard training CLI (unchanged)
- `src/training/augmentations.py` — Unchanged
- `src/training/metrics.py` — Unchanged
- `src/models/master/master_model.py` — Loaded for teacher only, no changes
- `src/models/master/distill_projections.py` — Used as-is
- `src/models/student/student_model.py` — Used as-is (projections attached at runtime)

## Channel Verification Table (Detailed)

### Backbone Level
```
Master distill_backbone_rgb: [S1(96), S2(192), S3(384), S4(768)]  → select [2:] → [384, 768]
backbone_projections(): teacher=[384, 768] → conv1x1 → student=[128, 256]
Student distill_backbone: [S3(128), S4(256)]
✅ Channel match verified
```

### FPN Neck Level
```
Master distill_fpn: [P3(256), P4(256), P5(256)]
fpn_projections(): teacher=[256, 256, 256] → conv1x1 → student=[128, 256, 256]
Student distill_fpn: [P3(128), P4(256), P5(256)]
✅ Channel match verified
```

### Head Levels (CLS and REG)
```
Master distill_head_cls: [cls_stem_P3(256), cls_stem_P4(256), cls_stem_P5(256)]
Master distill_head_reg: [reg_stem_P3(256), reg_stem_P4(256), reg_stem_P5(256)]
head_projections(): teacher=[256, 256, 256] → conv1x1 → student=[64, 128, 256]
Student distill_head_cls: [64, 128, 256]
Student distill_head_reg: [64, 128, 256]
✅ Channel match verified
```

### Spatial Resolution Match (640×640 input)
| Level | Teacher Spatial | Student Spatial | Stride | Match |
|-------|----------------|-----------------|--------|-------|
| P3/S3 | 80×80 | 80×80 | 8 | ✅ |
| P4/S4 | 40×40 | 40×40 | 16 | ✅ |
| P5 | 20×20 | 20×20 | 32 | ✅ |

## Inheritance Analysis

### Trainer Methods — Inherit vs Override

| Method | Action | Reason |
|--------|--------|--------|
| `__init__` | **OVERRIDE** | Load frozen teacher, instantiate 4 ProjectionLayers, attach as student submodules |
| `_train_epoch` | **OVERRIDE** | KD forward: teacher forward (no grad) → project → MSE loss + detection loss |
| `fit()` | **INHERIT** | Orchestration unchanged — single-phase KD training via `model_type="student"` path |
| `_train_phase()` | **INHERIT** | Student freeze/unfreeze are no-ops; projections are submodules so `model.parameters()` includes them; scheduler unchanged |
| `_validate()` | **INHERIT** | Validation uses student only (no teacher needed for eval); same mAP computation |
| `_save_checkpoint()` | **INHERIT** | `model.state_dict()` includes student + projection states (projections are registered submodules) |
| `_decode_predictions()` | **INHERIT** | Post-processing unchanged |

### Key Design Decisions

1. **Projections as submodules**: Attaching projections directly to the StudentModel via standard PyTorch attribute assignment (`self.model.kd_proj_backbone = backbone_projections()`) — this registers them in `model.parameters()`, `model.state_dict()`, and `model.train()`. Zero code needed in `_train_phase` or `_save_checkpoint`.

2. **Single-phase KD**: `fit()` dispatches via `config.model_type == "student"`, calling `_train_phase(phase=1, epochs=config.epochs, lr=config.lr, freeze_stages=0)`. Student's `freeze_backbone(0)` is a no-op — all params trainable from epoch 1.

3. **Teacher is frozen, not saved**: Teacher loaded once in `__init__`, set to `eval()` with all `requires_grad=False`. Not part of `self.model` — excluded from optimizer, checkpointing, and state_dict.

4. **KD loss composition**: `total_loss = det_loss + config.kd_weight * kd_loss`, normalized per batch. KDLoss operates on projected teacher features vs student features.

## Risks / Gotchas

### Risk 1: NaN in KD training
**Severity**: Medium  
The teacher checkpoint was trained with `amp=false` (FP16 caused NaN in YOLOv8Loss). KD training should also use `amp=false` unless empirically validated. KD loss adds gradient paths from projections → student backbone/neck/head, potentially amplifying numerical instability.

### Risk 2: Channel mismatch during loading
**Severity**: Low  
The `backbone_projections()` preset expects exactly 2 feature levels (S3, S4). If the master model outputs a different number of backbone stages (unlikely with frozen ConvNeXt-Tiny architecture), the projection forward would assert-fail. Mitigated by explicit `[2:]` slicing.

### Risk 3: Teacher checkpoint format
**Severity**: Low  
The checkpoint at `checkpoints/mastermodel_mango/best_model.pt` does NOT contain `model_type` in its config dict. Teacher loading must construct `MasterModel` from KDConfig fields (`num_classes`, `backbone_variant`), then load `model_state_dict`. The config dict in the checkpoint is a plain dict (not a `TrainingConfig` dataclass).

### Risk 4: Memory pressure (teacher + student in GPU)
**Severity**: Medium  
Both teacher (~25M params) and student (~3M params) must fit in GPU memory simultaneously during forward. With batch_size=2-4 and 640×640 images, this is ~2-3 GB. For batch_size=4, monitor GPU memory. If OOM, reduce batch_size or use `torch.no_grad()` + `pin_memory` for teacher features.

### Risk 5: KD weight tuning
**Severity**: Low  
The KD loss weight (`kd_weight`) must be tuned empirically. Too high → student overfits to teacher features and detection degrades. Too low → no distillation benefit. Start at 1.0 and tune via validation mAP.

## Recommendations

1. **Start with `kd_weight=1.0` and `kd_temperature=1.0`** — tune after first training run
2. **Disable AMP initially** (`amp: false`) — same NaN risk as master training
3. **Use `batch_size=2` first** — conservative GPU memory, increase if stable
4. **Use all 4 distillation levels** (backbone, FPN, head_cls, head_reg) — can ablate later
5. **KD checkpoint saves**: student state_dict + projection state_dict + KDConfig. Teacher checkpoint path is recorded in config only.
6. **Validation unchanged**: Student-only validation — teacher not used during eval
7. **No changes to existing files** — the subclass approach is verified safe

## Ready for Proposal

**Yes.** All architectural questions are resolved. The subclass approach has been verified against the actual codebase. Channel dimensions match across all 4 distillation levels. No modifications to existing files are needed.
