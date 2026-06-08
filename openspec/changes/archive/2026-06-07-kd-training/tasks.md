# Tasks: Cross-Modal Knowledge Distillation Training (kd-training)

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~455 (5 new files, 0 modified) |
| 400-line budget risk | Medium |
| 800-line budget risk | Low |
| Chained PRs recommended | Yes |
| Suggested split | PR 1 (Foundation: kd_config + kd_loss, ~85 lines) → PR 2 (Core: kd_trainer + kd_train + yaml, ~370 lines) |
| Delivery strategy | ask-on-risk |
| Chain strategy | stacked-to-main |

Decision needed before apply: Yes
Chained PRs recommended: Yes
Chain strategy: stacked-to-main
400-line budget risk: Medium

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | KDConfig + KDLoss — standalone, testable in isolation | PR 1 | base: main; ~85 lines; no deps on teacher/student models |
| 2 | KDTrainer + CLI + YAML — full training pipeline | PR 2 | base: main (after PR 1 merges); ~370 lines; depends on PR 1 |

## Phase 1: Foundation (Config + Loss)

- [x] 1.1 Create `src/training/kd_config.py` — `KDConfig(TrainingConfig)` dataclass with fields: `teacher_checkpoint` (str, required), `kd_weight` (float=1.0), `kd_temperature` (float=1.0), `distill_levels` (dict with backbone=1.0, fpn=0.1, head_cls=0.3, head_reg=0.5). Override defaults: `epochs=80`, `patience=30`, `amp=False`, `batch_size=4`, `model_type="student"`. Inherits `from_yaml` from `TrainingConfig`.
- [x] 1.2 Create `src/training/kd_loss.py` — `KDLoss(nn.Module)` with `__init__(level_weights: dict, temperature: float=1.0)` and `forward(teacher_feats: dict, student_feats: dict) -> (total_kd_loss, per_level_dict)`. MSE per level, weighted sum. Four levels: backbone, fpn, head_cls, head_reg. Each level's features are lists of tensors — compute MSE across all sub-levels within each group.

## Phase 2: Core Implementation (Trainer)

- [x] 2.1 Create `src/training/kd_trainer.py` — `KDTrainer(Trainer)` class. In `__init__`: (a) attach 4 projection groups to student model BEFORE `super().__init__()` — `model.kd_proj_backbone = backbone_projections()`, `model.kd_proj_fpn = fpn_projections()`, `model.kd_proj_head_cls = head_projections()`, `model.kd_proj_head_reg = head_projections()`; (b) call `super().__init__()` which moves model to device; (c) load teacher via `_load_teacher(config)` — instantiate `MasterModel`, load checkpoint, `eval()`, freeze all params, store as `self.teacher` (NOT a submodule); (d) create `self.kd_criterion = KDLoss(config.distill_levels, config.kd_temperature)`.
- [x] 2.2 Implement `KDTrainer._load_teacher(config)` — static/class method. Assert checkpoint file exists (`FileNotFoundError` if missing). Instantiate `MasterModel(num_classes=config.num_classes)`, load `state_dict` from checkpoint. Set `eval()`, freeze all `requires_grad=False`. Return teacher model.
- [x] 2.3 Override `KDTrainer._train_epoch(optimizer, epoch, phase)` — follow base `Trainer._train_epoch` pattern with these changes: (a) teacher forward under `torch.no_grad()` with `(rgb, nir)` → extract `distill_backbone_fused`, `distill_fpn`, `distill_head_cls`, `distill_head_reg`; (b) project each teacher feature group through corresponding `self.model.kd_proj_*`; (c) student forward with `rgb` only → extract same 4 distill keys; (d) compute `det_loss` via `self.criterion(student_output["preds"], targets)`; (e) compute `kd_loss, kd_per_level = self.kd_criterion(projected_teacher, student)`; (f) `total = det_loss + config.kd_weight * kd_loss`; (g) NaN guard, backward, grad clip, optimizer step — same pattern as base. Return dict with `cls_loss`, `box_loss`, `kd_loss`, `total_loss`.
- [x] 2.4 Verify checkpoint exclusion — `self.teacher` is stored as a plain attribute (not `nn.Module` submodule), so `self.model.state_dict()` in inherited `_save_checkpoint` naturally excludes teacher. Add a comment documenting this invariant. No code change needed in `_save_checkpoint`.

## Phase 3: Integration (CLI + Config + Verification)

- [x] 3.1 Create `src/training/kd_train.py` — CLI entry point mirroring `src/training/train.py`. `parse_args()` with `--config` (default `configs/kd_training.yaml`) and `--override` key=value pairs. `main()`: load `KDConfig.from_yaml()`, create `YOLODataset` (train/val) with transforms, build DataLoaders with `collate_fn`, instantiate `StudentModel(num_classes=config.num_classes)`, create `KDTrainer`, call `trainer.fit()`. Print results summary.
- [x] 3.2 Create `configs/kd_training.yaml` — all KDConfig fields: `teacher_checkpoint: checkpoints/mastermodel_mango/best_model.pt`, `kd_weight: 1.0`, `kd_temperature: 1.0`, `distill_levels: {backbone: 1.0, fpn: 0.1, head_cls: 0.3, head_reg: 0.5}`, `epochs: 80`, `patience: 30`, `amp: false`, `batch_size: 4`, `model_type: student`, plus inherited paths (`rgb-dir`, `nir-dir`, `labels-dir`, `output-dir: checkpoints/kd_student`).
- [x] 3.3 Smoke test — run `python -m src.training.kd_train --config configs/kd_training.yaml` for 1-2 epochs to verify: teacher loads without error, projections register in `model.state_dict()`, KD loss computes and decreases, no NaN/OOM, checkpoint saves without teacher weights.
