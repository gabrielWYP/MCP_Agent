# Tasks: Student Training Loop

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~270 (5 modified + 2 new files) |
| 400-line budget risk | Low |
| 800-line budget risk | Low |
| Chained PRs recommended | No |
| Suggested split | Single PR |
| Delivery strategy | ask-on-risk |
| Chain strategy | pending |

Decision needed before apply: No
Chained PRs recommended: No
Chain strategy: pending
400-line budget risk: Low

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | Config + stubs + curves + loop integration + CLI + tests | PR 1 | All changes fit single review; no split needed |

## Phase 1: Foundation (Config + Stubs)

- [x] 1.1 Add `model_type: str = "master"`, `epochs: int = 50`, `lr: float = 1e-3` fields to `TrainingConfig` in `src/training/config.py`. Add `__post_init__` validation that rejects `model_type` not in `{"master", "student"}` with `ValueError`.
- [x] 1.2 Add `freeze_backbone(freeze_stages: int = 0) -> None` and `unfreeze_backbone_stages(unfreeze_stages: list[int]) -> None` no-op stubs to `StudentModel` in `src/models/student/student_model.py` with docstrings containing "no-op" and "single-phase".
- [x] 1.3 Create `configs/training_student.yaml` with `model_type: student`, `epochs: 50`, `lr: 0.001`, `backbone_variant: tiny`, no `nir_dir`, `output_dir: checkpoints/student`.

## Phase 2: Core Integration (Loop + Metrics + CLI)

- [x] 2.1 Add `generate_training_curves(history: LossHistory, output_dir: Path) -> None` to `src/training/metrics.py`. Plot 3 subplots (cls_loss, box_loss, total_loss). Guard matplotlib import with try/except. Handle empty history with warning. Catch RuntimeError/ValueError for headless degradation.
- [x] 2.2 In `src/training/loop.py` `_train_epoch()`: add `if self.config.model_type == "student"` branch — call `self.model(rgb)` without NIR. Keep existing `self.model(rgb, nir)` for master.
- [x] 2.3 In `src/training/loop.py` `_validate()`: add same forward-call branching — `model(rgb)` for student, `model(rgb, nir)` for master.
- [x] 2.4 In `src/training/loop.py` `fit()`: when `model_type == "student"`, call `_train_phase(phase=1, epochs=config.epochs, lr=config.lr, freeze_stages=0)` (single-phase). Call `generate_training_curves(self.loss_history, self.output_dir)` after training completes.
- [x] 2.5 In `src/training/train.py`: add `--model {master,student}` CLI flag. When `model_type == "student"`, import and instantiate `StudentModel(num_classes=config.num_classes)` instead of `MasterModel`. Print student param count.

## Phase 3: Testing

- [x] 3.1 Create `tests/test_training_loop.py`. Test: `TrainingConfig(model_type="ensemble")` raises `ValueError`.
- [x] 3.2 Test: `StudentModel.freeze_backbone(3)` and `unfreeze_backbone_stages([3,4])` are no-ops — assert all params still have `requires_grad=True` after call.
- [x] 3.3 Test: `Trainer._train_epoch()` forward branching — mock model.forward, assert called with `(rgb,)` for student and `(rgb, nir)` for master.
- [x] 3.4 Test: `generate_training_curves()` writes PNG to temp dir with 3 subplots. Test empty history logs warning and writes no file. Test matplotlib import failure is graceful.
- [x] 3.5 Integration: run student 2-epoch loop via `train.py --config configs/training_student.yaml --override epochs=2`, assert checkpoint created. Verify master loop with `training_mango.yaml --override epochs_phase1=1` still works (no regression).

## Phase 4: Cleanup

- [x] 4.1 Update `fit()` print header to show "Training StudentModel" or "Training MasterModel" based on `model_type`.
- [x] 4.2 Verify `TrainingConfig.total_epochs` property returns correct value for student (uses `epochs` field) vs master (uses `epochs_phase1 + epochs_phase2`).
