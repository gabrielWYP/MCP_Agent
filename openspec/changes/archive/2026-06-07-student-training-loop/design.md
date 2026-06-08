# Design: Student Training Loop

## Technical Approach

Add `model_type` dispatch to the existing Trainer so it can run both MasterModel (two-phase, RGB+NIR) and StudentModel (single-phase, RGB-only). The approach is minimal branching: keep one Trainer class, add a `model_type` field in `TrainingConfig`, and branch forward calls and phase logic on that field. No class hierarchy change — the existing `fit()` → `_train_phase()` → `_train_epoch()` pipeline stays intact; only the forward signature and freeze/unfreeze calls vary. StudentModel gets no-op freeze stubs for API compatibility. Training curves auto-generate via a new `generate_training_curves()` function in `metrics.py` called at the end of `fit()`.

## Architecture Decisions

| Decision | Options | Tradeoffs | Chosen |
|----------|---------|-----------|--------|
| **Where to branch forward call** | A: Inside `_train_epoch()` / `_validate()` via `if self.config.model_type == "student"` | A: Minimal code, no new classes; single condition per forward path. B: Cleaner separation, but adds boilerplate and two classes for a single if-statement. | **A** — One Trainer, one if-statement. The cost of a subclass is higher than the benefit for a single forward-signature change. |
| **Freeze stub placement** | A: Stub methods on StudentModel (spec) | A: Explicit API contract, polymorphic dispatch, no `hasattr` guards. B: Fewer lines in StudentModel, but leaks Trainer logic into the training loop and breaks polymorphism. | **A** — `StudentModel.freeze_backbone()` and `unfreeze_backbone_stages()` are no-ops. Keeps Trainer code clean: `self.model.freeze_backbone(...)` works for both models. |
| **Training curves integration** | A: `generate_training_curves()` in `metrics.py` | A: Follows existing `metrics.py` pattern, no new file. B: Adds matplotlib import to `metrics.py` (acceptable, guarded). C: Clean separation, but overkill for one function. | **A** — Add `generate_training_curves(history, output_dir)` to `metrics.py`. Call from `Trainer.fit()` after training completes. Guard with try/except for headless servers. |
| **Config handling for student** | A: Separate `TrainingConfig` subclass | A: Clean but duplicates fields. B: Keeps one dataclass, makes NIR fields optional, adds `model_type`. | **B** — One `TrainingConfig` with `model_type: str = "master"`. NIR fields stay but are ignored by student. `backbone_variant` becomes optional (default `"tiny"`). Separate YAML for student hyperparams. |
| **Single-phase vs two-phase code path** | A: Refactor `_train_phase()` to accept a "single-phase" flag | A: Reuses scheduler logic, keeps code DRY. B: Student bypasses `_train_phase()` entirely. | **A** — `_train_phase()` already wraps cosine+warmup + freeze/unfreeze + epoch loop. For student, call `_train_phase(phase=1, epochs=total_epochs, lr=lr, freeze_stages=0)` with `model_type="student"`. Freeze is a no-op, so all params train. |

## Data Flow

```
Student Training Flow

TrainingConfig (model_type="student")
  │
  ▼
train.py ──► StudentModel ──► Trainer
  │                              │
  │                          _train_phase(phase=1,
  │                                        epochs=50,
  │                                        freeze_stages=0)
  │                              │
  │                          _train_epoch()
  │                              │
  │    ┌─────────────────────────────────────┐
  │    │ batch: {rgb, nir, bboxes, labels} │
  │    │   rgb ──► model(rgb)  (NIR ignored)│
  │    │   preds ──► YOLOv8Loss              │
  │    │   backward ──► optimizer.step()       │
  │    └─────────────────────────────────────┘
  │                              │
  │                          _validate()
  │                              │
  │    ┌─────────────────────────────────────┐
  │    │ rgb ──► model(rgb)                  │
  │    │ _decode_predictions() ──► compute_map()│
  │    └─────────────────────────────────────┘
  │                              │
  │                          LossHistory.update()
  │                              │
  │                          generate_training_curves()
  │                              │
  ▼                              ▼
  checkpoints/student/       training_curves.png
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/training/loop.py` | Modify | Add `model_type` branch in `__init__`, `fit()`, `_train_epoch()`, `_validate()`. Forward call: `model(rgb)` for student, `model(rgb, nir)` for master. Call `generate_training_curves()` at end of `fit()`. |
| `src/training/train.py` | Modify | Add `--model {master,student}` CLI flag. Instantiate `StudentModel` when `model_type="student"`. Print student param count. |
| `src/training/config.py` | Modify | Add `model_type: str = "master"`. Make `backbone_variant` optional with default `"tiny"`. Add `epochs: int = 50` (single-phase total) and `lr: float = 1e-3` as student-friendly aliases. Keep `epochs_phase1/2` for backward compat. |
| `src/models/student/student_model.py` | Modify | Add `freeze_backbone()` and `unfreeze_backbone_stages()` no-op stubs with docstrings. |
| `src/training/metrics.py` | Modify | Add `generate_training_curves(history, output_dir)` function. 3-subplot PNG (cls, box, total). Guard matplotlib import and display errors. |
| `configs/training_student.yaml` | Create | Student-specific config: `model_type: student`, `epochs: 50`, `lr: 0.001`, no `nir_dir`, `backbone_variant: tiny` (ignored). |
| `tests/test_training_loop.py` | Create | Unit tests for model_type dispatch, forward call branching, loss history, and curve generation. |

## Interfaces / Contracts

```python
# TrainingConfig additions
@dataclass
class TrainingConfig:
    model_type: str = "master"          # "master" | "student"
    epochs: int = 50                    # single-phase total (student)
    lr: float = 1e-3                    # single-phase LR (student)
    # existing fields remain; epochs_phase1/2/lr_phase1/2 used only for master

# StudentModel stubs
class StudentModel(nn.Module):
    def freeze_backbone(self, freeze_stages: int = 0) -> None:
        """No-op for single-phase student training (API compatibility)."""
        pass

    def unfreeze_backbone_stages(self, unfreeze_stages: list[int]) -> None:
        """No-op for single-phase student training (API compatibility)."""
        pass

# Trainer branching (pseudocode)
def _train_epoch(self, optimizer, epoch, phase):
    for batch in self.train_loader:
        rgb = batch["rgb"].to(self.device)
        if self.config.model_type == "student":
            output = self.model(rgb)
        else:
            nir = batch["nir"].to(self.device)
            output = self.model(rgb, nir)
        # ... loss, backward identical

# Metrics function
def generate_training_curves(history: LossHistory, output_dir: Path) -> None:
    """Generate training_curves.png from LossHistory."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib unavailable, skipping curves")
        return
    # ... 3 subplots, savefig
```

## Testing Strategy

| Layer | What to Test | Approach |
|-------|-------------|----------|
| Unit | `TrainingConfig` rejects invalid `model_type` | `pytest` with `pytest.raises(ValueError)` |
| Unit | `StudentModel.freeze_backbone()` is no-op | `assert all(p.requires_grad for p in model.parameters())` after call |
| Unit | `Trainer._train_epoch()` branches forward correctly | Mock `model.forward` and assert called with `(rgb,)` vs `(rgb, nir)` based on `model_type` |
| Unit | `generate_training_curves()` writes PNG | Temp dir, assert file exists, check shape |
| Unit | Headless server graceful degradation | Mock `matplotlib` import failure, assert no exception |
| Integration | Full student training loop runs 2 epochs | `train.py --config configs/training_student.yaml --override epochs=2` |
| Integration | Master training loop unchanged | Run existing `training_mango.yaml` with `epochs_phase1=1`, verify checkpoint created |

## Migration / Rollout

- No migration required — all changes are additive or branching.
- `model_type` defaults to `"master"`, so existing configs and CLI calls are unchanged.
- Rollback: `git revert` removes student branch logic; master training loop is completely unaffected.

## Open Questions

- [ ] Should `TrainingConfig` validate that `nir_dir` is present when `model_type="master"`? (Currently silent ignore.)
- [ ] Should student training use the same `YOLOv8Loss` or a lighter variant? (Using existing for now; deferred to KD phase.)
- [ ] Should `training_curves.png` include validation mAP overlay? (Deferred per proposal out-of-scope.)
