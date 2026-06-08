# Design: Cross-Modal Knowledge Distillation Training (kd-training)

## Technical Approach

Subclass `Trainer` into `KDTrainer` that loads a frozen `MasterModel` teacher and attaches four `ProjectionLayers` groups to the `StudentModel` as submodules. Override `__init__` (teacher load + projection attachment) and `_train_epoch` (teacher forward → project → MSE KD loss + detection loss). All other machinery — fit, validate, checkpoint, early stopping, logging — is inherited unchanged.

## Architecture Decisions

| Decision | Options | Tradeoffs | Chosen |
|----------|---------|-----------|--------|
| Teacher storage | `self.teacher` separate vs `self.model.teacher` | Separate: excluded from optimizer/checkpoint automatically. Submodule: requires manual filtering. | Separate `self.teacher` |
| Projection attachment timing | Before `super().__init__` vs after | Before: projections moved to device by `self.model.to(device)`. After: need manual `.to(device)`. | Before `super().__init__` |
| KDLoss scope | Pure KD MSE only vs combined with YOLOv8Loss | Pure: cleaner separation, reusable. Combined: single loss object. | Pure KDLoss only |
| KDConfig inheritance | `KDConfig(TrainingConfig)` vs composition | Inheritance: reuses `from_yaml`, `__post_init__`. Composition: more explicit. | Inheritance |
| Batch size | 2 (conservative) vs 4 (user request) | 2: safer OOM. 4: faster training, ~3 GB VRAM. | 4 (YAML-configurable, fallback to 2) |

## Data Flow

```
Batch (rgb, nir, bboxes, labels)
    │
    ├─► Teacher (no_grad) ──► rgb+nir ──► features [4 levels]
    │                              │
    │                              ▼
    │                    ProjectionLayers (1×1 conv)
    │                              │
    │                              ▼
    │                    Projected teacher features
    │                              │
    ▼                              │
Student (train) ◄─────────────────┘
    │
    ├─► Detection loss (YOLOv8Loss)
    └─► KD loss (MSE per level)
              │
              ▼
        total = det + kd_weight × Σ(w_level × MSE)
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/training/kd_config.py` | Create | `KDConfig(TrainingConfig)` with KD fields and overridden defaults |
| `src/training/kd_loss.py` | Create | `KDLoss(nn.Module)` — MSE per level with temperature |
| `src/training/kd_trainer.py` | Create | `KDTrainer(Trainer)` — teacher load, projection attach, KD forward |
| `src/training/kd_train.py` | Create | CLI entry point mirroring `train.py` |
| `configs/kd_training.yaml` | Create | KD hyperparameters with batch_size=4 |

## Component Design

### KDConfig

```python
@dataclass
class KDConfig(TrainingConfig):
    teacher_checkpoint: str
    kd_weight: float = 1.0
    kd_temperature: float = 1.0
    distill_levels: dict = field(default_factory=lambda: {
        "backbone": 1.0, "fpn": 0.1, "head_cls": 0.3, "head_reg": 0.5
    })
    epochs: int = 80
    patience: int = 30
    amp: bool = False
    batch_size: int = 4
    model_type: str = "student"
```

`from_yaml` inherited from `TrainingConfig` — `KDConfig(**data)` uses dataclass defaults for missing fields.

### KDLoss

```python
class KDLoss(nn.Module):
    def __init__(self, level_weights: dict, temperature: float = 1.0): ...
    def forward(self, teacher: dict, student: dict) -> tuple[Tensor, dict]:
        # teacher/student: {"backbone": [T], "fpn": [T], "head_cls": [T], "head_reg": [T]}
        # returns: total_kd_loss, {"backbone": float, ...}
```

### KDTrainer

```python
class KDTrainer(Trainer):
    def __init__(self, model, config, train_loader, val_loader):
        # Attach projections BEFORE super()
        model.kd_proj_backbone = backbone_projections()
        model.kd_proj_fpn = fpn_projections()
        model.kd_proj_head_cls = head_projections()
        model.kd_proj_head_reg = head_projections()
        super().__init__(model, config, train_loader, val_loader)
        self.teacher = self._load_teacher(config)
        self.kd_criterion = KDLoss(config.distill_levels, config.kd_temperature)

    def _train_epoch(self, optimizer, epoch, phase):
        # teacher no_grad → project → student → det_loss + kd_loss
        # NaN guard, backward, clip_grad — same pattern as base
```

### CLI Entry Point

Mirrors `train.py` — `KDConfig.from_yaml`, `StudentModel`, `KDTrainer.fit()`.

## Configuration Design

| Field | Type | Default | Source |
|-------|------|---------|--------|
| teacher_checkpoint | str | required | YAML |
| kd_weight | float | 1.0 | YAML |
| kd_temperature | float | 1.0 | YAML |
| distill_levels | dict | {backbone:1.0, fpn:0.1, head_cls:0.3, head_reg:0.5} | YAML |
| epochs | int | 80 | overridden from TrainingConfig(50) |
| patience | int | 30 | overridden from TrainingConfig(15) |
| amp | bool | false | overridden from TrainingConfig(true) |
| batch_size | int | 4 | overridden from TrainingConfig(2) |
| model_type | str | "student" | overridden from TrainingConfig("master") |

## Testing Strategy

| Layer | Test | Approach |
|-------|------|----------|
| Unit | KDConfig.from_yaml with partial YAML | Assert defaults applied |
| Unit | KDLoss.forward with dummy tensors | Assert MSE per level matches manual compute |
| Unit | KDTrainer.__init__ teacher loading | Mock checkpoint, assert frozen, assert not in optimizer |
| Integration | KDTrainer._train_epoch single batch | Assert loss dict contains kd_loss, no gradient in teacher |
| Integration | Checkpoint save/load | Assert teacher absent, projections present |

## Migration & Rollout

No migration required. Zero existing files modified. Rollback: delete the 5 new files. Existing `train.py` and `Trainer` remain unaffected.

## Open Questions

- None — all architectural questions resolved in exploration.
