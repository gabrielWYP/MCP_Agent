## Exploration: student-training-loop

### Summary

We are adding a StudentModel-only (no distillation) training loop to the existing codebase. The StudentModel (YOLOv8-Nano / CSPDarknetNano) is an RGB-only single-input model, but the current Trainer, dataset, config, and CLI are all hardwired for the dual-input MasterModel (ConvNeXt-Tiny, RGB+NIR). Three hard incompatibilities exist: (1) the Trainer calls `model(rgb, nir)` but Student only takes `model(x)`, (2) the Trainer calls `model.freeze_backbone()` and `model.unfreeze_backbone_stages()` — methods that do not exist on StudentModel, and (3) the dataset always loads NIR images that the Student does not consume, wasting I/O and memory. The YOLOv8Loss, metrics, and augmentations are fully model-agnostic and require no changes.

### Files that need changes

| File | Change | Effort |
|------|--------|--------|
| `src/training/loop.py` | Branch forward call: `model(rgb, nir)` vs `model(rgb)`, handle freeze/unfreeze dispatch (conditional or polymorphism), update `_validate` same way | Medium |
| `src/training/train.py` | Add `--model {master,student}` CLI flag, import and instantiate `StudentModel`, remove `backbone_variant` print when using student | Low |
| `src/training/config.py` | Add `model_type: str = "master"` field, make `nir_dir`, `nir_mean`, `nir_std`, `backbone_variant` conditionally optional (has_default or only required for master) | Low |
| `src/models/student/student_model.py` | Add `freeze_backbone(freeze_stages: int)` and `unfreeze_backbone_stages(unfreeze_stages: list[int])` methods. Map stages: stem (always frozen?), stage1=S1, stage2=S2, stage3=S3, stage4+sppf=S4 | Medium |
| `src/training/dataset.py` | Either: (a) add `include_nir: bool = True` parameter to `YOLODataset.__init__` and `collate_fn`, or (b) accept the overhead and just ignore `nir` in the batch (simpler, ~30-50% wasted I/O per image) | Low if option (b), Medium if option (a) |
| `configs/` | New `configs/training_student.yaml` with `model_type: student`, no NIR dirs, student-appropriate hyperparams | Low |

### Files that DON'T need changes

| File | Reason |
|------|--------|
| `src/training/loss.py` | `YOLOv8Loss.forward(predictions: list[Tensor], targets: dict)` — fully agnostic to model type. Both Master and Student produce `preds` as `[(B, nc+4, H_i, W_i)]`. |
| `src/training/metrics.py` | `compute_map()` operates on decoded pred_boxes/scores/labels — model-agnostic. |
| `src/training/augmentations.py` | `get_train_transforms()` and `get_val_transforms()` only transform RGB images (NIR gets letterbox only). No model-specific logic. |
| `src/models/student/backbone.py` | `CSPDarknetNano` works as-is — freeze/unfreeze methods belong at `StudentModel` level, not here. |
| `src/models/student/neck.py` | `PANet` — no changes needed. |
| `src/models/student/head.py` | `YOLOStudentHead` — already produces `preds` in Master-compatible format. |
| `src/models/__init__.py` | Already exports both `MasterModel` and `StudentModel`. |
| `src/models/master/` | No changes. Master training loop unaffected. |

### Risks / Gotchas

1. **Freeze semantics mismatch**: MasterModel freezes ConvNeXt "stages" with a simple numeric index. StudentModel's CSPDarknetNano has a different architecture (stem, stage1-4, sppf). Need a consistent 0-based stage model: `freeze_stages=4` should freeze stem+stage1+stage2+stage3+stage4 (the entire backbone). The Trainer currently passes `freeze_stages=4` for Phase 1 and `freeze_stages=2` for Phase 2. **Decision needed**: do we keep the 2-phase approach for student (freeze early CSPDarknet stages, then unfreeze stage3-4) or simplify to single-phase?

2. **NIR in batch dict**: If we skip NIR loading in the dataset, the batch dict will be missing the `nir` key. The current Trainer accesses `batch["nir"]` unconditionally. If we keep loading NIR (option b), we waste memory but avoid conditional logic in the Trainer. Given the dataset is small (~24 images), the overhead is negligible. Recommendation: **option (b) — keep loading NIR, just don't pass it to the student model**.

3. **Phase 2 may not apply**: Student YOLOv8-Nano is already small (~3M params). Freezing early stages and unfreezing later stages may not provide the same benefit as with the large ConvNeXt backbone. A single-phase training with a flat LR schedule might be more appropriate.

4. **Output dict key difference**: Master produces 8 output keys, Student produces 7. The Trainer only accesses `output["preds"]` for loss and `output["cls_preds"]` for decoding. Both models have these keys. The extra distillation keys (`distill_backbone_*`, etc.) are only used in a distillation pipeline — not relevant here.

5. **AMP NaN**: The current config has `amp: false` because YOLOv8Loss produces NaN with FP16. This affects both models equally and is already handled.

### Recommendations

1. **Minimal viable approach**: The simplest path is to add a `model_type` parameter to `Trainer.__init__` (or `TrainingConfig`), branch exactly two places in the Trainer (`_train_epoch` and `_validate` forward calls), add freeze/unfreeze methods to StudentModel (even if single-phase — for API compatibility), and accept the NIR I/O overhead (dataset stays as-is).

2. **Freeze/unfreeze for Student**: Implement as:
   ```python
   def freeze_backbone(self, freeze_stages: int = 0):
       modules = [self.backbone.stem, self.backbone.stage1, self.backbone.stage2,
                  self.backbone.stage3, self.backbone.stage4, self.backbone.sppf]
       for i, mod in enumerate(modules):
           if i < freeze_stages:
               for p in mod.parameters(): p.requires_grad = False
   ```
   This maps `freeze_stages=0..6` (stem + 4 stages + sppf). For Phase 1 with `freeze_stages=4`: freezes stem, stage1, stage2, stage3 — leaves stage4+sppf+neck+head trainable.

3. **CLI**: Add `--model student` flag. If omitted, default to `master` for backward compatibility.

4. **Config**: Add `model_type: str = "master"` to `TrainingConfig`. When `model_type == "student"`, `nir_dir`, `nir_mean`, `nir_std`, `backbone_variant` are ignored. The `from_yaml` classmethod handles this naturally since those keys just won't be in the student YAML.

5. **Testing**: Because `test_arch.py` is currently broken (level mismatch), there is no test infrastructure to validate this change. The verification phase will need at minimum a smoke test that instantiates StudentModel + Trainer + one batch and verifies no crash.

### Ready for Proposal
Yes — the exploration is complete. The core approach (branch-forward-call + add-freeze-methods + accept-NIR-overhead) is well-defined with manageable risk. Proceed to `sdd-propose`.
