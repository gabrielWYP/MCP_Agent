# Design: MasterModel Training Loop

## Technical Approach

Build a standalone `src/training/` module with a raw-PyTorch trainer orchestrating 2-phase fine-tuning of the multimodal MasterModel. Phase 1 freezes the entire ConvNeXt backbone and trains fusion+neck+head; Phase 2 unfreezes stages 3–4. A custom `YOLODataset` loads RGB+NIR pairs with YOLO-format labels, applies bbox-aware albumentations, and letterboxes to 640×640. The `YOLOv8Loss` uses a Task-Aligned Assigner (TAL) to match predictions to GT, then applies class-weighted BCE + CIoU. Validation computes mAP@0.5 and mAP@0.5:0.95 after every epoch. The design reuses the existing 2-phase pattern from `ml_pipeline/training/process.py` but keeps detection infrastructure in a clean, separate package.

## Architecture Decisions

| Decision | Choice | Alternatives | Rationale |
|----------|--------|--------------|-----------|
| File structure | `src/training/` new module | Put inside `src/ml_pipeline/training/` | Existing `ml_pipeline/training/` is ConvNeXt classification; mixing detection would couple unrelated domains. |
| Backbone variant | Add `_build_convnext_tiny_body()` + `variant` param | Always use Small | 28M vs 50M params reduces overfitting on 31 images. Channels [96,192,384,768] identical — zero downstream impact. |
| Phase orchestration | Single `Trainer` class switches phases internally | Two separate scripts | Matches existing `process.py` pattern; avoids checkpoint-reload boilerplate and state fragmentation. |
| Confidence filter | Load COCO JSON at dataset init, filter ≥0.5, cache in memory | Regenerate YOLO .txt files | YOLO .txt has no confidence column. Caching avoids repeated JSON parsing and keeps disk labels untouched. |
| Letterbox | Custom cv2 resize + pad in dataset | Albumentations `LongestMaxSize` + pad | Albumentations pad value isn't configurable to 114 gray easily; custom gives exact YOLO semantics. |
| TAL design | Separate `TaskAlignedAssigner` class | Inline inside loss | Testable in isolation; mirrors ultralytics structure; decouples matching from loss computation. |
| mAP implementation | Custom PR-curve + `torchvision.ops.box_iou` | pycocotools or torchmetrics | Avoids new heavy deps; ~60 lines; fits the "from-scratch PyTorch" constraint. |

## Data Flow

```
RGB/NIR images (.jpg)          YOLO labels (.txt) + COCO JSON (confidence)
       |                                |
       └──────── YOLODataset ───────────┘
            → letterbox 640×640
            → albumentations (spatial+photometric, bbox-aware)
            → normalize (ImageNet RGB, computed NIR)
            → return (rgb[3,640,640], nir[1,640,640], bboxes[N,4], labels[N], img_id)
                     |
              collate_fn (stack rgb/nir, keep bboxes/labels as list)
                     |
              MasterModel.forward(rgb, nir)
                     |
              [P3(B,6,80,80), P4(B,6,40,40), P5(B,6,20,20)]
                     |
              YOLOv8Loss
                → flatten preds → TAL assigner (top-k matching)
                → BCE cls loss (class-weighted) + CIoU reg loss
                → total = 7.5×reg + 0.5×cls
                     |
              backward → clip_grad_norm_(10.0) → optimizer step
                     |
              Validation (every epoch)
                → NMS + box decoding → mAP@0.5 / mAP@0.5:0.95
                → TensorBoard logging + early-stopping check
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/training/__init__.py` | Create | Package init |
| `src/training/config.py` | Create | `TrainingConfig` dataclass |
| `src/training/loop.py` | Create | `Trainer` class: 2-phase orchestration, AMP, early stopping, TensorBoard |
| `src/training/dataset.py` | Create | `YOLODataset`, letterbox, albumentations, collate_fn |
| `src/training/loss.py` | Create | `TaskAlignedAssigner`, `YOLOv8Loss` |
| `src/training/metrics.py` | Create | mAP@0.5, mAP@0.5:0.95, per-class AP, loss history |
| `src/training/train.py` | Create | CLI entry point |
| `src/models/master/backbone.py` | Modify | Add `_build_convnext_tiny_body()`; `DualConvNeXtBackbone(variant=...)` |
| `src/models/master/master_model.py` | Modify | Add `unfreeze_backbone_stages()`; accept `backbone_variant` |

## Interfaces / Contracts

### MasterModel unfreeze
```python
def unfreeze_backbone_stages(self, unfreeze_stages: list[int]):
    """Unfreeze specific stages (e.g., [2,3] for stages 3-4). Stems stay frozen."""
```

### TaskAlignedAssigner output
```python
class TaskAlignedAssigner:
    def __call__(self, pred_scores, pred_bboxes, gt_labels, gt_bboxes, anchors, strides):
        # Returns: target_classes (B, 8400), target_bboxes (B, 8400, 4),
        #          target_scores (B, 8400), fg_mask (B, 8400)
```

### Dataset output contract
```python
# __getitem__ returns:
rgb:   Tensor[3, 640, 640]   # ImageNet normalized
nir:   Tensor[1, 640, 640]   # dataset-specific normalized
bboxes: Tensor[N, 4]          # cxcywh in [0,1]
labels: Tensor[N]            # int64 class ids
img_id: int
```

### Loss aggregation
```python
total_loss = config.box_weight * ciou_loss + config.cls_weight * bce_loss
# Averaged over batch. Scalar output for backward().
```

## Testing Strategy

| Layer | What to Test | Approach |
|-------|-------------|----------|
| Unit | TAL assigner with known inputs/outputs | pytest: fixed pred/GT tensors, assert exact positive/negative masks |
| Unit | Letterbox resize + bbox rescaling | pytest: synthetic 1280×720 image, assert output 640×640 and bbox offsets |
| Unit | mAP computation | pytest: hand-crafted predictions/GT with known AP values |
| Integration | Full training step (forward → loss → backward) | Script: single batch through `Trainer.train_step()` with AMP on/off |
| E2E | 2-phase loop on tiny synthetic data | Script: 2 images, 2 epochs per phase, assert checkpoint saved and loss decreases |

## Migration / Rollout

No migration required. All changes are additive. The new `src/training/` package is independent; model modifications (`backbone.py`, `master_model.py`) are backward-compatible (Tiny variant is opt-in via `backbone_variant="tiny"`).

## Open Questions

- [ ] NIR normalization stats: compute real mean/std on the 31 NIR images before training, or use the existing fallback `[0.45, 0.22]`?
- [ ] Class-weighted sampling for imbalance: implement `WeightedRandomSampler` or defer to loss weighting only?
- [ ] Should the confidence filter be baked into the pre-generated YOLO .txt files instead of runtime COCO lookup?
