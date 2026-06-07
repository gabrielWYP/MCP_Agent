# Proposal: MasterModel Training Loop

## Intent

Train the multimodal MasterModel (RGB+NIR → DualConvNeXt → CrossModalFusion → DualFPN → YOLODetectionHead) on 31 mango damage images (214 bboxes) to produce a working detection pipeline. No training infrastructure exists yet — we must build the full stack from dataset to loss function to training loop.

## Scope

### In Scope
- DetectionDataset: YOLO-format label loader + bbox-aware albumentations + letterbox
- YOLOv8Loss: BCE (class-weighted) + CIoU + TAL assigner
- Training loop: 2-phase fine-tuning (Phase 1: frozen backbone, Phase 2: unfreeze stages 3-4)
- mAP evaluation (mAP@0.5, mAP@0.5:0.95, per-class AP)
- Checkpointing, TensorBoard logging, early stopping
- ConvNeXt-Tiny backbone variant (alongside current Small)
- Annotation confidence filter (threshold ≥ 0.5)

### Out of Scope
- Inference/prediction script (model.forward already works)
- ONNX/TensorRT export
- Deployment or serving
- Active learning / annotation refinement
- Hyperparameter tuning beyond sensible defaults
- DFL (Distribution Focal Loss) — deferred to v2
- Mosaic augmentation — requires ≥4 images, risky with 24 train samples

## Capabilities

### New Capabilities
- `training-loop`: 2-phase fine-tuning trainer with AMP, gradient clipping, cosine+warmup LR, early stopping
- `yolo-dataset`: DetectionDataset loading YOLO-format annotations with RGB+NIR pairs + bbox-aware transforms
- `yolo-loss`: YOLOv8-style loss (TAL assigner + class-weighted BCE + CIoU regression)
- `training-metrics`: mAP computation, loss curves, per-class AP, TensorBoard logging

### Modified Capabilities
- None. Model changes (ConvNeXt-Tiny variant, unfreeze_all) are implementation details within training-loop scope, not requirement changes to existing specs.

## Approach

**From-scratch PyTorch training loop** — no ultralytics dependency. Our multimodal input (RGB+NIR) doesn't fit ultralytics' single-image pipeline. Key decisions:

1. **Backbone**: Add ConvNeXt-Tiny option (28M vs 50M params). Smaller capacity = less overfitting on 31 images. Make it configurable — both Small and Tiny remain available.
2. **Dataset**: YOLO format (normalized cxcywh per image, already generated). Letterbox resize to 640×640 preserving aspect ratio.
3. **Loss**: TAL assigner → positive/negative assignment → class-weighted BCE (addressing 1:5.9 sano:danado imbalance) + CIoU regression. torchvision.ops.complete_box_iou_loss available.
4. **Filter**: Drop annotations below confidence 0.5 (removes ~20% noisy auto-annotations).
5. **Phase 1**: Freeze entire backbone (stems + stages 1-4). Train fusion + neck + head (~18.8M params), lr=1e-3, 10-20 epochs.
6. **Phase 2**: Unfreeze stages 3-4. Train full model, lr=1e-4, 20-40 epochs.
7. **Regularization**: Heavy augmentation (flip, rotate, scale, color jitter RGB-only), dropout (0.3-0.5 in fusion/neck), weight decay 1e-4, early stopping (patience=15).

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/training/` | New | Training package: trainer, loss, dataset, metrics, config |
| `src/models/master/backbone.py` | Modified | Add ConvNeXt-Tiny builder function |
| `src/models/master/master_model.py` | Modified | Add unfreeze_all(), configurable backbone variant |
| `data/annotations/yolo/` | Read | YOLO-format labels (already generated) |
| `data/cache/mango/` | Read | RGB+NIR raw images (already exist) |
| `checkpoints/` | New | Model checkpoint storage |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| 31 images → severe overfitting | High | 2-phase freeze, heavy augmentation, dropout, early stopping. This is proof-of-concept — accept limited generalization |
| Auto-annotation noise (conf 0.37-0.98) | Medium | Filter conf < 0.5; ~20% bboxes dropped, eliminating edge false positives |
| NIR black borders bias model | Medium | Monitor per-class AP; letterbox padding is neutral, borders are outside annotation regions |
| TAL assigner implementation bugs | Medium | Unit test with known inputs/outputs before training; reference ultralytics implementation |
| RTX 3080 OOM (10GB) | Low | AMP + batch=4 fits in ~700MB; gradient checkpointing as fallback |

## Rollback Plan

Training infrastructure is entirely new code under `src/training/`. Model modifications are additive (new backbone variant + unfreeze_all). Rollback: `git revert` the training PR; MasterModel remains unchanged. No data is modified.

## Dependencies

- Exploration: `sdd/mastermodel-training-loop/explore` (completed)
- YOLO-format annotations (already generated in `data/annotations/yolo/`)
- torchvision 0.27+ (for `complete_box_iou_loss`)

## Success Criteria

- [ ] Loss decreases monotonically on train set over Phase 1
- [ ] Phase 1 achieves val mAP@0.5 ≥ 0.3 (4-image val is high-variance)
- [ ] Phase 2 improves mAP over Phase 1
- [ ] Predictions on sample images show plausible detections
- [ ] Training loop completes without OOM on RTX 3080 with batch=4 + AMP