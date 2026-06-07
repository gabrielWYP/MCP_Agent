# Tasks: MasterModel Training Loop

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | 950–1150 |
| 400-line budget risk | High |
| Chained PRs recommended | Yes |
| Suggested split | PR 1 (foundation ~230) → PR 2 (data+loss+metrics ~460) → PR 3 (training loop ~320) |
| Delivery strategy | single-pr |
| Chain strategy | feature-branch-chain |

Decision needed before apply: Yes
Chained PRs recommended: Yes
Chain strategy: feature-branch-chain
400-line budget risk: High

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | Config, augmentations, backbone Tiny, model variant | PR 1 | Base: feature/mastermodel-training-loop; standalone testable |
| 2 | Dataset, letterbox, loss, metrics | PR 2 | Base: PR 1 branch; depends on config + augmentations |
| 3 | TrainingLoop, train.py, integration test | PR 3 | Base: PR 2 branch; depends on all above |

## Phase 1: Foundation

- [x] 1.1 Create `src/training/__init__.py` with public imports
- [x] 1.2 Create `src/training/config.py` — `TrainingConfig` dataclass (image_size, batch_size, phases, lr, amp, NIR stats, loss weights, early stopping)
- [x] 1.3 Create `src/training/augmentations.py` — `build_train_transform()` and `build_val_transform()` with bbox-aware albumentations (HFlip, VFlip, Rotate, ShiftScaleRotate, color jitter RGB-only)
- [x] 1.4 Modify `src/models/master/backbone.py` — add `_build_convnext_tiny_body()` and `variant` param to `DualConvNeXtBackbone.__init__`; Tiny stages channels `[96,192,384,768]`
- [x] 1.5 Modify `src/models/master/master_model.py` — add `backbone_variant` param, `unfreeze_backbone_stages(unfreeze_stages)` method; pass variant through to backbone

## Phase 2: Data Pipeline

- [x] 2.1 Create `src/training/dataset.py` — letterbox function (cv2 resize+pad to 640×640, pad=114, bbox scale+offset)
- [x] 2.2 Add `YOLODataset` class — load RGB+NIR pairs by stem substitution, parse YOLO .txt labels, filter by COCO confidence ≥0.5, apply augmentations + letterbox + normalize
- [x] 2.3 Add `collate_fn` and `WeightedRandomSampler` factory (inverse frequency weights: sano≈0.45, danado≈2.7)
- [x] 2.4 Unit test: letterbox output shape + bbox rescaling correctness (`tests/training/test_letterbox.py`)

## Phase 3: Loss

- [x] 3.1 Create `src/training/loss.py` — `TaskAlignedAssigner` (alignment metric, top-k=10, zero GT → all negative)
- [x] 3.2 Add `YOLOv8Loss` class — flatten multi-level preds, call TAL, compute class-weighted BCE + CIoU, aggregate `7.5×reg + 0.5×cls`
- [x] 3.3 Unit test: TAL assigner with synthetic pred/GT → assert fg_mask and target shapes

## Phase 4: Training Loop

- [x] 4.1 Create `src/training/metrics.py` — `compute_map_at_iou()`, `compute_map_50_95()`, per-class AP, `LossHistory`
- [x] 4.2 Unit test: mAP with hand-crafted predictions/GT → assert known AP values
- [x] 4.3 Create `src/training/loop.py` — `TrainingLoop` class (`train_step`, `val_step`, `train()`, `validate()`); Phase 1/2 switch, AMP, gradient clipping (max_norm=10.0), TensorBoard logging, early stopping (patience=15), best checkpoint save
- [x] 4.4 Create `src/training/train.py` — CLI entry point (argparse → TrainingConfig → DataLoader → TrainingLoop)

## Phase 5: Integration

- [x] 5.1 Integration test: full `train_step()` forward→loss→backward with AMP on and off (`tests/training/test_training_step.py`)
- [x] 5.2 E2E test: 2-phase loop on 2 synthetic images, 2 epochs per phase → assert checkpoint saved and loss decreases