## Exploration: mastermodel-training-loop

### Current State

MasterModel is a fully implemented multimodal YOLO-style detector (68.3M params, all trainable):

```
RGB(3,H,W) + NIR(1,H,W)
  → DualConvNeXtBackbone (shared weights, separate stems) → [S1..S4] × 2 streams
  → CrossModalFusion (cross-attention, Q=RGB, K/V=NIR) → [F1..F4]
  → DualFPN (two FPNs → per-level 1×1 conv fusion) → [P3, P4, P5]
  → YOLODetectionHead (decoupled cls+reg per level) → [(B,6,H_i,W_i)] × 3 levels
```

**Output shape per level**: `(B, 2 cls + 4 bbox, H/stride, W/stride)` — matching YOLOv8 convention.
Total anchor-free cells: 8,400 per 640×640 image (P3: 6400 + P4: 1600 + P5: 400).

Strides: [8, 16, 32]. Classes: `0=sano`, `1=danado`.

**Parameter breakdown:**

| Module | Params | % |
|--------|--------|---|
| Backbone (ConvNeXt-Small, dual stems) | 49,454,976 | 72.4% |
| CrossModalFusion (cross-attention × 4 stages) | 9,421,920 | 13.8% |
| DualFPN (two FPNs + fusion convs) | 5,855,488 | 8.6% |
| YOLODetectionHead (decoupled, 3 levels) | 3,545,106 | 5.2% |
| **Total** | **68,277,490** | 100% |

**`freeze_backbone()` behavior**: Always freezes both stems (rgb_stem, nir_stem) unconditionally, then freezes first `freeze_stages` shared stages (default=2). Leaves stages 3-4 + fusion + neck + head trainable.

**Environment**: torchvision 0.27 has `complete_box_iou_loss`, `generalized_box_iou_loss`, `box_iou`, and `nms` available. albumentations 2.0.8 installed. PyTorch 2.12. CUDA available on target GPU (RTX 3080 10GB), CPU-only on dev machine.

---

### Affected Areas

Files that will be created or modified:

- `src/training/` — **NEW** — training loop package (trainer, loss, dataset, metrics)
  - `src/training/loss.py` — YOLOv8 loss function (CIoU + BCE + TAL assigner)
  - `src/training/dataset.py` — DetectionDataset (RGB+NIR pair + bboxes)
  - `src/training/trainer.py` — Training loop with 2-phase fine-tuning
  - `src/training/metrics.py` — mAP computation (COCO-style)
  - `src/training/config.py` — Training hyperparameters
- `src/models/master/master_model.py` — May add `unfreeze_all()` method (currently missing, only has `freeze_backbone()`)
- `data/annotations/coco/` — Read COCO JSON for training data (already exists, no changes needed)
- `data/cache/mango/` — Read raw RGB+NIR image pairs (already exists, no changes needed)
- `checkpoints/` — **NEW** — Model checkpoint storage

---

### 1. MasterModel Trainability

**Freeze strategy**: Two-phase fine-tuning is critical for 31 images.

| Phase | Frozen | Trainable | Epochs | LR |
|-------|--------|-----------|--------|-----|
| Phase 1 | Backbone (all stems + stages 1-4) | Fusion + Neck + Head (~18.8M) | 10-20 | 1e-3 |
| Phase 2 | Backbone stems (always) | Everything else (~68M) | 20-40 | 1e-4 |

**Why this split**: ConvNeXt-Small has 49.5M parameters with ImageNet-pretrained weights. Training them on 24 images would destroy those features. Phase 1 gives the fusion, neck, and head time to converge on the task-specific patterns. Phase 2 fine-tunes backbone stages 1-4 at very low LR.

**`freeze_backbone()` issue**: The current implementation always freezes BOTH stems unconditionally and never unfreezes them. There's no `unfreeze_all()` method. Need to add one, or implement freeze logic with configurable granularity.

**Memory estimate (RTX 3080 10GB)**:
- fp32 weights: 273 MB
- AdamW states: 546 MB
- Gradients: 273 MB
- Activations (batch=2, 640×640): ~91 MB
- **Total fp32 (batch=2): ~1.18 GB**
- With AMP (automatic mixed precision): roughly 60% reduction → ~700 MB
- **Projected max batch size**: 8-12 with AMP, 4-6 without

---

### 2. Annotation Format

**COCO format** (data/annotations/coco/annotations_train.json):
- Categories: `{1: "mango", 2: "mango_damaged"}` → map to 0=sano, 1=danado
- Bboxes: `[x, y, w, h]` in absolute pixels (1280×720 images)
- Confidence: 0.375–1.0 (auto_mango=1.0, auto_nir=0.375–0.981)
- Segmentation: empty (not available)

**YOLO format** (data/annotations/yolo/labels/train/*.txt):
- One .txt per image, same filename as RGB image
- Each line: `class_id cx cy w h` — all normalized to [0,1]
- Example: `1 0.303125 0.176389 0.026562 0.075000`

**Class distribution (all splits)**:

| Split | Images | sano (0) | danado (1) | Total bboxes | Ratio |
|-------|--------|----------|------------|-------------|-------|
| Train | 24 | 24 | 129 | 153 | 1:5.4 |
| Val | 4 | 4 | 34 | 38 | 1:8.5 |
| Test | 3 | 3 | 20 | 23 | 1:6.7 |
| **All** | **31** | **31** | **183** | **214** | **1:5.9** |

**Recommendation**: Use YOLO format for training. It's simpler (1 file per image, normalized coords), no need to parse JSON, and the normalized format is invariant to image resizing. COCO format can be used for final mAP computation using pycocotools.

**Image dimensions**: All images are 1280×720. Training at 640×640 requires rescaling bboxes: `scale_x = 640/1280 = 0.5`, `scale_y = 640/720 ≈ 0.889`. This is non-square — we should **letterbox** (pad to square) or **stretch** to 640×640. Letterbox preserves aspect ratio better but adds padding.

---

### 3. YOLOv8 Loss Function

**Components needed**:

1. **Task-Aligned Assigner (TAL)**: Matches predictions to ground truth based on alignment metric:
   ```
   align_score = sqrt(cls_score × iou_score)  # per anchor-GT pair
   ```
   Selects top-k predictions per GT based on this score. Critical for convergence.

2. **Classification loss**: `BCEWithLogitsLoss` applied to assigned positive and negative samples. YOLOv8 predicts class probabilities per cell (not softmax across cells).

3. **Regression loss**: **CIoU (Complete IoU)** — `torchvision.ops.complete_box_iou_loss` is available. Includes center distance + aspect ratio penalties beyond IoU.

4. **Distribution Focal Loss (DFL)** — **SKIP for v1**. DFL predicts bbox edges as distributions. Adds complexity without being critical for initial training. Only necessary for competitive mAP.

**Implementation approach**: Write from scratch referencing ultralytics implementation. Key modules:
- `BatchTargetAssigner`: Implements TAL — takes (pred_cls, pred_bbox, gt_bbox, gt_cls) → (target_cls, target_bbox, fg_mask)
- `YOLOv8Loss`: Combines cls_loss + bbox_loss with configurable weights

**Available primitives in torchvision 0.27**:
- `torchvision.ops.complete_box_iou_loss(boxes1, boxes2)` — CIoU loss
- `torchvision.ops.box_iou(boxes1, boxes2)` — IoU matrix
- `torchvision.ops.generalized_box_iou_loss(boxes1, boxes2)` — GIoU loss (fallback)
- `torchvision.ops.nms(boxes, scores, iou_threshold)` — for post-processing

**Coordinate convention**: YOLOv8 uses `[cx, cy, w, h]` in grid-cell normalized coordinates. We'll need:
- `pred_bbox` in `[cx, cy, w, h]` format from the head (raw outputs)
- `gt_bbox` in same format, normalized to each grid level's scale

---

### 4. Dataset Design

**DetectionDataset** — returns per sample:
```python
{
    "rgb": Tensor[3, 640, 640],      # normalized RGB
    "nir": Tensor[1, 640, 640],      # normalized NIR
    "bboxes": Tensor[N, 4],          # [cx, cy, w, h] normalized to [0,1]
    "class_ids": Tensor[N],          # 0 or 1
    "image_id": int,                 # for evaluation
    "orig_size": (1280, 720),        # for rescaling predictions back
}
```

**Design decisions**:
- **Bbox-aware transforms**: albumentations supports `bbox_params` with format `'yolo'` (normalized cxcywh). All spatial transforms (flip, rotate, crop) will adjust bboxes automatically.
- **Image pairing**: NIR images share the same filename as RGB. Dataset loads `data/cache/mango/rgb/{name}.jpg` and `data/cache/mango/nir/{name}.jpg`.
- **Label loading**: Read from `data/annotations/yolo/labels/{split}/{name}.txt`.
- **Train augmentations**: Mosaic (optional, risky with 24 images), random resize crop, horizontal/vertical flip, rotation, shift/scale/rotate, color jitter (RGB only).
- **Val augmentations**: Letterbox resize to 640×640 + normalize only.
- **Normalization**: Use ImageNet stats for RGB `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`, NIR estimation `[0.45]` / `[0.22]`. Compute actual NIR stats from cached images.

**Letterbox logic**: Resize the longer side to 640, pad the shorter side to 640 with value 114 (gray). Adjust bboxes accordingly by scaling and shifting. This preserves aspect ratio — critical for the small dataset where stretching could distort damage patterns.

---

### 5. Training Infrastructure

**Patterns to reuse from `src/ml_pipeline/training/process.py`**:
- Cosine decay scheduler with linear warmup (`_build_scheduler`)
- Phase-based training with different LR per phase
- Checkpoint saving with model_state + config + metrics
- History tracking as JSON
- Best-model selection based on validation metric

**New additions needed for detection training**:

| Component | Implementation |
|-----------|---------------|
| **Validation metric** | mAP@0.5 and mAP@0.5:0.95 (COCO-style). Compute with pycocotools or custom. |
| **Checkpoint format** | `{model_state, config, epoch, val_metrics, optimizer_state, scheduler_state}` |
| **Logging** | TensorBoard via `torch.utils.tensorboard.SummaryWriter` — scalar metrics, image grids with bbox overlays, learning rate |
| **Early stopping** | Stop if val mAP@0.5 doesn't improve for N epochs (default=15). Restore best checkpoint. |
| **Learning rate** | Cosine decay with linear warmup (reuse `_build_scheduler`), one-cycle LR also good for small datasets |
| **Gradient clipping** | `nn.utils.clip_grad_norm_(max_norm=10.0)` — important for YOLO training stability |
| **Mixed precision** | `torch.cuda.amp.autocast` + `GradScaler` — essential for 10GB VRAM |
| **Gradient checkpointing** | Optional — `torch.utils.checkpoint` — trades compute for memory if needed |

**Training config (defaults)**:
```python
epochs: 60
batch_size: 4  # with AMP on 10GB
image_size: 640
lr_phase1: 1e-3  # head + neck + fusion
lr_phase2: 1e-4  # full model
weight_decay: 1e-4
warmup_epochs: 3
label_smoothing: 0.0  # BCE already provides regularization
grad_clip: 10.0
amp: True
```

---

### 6. Existing Code to Leverage

| Source | What to reuse |
|--------|---------------|
| `src/ml_pipeline/training/process.py` | `_build_scheduler()` (cosine+warmup), `_save_checkpoint()` pattern, phase-based training loop, history tracking |
| `src/ml_pipeline/training/dataset.py` | RGB+NIR loading with homography warp, ImageNet normalization stats, albumentations Compose patterns, `compute_nir_stats()` |
| `src/models/master/master_model.py` | `freeze_backbone()` method (needs `unfreeze_all()` addition), `count_parameters()` |
| `src/models/master/sanity_check.py` | Verifies all output shapes — can be used as architecture validation before training |
| `data/annotations/yolo/labels/` | Ready YOLO-format annotations, just need to load |
| `data/cache/mango/` | Preprocessed RGB+NIR image pairs (paired by filename) |

**What doesn't exist yet and needs to be built from scratch**:
- YOLOv8 loss function (CIoU + BCE + TAL assigner)
- DetectionDataset with bbox-aware transforms
- mAP computation (COCO evaluation protocol)
- Detection-specific training loop (per-level predictions, batched assigner)

---

### 7. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **31 images is critically small** — overfitting guaranteed | 🔴 HIGH | Extreme regularization: heavy dropout (0.3-0.5), strong augmentations, freeze backbone in phase 1, weight decay, early stopping. Consider synthetic data generation (cut-paste, mosaic, GAN). Accept that model may not generalize — this is proof-of-concept training. |
| **NIR annotations have variable quality** (confidence 0.375–0.981) | 🟡 MEDIUM | Filter annotations below confidence threshold (e.g., 0.5). 215 → ~170 bboxes. Train with confidence-weighted loss if cleaning is too aggressive. Low-confidence annotations are mostly tiny patches on edges — likely false positives from the auto-annotator. |
| **Class imbalance 1:5.9 (sano:danado)** | 🟡 MEDIUM | YOLO handles this better than classification (BCE per cell, not softmax). Still, use class-weighted BCE loss (danado weight=0.2, sano weight=0.8) or focal loss. The "sano" class is basically the whole mango — easy to learn. |
| **NIR black borders (~33% of image)** | 🟡 MEDIUM | The auto-annotations were generated from the visible portion of the NIR image. If the NIR has black borders (no overlap area), those regions produce no damage annotations. The model may learn that "no damage" regions correspond to black areas. Mitigation: verify annotation coverage vs border regions, possibly mask out border pixels during training. |
| **RGB+NIR memory at 640×640** | 🟢 LOW | ~1.2GB fp32, ~700MB AMP for batch=2. 10GB comfortably fits batch 4-8. If tight, enable gradient checkpointing on the backbone. |
| **Letterbox padding creates empty cells** | 🟢 LOW | YOLO handles this naturally — padded cells produce low confidence predictions that TAL assigner ignores. No special handling needed beyond correct bbox adjustment. |
| **No existing YOLOv8 loss implementation in codebase** | 🟡 MEDIUM | Must implement from scratch. Reference ultralytics implementation for correctness. TAL assigner is the hardest part — test it with known inputs/outputs. |
| **Validation metric: mAP is expensive to compute** | 🟢 LOW | Only 4 val images — computation is trivial. Use torchvision's `box_iou` and manual AP calculation for simplicity, or pycocotools for standard COCO metrics. |

---

### Approaches

#### 1. **From-scratch training loop with custom YOLOv8 loss** (RECOMMENDED)

Write all components: DetectionDataset, YOLOv8Loss with TAL assigner, training loop, mAP computation.

- **Pros**: Full control, no external dependencies (ultralytics is heavy), educational value, matches architecture exactly
- **Cons**: More code to write and debug, TAL assigner is non-trivial
- **Effort**: Medium-High (5-7 days)

#### 2. **Borrow ultralytics YOLOv8 loss/assigner, wrap with custom dataloader**

Install `ultralytics` and reuse their `v8DetectionLoss` and `TaskAlignedAssigner` classes. Write only the dataset and training loop.

- **Pros**: Less code, battle-tested loss implementation, proven convergence
- **Cons**: Heavy dependency (ultralytics is ~1GB with all models), version lock risk, may not work with custom multimodal input
- **Effort**: Medium (3-4 days)

#### 3. **Convert to two-stage detector (Faster R-CNN style) for training stability**

Replace YOLO head with a torchvision Faster R-CNN head. Use RPN + RoIAlign. Simpler loss function (RPN loss + Fast R-CNN loss).

- **Pros**: Simpler training, more stable on tiny datasets, torchvision has full implementation
- **Cons**: Major architectural change — breaks YOLO Nano compatibility for distillation, slower inference, defeats purpose of the dual-model design
- **Effort**: High (7-10 days)

---

### Recommendation

**Approach 1** — from-scratch training loop. The key reasons:

1. **No ultralytics dependency**: Our model has a multimodal input (RGB+NIR) that ultralytics doesn't support. Wrapping would require workarounds that defeat the purpose.
2. **TAL assigner is well-documented**: Multiple reference implementations exist (ultralytics, YOLOv6, PP-YOLOE). We implement the core algorithm (~150 lines).
3. **CIoU loss is available**: `torchvision.ops.complete_box_iou_loss` handles the regression loss — we just need the assigner + BCE classification.
4. **Educational value**: Understanding the loss function is critical for debugging and future improvements.

**Implementation order**:
1. DetectionDataset (load RGB+NIR + YOLO labels + bbox-aware transforms)
2. BatchTargetAssigner (TAL matching logic)
3. YOLOv8Loss (BCE cls + CIoU bbox)
4. Metrics (mAP@0.5, mAP@0.5:0.95)
5. Training loop (2-phase, checkpointing, logging)
6. Integration test (overfit single batch to verify loss decreases)

---

### Ready for Proposal

**Yes** — all 7 investigation areas are covered. The proposal should:

1. Scope the training loop to Phase 1 only (backbone frozen) — prove the pipeline works before committing to full fine-tuning
2. Set a realistic goal: mAP@0.5 ≥ 0.3 on val (given 4 images, this is high-variance but achievable)
3. Include a plan for annotation quality filtering (confidence threshold)
4. Define what "success" looks like: loss decreases, predictions on sample images show plausible detections
