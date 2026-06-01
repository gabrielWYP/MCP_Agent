# Exploration: Data Augmentation Pipeline for Mango Ripeness Detection

**Date**: 2026-06-01
**Investigator**: sdd-explore sub-agent
**Status**: Complete

---

## Executive Summary

The project already has a functional augmentation pipeline inside `src/ml_pipeline/training/dataset.py` (`MangoRipeness` + `get_transforms()`) that handles RGB+NIR stacking (4-channel) via albumentations. However, this pipeline operates **on-disk** from a local `data/` directory and stacks RGB+NIR into a single tensor — a critical mismatch with the **MasterModel** (`src/models/master/master_model.py`), which expects **separate** `(N,3,H,W)` and `(N,1,H,W)` tensors. The OCI extraction is also minimal: `src/storage_logic/logic.py` only **lists** objects via `s3fs` — it does not download, preprocess, or pair RGB+NIR images.

The new data augmentation pipeline must bridge three gaps: (1) OCI download + RGB/NIR pairing, (2) preprocessing + augmentation for the MasterModel's dual-stream format, and (3) LangGraph orchestrator integration. The recommended approach is a new `src/data_pipeline/` module with a LangGraph node placed between `check_for_new_data` (Node 1) and `validate_data` (Node 2).

---

## 1. OCI Extraction — Current State

### Files Analyzed
- `src/storage_logic/logic.py` (12 lines)
- `src/utils/ObjectStorage.py` (59 lines)
- `src/variables/configvariables.py` (14 lines)

### Findings
- **s3fs is the abstraction layer**: The project uses `s3fs.S3FileSystem` with OCI-compatible S3 endpoint (`S3_ENDPOINT_URL`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`) from `.env`.
- **Only listing, no downloading**: `check_new_objects(BUCKET_NAME)` calls `count_and_list_objects()` which runs `s3.ls()` — it counts objects but never downloads them.
- **No RGB/NIR pairing logic in storage**: The storage layer treats all objects uniformly; there's no awareness of the `{rgb,nir}` directory structure.
- **Calibration data structure mirrors expected layout**:
  ```
  calibracion/rgb/news/rgb/news_rgb_mango_rgb_1779644403.jpg
  calibracion/nir/news/nir/news_nir_mango_nir_1779644403.jpg
  ```
  Suggests OCI path would be `{bucket}/news/rgb/` and `{bucket}/news/nir/` with paired files sharing a common stem (e.g., `mango_rgb_1779644403` ↔ `mango_nir_1779644403`).

### Gap
No `download_from_oci()` function exists. The new pipeline needs:
- `list_oci_path(oci_path)` → return file listing
- `download_pair(rgb_key, nir_key, local_dir)` → download both images
- `discover_pairs(bucket, prefix)` → match RGB/NIR by filename stem

---

## 2. Existing Data Pipeline — Current State

### Files Analyzed
- `src/ml_pipeline/training/dataset.py` (290 lines)
- `src/ml_pipeline/training/process.py` (352 lines)
- `src/ml_pipeline/predict/process.py` (174 lines)

### Findings

#### Dataset Class (`MangoRipeness`)
- **On-disk structure assumption**: Expects `data/{train,val,test}/{sano,danado}/{rgb,nir}/*.jpg`
- **RGB+NIR pairing by filename**: Same stem in `rgb/` and `nir/` subdirectories
- **Homography warp**: Loads `.npy` matrix, warps NIR→RGB space via `cv2.warpPerspective()`
- **Channel stacking**: `np.concatenate([rgb, nir[..., np.newaxis]], axis=-1)` → `(H, W, 4)` — **this is for ConvNeXtTeacher, NOT MasterModel**
- **Classes**: `["sano", "danado"]` (binary classification)

#### Augmentation Pipeline (`get_transforms()`)
- Uses **albumentations** (proven with N-channel images)
- **Spatial augs**: `RandomResizedCrop`, `HorizontalFlip`, `VerticalFlip`, `RandomRotate90`, `ShiftScaleRotate`
- **Photometric augs**: `ColorJitter` (RGB only), `GaussianBlur`, `GaussNoise`
- **Normalization**: ImageNet mean/std for RGB channels + estimated NIR mean/std (`_MEAN_NIR=[0.45]`, `_STD_NIR=[0.22]`)
- **Train/val split**: Train gets full augmentation; val/test only gets `Resize + Normalize`
- **⚠️ Known issue**: `ColorJitter` only applies to first 3 channels — NIR channel is not color-augmented (correct behavior, documented in code)

#### Training Process (`process()`)
- 2-phase fine-tuning: (1) frozen backbone → (2) full fine-tuning
- `in_channels` parameter controls 3 vs 4 channel input
- Uses `build_dataloaders()` factory

#### Prediction Pipeline (`predict/process.py`)
- `prepare_image()` — single-image preprocessing matching training pipeline
- Same RGB+NIR stacking, homography warp, albumentations transform

### Gap
The current pipeline produces **stacked 4-channel tensors** `(C=4, H, W)` for the ConvNeXtTeacher. The MasterModel requires **separate** RGB `(3, H, W)` and NIR `(1, H, W)` tensors. Also, the existing pipeline assumes data is already local — no cloud download.

---

## 3. Master Model Input Format — Current State

### Files Analyzed
- `src/models/master/master_model.py` (189 lines)
- `src/models/master/backbone.py` (159 lines)
- `src/models/master/fusion.py` (197 lines)
- `src/models/master/neck.py` (168 lines)
- `src/models/master/head.py` (167 lines)

### Findings

#### Forward Signature
```python
def forward(self, rgb: torch.Tensor, nir: torch.Tensor) -> dict:
    # rgb: (N, 3, H, W) — normalized
    # nir: (N, 1, H, W) — normalized
```

**Critical**: The MasterModel takes **SEPARATE tensors**, not a stacked 4-channel tensor. This is because `DualConvNeXtBackbone` has:
- `rgb_stem`: Conv2d(3→96)
- `nir_stem`: Conv2d(1→96)
- Shared stages for both streams

#### Pipeline Flow
```
RGB (N,3,H,W) + NIR (N,1,H,W)
    → DualConvNeXtBackbone → [rgb_feats[4], nir_feats[4]]
    → CrossModalFusion (cross-attention) → [fused[4]]
    → DualFPN → [P2,P3,P4,P5] (256ch)
    → YOLODetectionHead → [(B,6,Hi,Wi) per level]
```

#### Output Format (Detection)
- `preds`: `[(B, 6, H3, W3), (B, 6, H4, W4), (B, 6, H5, W5)]` where 6 = 2 classes + 4 bbox coords
- Distillation features exposed: `distill_backbone_rgb`, `distill_backbone_fused`, `distill_fpn`, `distill_head_cls`, `distill_head_reg`

#### Known Architecture Bug
`DualFPN` produces **4 levels** [P2, P3, P4, P5] but `YOLODetectionHead` expects **3 levels** (strides [8, 16, 32] map to P3, P4, P5). The `test_arch.py` file has a level mismatch.

### Implication for Data Pipeline
The augmentation pipeline's output must be **two separate tensors**: `rgb_tensor (N,3,H,W)` and `nir_tensor (N,1,H,W)`. They must be:
- Same spatial dimensions (H, W)
- Individually normalized (RGB with ImageNet stats, NIR with custom stats)
- Augmented identically in spatial domain (same crop, same flip, same rotation)

---

## 4. Calibration Images — Current State

### Files Analyzed
- `calibracion/rgb/news/rgb/news_rgb_mango_rgb_1779644403.jpg` (1 image)
- `calibracion/nir/news/nir/news_nir_mango_nir_1779644403.jpg` (1 image)

### Findings
- **Single calibration pair only**: 1 RGB + 1 NIR image
- **Naming convention**: `news_{rgb|nir}_mango_{rgb|nir}_{timestamp}.jpg`
- **Directory structure**: `calibracion/{rgb|nir}/news/{rgb|nir}/`
- **Purpose**: These are Charuco board images for homography calibration (see `main.py` homography subcommand)
- **Pairing**: The timestamp `1779644403` is the common identifier; RGB has `_rgb_` in stem, NIR has `_nir_` in stem

### Implication
The OCI path structure likely mirrors this: `{bucket}/news/{rgb|nir}/` with paired files sharing a unique ID. The data augmentation pipeline needs to replicate this pairing logic when downloading from OCI.

---

## 5. Augmentation Needs for Mango Ripeness

### Current Augmentations (already implemented)

| Augmentation | Parameter | p | Value for Mango |
|---|---|---|---|
| RandomResizedCrop | scale=(0.7, 1.0) | 1.0 | ✅ Good — handles scale variation |
| HorizontalFlip | — | 0.5 | ✅ Mango orientation invariance |
| VerticalFlip | — | 0.3 | ⚠️ Low value — mangos rarely upside-down |
| RandomRotate90 | — | 0.5 | ✅ Rotation invariance |
| ShiftScaleRotate | shift=0.05, scale=0.1, rotate=15° | 0.4 | ✅ Subtle geometric variations |
| ColorJitter | brightness=0.3, contrast=0.3, sat=0.2, hue=0.05 | 0.5 | ✅ Ripeness color variation |
| GaussianBlur | (3, 5) | 0.2 | ✅ Motion blur / focus issues |
| GaussNoise | — | 0.2 | ✅ Sensor noise |

### Recommended Additions for Mango Domain

| Augmentation | Rationale | Priority |
|---|---|---|
| **Lighting extremes** (RandomBrightnessContrast with wider range) | Packing facilities have variable lighting (fluorescent → natural) | HIGH |
| **Shadow simulation** (RandomShadow / cutout) | Mangos cast shadows on conveyor belts | MEDIUM |
| **Mosaic/MixUp** | YOLO standard augmentation; combines 4 images into 1 — excellent for small datasets | HIGH |
| **HSV shift** (wider hue range for ripeness) | Green→yellow→orange ripeness spectrum needs full hue coverage | MEDIUM |
| **Perspective transform** | Slight angle changes from camera mounting variation | LOW |
| **ElasticTransform** | Simulates mango surface deformation | LOW |
| **NIR-specific augmentations**: | | |
| - Intensity scaling | NIR sensor gain variation | HIGH |
| - Speckle noise | NIR sensor artifact common in InGaAs cameras | MEDIUM |
| - Contrast Limited AHE | Enhances damage visibility in NIR | LOW |

### Critical Constraint
**Spatial augmentations MUST be identical for RGB and NIR**. If a crop is applied, both images must receive the same crop region. If a flip is applied, both must flip identically. The current albumentations pipeline handles this correctly for stacked images, but for separate tensors we need to either (a) stack temporarily for augmentation then split, or (b) use synchronized RandomParameter generation.

---

## 6. Placement — Where Should This Live?

### Approach A: `src/ml_pipeline/augmentation/` (alongside existing pipeline)
```
src/ml_pipeline/
├── training/
├── predict/
└── augmentation/
    ├── __init__.py
    ├── oci_extractor.py    # OCI download + pairing
    ├── preprocessor.py     # Resize, normalize, format
    ├── augmentor.py        # Augmentation pipeline
    └── vector_writer.py    # Output to disk/OCI
```

**Pros**: 
- Co-located with existing ML pipeline code
- Clear domain boundary: "ml_pipeline" contains all ML data processing
- Consistent with existing convention (`dataset.py` already handles data loading + augmentation)

**Cons**: 
- `augmentation` is misleading — this module does OCI extraction + preprocessing + augmentation + vector output
- Tight coupling with training assumptions

### Approach B: `src/data_pipeline/` (new domain)
```
src/data_pipeline/
├── __init__.py
├── oci_client.py          # OCI/S3 abstraction (extends storage_logic)
├── pair_discovery.py      # RGB+NIR matching logic
├── preprocessor.py        # Image preprocessing
├── augmentor.py           # Augmentation pipelines (separate RGB/NIR-aware)
├── vector_store.py        # Output writer
└── pipeline.py            # Orchestrator (process())
```

**Pros**:
- Clean separation: data engineering vs ML engineering
- `storage_logic/` is thin and reusable; `data_pipeline/` adds business logic
- Easier to test independently
- Clearer domain for LangGraph node integration
- Matches the scope: "extract from OCI → process → augment → output vectors"

**Cons**:
- New domain = new module to document
- Slightly more files
- `augmentor.py` would reimplement some logic from `dataset.py`

### Recommendation: **Approach B** (`src/data_pipeline/`)

The functionality spans OCI I/O, image preprocessing, augmentation, and format conversion — this is a data engineering concern, not an ML training concern. The `ml_pipeline/training/dataset.py` augmentations can remain as the training-time augmentation, while `data_pipeline/augmentor.py` handles the **offline augmentation** that generates a pre-augmented dataset from OCI raw images. This avoids coupling and follows the Single Responsibility Principle.

---

## 7. LangGraph Integration

### Current Graph Structure
```
START → check_for_new_data (Node 1)
            ↓ (has data)
        validate_data (Node 2)
            ↓ (quality >= 7.0)
        run_training_job (Node 3)
            ↓
        analyze_performance (Node 4)
            ↓
        check_deployment_decision (Node 5) → deploy_to_production (Node 7)
                                            → alert_human (Node 6)
```

### Current State Fields (`MLOpsState`)
```python
new_data_path, model_version, candidate_model_path,
candidate_metrics, production_metrics, deployment_decision,
failure_analysis_report, data_quality_score, analysis_reason, timestamp
```

### Proposed Integration

**Option A: New node before validation**
```
check_for_new_data → run_data_augmentation (NEW) → validate_data → ...
```
- Augmented data is available for VLM quality validation
- VLM can inspect augmented samples for quality

**Option B: Sub-step of training node**
- Data augmentation runs as part of `run_training_job`
- Simpler graph, but less visibility and control

**Recommendation: Option A** — new node `run_data_augmentation` between Node 1 and Node 2.

New state fields needed:
```python
augmented_data_path: str        # Local path to augmented dataset
output_vectors_path: str        # Path to saved tensors/vectors
augmentation_stats: dict        # {"total_pairs": N, "augmented": M, "failed": K}
```

The node function signature:
```python
def run_data_augmentation(state: MLOpsState) -> MLOpsState:
    # 1. List and download pairs from state["new_data_path"]
    # 2. Preprocess (resize, normalize, homography warp)
    # 3. Augment (spatial + photometric)
    # 4. Save augmented pairs locally
    # 5. Update state with paths and stats
```

---

## 8. Vector Output Format

### What "Multimodal Vectors" Means in Context

The MasterModel takes **separate RGB and NIR tensors**, so the "vectors" are:

```
For each augmented sample:
  rgb_tensor: torch.Tensor(3, H, W) float32  # Normalized RGB
  nir_tensor: torch.Tensor(1, H, W) float32  # Normalized NIR
  label: int                                   # 0=sano, 1=danado
  metadata: dict                               # Source OCI path, augmentations applied
```

### Storage Format Options

| Format | Pros | Cons | Recommendation |
|---|---|---|---|
| **PyTorch `.pt` files** | Direct `torch.load()`, preserves tensor metadata, fast I/O | Python/PyTorch only | ✅ **Primary** |
| **NumPy `.npy` files** | Universal, smaller, `np.load(mmap_mode)` | No built-in label/metadata | ⚠️ Secondary |
| **TFRecord** | TensorFlow ecosystem standard | Not PyTorch-native, requires protobuf | ❌ No |
| **HDF5** | Hierarchical, supports groups | Extra dependency (h5py), slower | ❌ No |
| **Raw PNG/JPG** | Human-inspectable, universal | No normalization preserved, disk-heavy | ❌ For debug only |

### Recommended Structure
```
data/augmented/
├── train/
│   ├── rgb/
│   │   ├── mango_001_aug0.pt     # Tensor (3, 640, 640)
│   │   └── mango_001_aug1.pt
│   ├── nir/
│   │   ├── mango_001_aug0.pt     # Tensor (1, 640, 640)
│   │   └── mango_001_aug1.pt
│   └── labels.json               # {"mango_001_aug0": {"label": 0, "source": "oci://..."}}
├── val/
└── test/
```

Or, alternatively, a single **`.pt` archive per split**:
```python
torch.save({
    "rgb": torch.stack(rgb_tensors),      # (N, 3, H, W)
    "nir": torch.stack(nir_tensors),      # (N, 1, H, W)
    "labels": torch.tensor(labels),       # (N,)
    "metadata": metadata_list,
}, "data/augmented/train.pt")
```

**Recommendation**: Use per-image `.pt` files for flexibility (incremental processing, partial loads) + a `dataset_index.json` for the DataLoader. For final training, the DataLoader can batch-load on-the-fly. A single archive `.pt` is better for transfer but harder to reprocess individual samples.

---

## Dependencies and Risks

### Dependencies
| Dependency | Status | Notes |
|---|---|---|
| `s3fs` | ✅ Already in requirements.txt | Used by `ObjectStorage.py` |
| `albumentations` | ✅ Already in requirements.txt | Proven with N-channel images |
| `opencv-contrib-python` | ✅ Already installed | Used for homography warp, I/O |
| `torch`, `torchvision` | ✅ Already installed | Tensor operations |
| `numpy` | ✅ Already installed | Array operations |
| `oci` (Oracle SDK) | ❌ NOT installed | `s3fs` uses S3-compatible API, OCI SDK may not be needed if OCI bucket supports S3 protocol |

### Risks

1. **RGB/NIR synchronization during augmentation**: If spatial transforms aren't perfectly synchronized, the cross-modal fusion in MasterModel will receive misaligned features. Mitigation: use `albumentations` on a temporary 4-channel stack, then split channels 0:3 (RGB) and 3:4 (NIR).

2. **OCI latency for large datasets**: Downloading thousands of mango images from OCI could take minutes to hours. Mitigation: implement async downloads with progress tracking, cache already-downloaded pairs.

3. **NIR normalization stats are estimates**: Current `_MEAN_NIR=0.45`, `_STD_NIR=0.22` are placeholders. The actual dataset's NIR stats depend on the camera sensor. Mitigation: run `compute_nir_stats()` from `dataset.py` on the augmented dataset before training.

4. **Homography matrix quality**: If the Charuco calibration quality is poor, the NIR→RGB warp introduces misalignment. Mitigation: validate warp quality on sample pairs before full pipeline execution.

5. **MasterModel's P2→P3 level mismatch bug**: `DualFPN` produces 4 levels (P2-P5) but `YOLODetectionHead` expects 3 (P3-P5). This exists in the current code and needs fixing before the data pipeline's output can be used for MasterModel training.

6. **Memory constraints**: Augmenting a full dataset (e.g., 1000 images × 10 augmentations = 10,000 tensor files) could exhaust disk space. Mitigation: configurable augmentation factor, lazy generation during training instead of pre-generation.

---

## Approaches Comparison

| | Approach A: Extend existing dataset.py | Approach B: New data_pipeline domain |
|---|---|---|
| **Effort** | Low — add OCI download + MasterModel format to existing code | Medium — new module with clean abstractions |
| **OCI integration** | Add download methods to `dataset.py` or `storage_logic/` | Dedicated `oci_client.py` in `data_pipeline/` |
| **Augmentation** | Reuse `get_transforms()`, add MasterModel output format | New `augmentor.py` with dual-stream awareness |
| **LangGraph node** | Add as pre-step in `run_training_job` | New dedicated node with own state fields |
| **Testability** | Harder — coupled to training assumptions | Easier — independent module, clear I/O boundaries |
| **Future-proof** | Ties augmentation to training format | Clean separation of data engineering from ML |
| **Risk** | Breaking existing ConvNeXtTeacher pipeline | Creating duplicate augmentation logic |

### Recommendation: **Approach B** (`src/data_pipeline/`)

The clean domain separation justifies the medium effort. The existing `dataset.py` augmentations remain for training-time augmentation; the new `data_pipeline` handles the offline "raw OCI → augmented vectors" workflow. This also aligns with the MLOps architecture where data preparation is a distinct phase before training.

---

## Affected Areas

- `src/storage_logic/logic.py` — needs download capability (reuse `s3fs` from `ObjectStorage.py`)
- `src/ml_pipeline/training/dataset.py` — reference for augmentation patterns, normalization stats, homography loading
- `src/models/master/master_model.py` — **defines the target input format** (separate RGB + NIR tensors)
- `src/agent/graph_nodes.py` — new node `run_data_augmentation` + conditional edges
- `src/agent/state.py` — new fields in `MLOpsState`
- `src/agent/process.py` — updated graph construction
- `openspec/config.yaml` — update testing section when pipeline is built
- `src/models/master/neck.py` — P2→P3 mismatch bug must be fixed before MasterModel can consume augmented data
- `requirements.txt` — may need `aiofiles` for async OCI downloads (optional)

---

## Ready for Proposal

**Yes** — the architecture is well-understood, the gaps are clearly identified, and the recommendation is actionable. The orchestrator should proceed to `sdd-propose`.

### Key Points for Proposal
1. New module: `src/data_pipeline/` with OCI client, pair discovery, preprocessor, augmentor, vector store
2. New LangGraph node: `run_data_augmentation` between Node 1 and Node 2
3. Target format: separate RGB (3, H, W) + NIR (1, H, W) `.pt` tensors
4. Spatial augmentation synchronization via temporary 4-channel stack
5. Prerequisite: fix DualFPN→YOLODetectionHead level mismatch (P2 vs P3)
6. Configurable augmentation factor, async OCI downloads, progress tracking
