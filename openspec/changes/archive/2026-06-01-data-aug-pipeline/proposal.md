# Proposal: Data Augmentation Pipeline for Mango Ripeness Detection

## Intent

The MLOps orchestrator detects new mango images on OCI but cannot download, preprocess, or augment them. The existing training pipeline (`dataset.py`) stacks RGB+NIR into a single 4-channel tensor for ConvNeXtTeacher, while the MasterModel requires **separate** `(N,3,H,W)` and `(N,1,H,W)` dual-stream inputs. Without this pipeline, the MasterModel cannot consume OCI-sourced data, blocking the entire training loop.

## Scope

### In Scope
- OCI image extraction with RGB/NIR pair discovery and download
- Preprocessing pipeline (resize, homography warp, format conversion)
- Data augmentation producing MasterModel-compatible dual-stream tensors
- Per-image `.pt` vector output with `dataset_index.json` manifest
- New LangGraph node `run_data_augmentation` inserted between `check_for_new_data` and `validate_data`
- New state fields: `augmented_data_path`, `output_vectors_path`, `augmentation_stats`

### Out of Scope
- Model training or architecture changes (MasterModel, ConvNeXtTeacher untouched)
- Model deployment or serving
- DualFPN level-mismatch fix (separate change, prerequisite)
- VLM-based data quality validation (Node 2 already covers this)
- Mosaic/MixUp augmentations (deferred to enhancement phase)

## Capabilities

### New Capabilities
- `data-extraction`: OCI path listing, RGB/NIR pair discovery by filename stem, and parallel download via s3fs
- `data-preprocessing`: Resize, homography warp (NIR→RGB), format conversion to per-sample dual-stream tensors
- `data-augmentation`: Spatially-synchronized RGB+NIR augmentation producing separate (3,H,W) and (1,H,W) tensors matching MasterModel input spec

### Modified Capabilities
- `testing-teacher-arch`: Test suite MUST be updated to validate that data pipeline output tensors conform to MasterModel's `forward(rgb, nir)` signature (new scenarios for tensor shape validation)

## Approach

New `src/data_pipeline/` module with three layers:

1. **Extraction** (`oci_client.py`, `pair_discovery.py`): Extends `ObjectStorage` with download capability; discovers RGB/NIR pairs by shared filename stem (mirroring `calibracion/` naming convention).

2. **Preprocessing** (`preprocessor.py`): Loads paired images, applies homography warp (reusing `dataset.py` pattern), resizes to target dimensions, normalizes with ImageNet stats (RGB) + estimated NIR stats.

3. **Augmentation** (`augmentor.py`): Applies albumentations on a temporary 4-channel stack (spatial transforms are synchronized), then splits into separate RGB and NIR tensors. Photometric augmentations are modality-aware (ColorJitter on RGB only, intensity scaling on NIR only).

Output: per-image `.pt` files + `dataset_index.json`, written by `vector_store.py`.

Orchestration: `pipeline.py` exposes `run_data_augmentation(state)` as a LangGraph node, inserted between Node 1 (`check_for_new_data`) and Node 2 (`validate_data`).

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/data_pipeline/` | New | Entire module: oci_client, pair_discovery, preprocessor, augmentor, vector_store, pipeline |
| `src/agent/graph_nodes.py` | Modified | Add `run_data_augmentation` node function |
| `src/agent/state.py` | Modified | Add `augmented_data_path`, `output_vectors_path`, `augmentation_stats` fields |
| `src/agent/process.py` | Modified | Wire new node into graph |
| `openspec/specs/testing-teacher-arch/spec.md` | Modified | Add scenarios for pipeline output tensor validation |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| RGB/NIR spatial desync during augmentation | Medium | Stack into 4-channel before augmenting, split after — proven pattern from albumentations |
| OCI download latency for large datasets | Medium | Async download with progress tracking; cache previously downloaded pairs locally |
| NIR normalization stats are placeholder values | High | Run `compute_nir_stats()` from `dataset.py` on downloaded dataset before training; store computed stats in manifest |
| Homography matrix quality | Low | Validate warp on sample pairs; fall back to simple resize if matrix is absent or corrupt |
| Memory exhaustion with large augmentation factor | Low | Configurable augmentation multiplier; lazy tensor generation via DataLoader |

## Rollback Plan

1. Remove `run_data_augmentation` node from LangGraph graph; reconnect `check_for_new_data → validate_data` directly.
2. Delete `src/data_pipeline/` module.
3. Remove state fields from `MLOpsState`. Existing training pipeline (`dataset.py`) remains fully functional.

## Dependencies

- Prerequisite: DualFPN level-mismatch fix (P2→P3) must be resolved before augmented data feeds MasterModel training
- `s3fs` (already installed) for OCI S3-compatible access
- `albumentations` (already installed) for augmentation transforms
- `opencv-contrib-python` (already installed) for homography warp

## Success Criteria

- [ ] OCI client discovers and downloads paired RGB/NIR images from user-provided path
- [ ] Preprocessor applies homography warp and resize, producing aligned pairs
- [ ] Augmentor outputs separate RGB `(3,H,W)` and NIR `(1,H,W)` tensors validated against MasterModel `forward()` signature
- [ ] Spatial augmentations are pixel-aligned between RGB and NIR (≤1px deviation)
- [ ] LangGraph node integrates cleanly; graph routes `check_for_new_data → run_data_augmentation → validate_data`
- [ ] `dataset_index.json` manifest accurately records all augmented samples with source OCI paths