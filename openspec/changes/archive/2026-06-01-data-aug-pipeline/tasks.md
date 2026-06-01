# Tasks: Data Augmentation Pipeline

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | 600–800 |
| 800-line budget risk | Medium |
| Chained PRs recommended | No (within budget if kept tight) |
| Suggested split | Single PR (all data_pipeline modules + tests) |
| Delivery strategy | ask-always |
| Chain strategy | pending |

Decision needed before apply: Yes
Chained PRs recommended: No
Chain strategy: pending
800-line budget risk: Medium

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | Config + types + ObjectStorage extension | PR 1 | Foundation; no deps |
| 2 | OCI download + pair discovery | PR 2 | Depends on PR 1 |
| 3 | Preprocessor + augmentor | PR 3 | Depends on PR 2 |
| 4 | Vector store + pipeline orchestrator | PR 4 | Depends on PR 3 |
| 5 | Tests for all components | PR 5 | Depends on PR 1–4 |

## Phase 1: Foundation / Config (Files: config.py, __init__.py, ObjectStorage.py)

- [x] 1.1 Create `src/data_pipeline/__init__.py` with package docstring and exports for `DataAugmentationPipeline`, `PipelineConfig`
- [x] 1.2 Create `src/data_pipeline/config.py` with `PipelineConfig` dataclass (OCI endpoint, bucket, prefix, cache_dir, target_size, augmentation_factor, spatial_intensity, photometric_intensity, homography_path, nir_stats) + YAML/env loader
- [x] 1.3 Add `download_object(key, local_path)` method to `src/utils/ObjectStorage.py` using existing `get_s3fs()` with size-based skip if file exists

## Phase 2: Extraction (Files: oci_client.py, pair_discovery.py)

- [x] 2.1 Create `src/data_pipeline/oci_client.py` with `OCIManager` class wrapping s3fs: `list_objects(prefix)` with pagination, exponential backoff retry (3 attempts, 2^n delay), `download_pair(rgb_key, nir_key, cache_dir)` delegating to ObjectStorage
- [x] 2.2 Create `src/data_pipeline/pair_discovery.py` with `PairDiscovery` class: `match_by_stem(objects_list)` returns `matched_pairs: list[PairedSample]` and `unmatched_rgb: list[str]` using `{class}/{rgb|nir}/{stem}.{ext}` convention
- [x] 2.3 Test: parametrized pair discovery with all-matched, missing-NIR, missing-both, multi-class scenarios

## Phase 3: Preprocessing (Files: preprocessor.py)

- [x] 3.1 Create `src/data_pipeline/preprocessor.py` with `Preprocessor` class: `__init__(target_size, homography_path, nir_stats)` loads H matrix once; `process_pair(rgb_path, nir_path)` loads RGB (H,W,3) BGR→RGB, NIR (H,W) grayscale, applies `cv2.warpPerspective` or `cv2.resize` fallback, resizes to target, normalizes with ImageNet stats (RGB) and manifest stats (NIR), returns `(rgb_tensor(3,H,W), nir_tensor(1,H,W), metadata)`
- [x] 3.2 Test: valid pair produces correct shapes `(3,640,640)` and `(1,640,640)` float32
- [x] 3.3 Test: corrupt/zero-byte NIR is logged and skipped without crash
- [x] 3.4 Test: missing homography triggers fallback resize with warning logged

## Phase 4: Augmentation (Files: augmentor.py)

- [x] 4.1 Create `src/data_pipeline/augmentor.py` with `Augmentor` class: `augment_pair(rgb_tensor, nir_tensor, label, class_name, config)` stacks to (H,W,4), applies spatial transforms via albumentations (flip, rotate, shift-scale), splits back to rgb(3,H,W) + nir(1,H,W), applies modality-aware photometric (ColorJitter on RGB only, brightness/contrast on NIR), returns `list[AugmentedSample]` × augmentation_factor
- [x] 4.2 Test: 4-channel stack → split preserves pixel alignment after HorizontalFlip + RandomRotate90 (checkerboard synthetic input)
- [x] 4.3 Test: augmentation_factor=3 from 10 originals produces 30 samples
- [x] 4.4 Test: augmentation_factor=1 produces identity-only (no geometric variation)

## Phase 5: Vector Store + Pipeline (Files: vector_store.py, pipeline.py)

- [x] 5.1 Create `src/data_pipeline/vector_store.py` with `VectorStore` class: `save_sample(sample, output_dir)` writes `{stem}_aug{N}.pt` with `{rgb, nir, label, class, meta}` dict; `write_manifest(entries, output_dir)` overwrites `dataset_index.json` idempotently
- [x] 5.2 Test: save → load roundtrip preserves tensor equality and metadata
- [x] 5.3 Test: manifest with 30 entries references existing .pt files on disk
- [x] 5.4 Create `src/data_pipeline/pipeline.py` with `DataAugmentationPipeline.run(config)`: chains OCI list → pair discovery → download → preprocess → augment → save; streams one pair at a time; writes final manifest; returns stats dict `{downloaded, processed, augmented, skipped, unmatched}`
- [x] 5.5 Integration test: full pipeline with 2-class dummy pairs in tmpdir; assert manifest count, tensor shapes, and stats

## Phase 6: Configuration

- [x] 6.1 Create `pipeline_config.yaml` example at project root with all configurable params (OCI paths, dims, augmentation params, cache dir, homography path)
- [x] 6.2 Test: config loader reads YAML and applies defaults for missing fields
