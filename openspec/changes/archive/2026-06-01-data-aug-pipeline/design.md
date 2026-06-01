# Design: Data Augmentation Pipeline for MasterModel

## Technical Approach

New `src/data_pipeline/` module bridging OCI object storage to MasterModel-ready dual-stream tensors. Pipeline runs as a dedicated LangGraph node between `check_for_new_data` and `validate_data`.

Architecture: **4-channel temp stack → split** for spatial sync, then **modality-aware photometric** transforms on separate tensors. This reuses the proven albumentations pattern from `dataset.py` while adapting output to MasterModel's `forward(rgb, nir)` signature.

## Architecture Decisions

| Decision | Choice | Alternatives | Rationale |
|----------|--------|-------------|-----------|
| Spatial sync strategy | 4-channel temp stack → split | Separate transforms with seed sync | Albumentations natively supports N-channel images; deterministic split guarantees ≤1px alignment without custom random seed management |
| NIR normalization | Placeholder stats + computed manifest | Pre-compute once globally | `dataset.py` already has `compute_nir_stats()`; pipeline writes computed stats to manifest so training can use real values; placeholder prevents blocking |
| Homography loading | Load `.npy` once in `Preprocessor` init, apply per-pair | Load per-image | Matrix is static for a camera rig; avoids repeated disk I/O |
| OCI download | Parallel async via `ThreadPoolExecutor` + s3fs | AsyncIO/aiofiles | s3fs is already in use; threads are simpler and sufficient for I/O-bound downloads; `max_workers=8` default |
| Cache strategy | Local disk mirror of OCI structure with size-based dedup | Redis/in-memory cache | Images are large (MBs); local disk is cheapest and persists across runs; size-check skips re-download |
| Memory strategy | Stream pairs one-at-a-time through load→warp→augment→save | Batch in memory | Prevents OOM with large augmentation factors; only 1 pair + aug variants resident at once |
| Vector output | Per-image `.pt` + `dataset_index.json` | Single archive `.pt` | Per-image enables incremental reprocessing, partial recovery, and random-access DataLoader; manifest is idempotent overwrite |
| LangGraph wiring | New node in `process.py` builder | Inline in training node | Graph nodes exist but graph isn't wired yet in mainline; builder in `process.py` is the minimal viable integration |

## Data Flow

```
OCI Bucket (news/rgb/, news/nir/)
    │
    ▼
OCIManager.list_objects()  ──→  PairDiscovery.match_by_stem()
    │                              (matched_pairs, unmatched_rgb)
    ▼
ParallelDownloader.download_pairs()  ──→  Local cache: cache_dir/{class}/{rgb|nir}/
    │
    ▼
Preprocessor.process_pair():
    - Load RGB (H,W,3), NIR (H,W)
    - cv2.warpPerspective(NIR, H_matrix, (w,h))  [or cv2.resize fallback]
    - Resize both to (target_h, target_w)
    - Normalize RGB with ImageNet stats; NIR with manifest stats
    - Output: rgb_arr (H,W,3), nir_arr (H,W,1) float32
    │
    ▼
Augmentor.augment_pair():
    - Stack → (H,W,4)
    - Apply spatial transforms (albumentations)
    - Split → rgb_aug (3,H,W), nir_aug (1,H,W)
    - Apply photometric transforms per modality
    │
    ▼
VectorStore.save_sample():
    - torch.save({rgb, nir, label, class, meta}) → {stem}_aug{N}.pt
    - Append to manifest entries
    │
    ▼
Pipeline.run() writes dataset_index.json
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/data_pipeline/__init__.py` | Create | Package init, exports `run_data_augmentation` |
| `src/data_pipeline/oci_client.py` | Create | `OCIManager` wraps s3fs with retry, listing, metadata |
| `src/data_pipeline/pair_discovery.py` | Create | `PairDiscovery.match_by_stem()` returns matched/unmatched |
| `src/data_pipeline/downloader.py` | Create | `ParallelDownloader` with ThreadPoolExecutor, size-dedup cache |
| `src/data_pipeline/preprocessor.py` | Create | `Preprocessor` loads, warps, resizes, normalizes; dual-stream output |
| `src/data_pipeline/augmentor.py` | Create | `Augmentor` 4-channel stack/spatial sync + split + modality-aware photometric |
| `src/data_pipeline/vector_store.py` | Create | `VectorStore` writes per-image `.pt` + accumulates manifest |
| `src/data_pipeline/pipeline.py` | Create | `DataAugmentationPipeline` orchestrates extraction→vectors; exposes `run(state)` for LangGraph |
| `src/data_pipeline/config.py` | Create | YAML/env config dataclass: OCI paths, augmentation params, image dims, cache dir |
| `src/agent/graph_nodes.py` | Modify | Add `run_data_augmentation(state)` node wrapper |
| `src/agent/state.py` | Modify | Add `augmented_data_path`, `output_vectors_path`, `augmentation_stats` to `MLOpsState` |
| `src/agent/process.py` | Modify | Build `StateGraph`, wire nodes: check → augment → validate → … |
| `src/utils/ObjectStorage.py` | Modify | Add `download_object(key, local_path)` to existing s3fs wrapper |

## Interfaces / Contracts

```python
# pair_discovery.py
class PairedSample(TypedDict):
    class_name: str
    stem: str
    rgb_key: str
    nir_key: str

# augmentor.py
class AugmentedSample(NamedTuple):
    rgb: torch.Tensor   # (3, H, W) float32
    nir: torch.Tensor   # (1, H, W) float32
    label: int
    class_name: str
    metadata: dict

# vector_store.py
class ManifestEntry(TypedDict):
    source_oci_key: str
    local_path: str
    label: int
    class_name: str
    aug_id: int
    homography_applied: bool
```

## Testing Strategy

| Layer | What to Test | Approach |
|-------|-------------|----------|
| Unit | Pair discovery stem matching | Parametrized pytest with various filename patterns |
| Unit | Homography warp vs resize fallback | Mock matrix, assert output shapes, warn logs |
| Unit | 4-channel stack split preserves alignment | Synthetic checkerboard; assert pixel correspondence after flip/rotate |
| Unit | VectorStore `.pt` roundtrip | Save → load → assert tensor equality and metadata |
| Integration | Full pipeline with local dummy images | 2-class dummy pairs in tmpdir; run pipeline; validate manifest count and tensor shapes |
| Integration | OCI retry behavior | Mock s3fs to fail twice then succeed; assert 3 attempts |
| E2E | LangGraph node integration | Build graph with new node, invoke with test state, assert state fields populated |

## Migration / Rollout

No data migration required — this is a new offline pipeline generating new artifacts. Rollback: remove node from graph, delete `src/data_pipeline/`, remove state fields. Existing `dataset.py` and training flow remain untouched.

Prerequisite blocker: **DualFPN P2→P3 level mismatch** must be fixed before MasterModel can consume augmented data for training.

## Open Questions

- [ ] What is the exact OCI prefix structure? Assumed `{bucket}/news/{rgb,nir}/` based on `calibracion/`; confirm before implementation.
- [ ] Should computed NIR stats be persisted back to a shared config file, or only in the per-run manifest?
- [ ] How should missing pairs be surfaced — log-only, or should `augmentation_stats` include a list of unmatched stems for human review?
