# Design: Pipeline E2E with Homography Alignment

## Technical Approach

Run the existing `DataAugmentationPipeline` as a validation exercise using `config/augmentation.yaml` (factor=3, timestamp_id matching, homography path set). The pipeline streams 31 cached pairs through `Preprocessor` (homography warp with `H⁻¹`), `Augmentor` (spatial+photometric), and `VectorStore` (`.pt` + manifest). No code changes — only execution and verification.

## Architecture Decisions

| Decision | Options | Tradeoffs | Choice |
|----------|---------|-----------|--------|
| Clear old output first | Delete `data/augmented/` vs. timestamped subdir | Deletion is destructive but guarantees clean state; timestamped subdir preserves history but risks ~194 files and manual cleanup. | Delete `data/augmented/` before run. |
| Verification scope | Full load of all 93 tensors vs. spot-check | Loading all is definitive but heavy; spot-check (3–5 samples) is lightweight and sufficient for shape/range validation. | Spot-check 5 random `.pt` files + full manifest parse. |
| Rollback on failure | Auto-delete partial output vs. leave for inspection | Auto-delete keeps disk clean but destroys debug evidence; leaving partial output aids diagnosis. | Leave partial output on failure; manual cleanup after root-cause. |
| Execution entry point | `python -m src.data_pipeline.pipeline` vs. `src/main.py mlops` | No dedicated pipeline CLI exists; `main.py` runs the full MLOps agent (overhead). | Use a minimal ad-hoc runner script (`run_pipeline.py`) invoking `DataAugmentationPipeline.run(local_pairs=...)`. |

## Data Flow

```
Config (augmentation.yaml)
         │
         ▼
DataAugmentationPipeline
         │
         ├──► local_pairs (31 cached pairs from data/cache/)
         │           │
         │           ▼
         │    Preprocessor.process_pair()
         │      ├── Load RGB → (H,W,3) uint8
         │      ├── Load NIR → (H,W) uint8
         │      ├── align_nir() → warpPerspective(H⁻¹)
         │      ├── resize → (640,640)
         │      └── normalize → tensors (3,640,640) + (1,640,640)
         │           │
         │           ▼
         │    Augmentor.augment_pair(factor=3)
         │      ├── Stack RGB+NIR → 4-channel
         │      ├── Spatial transforms (albumentations)
         │      ├── Split → RGB / NIR
         │      ├── Photometric transforms (modality-aware)
         │      └── Denormalize → 3 normalized tensors
         │           │
         │           ▼
         │    VectorStore.save_sample()
         │      └── {stem}_aug{id}.pt
         │
         └──► VectorStore.write_manifest()
                └── dataset_index.json
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `data/augmented/` | Modify | Cleared then repopulated with ~93 `.pt` files |
| `data/augmented/dataset_index.json` | Modify | Overwritten with 93 entries, `homography_applied: true` |
| `run_pipeline.py` | Create | Ad-hoc runner: loads cached pairs, instantiates pipeline, runs, prints stats |

## Interfaces / Contracts

### `run_pipeline.py` — Minimal Runner

```python
from pathlib import Path
from src.data_pipeline import PipelineConfig, DataAugmentationPipeline

config = PipelineConfig.from_yaml("config/augmentation.yaml")
pipeline = DataAugmentationPipeline(config)

# Build local_pairs from data/cache/ (discovered via glob)
local_pairs = [...]  # 31 dicts: {class_name, stem, rgb_path, nir_path}

stats = pipeline.run(local_pairs=local_pairs)
print(stats)
```

### Verification Gates

1. **File count**: `len(list(Path("data/augmented").glob("*.pt"))) == 93`
2. **Manifest entries**: `len(json.load(open("data/augmented/dataset_index.json"))) == 93`
3. **Log scan**: No `"No homography available"` in stderr/stdout
4. **Tensor shapes**: Spot-check 5 samples → `rgb.shape == (3,640,640)`, `nir.shape == (1,640,640)`
5. **Value ranges**: RGB in ImageNet-normalized range (~ -2.5 to +2.5); NIR in dataset-normalized range
6. **Cache integrity**: No new downloads (all pairs already in `data/cache/`)

## Testing Strategy

| Layer | What to Test | Approach |
|-------|-------------|----------|
| Unit | Preprocessor homography direction | Already fixed and covered by existing `tests/data_pipeline/test_preprocessor.py` |
| Integration | Full pipeline with local pairs | Execute `run_pipeline.py`, assert on `stats` and file system |
| E2E | End-to-end with cached data | Run once, verify all 6 verification gates above |

## Migration / Rollout

No migration required. This is a clean-slate validation run against cached data.

**Rollback**: Delete `data/augmented/` contents. The previous factor=1 run is unrecoverable but reproducible by setting `augmentation_factor: 1`.

## Open Questions

- [ ] Should we add `run_pipeline.py` to `.gitignore` or commit it as a utility? (Suggested: commit as `scripts/run_pipeline.py` for reuse.)
- [ ] Black NIR border artifact (~33% out-of-bounds from camera offset) — accept as-is or document threshold?
