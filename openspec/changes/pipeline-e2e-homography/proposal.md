# Proposal: Pipeline E2E with Homography Alignment

## Intent

Validate the full `DataAugmentationPipeline` produces correct augmented data with homography-based NIR→RGB alignment. The code fixes (homography direction bug → `H⁻¹` warp, `homography_path` config) are already applied. This change runs the pipeline and verifies output integrity — integration-test-level execution, not a code change.

## Scope

### In Scope
- Run `DataAugmentationPipeline` with `config/augmentation.yaml` as-is (homography path set, factor=3, timestamp_id matching)
- Verify ~31 pairs × 3 factor = ~93 augmented `.pt` files produced
- Verify no "No homography available" warnings (homography applies correctly)
- Verify `dataset_index.json` manifest is complete and well-formed
- Spot-check output tensor dimensions `(rgb: [3,640,640], nir: [1,640,640])` and value ranges
- Document the black NIR border behavior (~33% out-of-bounds pixels from camera offset) as expected

### Out of Scope
- Training MasterModel with produced data
- Labeling (danado/sano classification)
- Model evaluation
- Border cropping/padding (deferred — model should learn to handle partial NIR)
- Adding homography direction unit tests (separate change)

## Capabilities

### New Capabilities
None — no new spec-level behavior is introduced. This is a validation run of existing capabilities.

### Modified Capabilities
None — all code changes were applied as bug fixes within existing requirements. The `data-preprocessing` spec already requires NIR→RGB alignment via homography; the bug fix corrects implementation, not spec.

## Approach

1. **Clear** existing `data/augmented/` output from the factor=1 test run
2. **Run** `python -m src.data_pipeline.pipeline` (or orchestrator entry point) with current config
3. **Inspect** stdout/stderr for homography warnings (should see none)
4. **Count** output `.pt` files → expect ~93 (31 pairs × 3)
5. **Load** `dataset_index.json` → verify entries, `homography_applied=True`, correct metadata
6. **Sample** 3–5 `.pt` files → check `rgb.shape == [3,640,640]`, `nir.shape == [1,640,640]`, values in [0,1] (normalized)
7. **Document** black border percentage as expected camera-offset artifact

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `data/augmented/` | Replaced | 31 files overwritten → ~93 files |
| `data/augmented/dataset_index.json` | Replaced | Updated manifest with homography flag |
| `config/augmentation.yaml` | Already modified | `homography_path` already set (no further changes) |
| `src/data_pipeline/preprocessor.py` | Already fixed | `np.linalg.inv(self.H)` already applied |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|-------------|
| Pipeline crash mid-run | Low | Cached downloads survive re-run; pipeline is idempotent |
| Black NIR borders confuse downstream | Medium | Document as expected; defer to model-level handling |
| Disk fills during run | Low | 925 GB free, ~582 MB projected |
| OCI credentials expire | Low | Already configured and cached |

## Rollback Plan

Delete `data/augmented/` contents. The previous factor=1 run cannot be recovered (overwritten), but can be reproduced by resetting `homography_path: null` and `augmentation_factor: 1`.

## Dependencies

- Exploration: `sdd/pipeline-e2e-homography/explore` (completed)
- Bug fix applied: `preprocessor.py` H→H⁻¹ (line 142)
- Config applied: `homography_path` set in `config/augmentation.yaml` (line 14)

## Success Criteria

- [ ] Pipeline completes without errors
- [ ] ~93 `.pt` files in `data/augmented/` (31 pairs × 3 augmentation factor)
- [ ] Zero "No homography available" warnings in logs
- [ ] `dataset_index.json` has entries for all output files with `homography_applied: true`
- [ ] Spot-checked tensors: `rgb.shape=[3,640,640]`, `nir.shape=[1,640,640]`, values in normalized range