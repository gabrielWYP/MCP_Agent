# Tasks: Pipeline E2E with Homography Alignment

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | 10–30 (script already exists, minor fixes only) |
| 400-line budget risk | Low |
| Chained PRs recommended | No |
| Suggested split | Single PR |
| Delivery strategy | single-pr (800-line budget) |
| Chain strategy | size-exception |

Decision needed before apply: No
Chained PRs recommended: No
Chain strategy: size-exception
400-line budget risk: Low

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | Run pipeline, verify all 6 gates, fix any runner issues | PR 1 | Single PR; base: main; includes runner fixes + verification log |

## Phase 1: Pre-Run Validation

- [x] 1.1 Verify `scripts/run_pipeline.py` runs without import errors (`python3 scripts/run_pipeline.py --help`)
- [x] 1.2 Verify `config/augmentation.yaml` has `homography_path`, `augmentation_factor: 3`, `pair_matching: timestamp_id`
- [x] 1.3 Verify `data/cache/` contains 31+ cached pairs (no new downloads expected)
- [x] 1.4 Delete contents of `data/augmented/` to ensure clean-slate run

## Phase 2: Pipeline Execution

- [x] 2.1 Run `python3 scripts/run_pipeline.py` and capture stdout + stderr
- [x] 2.2 Confirm pipeline exits with code 0 (`stats["skipped"] == 0` and `stats["augmented"] > 0`)
- [x] 2.3 Scan captured stderr/stdout for `"No homography available"` warnings — expect zero matches

## Phase 3: Output Verification (6 Gates)

- [x] 3.1 Gate 1 — File count: assert `len(glob("data/augmented/*.pt")) ≈ 93` (31 pairs × 3 factor)
- [x] 3.2 Gate 2 — Manifest: load `data/augmented/dataset_index.json`, verify entry count matches `.pt` file count, all entries have `homography_applied: true`
- [x] 3.3 Gate 3 — Homography applied: zero `"No homography available"` in logs (re-confirmed from Phase 2)
- [x] 3.4 Gate 4 — Tensor shapes: load 5 random `.pt` files, assert `rgb.shape == (3,640,640)` and `nir.shape == (1,640,640)`
- [x] 3.5 Gate 5 — Value ranges: from same 5 samples, assert RGB in ImageNet-normalized range (~-2.5 to +2.5), NIR in dataset-normalized range
- [x] 3.6 Gate 6 — Cache integrity: confirm no new files in `data/cache/` (all pairs pre-cached)

## Phase 4: Documentation & Cleanup

- [x] 4.1 Document black NIR border behavior (~33% out-of-bounds pixels) as expected artifact in verification notes
- [x] 4.2 Commit `scripts/run_pipeline.py` if not already tracked (utility script for reuse)
- [x] 4.3 On failure: leave `data/augmented/` contents for debugging; document which gates failed