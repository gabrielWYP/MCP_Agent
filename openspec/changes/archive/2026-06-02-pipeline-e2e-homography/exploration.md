## Exploration: pipeline-e2e-homography

### Current State
The data augmentation pipeline was run in test mode with `augmentation_factor=1` and no homography alignment. It produced 31 `.pt` files (one per pair, `_aug0` only) with `homography_applied=False`. The pipeline components — OCI listing, pair discovery by timestamp ID, parallel download, preprocessing, augmentation, and vector store — all work correctly. The existing run used resize fallback for NIR alignment.

### Affected Areas
- `src/data_pipeline/preprocessor.py` — **has a homography direction bug** (see Component Status below)
- `config/augmentation.yaml` — needs `homography_path` set
- `data/augmented/` — output will be overwritten/expanded (31→93 files)
- `data/cache/mango/` — 31 RGB+31 NIR images already cached (no re-download needed)
- `tests/data_pipeline/test_preprocessor.py` — has a test for valid homography but uses identity matrix (hides the direction bug)

### Component Status

| Component | Status | Details |
|-----------|--------|---------|
| Homography matrix | ✅ Valid | 3×3 float64, det=0.92, non-singular. Near-identity with ~120px translation offset. |
| Preprocessor (load) | ✅ Ready | `config.homography_path` → `np.load()` → `self.H` works correctly |
| Preprocessor (warp) | ❌ **BUG** | Uses `cv2.warpPerspective(nir, H, (w,h))` but should use `cv2.warpPerspective(nir, np.linalg.inv(H), (w,h))`. H maps RGB→NIR (from `cv2.findHomography(pts_rgb, pts_nir)`), but `warpPerspective` expects M to map src→dst. The bug causes 34.5% of pixels to go out-of-bounds and 88.6% of pixels differ >5 from the correct warp. |
| Augmentation | ✅ Ready | `augmentation_factor=3` (already set in YAML) produces identity + 2 random transforms. Spatial (RandomResizedCrop, flips, rotate) applied to 4-channel stack. Photometric (ColorJitter, Blur, Noise) applied per-modality. |
| Pipeline orchestrator | ✅ Ready | Chains all components correctly. Handles corrupt pairs gracefully (skips). Writes manifest. |
| Pair discovery | ✅ Ready | `timestamp_id` mode correctly matches `mango_rgb_{id}` / `mango_nir_{id}` patterns |
| OCI download | ✅ Ready | 31 pairs already cached. Retry with exponential backoff (3 attempts). |
| Vector store | ✅ Ready | Saves `.pt` files (6.25 MB each) + `dataset_index.json` manifest with homography flag |
| Data quality | ✅ Excellent | All 31 pairs valid, 720×1280, 31/31 matched RGB+NIR |
| Disk space | ✅ Safe | 925 GB free. Projected: ~582 MB for 93 files (31×3). Current: 194 MB. |
| Existing tests | ⚠️ Partial | `test_preprocessor.py` uses identity matrix (doesn't catch direction bug). `test_pipeline.py` covers full flow with mock pairs. |

### Homography Direction Bug — Detailed Analysis

**Root cause**: `homografia_script.py` computes H with `cv2.findHomography(pts_rgb, pts_nir)`, mapping RGB→NIR. The preprocessor calls `cv2.warpPerspective(nir, H, (w, h))`, which internally computes `dst(x,y) = src(H⁻¹ * [x,y,1])`. Since H maps RGB→NIR, H⁻¹ maps NIR→RGB. So the code is warping NIR coordinates *back to RGB space* instead of forward-mapping RGB coordinates to NIR space.

**Impact**: Using the actual matrix on real images shows:
- Black pixels (out-of-bounds): 34.5% of output
- Pixels differing >5 from correct warp: 88.6%
- Mean absolute difference: 82.3 (out of 255)
- This is NOT a cosmetic issue — the alignment is substantially wrong

**Fix** (1 line):
```python
# preprocessor.py, line 141 — CHANGE:
warped = cv2.warpPerspective(nir, self.H, (w, h))
# TO:
warped = cv2.warpPerspective(nir, np.linalg.inv(self.H), (w, h))
```

**Verification with fix**: Fixing the direction reduces black pixels to 33.6% (still expected near image borders due to translation). The mean pixel value shifts from 63.1 to 79.6 (NIR content is correctly positioned).

**Note on black border**: Even with the fix, ~33% of pixels are black (out-of-bounds) because the cameras have a ~120px physical offset. This means the NIR image doesn't cover the full RGB FOV. This is physically correct and must be handled downstream (the model should learn to accept partially-valid NIR input, or cropping/padding could be added in a follow-up).

### Config Changes Needed

Edit `config/augmentation.yaml`:

```yaml
# Line 14 — CHANGE:
homography_path: null
# TO:
homography_path: "notebooks/matriz_homografia_aruco.npy"
```

All other settings are already correct for this run:
- `augmentation_factor: 3` ✅
- `target_size: [640, 640]` ✅
- `pair_matching: "timestamp_id"` ✅
- `min_timestamp: 1779580800` ✅ (May 24, 2026 onwards)

### Approaches

1. **Fix preprocessor + config → run pipeline** (Recommended)
   - Pros: One-line fix, config change is trivial, cached data means no OCI download needed, ~3 min runtime
   - Cons: Black NIR borders (~33%) remain — downstream model must handle
   - Effort: Low

2. **Fix preprocessor + add border cropping → run pipeline**
   - Pros: No black borders in output, cleaner training data
   - Cons: Loses valid RGB pixels at edges, adds complexity, requires determining crop region from homography
   - Effort: Medium

3. **Run as-is with H (current bugged direction)**
   - Pros: No code changes
   - Cons: NIR alignment is WRONG for 88.6% of pixels. Training data would be corrupted. Do not do this.
   - Effort: N/A (not recommended)

### Recommendation

**Approach 1** — Fix the preprocessor direction bug (1 line) + set `homography_path` in config. Then run the pipeline. This is the minimum viable fix to get correct homography alignment.

The black NIR borders are a known consequence of the camera offset. They should be handled at the model/training level or in a follow-up change. Adding cropping/padding logic to the preprocessor now would delay the run without understanding what the model actually needs.

### Risks
- **Homography direction bug** (HIGH) — Must fix before running. The current code produces significantly corrupted NIR alignment.
- **Black NIR borders** (MEDIUM) — ~33% of NIR pixels will be black after correct homography warp due to camera offset. The downstream model (`MasterModel`) must handle partially-valid NIR input. If the model was trained expecting full-coverage NIR, this could degrade accuracy.
- **Security: hardcoded credentials** (LOW) — OCI access key and secret are in `config/augmentation.yaml`. Existing practice, but should be moved to env vars in a future change.
- **Source OCI key missing** (LOW) — The current manifest has `source_oci_key: ""` because the download step doesn't propagate it. The field exists in `ManifestEntry` but isn't populated. This is a pre-existing gap, not new to this change.
- **No test for real homography direction** (MEDIUM) — `test_preprocessor.py` uses identity matrix for homography tests, which masks direction bugs. A test with a non-identity matrix would have caught this.

### Ready for Proposal
**No** — Requires one code fix first (homography direction bug in `preprocessor.py`, line 141). After that fix is applied, the change is ready to propose and run. The config change is trivial (one line).
