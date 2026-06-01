# Proposal: Fix DualFPN → YOLODetectionHead Level Mismatch

## Intent

The `DualFPN` neck produces 4 pyramid levels [P2, P3, P4, P5] but `YOLODetectionHead` expects exactly 3 strides [8, 16, 32] corresponding to [P3, P4, P5]. The assertion at `head.py:136` crashes with "Expected 3 FPN levels, got 4", making the MasterModel untrainable. This fix aligns the neck output with the head's contract by trimming P2 — the stride-4 level adds 4× compute with marginal benefit for mango damage detection.

## Scope

### In Scope
- Modify `DualFPN.__init__` to create 3 fusion_convs instead of 4
- Modify `DualFPN.forward()` to return `pyramid[1:]` (P3, P4, P5 only)
- Update `SingleFPN` and `DualFPN` docstrings to reflect 3-level output
- Fix `test_arch.py` to work with the corrected 3-level pyramid
- Verify `master_model.py` remains consistent (already documents P3–P5)

### Out of Scope
- Changing YOLODetectionHead to accept 4 levels (rejected: P2 not useful)
- Modifying backbone, fusion, or distillation modules
- Adding new detection strides or anchor configurations
- Performance benchmarking (separate concern)

## Capabilities

### New Capabilities
None — this is a bug fix, not a new feature.

### Modified Capabilities
- `testing-teacher-arch`: Requirement "MasterModel Integration" scenario "Full forward pass on CPU" MUST pass now that the pyramid has 3 matching levels. Requirement "YOLO Detection Head Output Format" implicitly assumes 3-level input, which is now satisfied.

## Approach

Trim P2 from DualFPN output. Two targeted edits in `neck.py`:

1. **`DualFPN.__init__`**: Change `for _ in range(4)` to `for _ in range(3)` in the `fusion_convs` ModuleList. The fusion now only operates on P3-P5 levels.

2. **`DualFPN.forward()`**: After building `unified_pyramid` with all 4 levels, return `unified_pyramid[1:]` to slice off P2.

This is minimal because P2 is computed internally by each SingleFPN (required for top-down pathway to generate P3), so the SingleFPN stays unchanged — we just don't pass P2 downstream.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/models/master/neck.py:132-140` | Modified | `fusion_convs` reduced from 4 to 3 levels |
| `src/models/master/neck.py:168` | Modified | `forward()` returns `pyramid[1:]` instead of full pyramid |
| `src/models/master/neck.py:1-18` | Modified | Module docstring updated to reflect [P3, P4, P5] output |
| `src/models/master/neck.py:152-153` | Modified | `forward()` docstring updated |
| `src/models/master/test_arch.py:41-43` | Modified | Test prints: FPN pyramid assertion/shape now expects 3 levels |
| `src/models/master/master_model.py` | Verified | No code change — docstring already says P3..P5 |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| P2 features lost for downstream tasks (e.g., segmentation) | Low | SingleFPN still computes P2 internally; can be exposed separately if needed |
| Pre-trained checkpoint incompatibility (if weights exist for 4 fusion_convs) | Low | No trained weights exist yet — this is Phase 1 architecture validation |
| Distillation training scripts assume 4-level pyramid | Low | No training scripts yet; `distill_projections.py` already uses 3-level FPN projections |

## Rollback Plan

Revert the two changes in `neck.py`: restore `range(4)` and remove `[1:]` slice. This restores 4-level output. If a future need for P2 arises, expose it via a separate method or parameter rather than changing the default output.

## Dependencies

- None — this is a self-contained architecture fix with no external dependencies.

## Success Criteria

- [ ] `DualFPN.forward()` returns exactly 3 tensors [P3, P4, P5] at 256 channels
- [ ] `MasterModel` forward pass completes without assertion error
- [ ] `test_arch.py` runs end-to-end successfully (all module + full model tests)
- [ ] `YOLODetectionHead` receives correct 3-level pyramid with matching strides [8, 16, 32]