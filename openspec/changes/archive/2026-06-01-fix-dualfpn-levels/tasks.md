# Tasks: Fix DualFPN Level Mismatch

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~30-50 |
| 400-line budget risk | Low |
| Chained PRs recommended | No |
| Suggested split | Single PR |
| Delivery strategy | ask-always |
| Chain strategy | not needed |

Decision needed before apply: No
Chained PRs recommended: No
Chain strategy: not needed
400-line budget risk: Low

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | Trim P2 from DualFPN output, fix test | PR 1 | Single self-contained bugfix |

## Phase 1: Foundation — Trim DualFPN Output

- [x] 1.1 In `src/models/master/neck.py`, change `DualFPN.__init__` fusion_convs from `range(4)` to `range(3)`
- [x] 1.2 In `src/models/master/neck.py`, update `DualFPN.forward()` to iterate with `zip(rgb_pyramid[1:], nir_pyramid[1:], self.fusion_convs)` instead of enumerate over all 4 levels
- [x] 1.3 Update module-level docstring in `neck.py` to reflect output `[P3, P4, P5]` instead of `[P2, P3, P4, P5]`
- [x] 1.4 Update `DualFPN.forward()` docstring return description to `[P3, P4, P5]`

## Phase 2: Testing — Fix and Verify

- [x] 2.1 In `src/models/master/test_arch.py`, add explicit `assert len(pyramid) == 3` after `neck(fused, nir_pass)` call
- [x] 2.2 Run `python -m src.models.master.test_arch` and confirm full pipeline passes with 3-level pyramid
- [x] 2.3 Verify MasterModel forward returns dict with 8 keys and no assertion errors from YOLODetectionHead
