# Tasks: YOLOv8-Nano Student Model

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~700 (range 650–850) |
| 400-line budget risk | High |
| Chained PRs recommended | Yes |
| Suggested split | PR 1 → PR 2 → PR 3 |
| Delivery strategy | ask-always |
| Chain strategy | pending (ask user) |

Decision needed before apply: Yes
Chained PRs recommended: Yes
Chain strategy: pending
400-line budget risk: High

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | Foundation blocks + CSPDarknet-Nano backbone + backbone tests | PR 1 | base: main; ~260 lines; verifiable via stage shape assertions |
| 2 | PANet neck + DecoupledHead + neck/head unit tests | PR 2 | depends on PR 1; ~330 lines; tests wire backbone→neck→head |
| 3 | StudentModel integration + exports + integration tests | PR 3 | depends on PR 2; ~210 lines; 7-key forward, projection compat, param count |

## Phase 1: Foundation & Backbone

- [x] 1.1 Create `src/models/student/__init__.py` — empty package init
- [x] 1.2 Create `src/models/student/backbone.py` with: `Conv_BN_SiLU(in,out,k,s)`, `Bottleneck(in,out)` (1×1→3×3 + residual), `C2f(in,out,n)` (split 2 branches + n bottlenecks + concat), `SPPF(in,out,k=5)` (3×MaxPool + concat + 1×1), `CSPDarknetNano` (Stem Conv 3→16 s=2 + Conv 16→32 s=2; S1: C2f(32,n=1); S2: Conv+C2f(64,n=2); S3: Conv+C2f(128,n=2); S4: Conv+C2f(256,n=1)+SPPF). Forward returns `(stages_list, (s3, s4))` for neck and distill.
- [x] 1.3 Create `tests/models/student/__init__.py` and `tests/models/student/test_backbone.py` — forward `(2,3,640,640)`, assert stage channels `[32,64,128,256]` and spatial sizes `[160,80,40,20]`; assert S3=128ch, S4=256ch match `backbone_projections` student_channels

## Phase 2: Neck & Head

- [ ] 2.1 Create `src/models/student/neck.py` — `PANet` with lateral convs (S2→128, S3→256, S4→256), top-down path (upsample + concat + C2f), bottom-up path (Conv s=2 downsample + concat + C2f). Output: `[P3(128,80×80), P4(256,40×40), P5(256,20×20)]`
- [ ] 2.2 Create `src/models/student/head.py` — `DecoupledHead(fpn_ch, stem_ch, num_classes)` with cls_stem (3×3 conv→SiLU), reg_stem (3×3 conv→SiLU), cls_pred (1×1→num_classes), reg_pred (1×1→4). `YOLOStudentHead` instantiates 3 heads with `fpn_channels=[128,256,256]`, `stem_channels=[64,128,256]`. Forward returns preds, cls_preds, reg_preds, distill_head_cls, distill_head_reg
- [ ] 2.3 Create `tests/models/student/test_neck.py` — feed backbone features, assert output channels `[128,256,256]` at spatial sizes `[80,40,20]`
- [ ] 2.4 Create `tests/models/student/test_head.py` — feed FPN features, assert `preds[i]=(B,6,H,W)`, `cls=(B,2,H,W)`, `reg=(B,4,H,W)`; distill stems at `[64,128,256]`ch

## Phase 3: Integration & Verification

- [ ] 3.1 Create `src/models/student/student_model.py` — `StudentModel` composes backbone→neck→head. Forward returns 7-key dict. `count_parameters()` method. CPU eval mode by default
- [ ] 3.2 Update `src/models/__init__.py` — add `StudentModel` to imports and `__all__`; update `src/models/student/__init__.py` with public API exports
- [ ] 3.3 Create `tests/models/student/conftest.py` — shared fixtures: `device` (CPU), `batch_input` (N,3,640,640), `student_model`
- [ ] 3.4 Create `tests/models/student/test_student_model.py` — verify 7-key dict with all shapes per spec table; projection compatibility (distill outputs pass through `ProjectionLayers`); param count 2.7M–3.7M
