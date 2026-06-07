# Proposal: YOLOv8-Nano Student Model with Knowledge Distillation Interface

## Intent

The teacher pipeline (`src/models/master/`) is complete with 8 distillation output keys and projection layers pre-configured for student channels `[128, 256, 256]` at FPN levels. However, the student model receiving those projections does not exist. We need a YOLOv8-Nano implementation in PyTorch that produces the exact same output format as the teacher head, enabling direct feature-level KD training in a future phase.

## Scope

### In Scope
- YOLOv8-Nano backbone (CSPDarknet-Nano: Conv stems + C2f blocks + SPPF) in `src/models/student/backbone.py`
- PANet neck (FPN top-down + bottom-up, 128ch uniform → P4/P5 at 256ch) in `src/models/student/neck.py`
- Decoupled detection head (cls_stem/reg_stem per level, expose `distill_cls`/`distill_reg`) in `src/models/student/head.py`
- `StudentModel` integration class exposing 6 distill-compatible output keys in `src/models/student/student_model.py`
- Shape validation tests in `tests/models/student/` matching `tests/models/master/` patterns
- `__init__.py` with public API exports

### Out of Scope
- KD training loop / loss functions (future change)
- Detection dataset loader (YOLO format)
- YOLO loss, DFL, task-aligned matching, bbox loss
- Pretrained weight loading or transfer learning
- Inference optimization / ONNX export

## Capabilities

### New Capabilities
- `yolo-nano-student`: YOLOv8-Nano architecture (backbone + PANet neck + decoupled head) producing teacher-compatible output format for knowledge distillation

### Modified Capabilities
- None — this is a new module with no spec-level changes to existing capabilities

## Approach

Implement YOLOv8-Nano from scratch in PyTorch (no ultralytics dependency), matching the teacher's interface contract:
- **Backbone**: 4-stage CSPDarknet-Nano with C2f blocks, SPPF, output channels `[16, 32, 64, 128]`
- **Neck**: PANet producing 3 pyramid levels at channels `[128, 256, 256]` (F3, F4, F5) — matching `ProjectionLayers` student_channels defaults
- **Head**: `DecoupledHead` per level (cls_stem 3×3→cls_pred 1×1, reg_stem 3×3→reg_pred 1×1), exposing `distill_cls`/`distill_reg` stems
- **Output dict**: 6 keys mirroring teacher: `preds`, `cls_preds`, `reg_preds`, `distill_backbone`, `distill_fpn`, `distill_head_cls`, `distill_head_reg`

Single-modality (RGB-only) input `(N, 3, 640, 640)`. ~3.2M params total.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/models/student/` | New | Entire student model package |
| `tests/models/student/` | New | Shape validation test suite |
| `src/models/master/distill_projections.py` | None | Already has correct student_channels defaults |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Channel mismatch with projection layers | Low | ProjectionLayers already defines student_channels=[128,256,256]; student must match exactly |
| C2f block implementation differs from YOLOv8 reference | Med | Follow ultralytics C2f structure precisely; validate in tests |
| Neck channel routing produces wrong spatial dims | Med | PANet concat requires matching spatial sizes; test with 640×640 synthetic input |
| Head stem channels don't match head_projections defaults | Low | head_projections expects [64,128,256]; head cls/reg stems must output those channels |

## Rollback Plan

Delete `src/models/student/` and `tests/models/student/`. No existing code is modified — this is purely additive.

## Dependencies

- PyTorch 2.1+ (already in requirements.txt)
- pytest (already configured)

## Success Criteria

- [ ] `StudentModel()` forward pass on CPU with `(N,3,640,640)` produces dict with 6 expected keys
- [ ] `preds[i]` shape = `(N, 6, H_i, W_i)` at strides [8,16,32] — matches teacher format
- [ ] FPN output channels = `[128, 256, 256]` — matches ProjectionLayers defaults
- [ ] Head stem channels = `[64, 128, 256]` — matches head_projections defaults
- [ ] Total params ~3.2M (tolerance ±0.5M)
- [ ] All shape validation tests pass on CPU, no network calls