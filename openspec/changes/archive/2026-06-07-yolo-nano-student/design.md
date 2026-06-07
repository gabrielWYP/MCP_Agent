# Design: YOLOv8-Nano Student Model

## Technical Approach

Implement a YOLOv8-Nano student model in PyTorch (no ultralytics dependency) that produces the exact same output dict format as the teacher `MasterModel`, enabling direct feature-level KD. Four modules: CSPDarknet-Nano backbone, PANet neck, DecoupledHead per FPN level, and `StudentModel` integration class. All channel dimensions locked to `ProjectionLayers` defaults: FPN `[128, 256, 256]`, backbone distill `[128, 256]`, head stems `[64, 128, 256]`.

## Architecture Decisions

### Decision: PANet over simple FPN

| Option | Tradeoff | Decision |
|--------|----------|----------|
| Simple FPN | Top-down only; semantic info flows down but no bottom-up localization path | Rejected |
| PANet | Top-down + bottom-up; bidirectional feature fusion, standard in YOLOv8 | **Chosen** |
| BiFPN | Learnable fusion weights; adds complexity for marginal gain on small model | Rejected |

Bottom-up path propagates fine-grained localization from P3→P5, critical for detection at multiple scales.

### Decision: Asymmetric FPN channels [128, 256, 256]

| Option | Tradeoff | Decision |
|--------|----------|----------|
| Uniform 128ch | Matches earlier Nano configs; under-capacity at P4/P5 | Rejected |
| Uniform 256ch | Over-capacity at P3 (80×80 = 6400 pixels) → excessive FLOPs | Rejected |
| [128, 256, 256] | Contracted with `ProjectionLayers`; lightweight P3, sufficient P4/P5 | **Chosen** |

This is not arbitrary — `fpn_projections` already defines `student_channels=[128, 256, 256]`. P3 runs on 6400 pixels; halving channels saves ~4× compute there.

### Decision: C2f over C3

| Option | Tradeoff | Decision |
|--------|----------|----------|
| C3 (YOLOv5) | Single bottleneck branch; heavier; weaker gradient flow | Rejected |
| C2f (YOLOv8) | Split into 2 branches + bottlenecks with concat; better gradient flow, fewer params | **Chosen** |
| ELAN | More branches; overkill for Nano width | Rejected |

C2f matches the YOLOv8 reference architecture and is the standard block for Nanos.

### Decision: Head stem channels per level [64, 128, 256]

Driven by `head_projections` contract (`student_channels=[64, 128, 256]`). Each `DecoupledHead` receives `fpn_ch[i]` channels and projects to `stem_ch[i]` via the 3×3 stem conv, then 1×1 pred conv. No alternative — must match the pre-configured projection layers.

## Data Flow

```
Input (B,3,640,640)
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ CSPDarknet-Nano                                      │
│  Stem: Conv(3→16,s=2) → Conv(16→32,s=2)            │
│  S1: C2f(32,n=1)       → 32ch,  160×160, stride 4   │
│  S2: Conv+C2f(64,n=2)  → 64ch,   80×80,  stride 8   │
│  S3: Conv+C2f(128,n=2) → 128ch,  40×40,  stride 16  │ ← distill_backbone[0]
│  S4: Conv+C2f(256,n=1)  → 256ch,  20×20,  stride 32  │ ← distill_backbone[1]
│      +SPPF(256,k=5)                                 │
└──────────────────────────────────────────────────────┘
    │ [S2, S3, S4]     S1 not used by neck
    ▼
┌──────────────────────────────────────────────────────┐
│ PANet Neck                                           │
│  Lateral: S2→128, S3→256, S4→256                    │
│  Top-down:  P5 ──upsample──┐ concat→C2f(256)→P4      │
│             P4 ──upsample──┐ concat→C2f(128)→P3      │
│  Bottom-up: P3 ──Conv s=2──┐ concat→C2f(256)→P4      │
│             P4 ──Conv s=2──┐ concat→C2f(256)→P5      │
│                                                      │
│  Output: [P3(128,80×80), P4(256,40×40), P5(256,20×20)]│
└──────────────────────────────────────────────────────┘
    │ [P3, P4, P5]
    ▼
┌──────────────────────────────────────────────────────┐
│ DecoupledHead × 3                                    │
│  P3: cls_stem(128→64)+cls_pred(64→2)                │
│      reg_stem(128→64)+reg_pred(64→4)                 │
│  P4: cls_stem(256→128)+cls_pred(128→2)               │
│      reg_stem(256→128)+reg_pred(128→4)               │
│  P5: cls_stem(256→256)+cls_pred(256→2)               │
│      reg_stem(256→256)+reg_pred(256→4)               │
└──────────────────────────────────────────────────────┘
    │
    ▼
  Output dict (7 keys)
```

## Channel Alignment Table

| Level | Teacher Ch | Projection | Student Ch | Student Source |
|-------|-----------|-------------|-------------|----------------|
| FPN P3 | 256 | fpn_projections | 128 | Neck P3 out |
| FPN P4 | 256 | fpn_projections | 256 | Neck P4 out |
| FPN P5 | 256 | fpn_projections | 256 | Neck P5 out |
| Backbone S3 | 384 | backbone_projections | 128 | Backbone S3 out |
| Backbone S4 | 768 | backbone_projections | 256 | Backbone S4 out |
| Head cls P3 | 256 | head_projections | 64 | cls_stem out |
| Head cls P4 | 256 | head_projections | 128 | cls_stem out |
| Head cls P5 | 256 | head_projections | 256 | cls_stem out |
| Head reg P3 | 256 | head_projections | 64 | reg_stem out |
| Head reg P4 | 256 | head_projections | 128 | reg_stem out |
| Head reg P5 | 256 | head_projections | 256 | reg_stem out |

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/models/student/backbone.py` | Create | CSPDarknet-Nano: Conv stems + 4 C2f stages + SPPF |
| `src/models/student/neck.py` | Create | PANet: top-down FPN + bottom-up path, lateral convs |
| `src/models/student/head.py` | Create | DecoupledHead per level with cls/reg stems, distill outputs |
| `src/models/student/student_model.py` | Create | Composes backbone→neck→head, returns 7-key dict |
| `src/models/student/__init__.py` | Create | Public API: StudentModel, CSPDarknetNano, PANet, YOLOStudentHead |
| `src/models/__init__.py` | Modify | Add `StudentModel` export |
| `tests/models/student/__init__.py` | Create | Package init |
| `tests/models/student/test_backbone.py` | Create | Shape correctness, S3/S4 distill outputs, stage channels |
| `tests/models/student/test_neck.py` | Create | PANet output shapes match [128,256,256], spatial sizes |
| `tests/models/student/test_head.py` | Create | DecoupledHead shapes, distill stem channels |
| `tests/models/student/test_student_model.py` | Create | 7-key forward, projection compatibility, param count |
| `tests/models/student/conftest.py` | Create | Shared fixtures: model, input tensor, CPU guard |

## Interfaces / Contracts

```python
# StudentModel.forward contract
def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
    """
    Args: x — (B, 3, 640, 640) RGB
    Returns dict with keys:
      'preds'            : [(B,6,80,80), (B,6,40,40), (B,6,20,20)]
      'cls_preds'        : [(B,2,80,80), (B,2,40,40), (B,2,20,20)]
      'reg_preds'        : [(B,4,80,80), (B,4,40,40), (B,4,20,20)]
      'distill_backbone' : [(B,128,80,80), (B,256,40,40)]
      'distill_fpn'      : [(B,128,80,80), (B,256,40,40), (B,256,20,20)]
      'distill_head_cls' : [(B,64,80,80), (B,128,40,40), (B,256,20,20)]
      'distill_head_reg' : [(B,64,80,80), (B,128,40,40), (B,256,20,20)]
    """
```

## Testing Strategy

| Layer | What | Approach |
|-------|------|----------|
| Unit | Backbone stage shapes | Forward `(2,3,640,640)`; assert each stage channel/size |
| Unit | Backbone distill outputs | S3=128ch, S4=256ch match `backbone_projections` contract |
| Unit | PANet neck shapes | Output [128,256,256] at [80×80, 40×40, 20×20] |
| Unit | Head prediction shapes | `preds[i]=(B,6,H,W)`, `cls=(B,2,H,W)`, `reg=(B,4,H,W)` |
| Unit | Head stem channels | `distill_cls[i]` and `distill_reg[i]` at [64,128,256]ch |
| Integration | StudentModel forward 7 keys | All shapes match contract table above |
| Integration | Projection compatibility | Student distill outputs pass through `fpn_projections()`, `backbone_projections()`, `head_projections()` without error |
| Integration | Param count ~3.2M | Assert 2.7M ≤ params ≤ 3.7M |

## Migration / Rollback

No migration required. Pure addition: `src/models/student/` and `tests/models/student/`. Rollback: delete those directories. No existing files modified except `src/models/__init__.py` (add export).

## Open Questions

- [ ] C2f repeat counts (n=1,2,2,1 for backbone, n=1 for all neck) — confirm param count falls in 3.2M ±0.5M range during implementation