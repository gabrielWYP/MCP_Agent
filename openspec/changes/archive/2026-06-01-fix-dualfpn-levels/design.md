# Design: Fix DualFPN → YOLODetectionHead Level Mismatch

## Technical Approach

Trim P2 from `DualFPN` output so the neck emits exactly 3 levels [P3, P4, P5], matching `YOLODetectionHead.strides = [8, 16, 32]`. `SingleFPN` remains unchanged — it still computes P2 internally as part of the top-down additive pathway, but `DualFPN` no longer fuses or returns it.

## Architecture Decisions

| Decision | Choice | Alternatives | Rationale |
|----------|--------|--------------|-----------|
| Trim mechanism | 3 fusion convs + zip over `pyramid[1:]` | Keep 4 convs, return `pyramid[1:]` | Avoids fusing P2 that gets discarded, saving parameters and compute. Correctness is explicit in the loop, not via post-hoc slice. |
| SingleFPN changes | None | Add `num_levels` param | P2 is still required for the top-down pathway. Modifying `SingleFPN` would break FPN internals. |
| Docstring scope | Update `DualFPN` only | Also scrub `SingleFPN` docstring | `SingleFPN` still outputs 4 levels; its docstring must remain accurate. |

## Data Flow

```
Backbone [S1..S4]
    │
    ├─→ SingleFPN ──→ [P2, P3, P4, P5]
    │                      │
    │                      └── P2 dropped at DualFPN boundary
    │                      │
    └─→ SingleFPN ──→ [P2, P3, P4, P5]
                           │
DualFPN.forward: zip(P3..P5, fusion_convs[0..2])
                           │
                           └──→ [P3, P4, P5] ──→ YOLODetectionHead
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/models/master/neck.py` | Modify | `DualFPN.__init__`: `range(4)` → `range(3)` in `fusion_convs`. `DualFPN.forward`: iterate with `zip(rgb_pyramid[1:], nir_pyramid[1:], self.fusion_convs)`. Update module-level and `forward` docstrings to `[P3, P4, P5]`. |
| `src/models/master/test_arch.py` | Modify | Add explicit `assert len(pyramid) == 3` after `neck(fused, nir_pass)` to prevent regression. |

## Interfaces / Contracts

- **`DualFPN.forward()`**: Returns `list[Tensor]` of length 3, levels `[P3, P4, P5]`, each `(B, 256, H_i, W_i)`.
- **`SingleFPN.forward()`**: Unchanged — still returns `list[Tensor]` of length 4, levels `[P2, P3, P4, P5]`.
- No changes to `YOLODetectionHead`, `MasterModel`, or `ProjectionLayers` — they already expect 3 levels.

## Testing Strategy

| Layer | What to Test | Approach |
|-------|-------------|----------|
| Unit | `DualFPN` returns 3 tensors | `test_arch.py` script |
| Integration | End-to-end `MasterModel` forward | `test_arch.py` full pipeline |

Run: `python -m src.models.master.test_arch`

## Migration / Rollout

No migration required. No trained weights exist.

## Open Questions

None.
