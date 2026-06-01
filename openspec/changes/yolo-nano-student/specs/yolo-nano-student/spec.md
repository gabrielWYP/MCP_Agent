# YOLOv8-Nano Student Specification

## Purpose

YOLOv8-Nano student model (backbone + PANet neck + decoupled head) producing teacher-compatible output for knowledge distillation. Single-modality RGB input (N,3,640,640).

## Channel Contract

| Level | Teacher Ch | Student Ch | Projection Source |
|-------|-----------|-----------|-------------------|
| FPN P3 | 256 | 128 | `fpn_projections` |
| FPN P4 | 256 | 256 | `fpn_projections` |
| FPN P5 | 256 | 256 | `fpn_projections` |
| Backbone S3→S4 | 384, 768 | 128, 256 | `backbone_projections` |
| Head stem P3→P5 | 256×3 | 64, 128, 256 | `head_projections` |

## Interface: StudentModel.forward(x)

| Key | Shape (×3 levels) | Note |
|-----|--------------------|------|
| `preds` | (N,6,80,80), (N,6,40,40), (N,6,20,20) | cls+reg concatenated |
| `cls_preds` | (N,2,80,80), (N,2,40,40), (N,2,20,20) | 2-class logits |
| `reg_preds` | (N,4,80,80), (N,4,40,40), (N,4,20,20) | xywh bbox |
| `distill_backbone` | (N,128,80,80), (N,256,40,40) | S3, S4 only |
| `distill_fpn` | (N,128,80,80), (N,256,40,40), (N,256,20,20) | P3, P4, P5 |
| `distill_head_cls` | (N,64,80,80), (N,128,40,40), (N,256,20,20) | cls_stem outputs |
| `distill_head_reg` | (N,64,80,80), (N,128,40,40), (N,256,20,20) | reg_stem outputs |

Strides: P3=8, P4=16, P5=32. Shapes assume input (N,3,640,640).

## Requirements

### Requirement: CSPDarknet-Nano Backbone

SHALL provide a 4-stage CSPDarknet-Nano backbone with SPPF on the final stage. Input: (N,3,640,640). Output channels per stage: [16,32,64,128]. MUST expose stages S3 (128ch) and S4 (256ch) as `distill_backbone`.

- GIVEN default backbone initialization
- WHEN forwarding (2,3,640,640)
- THEN stage channels are [16,32,64,128] and spatial sizes [160,80,40,20]

- GIVEN backbone S3 and S4 outputs
- THEN S3=128ch and S4=256ch match `backbone_projections` student_channels

### Requirement: PANet Neck

SHALL provide PANet neck producing 3 FPN levels at channels [128,256,256] matching `fpn_projections` defaults. MUST use top-down + bottom-up aggregation.

- GIVEN backbone features [S1..S4]
- WHEN processed through PANet neck
- THEN 3 outputs at channels [128,256,256] and spatial sizes [80,40,20]

- GIVEN FPN outputs
- THEN channels match `fpn_projections` student_channels [128,256,256]

### Requirement: Decoupled Detection Head

SHALL provide a decoupled head per FPN level with cls_stem (3×3) and reg_stem (3×3) branches. Head stem channels per level MUST be [64,128,256] matching `head_projections` defaults.

- GIVEN FPN [P3,P4,P5] at channels [128,256,256]
- WHEN processed through decoupled head
- THEN preds[i] shape (N,6,H_i,W_i) at strides [8,16,32]

- GIVEN head cls/reg distill outputs
- THEN stem channels are [64,128,256] matching `head_projections` student_channels

### Requirement: StudentModel Integration

SHALL compose backbone + neck + head with `forward(x)` returning a dict with 7 keys. Params ≈ 3.2M (±0.5M).

- GIVEN `StudentModel()` default config
- WHEN forwarding (2,3,640,640)
- THEN dict has keys: preds, cls_preds, reg_preds, distill_backbone, distill_fpn, distill_head_cls, distill_head_reg

- GIVEN `StudentModel()` on CPU eval mode
- WHEN forwarding torch.randn(1,3,640,640)
- THEN all shapes match interface table; no CUDA/network calls

- GIVEN student `distill_fpn` and teacher `fpn_projections()`
- WHEN student FPN features passed to `ProjectionLayers.forward()`
- THEN projection completes without shape errors