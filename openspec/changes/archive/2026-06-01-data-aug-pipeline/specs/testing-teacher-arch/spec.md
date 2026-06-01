# Delta for testing-teacher-arch

## MODIFIED Requirements

### Requirement: MasterModel Integration

The full pipeline SHALL produce all required dict keys and `freeze_backbone` SHALL correctly control gradient flow.

- DualFPN MUST output exactly 3 pyramid levels [P3, P4, P5] before passing to YOLODetectionHead.
- Forward MUST return dict with 8 keys: preds, cls_preds, reg_preds, distill_backbone_rgb, distill_backbone_fused, distill_fpn, distill_head_cls, distill_head_reg.
- `freeze_backbone(n)` MUST set requires_grad=False on stems + first n stages.
- `count_parameters()` MUST return positive counts per module.
- Data pipeline output tensors (rgb=(N,3,H,W), nir=(N,1,H,W)) MUST be accepted by MasterModel.forward() without shape errors when produced by the augmentation pipeline.

(Previously: No mention of data pipeline tensor validation; MasterModel integration tested with random synthetic tensors only.)

#### Scenario: Full forward pass on CPU

- GIVEN MasterModel(pretrained_backbone=False) on CPU
- WHEN forward(rgb=(B,3,640,640), nir=(B,1,640,640))
- THEN output dict has exactly 8 keys; all tensor shapes match specification
- AND DualFPN passes exactly 3 pyramid tensors to YOLODetectionHead without assertion error

#### Scenario: Freeze backbone disables correct stages

- GIVEN MasterModel with freeze_backbone(freeze_stages=2)
- THEN stems and stages[0..1] have requires_grad=False; stages[2..3], fusion, neck, head have requires_grad=True

#### Scenario: Data pipeline tensors accepted by MasterModel forward

- GIVEN data pipeline output of rgb_tensor shape (3, 640, 640) and nir_tensor shape (1, 640, 640) loaded from a .pt file
- WHEN unsqueezed to batch dim and passed to MasterModel.forward(rgb, nir)
- THEN the model produces a valid output dict with all 8 keys
- AND no shape mismatch assertion is raised

#### Scenario: Preprocessed NIR single-channel accepted by NIR stem

- GIVEN a preprocessed NIR tensor of shape (1, 640, 640) from the preprocessing pipeline
- WHEN passed through DualConvNeXtBackbone.nir_stem
- THEN the output shape is (1, 96, 160, 160) without error