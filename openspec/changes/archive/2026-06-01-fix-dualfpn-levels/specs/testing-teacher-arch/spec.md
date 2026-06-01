# Delta for testing-teacher-arch

## ADDED Requirements

### Requirement: DualFPN Pyramid Output

`DualFPN` SHALL output exactly 3 pyramid levels [P3, P4, P5] at strides [8, 16, 32].

- `DualFPN.forward()` MUST return a list of exactly 3 tensors, each with 256 channels.
- P2 (stride 4) SHALL still be computed internally for the top-down pathway but MUST NOT appear in output.
- Output strides MUST correspond to [8, 16, 32] for P3, P4, P5 respectively.

#### Scenario: DualFPN returns 3-level pyramid

- GIVEN DualFPN initialized with 256 channels
- WHEN forward receives fused features from backbone stages
- THEN output is exactly 3 tensors [P3, P4, P5] with shapes matching strides [8, 16, 32]
- AND P2 is excluded from output despite being computed internally

#### Scenario: DualFPN output feeds YOLODetectionHead without error

- GIVEN DualFPN output of 3 tensors [P3, P4, P5]
- WHEN output is passed to YOLODetectionHead
- THEN no assertion error occurs; head processes all 3 levels correctly

## MODIFIED Requirements

### Requirement: YOLO Detection Head Output Format

`YOLODetectionHead` SHALL produce anchor-free detections in YOLO-compatible format with decoupled cls/reg branches.

- Input MUST be exactly 3 FPN levels at strides [8, 16, 32] corresponding to [P3, P4, P5].
- `preds`: 3 tensors (B, nc+4, H_i, W_i) at strides [8,16,32], nc=2.
- `cls_preds`: 3 tensors (B, 2, H_i, W_i). `reg_preds`: 3 tensors (B, 4, H_i, W_i).
- `distill_cls`/`distill_reg` features MUST be exposed per level for KD.

(Previously: input contract was implicit — now explicitly requires exactly 3 FPN levels)

#### Scenario: Head output shapes match YOLO format

- GIVEN YOLODetectionHead(fpn_channels=256, num_classes=2)
- WHEN forward receives pyramid [P3,P4,P5] at strides [8,16,32]
- THEN preds[i] has 6 channels, cls_preds[i] has 2 channels, reg_preds[i] has 4 channels

#### Scenario: Decoupled branches produce non-degenerate outputs

- GIVEN DecoupledHead with random weights
- WHEN forward processes a 256-channel feature map
- THEN cls_out and reg_out are non-zero (independent branches, not collapsed)

#### Scenario: Head rejects mismatched FPN level count

- GIVEN YOLODetectionHead configured for 3 strides
- WHEN forward receives a pyramid with ≠ 3 levels
- THEN an assertion error MUST be raised indicating expected vs actual level count

### Requirement: MasterModel Integration

The full pipeline SHALL produce all required dict keys and `freeze_backbone` SHALL correctly control gradient flow.

- DualFPN MUST output exactly 3 pyramid levels [P3, P4, P5] before passing to YOLODetectionHead.
- Forward MUST return dict with 8 keys: preds, cls_preds, reg_preds, distill_backbone_rgb, distill_backbone_fused, distill_fpn, distill_head_cls, distill_head_reg.
- `freeze_backbone(n)` MUST set requires_grad=False on stems + first n stages.
- `count_parameters()` MUST return positive counts per module.

(Previously: no explicit DualFPN→Head contract; now specifies 3-level requirement)

#### Scenario: Full forward pass on CPU

- GIVEN MasterModel(pretrained_backbone=False) on CPU
- WHEN forward(rgb=(B,3,640,640), nir=(B,1,640,640))
- THEN output dict has exactly 8 keys; all tensor shapes match specification
- AND DualFPN passes exactly 3 pyramid tensors to YOLODetectionHead without assertion error

#### Scenario: Freeze backbone disables correct stages

- GIVEN MasterModel with freeze_backbone(freeze_stages=2)
- THEN stems and stages[0..1] have requires_grad=False; stages[2..3], fusion, neck, head have requires_grad=True