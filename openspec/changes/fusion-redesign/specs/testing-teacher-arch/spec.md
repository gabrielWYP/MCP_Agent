# Delta for testing-teacher-arch

## MODIFIED Requirements

### Requirement: ConvNeXtV2 Backbone Shapes

`EarlyFusionBackbone(pretrained=False)` SHALL produce correct channel and spatial shapes
for the single 4-channel stream.

- The 4-channel stem MUST output (N, 96, H/4, W/4).
- 4 stage output channels: [96, 192, 384, 768].
- For 640x640 input, spatial dims: [160^2, 80^2, 40^2, 20^2].
(Previously: two streams — separate RGB (3ch) and NIR (1ch) stems feeding shared stages)

#### Scenario: Backbone forward shape correctness
- GIVEN EarlyFusionBackbone(pretrained=False) on CPU
- WHEN forward(x=(B,4,640,640))
- THEN output has 4 tensors, channels [96,192,384,768], at correct spatial dims

#### Scenario: 4-channel stem accepts stacked input
- GIVEN EarlyFusionBackbone(pretrained=False) on CPU
- WHEN forward receives a stacked 4-channel tensor (B,4,640,640)
- THEN stem output shape is (B,96,160,160) without error

### Requirement: FPN Neck Output

`FPNNeck`, wrapping unchanged `SingleFPN`, SHALL output a configurable list of pyramid
levels driven by `head_strides`. Default configuration emits 4 levels [P2, P3, P4, P5] at
strides [4, 8, 16, 32]. `FPNNeck.forward()` MUST return a list whose length and strides
match the configured `head_strides`, each with 256 channels, and every returned level MUST
participate in the loss.
(Previously: `DualFPN` SHALL output exactly 3 pyramid levels [P3, P4, P5] at strides
[8, 16, 32]; P2 was computed internally but excluded from output and had no gradient path)

#### Scenario: FPNNeck returns the configured pyramid
- GIVEN FPNNeck initialized with head_strides=[4,8,16,32]
- WHEN forward receives backbone features
- THEN output is exactly 4 tensors [P2,P3,P4,P5] matching strides [4,8,16,32], 256 channels each
- AND every returned tensor has a live gradient path to the loss

#### Scenario: FPNNeck output feeds YOLODetectionHead without error
- GIVEN FPNNeck output matching the configured `head_strides`
- WHEN output is passed to YOLODetectionHead
- THEN no assertion error occurs; head processes all configured levels correctly

### Requirement: YOLO Detection Head Output Format

`YOLODetectionHead` SHALL produce anchor-free detections in YOLO-compatible format with
decoupled cls/reg branches, generalised over a configurable strides list.

- Input MUST be exactly `len(head_strides)` FPN levels matching the configured strides.
- `preds`: one tensor per level (B, nc+4, H_i, W_i). `cls_preds`: (B, 2, H_i, W_i).
  `reg_preds`: (B, 4, H_i, W_i).
- `distill_cls`/`distill_reg` features MUST be exposed per level for KD, computed from the
  same stem call used for the prediction (no duplicate stem computation).
(Previously: input MUST be exactly 3 FPN levels at strides [8, 16, 32])

#### Scenario: Head output shapes match YOLO format for the configured levels
- GIVEN YOLODetectionHead(fpn_channels=256, num_classes=2, strides=[4,8,16,32])
- WHEN forward receives pyramid [P2,P3,P4,P5]
- THEN preds[i] has 6 channels, cls_preds[i] has 2 channels, reg_preds[i] has 4 channels, for
  all 4 levels

#### Scenario: Head rejects mismatched FPN level count
- GIVEN YOLODetectionHead configured for a given `head_strides` list
- WHEN forward receives a pyramid with a different level count
- THEN an assertion error MUST be raised indicating expected vs actual level count

## REMOVED Requirements

### Requirement: Cross-Modal Fusion Shape Preservation

(Reason: `StageAttentionFusion` and `CrossModalFusion` are deleted; the two-branch
cross-attention design is replaced by single-stream fusion at the 4-channel input stem)
(Migration: None — no runtime fusion module remains; fusion behavior is covered by the
`multimodal-early-fusion` spec's input-contract and stem-inflation requirements)

## MODIFIED Requirements

### Requirement: MasterModel Integration

The full pipeline SHALL produce all required dict keys and `freeze_backbone` SHALL
correctly control gradient flow.

- `FPNNeck` MUST output levels matching the configured `head_strides` before passing to
  YOLODetectionHead.
- Forward MUST return a dict with 7 keys: preds, cls_preds, reg_preds, distill_backbone,
  distill_fpn, distill_head_cls, distill_head_reg.
- `freeze_backbone(n)` MUST set requires_grad=False on the single stem plus the first n
  stages, with no stem/stage asymmetry.
- `count_parameters()` MUST return positive counts per module.
- Data pipeline output (stacked 4-channel tensor, (N,4,H,W)) MUST be accepted by
  MasterModel.forward() without shape errors.
(Previously: 8-key dict including `distill_backbone_rgb` and `distill_backbone_fused`;
separate `rgb_stem`/`nir_stem` freeze asymmetry; depended on `DualFPN`'s fixed 3-level output)

#### Scenario: Full forward pass on CPU
- GIVEN MasterModel(pretrained_backbone=False) on CPU
- WHEN forward(x=(B,4,640,640))
- THEN output dict has exactly 7 keys; all tensor shapes match specification
- AND FPNNeck passes the configured number of levels to YOLODetectionHead without error

#### Scenario: Freeze backbone disables correct stages
- GIVEN MasterModel with freeze_backbone(freeze_stages=2)
- THEN the stem and stages[0..1] have requires_grad=False; stages[2..3], neck, head have
  requires_grad=True

#### Scenario: Stacked 4-channel tensor accepted by MasterModel forward
- GIVEN a stacked 4-channel tensor of shape (4, 640, 640) from the preprocessing pipeline
- WHEN unsqueezed to batch dim and passed to MasterModel.forward(x)
- THEN the model produces a valid 7-key output dict with no shape mismatch assertion
