# Teacher Architecture Validation Specification

## Purpose

Validate the MasterModel teacher pipeline (ConvNeXtV2 backbone → cross-modal fusion → dual FPN → YOLO head) using CPU-only synthetic tensors. No pretrained weights, GPU, S3, or real datasets.

## Requirements

### Requirement: ConvNeXtV2 Backbone Shapes

`DualConvNeXtBackbone(pretrained=False)` SHALL produce correct channel and spatial shapes for both streams.

- RGB stem (3ch→96ch) and NIR stem (1ch→96ch) MUST output (N, 96, H/4, W/4).
- 4 stage output channels: [96, 192, 384, 768].
- For 640×640 input, spatial dims: [160², 80², 40², 20²].

#### Scenario: Backbone forward shape correctness

- GIVEN DualConvNeXtBackbone(pretrained=False) on CPU
- WHEN forward(rgb=(B,3,640,640), nir=(B,1,640,640))
- THEN rgb_features and nir_features each have 4 tensors, channels [96,192,384,768] at correct spatial dims

#### Scenario: NIR stem accepts single-channel input

- GIVEN DualConvNeXtBackbone(pretrained=False) on CPU
- WHEN forward receives nir tensor (B,1,640,640)
- THEN nir stem output shape is (B,96,160,160) without error

#### Scenario: Distinct streams despite shared weights

- GIVEN same backbone with non-identical RGB and NIR inputs
- THEN rgb_features[i] and nir_features[i] MUST differ (different stem activations through shared stages)

### Requirement: Cross-Modal Fusion Shape Preservation

`CrossModalFusion` SHALL output fused features preserving input spatial dimensions and pass NIR features through unchanged.

- Each `StageAttentionFusion` MUST output (N, C, H, W) matching input spatial dims.
- `max_tokens_side` pooling MUST prevent OOM on CPU for large feature maps.
- NIR features MUST pass through unchanged for the parallel FPN path.

#### Scenario: Fusion preserves spatial dimensions

- GIVEN CrossModalFusion(channels=[96,192,384,768])
- WHEN forward receives backbone features for 640×640 input
- THEN fused_features[i] matches rgb_features[i] spatial dims; nir_features pass through unchanged

#### Scenario: Small feature maps skip pooling

- GIVEN StageAttentionFusion(max_tokens_side=20) with H≤20, W≤20 inputs
- WHEN forward processes without pooling
- THEN output shape equals (N, C, H, W) input — no upsampling

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

### Requirement: YOLO Detection Head Output Format

`YOLODetectionHead` SHALL produce anchor-free detections in YOLO-compatible format with decoupled cls/reg branches.

- Input MUST be exactly 3 FPN levels at strides [8, 16, 32] corresponding to [P3, P4, P5].
- `preds`: 3 tensors (B, nc+4, H_i, W_i) at strides [8,16,32], nc=2.
- `cls_preds`: 3 tensors (B, 2, H_i, W_i). `reg_preds`: 3 tensors (B, 4, H_i, W_i).
- `distill_cls`/`distill_reg` features MUST be exposed per level for KD.

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
- Data pipeline output tensors (rgb=(N,3,H,W), nir=(N,1,H,W)) MUST be accepted by MasterModel.forward() without shape errors when produced by the augmentation pipeline.

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

### Requirement: CPU-Only Test Execution

The test suite MUST run on CPU without network, pretrained downloads, or real data.

- All models MUST use pretrained=False.
- Input tensors MUST use torch.randn with deterministic seeds.
- Shared module fixtures via conftest.py SHOULD reduce instantiation overhead.
- Tests >5s MUST use @pytest.mark.slow.

#### Scenario: No network calls during CI

- GIVEN pytest on CI with no network access
- WHEN suite executes
- THEN no test downloads pretrained weights or connects externally

#### Scenario: Deterministic shape assertions

- GIVEN torch.manual_seed(42) in session fixture
- WHEN backbone forward runs twice
- THEN output shapes are identical across runs
