# Multimodal Early Fusion Specification

## Purpose

Defines the single-stream, 4-channel RGB+NIR backbone that replaces the two-branch
cross-attention design (`DualConvNeXtBackbone`, `DualFPN`, `StageAttentionFusion`,
`CrossModalFusion`). Encodes the input contract, pretrained-stem inflation, and the
gradient-reachability and resolution invariants the removed design violated (D1-D4).

## Requirements

### Requirement: 4-Channel Input Contract and Mean-Preserving Stem Inflation

The model MUST accept one 4-channel tensor per sample: channels 0-2 are RGB under
ImageNet normalisation; channel 3 is NIR under `nir_mean=0.0569`, `nir_std=0.0546`.
The model MUST NOT accept RGB and NIR as two separate forward arguments. Loading
ImageNet pretrained weights into the 4-channel stem MUST use mean-preserving
inflation: `W_new[:, :3] = W_imagenet * 3/4`, `W_new[:, 3] = mean(W_imagenet, dim=1) * 3/4`.
Stem bias and stem LayerNorm MUST be copied verbatim.

#### Scenario: Stacked 4-channel tensor accepted
- GIVEN a sample with RGB under ImageNet stats and NIR under `nir_mean`/`nir_std`
- WHEN the two are stacked into one (4, H, W) tensor and passed to the model
- THEN the model consumes it in a single forward call, with no separate NIR argument

#### Scenario: Inflated weights are mean-preserving
- GIVEN ImageNet pretrained stem weights `W` of shape (96, 3, 4, 4)
- WHEN the stem is inflated to 4 input channels
- THEN `W_new[:, :3] == W * 3/4` and `W_new[:, 3] == mean(W, dim=1) * 3/4` elementwise
- AND stem bias and LayerNorm are unchanged from the pretrained checkpoint

#### Scenario: Inflation preserves pretrained activation statistics
- GIVEN the inflated stem and a reference RGB-only pretrained stem, both run on real batches
- WHEN per-stage activation mean/std are compared
- THEN the inflated stem's statistics fall within 2x the reference's own across-image std at
  every stage, and closer to the reference than a naive (non-rescaled) copy

### Requirement: Single-Stream Backbone With No Cross-Modal Module

The backbone MUST be a single ConvNeXt stream: one 4-channel stem, one stage stack, one
forward pass, emitting 4 stage outputs at channels `[96, 192, 384, 768]`. The system MUST
NOT contain a second stem, a second stage stack, or any cross-modal attention module.

#### Scenario: One forward pass, no fusion module in the call graph
- GIVEN the single-stream backbone
- WHEN a batch is processed
- THEN exactly one stem and one stage stack execute, producing 4 feature maps, and no
  attention-based fusion module appears anywhere in the call graph

#### Scenario: NIR-reachable trainable capacity
- GIVEN the single-stream backbone with 4-channel input
- WHEN parameters receiving non-zero gradient from a NIR-only input perturbation are counted
- THEN the count is at least 27,000,000 (up from the measured 1,824 in the removed design)

### Requirement: Every Emitted Pyramid Level Is Consumed

Every pyramid level the neck emits MUST have a live gradient path to the loss. The system
MUST NOT compute a pyramid level and discard it before the head, and no lossy
resolution-collapsing pooling (e.g. `adaptive_avg_pool2d`) MAY sit between the backbone
and the head.

#### Scenario: No orphaned pyramid level
- GIVEN the neck emits N levels for the configured `head_strides`
- WHEN a backward pass runs after a full forward pass
- THEN every emitted level's neck parameters receive a non-None gradient

#### Scenario: Finest level reaches usable resolution
- GIVEN the assembled backbone-to-head path
- WHEN the finest emitted level's spatial cell size is measured at 640px input
- THEN it is at most 8 px, and no `adaptive_avg_pool2d` or equivalent pooling exists on the path

### Requirement: Configurable Head Strides, Stem Computed Once Per Level

The detection head MUST accept a configurable `head_strides` list (default
`[4, 8, 16, 32]`) and build one `DecoupledHead` per configured stride. Each
`DecoupledHead`'s `cls_stem`/`reg_stem` MUST run exactly once per level per forward pass;
the same stem output MUST be reused for the prediction and the distillation feature.

#### Scenario: Stride-4 level ablated by config
- GIVEN `head_strides=[8, 16, 32]`
- WHEN the model is built
- THEN exactly 3 `DecoupledHead` instances exist and no stride-4 level exists downstream

#### Scenario: Stem not recomputed for distillation
- GIVEN a forward pass with distillation features requested
- WHEN `cls_stem`/`reg_stem` execution is counted per level
- THEN each runs exactly once; the distillation feature reuses that single stem output

### Requirement: Consistent Stride Reporting Across Tooling

Any script that decodes a checkpoint's pyramid predictions MUST read the stride list from
the model configuration or checkpoint metadata, not from a hardcoded list.

#### Scenario: Evaluation and visualisation read checkpoint strides
- GIVEN a checkpoint trained with `head_strides=[4, 8, 16, 32]`
- WHEN `evaluate_checkpoint.py` or `visualize_damage_predictions.py` loads it
- THEN decoding uses the checkpoint's actual 4 strides, not a hardcoded `[8, 16, 32]`

#### Scenario: 3-level and 4-level checkpoints both decode correctly
- GIVEN one 3-level and one 4-level checkpoint evaluated by the same script
- WHEN each is decoded
- THEN each uses its own correct stride list, with no misalignment between predictions and
  anchor grids
