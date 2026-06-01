# Data Augmentation Specification

## Purpose

Apply spatially-synchronized geometric and photometric augmentations to RGB+NIR image pairs, producing MasterModel-compatible dual-stream tensors with a configurable augmentation factor, and write per-image vector outputs with dataset manifest.

## Requirements

### Requirement: Spatially-Synchronized Augmentation

The system SHALL apply identical spatial transforms to RGB and NIR simultaneously to preserve pixel alignment.

- RGB and NIR MUST be stacked into a temporary 4-channel array (H, W, 4) before spatial augmentation.
- Spatial transforms (rotations, flips, crops) MUST operate on the 4-channel stack to guarantee sub-pixel alignment.
- After augmentation, the stack MUST be split back into separate rgb_tensor (3, H, W) and nir_tensor (1, H, W).
- The deviation between RGB and NIR spatial coordinates after augmentation MUST NOT exceed 1 pixel.

#### Scenario: Geometric augmentations preserve pixel alignment

- GIVEN a 4-channel stacked input (H, W, 4) with horizontal flip and rotation applied
- WHEN the augmented stack is split into rgb (3, H, W) and nir (1, H, W)
- THEN for any pixel (i, j) in rgb, nir[i, j] corresponds to the same physical point in the original image

#### Scenario: Augmentation factor produces correct sample count

- GIVEN 10 paired samples and augmentation_factor=3
- WHEN augmentation pipeline runs
- THEN output contains 30 augmented samples (10 original + 20 augmented variants)
- AND each variant includes a copy of the original (identity augmentation)

### Requirement: Modality-Aware Photometric Transforms

The system SHALL apply photometric transforms differently per modality.

- For RGB channels: ColorJitter (brightness, contrast, saturation, hue) and GaussianBlur MAY be applied.
- For NIR channel: brightness and contrast jitter MAY be applied; color transforms (saturation, hue) MUST NOT be applied to NIR.
- Photometric transforms MUST be applied after the 4-channel stack is split, independently per modality.

#### Scenario: NIR channel receives only intensity transforms

- GIVEN augmentation config with brightness_limit=(0.7, 1.3) and contrast_limit=(0.7, 1.3)
- WHEN NIR augmentation is applied
- THEN NIR intensity values are scaled within the configured range
- AND no color-space transformation is applied to the single-channel NIR

#### Scenario: RGB channel receives full photometric augmentation

- GIVEN augmentation config with ColorJitter enabled
- WHEN RGB augmentation is applied
- THEN brightness, contrast, saturation, and hue are randomly varied within configured limits

### Requirement: Vector Store Output Format

The system SHALL write per-image `.pt` files and a `dataset_index.json` manifest.

- Each augmented sample MUST be saved as `{stem}_aug{N}.pt` containing `{"rgb": tensor(3,H,W), "nir": tensor(1,H,W), "label": int, "class": str}`.
- The manifest MUST list all output files with: source_oci_key, local_path, label, class_name, augmentation_id, and whether homography was applied.
- If a previous manifest exists in the output directory, it MUST be overwritten (idempotent re-run).

#### Scenario: Per-image output validates against MasterModel

- GIVEN a vector store output for sample "sano/img001_aug0.pt"
- WHEN the .pt file is loaded and its rgb/nir tensors passed to MasterModel.forward()
- THEN the model produces valid detection output without shape errors

#### Scenario: Manifest accurately records all samples

- GIVEN augmentation produces 30 samples from 10 originals with augmentation_factor=3
- WHEN dataset_index.json is written
- THEN it contains exactly 30 entries with unique augmentation_ids
- AND each entry references an existing .pt file on disk

### Requirement: Configurable Augmentation Intensity

The system SHALL support configurable augmentation parameters via a config dictionary.

- `augmentation_factor`: integer multiplier for total samples (MUST be ≥1; 1 = no augmentation).
- `spatial_intensity`: float controlling rotation/scale/shift limits (default 0.4).
- `photometric_intensity`: float controlling brightness/contrast limits (default 0.3).
- Default values MUST produce a sensible baseline augmentation for mango ripeness detection.

#### Scenario: augmentation_factor=1 produces originals only

- GIVEN augmentation_factor=1 and 10 paired samples
- WHEN augmentation pipeline runs
- THEN output contains exactly 10 samples (identity transform only, no augmentation)