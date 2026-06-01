# Data Preprocessing Specification

## Purpose

Transform raw downloaded RGB+NIR image pairs into spatially-aligned, normalized, MasterModel-ready tensors via resize, homography warp, and dual-stream format conversion.

## Requirements

### Requirement: Image Loading and Validation

The system SHALL load cached RGB and NIR images from local disk and validate basic integrity.

- RGB images MUST be loaded as (H, W, 3) uint8 arrays via OpenCV (BGR→RGB conversion).
- NIR images MUST be loaded as (H, W) uint8 grayscale arrays.
- Corrupt or unreadable images MUST be logged and skipped (not crash the pipeline).

#### Scenario: Valid paired images loaded

- GIVEN cached rgb_path and nir_path pointing to valid JPEG files
- WHEN preprocessor loads the pair
- THEN rgb_array shape is (H, W, 3) uint8 and nir_array shape is (H, W) uint8

#### Scenario: Corrupt image skipped gracefully

- GIVEN nir_path points to a corrupt or zero-byte file
- WHEN preprocessor attempts to load the NIR image
- THEN the pair is logged as invalid with file path and error detail
- AND the pair is excluded from the output dataset without crashing

### Requirement: NIR→RGB Spatial Alignment via Homography

The system SHALL align NIR images to RGB spatial coordinates using a homography matrix.

- If a valid homography matrix (.npy) is provided, NIR MUST be warped via `cv2.warpPerspective(H, (w, h))` to match RGB dimensions.
- If no homography matrix is available or the matrix is invalid, the system SHALL fall back to simple `cv2.resize(nir, (w, h))` and log a warning.
- Warped NIR output MUST have the same (H, W) spatial dimensions as the paired RGB image.

#### Scenario: Homography warp produces aligned NIR

- GIVEN a valid homography matrix and NIR image (640×480)
- WHEN alignment is applied to an RGB target of (640, 640)
- THEN warped NIR shape is (640, 640) matching RGB spatial dimensions

#### Scenario: Missing homography triggers fallback resize

- GIVEN homography_path=None or file does not exist
- WHEN alignment is attempted
- THEN NIR is resized to match RGB dimensions via cv2.resize
- AND a warning is logged indicating fallback was used

### Requirement: Resize and Normalization

The system SHALL resize images to configurable target dimensions and normalize per-channel.

- Both RGB and resized/warped NIR MUST be resized to (target_h, target_w), default (640, 640).
- RGB normalization MUST use ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
- NIR normalization MUST use dataset-computed stats (or placeholder mean=0.45, std=0.22 as fallback).
- The normalization step MUST be applied after resize, consistent with albumentations order.

#### Scenario: Default preprocessing produces MasterModel-compatible shapes

- GIVEN a paired 800×600 RGB+NIR input with default config (640×640)
- WHEN preprocessing pipeline completes
- THEN rgb_tensor shape is (3, 640, 640) float32
- AND nir_tensor shape is (1, 640, 640) float32
- AND both are normalized per their respective mean/std

#### Scenario: Configurable target size

- GIVEN target_size=(416, 416) in config
- WHEN preprocessing completes
- THEN rgb_tensor shape is (3, 416, 416) and nir_tensor shape is (1, 416, 416)

### Requirement: Dual-Stream Output Format

The preprocessor SHALL output separate RGB and NIR tensors matching MasterModel's `forward(rgb, nir)` signature.

- Output MUST be two separate tensors: `rgb_tensor` of shape `(3, H, W)` and `nir_tensor` of shape `(1, H, W)`.
- The system MUST NOT stack them into a single 4-channel tensor (that is the ConvNeXtTeacher format, not MasterModel).
- Each preprocessed sample MUST include metadata: source class, filename stem, and whether homography was applied.

#### Scenario: Output tensors match MasterModel forward signature

- GIVEN preprocessed sample from input pair (rgb, nir)
- WHEN rgb_tensor and nir_tensor are passed to MasterModel.forward()
- THEN the model accepts them without shape mismatch error