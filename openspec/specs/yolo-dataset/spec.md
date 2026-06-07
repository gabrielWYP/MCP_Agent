# yolo-dataset Specification

## Purpose

Dataset class that loads RGB+NIR image pairs with YOLO-format annotations, applies bbox-aware augmentations, and produces tensors suitable for the MasterModel training loop.

## Requirements

### Requirement: Image Pair Loading

The dataset MUST load RGB and NIR image pairs from `data/cache/mango/rgb/` and `data/cache/mango/nir/` respectively, matching pairs by filename stem (applying `_rgb` → `_nir` substitution consistent with the annotation pipeline).

#### Scenario: Matching RGB and NIR by stem

- GIVEN an RGB image at `data/cache/mango/rgb/mango_001_rgb.jpg`
- WHEN the dataset resolves the corresponding NIR path
- THEN it SHALL load `data/cache/mango/nir/mango_001_nir.jpg`

#### Scenario: Missing NIR pair

- GIVEN an RGB image with no matching NIR file
- WHEN the dataset attempts to load the pair
- THEN a clear error SHALL be raised identifying the missing file

### Requirement: YOLO Label Parsing

The dataset MUST parse YOLO-format label files from `data/annotations/yolo/labels/{split}/`, where each line is `class_id cx cy w h` (normalized [0,1]). Annotations with confidence < 0.5 SHALL be filtered out.

#### Scenario: Standard label loading

- GIVEN label file `train/mango_001_rgb.txt` with 5 bounding boxes
- WHEN the dataset loads labels for that image
- THEN it SHALL return all 5 bboxes as a Tensor[N, 4] and class labels as Tensor[N]

#### Scenario: Low-confidence annotation filtering

- GIVEN a label file containing annotations with confidence scores
- WHEN confidence < 0.5 for any annotation
- THEN those annotations SHALL be excluded from the returned labels

#### Scenario: Empty label file (no objects)

- GIVEN an image with an empty or missing label file
- WHEN the dataset loads it
- THEN it SHALL return empty bboxes Tensor[0, 4] and empty class_labels Tensor[0]

### Requirement: Bbox-Aware Augmentation

The dataset MUST apply albumentations transforms with `bbox_params(format="yolo")` so that spatial transforms (flip, rotate, crop) adjust bounding box coordinates accordingly. Train and val SHALL have separate augmentation pipelines.

#### Scenario: Train augmentation applies spatial transforms

- GIVEN train split is selected
- WHEN a sample is loaded
- THEN albumentations SHALL apply: HorizontalFlip, VerticalFlip, Rotate, ShiftScaleRotate, and color jitter (RGB only)

#### Scenario: Val augmentation is letterbox-only

- GIVEN val or test split is selected
- WHEN a sample is loaded
- THEN only letterbox resize + normalization SHALL be applied (no random augmentation)

#### Scenario: Augmentation adjusts bboxes

- GIVEN a bbox at [0.5, 0.5, 0.2, 0.3] and HorizontalFlip is applied
- WHEN the transform executes
- THEN the bbox SHALL be mirrored to [0.3, 0.5, 0.2, 0.3] (cx adjusted, w preserved)

### Requirement: Letterbox Resizing

The dataset MUST resize images to 640×640 using letterboxing: scale the longer side to 640, pad the shorter side with value 114 (gray). Bounding boxes SHALL be rescaled and offset accordingly.

#### Scenario: Letterbox non-square input

- GIVEN input image 1280×720
- WHEN letterbox resize to 640×640
- THEN the image SHALL be scaled by 0.5 (720→640) and padded 160px horizontally
- AND bboxes SHALL have their cx shifted by the padding offset and coordinates rescaled

### Requirement: Output Contract

Each sample MUST return: (rgb_tensor[3,640,640], nir_tensor[1,640,640], bboxes[N,4], class_labels[N], image_id). RGB SHALL be normalized with ImageNet stats; NIR SHALL be normalized with computed stats.

#### Scenario: Standard output format

- GIVEN a sample with 3 objects
- WHEN __getitem__ is called
- THEN it SHALL return rgb Tensor[3,640,640], nir Tensor[1,640,640], bboxes Tensor[3,4], class_labels Tensor[3], image_id int
