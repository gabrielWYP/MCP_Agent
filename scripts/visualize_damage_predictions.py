#!/usr/bin/env python3
"""Visualize damage predictions vs ground truth for mango detection models.

Loads a trained MasterModel (RGB+NIR) or StudentModel (RGB-only) checkpoint
and produces side-by-side comparison images showing:
  - Left panel:  RGB image with ground truth boxes from YOLO label files
  - Right panel: RGB image with model predictions (after NMS-free decoding)

Class colors:
  - Green (class 0): mango / sano
  - Red   (class 1): damage / danado

This script helps verify whether damage annotations (class 1) are correctly
aligned with the images, or if the model is simply failing to learn them.

Example usage:
    # MasterModel (RGB+NIR)
    python scripts/visualize_damage_predictions.py \\
        --checkpoint checkpoints/mastermodel/best_model.pt \\
        --config configs/training_mango.yaml \\
        --rgb-dir data/cache/mango/rgb \\
        --nir-dir data/cache/mango/nir \\
        --labels-dir data/annotations/yolo/labels \\
        --model-type master \\
        --split val \\
        --num-samples 20

    # StudentModel (RGB-only)
    python scripts/visualize_damage_predictions.py \\
        --checkpoint checkpoints/student/best_model.pt \\
        --config configs/training_student.yaml \\
        --rgb-dir data/cache/mango/rgb \\
        --labels-dir data/annotations/yolo/labels \\
        --model-type student \\
        --split val \\
        --conf-threshold 0.3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.master.master_model import MasterModel
from src.models.student.student_model import StudentModel
from src.training.config import TrainingConfig
from src.training.dataset import letterbox

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# BGR colors for OpenCV drawing
CLASS_COLORS = {
    0: (0, 255, 0),    # green  — mango / sano
    1: (0, 0, 255),    # red    — damage / danado
}
CLASS_NAMES = {0: "mango", 1: "damage"}

STRIDES = [8, 16, 32]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize model predictions vs ground truth for mango damage detection."
    )
    parser.add_argument(
        "--checkpoint", required=True, type=str,
        help="Path to .pt checkpoint file.",
    )
    parser.add_argument(
        "--config", required=True, type=str,
        help="Path to YAML training config (training_mango.yaml, training_student.yaml, etc.).",
    )
    parser.add_argument(
        "--rgb-dir", required=True, type=str,
        help="Directory containing RGB images.",
    )
    parser.add_argument(
        "--nir-dir", type=str, default=None,
        help="Directory containing NIR images (required for MasterModel).",
    )
    parser.add_argument(
        "--labels-dir", required=True, type=str,
        help="Directory containing YOLO label files (with split subdirectories).",
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val", "test"],
        help="Dataset split to visualize (default: val).",
    )
    parser.add_argument(
        "--model-type", required=True, type=str, choices=["master", "student"],
        help="Model architecture: master (RGB+NIR) or student (RGB-only).",
    )
    parser.add_argument(
        "--num-samples", type=int, default=20,
        help="Number of images to visualize (default: 20).",
    )
    parser.add_argument(
        "--conf-threshold", type=float, default=0.25,
        help="Confidence threshold for predictions (default: 0.25).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="visualizations/damage_preds",
        help="Directory to save output PNGs (default: visualizations/damage_preds/).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cuda or cpu (default: auto-detect).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    model_type: str,
    checkpoint_path: str,
    config: TrainingConfig,
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate the model and load checkpoint weights.

    Handles both checkpoint dicts (with 'model_state_dict' key) and raw
    state dicts.
    """
    if model_type == "master":
        model = MasterModel(
            num_classes=config.num_classes,
            pretrained_backbone=False,
            backbone_variant=config.backbone_variant,
        )
    else:
        model = StudentModel(num_classes=config.num_classes)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        logger.info(
            "Loaded checkpoint dict (epoch=%s, best_map50=%s)",
            checkpoint.get("epoch", "?"),
            checkpoint.get("best_map50", "?"),
        )
    else:
        state_dict = checkpoint
        logger.info("Loaded raw state_dict.")

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    logger.info("Model loaded: %s (%s parameters)", model_type, sum(p.numel() for p in model.parameters()))
    return model


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------


def discover_image_pairs(
    rgb_dir: Path,
    nir_dir: Path | None,
    labels_dir: Path,
    split: str,
    model_type: str,
) -> list[dict]:
    """Find RGB (+ NIR) image pairs that have corresponding label files.

    Returns:
        List of dicts with keys: rgb_path, nir_path (or None), label_path, image_id.
    """
    labels_split_dir = labels_dir / split
    if not labels_split_dir.exists():
        # Fallback: labels_dir may already point at the split directory itself
        logger.warning("Labels split dir not found: %s — trying labels_dir directly.", labels_split_dir)
        labels_split_dir = labels_dir

    rgb_files = sorted(rgb_dir.glob("*.jpg"))
    if not rgb_files:
        rgb_files = sorted(rgb_dir.glob("*.png"))

    pairs = []
    for idx, rgb_path in enumerate(rgb_files):
        # Match label file
        label_path = labels_split_dir / f"{rgb_path.stem}.txt"
        if not label_path.exists():
            continue

        nir_path = None
        if model_type == "master":
            nir_name = rgb_path.name.replace("_rgb", "_nir")
            nir_path = nir_dir / nir_name if nir_dir else None
            if nir_path is None or not nir_path.exists():
                logger.debug("Missing NIR for %s, skipping.", rgb_path.name)
                continue

        pairs.append({
            "rgb_path": rgb_path,
            "nir_path": nir_path,
            "label_path": label_path,
            "image_id": idx,
            "stem": rgb_path.stem,
        })

    return pairs


# ---------------------------------------------------------------------------
# Image preprocessing (mirrors YOLODataset.__getitem__ for val split)
# ---------------------------------------------------------------------------


def preprocess_rgb(rgb_img: np.ndarray, image_size: int, letterbox_value: int) -> tuple[torch.Tensor, float, int, int]:
    """Letterbox + ImageNet normalize an RGB image for model input.

    Args:
        rgb_img: (H, W, 3) RGB uint8 image.
        image_size: Target square size.
        letterbox_value: Padding pixel value.

    Returns:
        rgb_tensor: (1, 3, image_size, image_size) normalized tensor.
        scale: Letterbox scale factor.
        pad_x: Horizontal padding offset.
        pad_y: Vertical padding offset.
    """
    rgb_lb, scale, pad_x, pad_y = letterbox(rgb_img, image_size, letterbox_value)

    # ImageNet normalization
    img = rgb_lb.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, scale, pad_x, pad_y


def preprocess_nir(
    nir_img: np.ndarray, image_size: int, nir_mean: float, nir_std: float
) -> torch.Tensor:
    """Letterbox + normalize a NIR grayscale image for model input.

    Args:
        nir_img: (H, W) grayscale uint8 image.
        image_size: Target square size.
        nir_mean: Dataset NIR normalization mean.
        nir_std: Dataset NIR normalization std.

    Returns:
        nir_tensor: (1, 1, image_size, image_size) normalized tensor.
    """
    nir_lb, _, _, _ = letterbox(nir_img, image_size, int(nir_mean * 255))
    img = nir_lb.astype(np.float32) / 255.0
    img = (img - nir_mean) / nir_std
    return torch.from_numpy(img[np.newaxis, np.newaxis, :, :]).float()


# ---------------------------------------------------------------------------
# Prediction decoding (mirrors Trainer._decode_predictions)
# ---------------------------------------------------------------------------


def decode_predictions(
    output: dict,
    num_classes: int,
    image_size: int,
    conf_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode raw model output into filtered boxes, scores, and labels.

    Applies sigmoid to classification logits, decodes anchor-free bbox
    deltas, and filters by confidence threshold.

    Args:
        output: Model forward output dict with 'cls_preds' and 'preds'.
        num_classes: Number of detection classes.
        image_size: Input image size (for normalization).
        conf_threshold: Minimum confidence to keep a prediction.

    Returns:
        boxes: (P, 4) cxcywh in letterbox pixel coordinates.
        scores: (P,) confidence scores.
        labels: (P,) integer class IDs.
    """
    preds = output["preds"]       # list of (B, nc+4, H, W)
    cls_preds = output["cls_preds"]  # list of (B, nc, H, W)

    all_boxes = []
    all_scores = []
    all_labels = []

    for pred, cls_pred, stride in zip(preds, cls_preds, STRIDES):
        # pred: (1, nc+4, H, W), cls_pred: (1, nc, H, W)
        H, W = pred.shape[2], pred.shape[3]

        p = pred[0]      # (nc+4, H, W)
        c = cls_pred[0]  # (nc, H, W)

        # Classification scores (sigmoid)
        scores = c.sigmoid()  # (nc, H, W)
        max_scores, max_labels = scores.max(dim=0)  # (H, W)

        # Regression deltas
        reg = p[num_classes:]  # (4, H, W)

        # Filter by confidence
        mask = max_scores > conf_threshold
        if not mask.any():
            continue

        ys, xs = mask.nonzero(as_tuple=True)

        # Decode bboxes (anchor-free)
        dx = reg[0, ys, xs]
        dy = reg[1, ys, xs]
        w = reg[2, ys, xs].exp()
        h = reg[3, ys, xs].exp()

        # Anchor centers → pixel coordinates in letterbox space
        anchor_x = (xs.float() + 0.5) * stride
        anchor_y = (ys.float() + 0.5) * stride

        cx = anchor_x + dx
        cy = anchor_y + dy

        boxes = torch.stack([cx, cy, w, h], dim=1)  # (K, 4) pixel coords

        all_boxes.append(boxes)
        all_scores.append(max_scores[mask])
        all_labels.append(max_labels[mask])

    if all_boxes:
        boxes = torch.cat(all_boxes).cpu().numpy()
        scores = torch.cat(all_scores).cpu().numpy()
        labels = torch.cat(all_labels).cpu().numpy()
    else:
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int64)

    return boxes, scores, labels


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------


def load_yolo_labels(label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a YOLO-format label file.

    Format per line: class_id cx cy w h (normalized 0-1).

    Returns:
        bboxes: (N, 4) [cx, cy, w, h] normalized to original image.
        labels: (N,) integer class IDs.
    """
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    bboxes, labels = [], []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            if w <= 0 or h <= 0:
                continue
            bboxes.append([cx, cy, w, h])
            labels.append(cls_id)

    if not bboxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    return np.array(bboxes, dtype=np.float32), np.array(labels, dtype=np.int64)


def gt_to_letterbox_pixels(
    bboxes: np.ndarray,
    orig_w: int,
    orig_h: int,
    scale: float,
    pad_x: int,
    pad_y: int,
) -> np.ndarray:
    """Convert YOLO-normalized GT bboxes to pixel coords in letterbox space.

    Args:
        bboxes: (N, 4) [cx, cy, w, h] normalized to original image.
        orig_w, orig_h: Original image dimensions.
        scale: Letterbox scale factor.
        pad_x, pad_y: Letterbox padding offsets.

    Returns:
        (N, 4) [cx, cy, w, h] in letterbox pixel coordinates.
    """
    if len(bboxes) == 0:
        return bboxes

    result = np.zeros_like(bboxes)
    # Original pixel coords
    cx_px = bboxes[:, 0] * orig_w
    cy_px = bboxes[:, 1] * orig_h
    w_px = bboxes[:, 2] * orig_w
    h_px = bboxes[:, 3] * orig_h

    # Apply letterbox transform
    result[:, 0] = cx_px * scale + pad_x
    result[:, 1] = cy_px * scale + pad_y
    result[:, 2] = w_px * scale
    result[:, 3] = h_px * scale

    return result


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    if len(boxes) == 0:
        return boxes
    result = np.zeros_like(boxes)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return result


def draw_boxes(
    image: np.ndarray,
    boxes_cxcywh: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray | None = None,
    title: str = "",
) -> np.ndarray:
    """Draw bounding boxes on an image with class-colored rectangles and labels.

    Args:
        image: (H, W, 3) BGR uint8 image to draw on (modified in place).
        boxes_cxcywh: (N, 4) [cx, cy, w, h] in pixel coordinates.
        labels: (N,) integer class IDs.
        scores: (N,) optional confidence scores to display.
        title: Title text to draw at the top.

    Returns:
        The annotated image.
    """
    if len(boxes_cxcywh) > 0:
        boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh)
        for i, (box, cls_id) in enumerate(zip(boxes_xyxy, labels)):
            color = CLASS_COLORS.get(int(cls_id), (255, 255, 255))
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Label text
            cls_name = CLASS_NAMES.get(int(cls_id), f"cls{cls_id}")
            if scores is not None:
                text = f"{cls_name} {scores[i]:.2f}"
            else:
                text = cls_name

            # Text background
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(
                image, text, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

    # Title
    if title:
        cv2.putText(
            image, title, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
        )

    return image


def create_side_by_side(
    rgb_lb: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
) -> np.ndarray:
    """Create a side-by-side comparison image: GT (left) vs Predictions (right).

    Args:
        rgb_lb: (H, W, 3) BGR letterboxed image.
        gt_boxes: (N, 4) GT cxcywh in letterbox pixel coords.
        gt_labels: (N,) GT class IDs.
        pred_boxes: (P, 4) predicted cxcywh in letterbox pixel coords.
        pred_labels: (P,) predicted class IDs.
        pred_scores: (P,) predicted confidence scores.

    Returns:
        (H, 2*W, 3) BGR side-by-side image.
    """
    left = rgb_lb.copy()
    right = rgb_lb.copy()

    draw_boxes(left, gt_boxes, gt_labels, scores=None, title="Ground Truth")
    draw_boxes(right, pred_boxes, pred_labels, scores=pred_scores, title="Predictions")

    # Add GT/pred counts to titles
    gt_mango = int((gt_labels == 0).sum()) if len(gt_labels) > 0 else 0
    gt_damage = int((gt_labels == 1).sum()) if len(gt_labels) > 0 else 0
    pred_mango = int((pred_labels == 0).sum()) if len(pred_labels) > 0 else 0
    pred_damage = int((pred_labels == 1).sum()) if len(pred_labels) > 0 else 0

    gt_summary = f"GT: {gt_mango} mango, {gt_damage} damage"
    pred_summary = f"Pred: {pred_mango} mango, {pred_damage} damage"

    cv2.putText(
        left, gt_summary, (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA,
    )
    cv2.putText(
        right, pred_summary, (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA,
    )

    return np.hstack([left, right])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Config
    config = TrainingConfig.from_yaml(args.config)
    logger.info("Config: model_type=%s, image_size=%d, num_classes=%d", config.model_type, config.image_size, config.num_classes)

    # Validate NIR requirement
    if args.model_type == "master" and args.nir_dir is None:
        logger.error("--nir-dir is required for MasterModel (--model-type master).")
        return 1

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # Load model
    model = load_model(args.model_type, args.checkpoint, config, device)

    # Discover images
    rgb_dir = Path(args.rgb_dir)
    nir_dir = Path(args.nir_dir) if args.nir_dir else None
    labels_dir = Path(args.labels_dir)

    pairs = discover_image_pairs(rgb_dir, nir_dir, labels_dir, args.split, args.model_type)
    if not pairs:
        logger.error("No image pairs found. Check --rgb-dir, --nir-dir, --labels-dir, and --split.")
        return 1

    logger.info("Found %d image pairs with labels in split '%s'.", len(pairs), args.split)

    # Sample images
    num_samples = min(args.num_samples, len(pairs))
    if num_samples < len(pairs):
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(len(pairs), size=num_samples, replace=False)
        indices.sort()
        selected = [pairs[i] for i in indices]
    else:
        selected = pairs

    logger.info("Visualizing %d images.", len(selected))

    # Process each image
    success_count = 0
    error_count = 0

    for i, pair in enumerate(selected):
        stem = pair["stem"]
        try:
            # Load RGB image
            rgb_img = cv2.imread(str(pair["rgb_path"]))  # BGR
            if rgb_img is None:
                logger.warning("[%d/%d] Cannot read RGB: %s — skipping.", i + 1, len(selected), pair["rgb_path"])
                error_count += 1
                continue

            rgb_rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = rgb_rgb.shape[:2]

            # Preprocess RGB
            rgb_tensor, scale, pad_x, pad_y = preprocess_rgb(
                rgb_rgb, config.image_size, config.letterbox_value
            )
            rgb_tensor = rgb_tensor.to(device)

            # Preprocess NIR if needed
            nir_tensor = None
            if args.model_type == "master":
                nir_img = cv2.imread(str(pair["nir_path"]), cv2.IMREAD_GRAYSCALE)
                if nir_img is None:
                    logger.warning("[%d/%d] Cannot read NIR: %s — skipping.", i + 1, len(selected), pair["nir_path"])
                    error_count += 1
                    continue
                nir_tensor = preprocess_nir(nir_img, config.image_size, config.nir_mean, config.nir_std)
                nir_tensor = nir_tensor.to(device)

            # Inference
            with torch.no_grad():
                if args.model_type == "master":
                    output = model(rgb_tensor, nir_tensor)
                else:
                    output = model(rgb_tensor)

            # Decode predictions (in letterbox pixel coordinates)
            pred_boxes, pred_scores, pred_labels = decode_predictions(
                output, config.num_classes, config.image_size, args.conf_threshold
            )

            # Load ground truth
            gt_bboxes, gt_labels_arr = load_yolo_labels(pair["label_path"])

            # Convert GT to letterbox pixel coordinates
            gt_boxes_lb = gt_to_letterbox_pixels(gt_bboxes, orig_w, orig_h, scale, pad_x, pad_y)

            # Build letterbox BGR image for drawing
            rgb_lb_bgr, _, _, _ = letterbox(rgb_img, config.image_size, config.letterbox_value)

            # Create side-by-side visualization
            comparison = create_side_by_side(
                rgb_lb_bgr,
                gt_boxes_lb,
                gt_labels_arr,
                pred_boxes,
                pred_labels,
                pred_scores,
            )

            # Save
            out_path = output_dir / f"{stem}_gt_vs_pred.png"
            cv2.imwrite(str(out_path), comparison)

            # Log summary
            gt_mango = int((gt_labels_arr == 0).sum()) if len(gt_labels_arr) > 0 else 0
            gt_damage = int((gt_labels_arr == 1).sum()) if len(gt_labels_arr) > 0 else 0
            pred_mango = int((pred_labels == 0).sum()) if len(pred_labels) > 0 else 0
            pred_damage = int((pred_labels == 1).sum()) if len(pred_labels) > 0 else 0

            logger.info(
                "[%d/%d] %s — GT: %d mango, %d damage | Pred: %d mango, %d damage → %s",
                i + 1, len(selected), stem,
                gt_mango, gt_damage, pred_mango, pred_damage,
                out_path.name,
            )
            success_count += 1

        except Exception as e:
            logger.error("[%d/%d] Error processing %s: %s", i + 1, len(selected), stem, e, exc_info=True)
            error_count += 1

    logger.info(
        "Done. %d/%d images visualized successfully, %d errors.",
        success_count, len(selected), error_count,
    )
    logger.info("Output saved to: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
