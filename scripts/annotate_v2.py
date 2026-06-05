#!/usr/bin/env python3
"""
V2 Annotation: Florence-2 mango bbox + damage detection inside ROI (RGB + NIR).

Strategy:
    1. Read existing mango bbox (class 0) from YOLO labels (Florence-2 generated)
    2. Crop the mango ROI from both RGB and NIR images
    3. Detect damage inside the ROI using:
       a) NIR: intensity threshold (dark regions = damage)
       b) RGB: brown/dark color segmentation (HSV + Lab)
    4. Combine both, filter by size, project back to full image coords
    5. Write updated YOLO labels

Usage:
    python3 scripts/annotate_v2.py \
        --rgb-dir data/cache/mango/rgb \
        --nir-dir data/cache/mango/nir \
        --labels-dir data/annotations/yolo/labels \
        --output-dir data/annotations/yolo/labels \
        --debug-dir data/annotations/debug_v2
"""
import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def read_mango_bbox(label_path: Path, img_w: int, img_h: int) -> tuple | None:
    """Read class 0 (mango) bbox from YOLO label file. Returns (x1, y1, x2, y2) in pixels."""
    if not label_path.exists():
        return None
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and int(parts[0]) == 0:
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((cx - w / 2) * img_w)
                y1 = int((cy - h / 2) * img_h)
                x2 = int((cx + w / 2) * img_w)
                y2 = int((cy + h / 2) * img_h)
                return (max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2))
    return None


def detect_damage_nir_roi(nir_roi: np.ndarray, min_area: int = 60, percentile: int = 10) -> list[tuple]:
    """Detect damage regions in NIR crop using local anomaly detection.

    Damage in NIR appears as localized dark spots. Uses:
    1. CLAHE enhancement
    2. Local contrast: pixel vs local mean in sliding window
    3. Adaptive thresholding on the contrast map
    4. Morphological cleanup + area filter

    Returns: list of (x1, y1, x2, y2) bboxes in ROI-local coords.
    """
    if nir_roi.size == 0 or nir_roi.shape[0] < 10 or nir_roi.shape[1] < 10:
        return []

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(nir_roi)

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)

    # --- Local anomaly detection ---
    # Compute local mean using a large kernel (30x30 or 1/8 of ROI size)
    kernel_size = max(15, min(nir_roi.shape[0], nir_roi.shape[1]) // 6)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    local_mean = cv2.filter2D(blurred, -1, kernel)

    # Local contrast: how much darker is each pixel vs its neighborhood
    local_contrast = local_mean - blurred.astype(np.float32)  # positive = darker than neighbors
    local_contrast = np.clip(local_contrast, 0, 255)

    # Normalize contrast to 0-255
    if local_contrast.max() > 0:
        local_contrast = (local_contrast / local_contrast.max() * 255).astype(np.uint8)

    # Threshold: pixels significantly darker than local mean
    contrast_thresh = np.percentile(local_contrast[local_contrast > 0], 90) if local_contrast.max() > 0 else 255
    _, anomaly_mask = cv2.threshold(local_contrast, max(15, contrast_thresh * 0.6), 255, cv2.THRESH_BINARY)
    anomaly_mask = anomaly_mask.astype(np.uint8)

    # Also: global dark threshold (absolute)
    global_thresh = np.percentile(enhanced, percentile)
    _, dark_mask = cv2.threshold(enhanced, global_thresh, 255, cv2.THRESH_BINARY_INV)

    # Combine: anomaly OR globally dark
    damage_mask = cv2.bitwise_or(anomaly_mask, dark_mask)

    # Morphological cleanup
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_OPEN, kernel_small)
    damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_CLOSE, kernel_med)

    # Extract contours
    contours, _ = cv2.findContours(damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)

        # Solidity check: damage regions tend to be solid blobs
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1)
        if solidity < 0.3:  # Skip very non-convex shapes (noise)
            continue

        bboxes.append((x, y, x + w, y + h))

    return bboxes


def detect_damage_rgb_roi(rgb_roi: np.ndarray, min_area: int = 80) -> list[tuple]:
    """Detect damage regions in RGB crop using color segmentation.

    Mango damage appears as brown/dark spots against yellow/green healthy tissue.
    Uses HSV for brown detection + Lab for darkness.

    Returns: list of (x1, y1, x2, y2) bboxes in ROI-local coords.
    """
    if rgb_roi.size == 0:
        return []

    # Convert to HSV for brown/dark detection
    hsv = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2HSV)

    # Brown range in HSV (low saturation, medium-low value)
    # Brown = hue 10-25, sat 50-255, val 20-200
    mask_brown1 = cv2.inRange(hsv, (8, 40, 20), (25, 255, 180))
    # Dark brown/black = hue 0-15, sat 0-100, val 0-100
    mask_brown2 = cv2.inRange(hsv, (0, 0, 0), (15, 100, 80))

    damage_mask = cv2.bitwise_or(mask_brown1, mask_brown2)

    # Also use Lab color space for darkness detection
    lab = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    # Dark pixels in L channel (damage is darker than healthy tissue)
    _, dark_mask = cv2.threshold(L, 80, 255, cv2.THRESH_BINARY_INV)
    damage_mask = cv2.bitwise_or(damage_mask, dark_mask)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_OPEN, kernel)
    damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)

        # Aspect ratio filter: damage spots are reasonably compact
        aspect_ratio = max(w, h) / (min(w, h) + 1)
        if aspect_ratio > 4:
            continue

        bboxes.append((x, y, x + w, y + h))

    return bboxes


def merge_nearby_bboxes(bboxes: list[tuple], distance_thresh: int = 15) -> list[tuple]:
    """Merge bboxes that are very close to each other (likely same damage region)."""
    if len(bboxes) <= 1:
        return bboxes

    merged = []
    used = set()

    for i, b1 in enumerate(bboxes):
        if i in used:
            continue
        x1_1, y1_1, x2_1, y2_1 = b1

        for j, b2 in enumerate(bboxes):
            if j <= i or j in used:
                continue
            x1_2, y1_2, x2_2, y2_2 = b2

            # Check if close
            cx1, cy1 = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
            cx2, cy2 = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
            dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

            if dist < distance_thresh:
                # Merge
                x1_1 = min(x1_1, x1_2)
                y1_1 = min(y1_1, y1_2)
                x2_1 = max(x2_1, x2_2)
                y2_1 = max(y2_1, y2_2)
                used.add(j)

        merged.append((x1_1, y1_1, x2_1, y2_1))
        used.add(i)

    return merged


def bbox_to_yolo(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> tuple:
    """Convert pixel xyxy to YOLO cxcywh normalized."""
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return (cx, cy, w, h)


def draw_debug(rgb_img: np.ndarray, mango_bbox: tuple, damage_bboxes: list[tuple],
               out_path: Path, img_w: int, img_h: int):
    """Draw debug visualization."""
    debug = rgb_img.copy()

    # Draw mango bbox (green)
    if mango_bbox:
        x1, y1, x2, y2 = mango_bbox
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(debug, "mango", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw damage bboxes (red)
    for bx1, by1, bx2, by2 in damage_bboxes:
        cv2.rectangle(debug, (bx1, by1), (bx2, by2), (0, 0, 255), 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(debug, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description="V2 Annotation: damage inside mango ROI")
    parser.add_argument("--rgb-dir", default="data/cache/mango/rgb")
    parser.add_argument("--nir-dir", default="data/cache/mango/nir")
    parser.add_argument("--labels-dir", default="data/annotations/yolo/labels")
    parser.add_argument("--output-dir", default="data/annotations/yolo/labels")
    parser.add_argument("--debug-dir", default="data/annotations/debug_v2")
    parser.add_argument("--min-area-nir", type=int, default=80,
                        help="Min area (px²) for NIR damage regions")
    parser.add_argument("--min-area-rgb", type=int, default=60,
                        help="Min area (px²) for RGB damage regions")
    parser.add_argument("--nir-percentile", type=int, default=18,
                        help="Percentile for NIR dark threshold (lower = stricter)")
    args = parser.parse_args()

    labels_root = Path(args.labels_dir)
    output_root = Path(args.output_dir)
    rgb_dir = Path(args.rgb_dir)
    nir_dir = Path(args.nir_dir)
    debug_dir = Path(args.debug_dir)

    total_updated = 0
    total_damage_bboxes = 0

    for split in ["train", "val", "test"]:
        split_labels = labels_root / split
        out_split = output_root / split
        out_split.mkdir(parents=True, exist_ok=True)

        label_files = sorted(split_labels.glob("*.txt"))
        if not label_files:
            continue

        logger.info(f"\n--- {split}: {len(label_files)} images ---")

        for label_file in label_files:
            stem = label_file.stem
            rgb_path = rgb_dir / f"{stem}.jpg"
            nir_path = nir_dir / f"{stem.replace('_rgb', '_nir')}.jpg"

            if not rgb_path.exists() or not nir_path.exists():
                logger.warning(f"  Missing image for {stem}")
                continue

            # Load images
            rgb_img = cv2.imread(str(rgb_path))
            nir_img = cv2.imread(str(nir_path), cv2.IMREAD_GRAYSCALE)
            if rgb_img is None or nir_img is None:
                continue

            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            img_h, img_w = rgb_img.shape[:2]

            # Read mango bbox (Florence-2)
            mango_bbox = read_mango_bbox(label_file, img_w, img_h)
            if mango_bbox is None:
                logger.warning(f"  No mango bbox in {stem}")
                continue

            mx1, my1, mx2, my2 = mango_bbox

            # Crop ROI from both modalities
            # NIR: may have different resolution, need to scale bbox
            nir_h, nir_w = nir_img.shape[:2]
            scale_x = nir_w / img_w
            scale_y = nir_h / img_h

            nir_mx1 = int(mx1 * scale_x)
            nir_my1 = int(my1 * scale_y)
            nir_mx2 = int(mx2 * scale_x)
            nir_my2 = int(my2 * scale_y)

            rgb_roi = rgb_img[my1:my2, mx1:mx2]
            nir_roi = nir_img[nir_my1:nir_my2, nir_mx1:nir_mx2]

            # Detect damage in both modalities
            damage_nir = detect_damage_nir_roi(nir_roi, args.min_area_nir, args.nir_percentile)
            damage_rgb = detect_damage_rgb_roi(rgb_roi, args.min_area_rgb)

            # Convert NIR damage to RGB coords (scale back)
            damage_nir_rgb = []
            for nx1, ny1, nx2, ny2 in damage_nir:
                damage_nir_rgb.append((
                    int(mx1 + nx1 / scale_x),
                    int(my1 + ny1 / scale_y),
                    int(mx1 + nx2 / scale_x),
                    int(my1 + ny2 / scale_y),
                ))

            # Convert RGB damage to full image coords
            damage_rgb_full = []
            for rx1, ry1, rx2, ry2 in damage_rgb:
                damage_rgb_full.append((
                    mx1 + rx1, my1 + ry1,
                    mx1 + rx2, my1 + ry2,
                ))

            # Combine and merge
            all_damage = damage_nir_rgb + damage_rgb_full
            all_damage = merge_nearby_bboxes(all_damage, distance_thresh=20)

            # Filter: damage bboxes must be within mango ROI (with small margin)
            margin = 10
            filtered = []
            for bx1, by1, bx2, by2 in all_damage:
                if (bx1 >= mx1 - margin and by1 >= my1 - margin and
                    bx2 <= mx2 + margin and by2 <= my2 + margin):
                    # Skip if bbox is too large (>80% of mango)
                    area_dmg = (bx2 - bx1) * (by2 - by1)
                    area_mango = (mx2 - mx1) * (my2 - my1)
                    if area_dmg < area_mango * 0.7:
                        filtered.append((bx1, by1, bx2, by2))

            # Write updated label: mango bbox + filtered damage bboxes
            cx_m, cy_m, w_m, h_m = bbox_to_yolo(mx1, my1, mx2, my2, img_w, img_h)
            out_path = out_split / label_file.name

            with open(out_path, "w") as f:
                f.write(f"0 {cx_m:.6f} {cy_m:.6f} {w_m:.6f} {h_m:.6f}\n")
                for bx1, by1, bx2, by2 in filtered:
                    cx_d, cy_d, w_d, h_d = bbox_to_yolo(bx1, by1, bx2, by2, img_w, img_h)
                    f.write(f"1 {cx_d:.6f} {cy_d:.6f} {w_d:.6f} {h_d:.6f}\n")

            total_damage_bboxes += len(filtered)
            total_updated += 1

            # Debug image (first 3 per split)
            label_idx = label_files.index(label_file)
            if label_idx < 3:
                debug_path = debug_dir / f"{split}_{stem}.jpg"
                draw_debug(rgb_img, mango_bbox, filtered, debug_path, img_w, img_h)

        logger.info(f"  {split}: {total_updated} images, {total_damage_bboxes} damage bboxes")

    print(f"\n{'='*50}")
    print(f"Total images updated: {total_updated}")
    print(f"Total damage bboxes:   {total_damage_bboxes}")
    print(f"Debug images:          {debug_dir}/")
    print(f"{'='*50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
