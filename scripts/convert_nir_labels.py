#!/usr/bin/env python3
"""
Convert Label Studio NIR annotations → YOLO labels (with homography projection to RGB).

Usage:
    python3 scripts/convert_nir_labels.py \
        --export data/label_studio_nir/export.json \
        --homography notebooks/matriz_homografia_aruco.npy \
        --output-dir data/annotations/yolo/labels
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def project_nir_to_rgb(
    nir_bbox_xyxy: tuple[float, float, float, float],
    H_inv: np.ndarray,
    rgb_w: int,
    rgb_h: int,
) -> tuple[int, int, int, int] | None:
    """Project a bbox from NIR coords to RGB coords using inverse homography.

    H maps RGB→NIR, so H_inv maps NIR→RGB.
    Projects the 4 corners and takes the bounding rectangle.
    """
    x1, y1, x2, y2 = nir_bbox_xyxy
    corners = np.array([
        [x1, y1, 1],
        [x2, y1, 1],
        [x2, y2, 1],
        [x1, y2, 1],
    ], dtype=np.float32).T  # (3, 4)

    projected = H_inv @ corners  # (3, 4)
    valid = np.abs(projected[2]) > 1e-8
    if not np.all(valid):
        return None

    projected = projected[:2] / projected[2]  # normalize homogeneous

    px1 = max(0, int(projected[0].min()))
    py1 = max(0, int(projected[1].min()))
    px2 = min(rgb_w, int(projected[0].max()))
    py2 = min(rgb_h, int(projected[1].max()))

    if px2 <= px1 or py2 <= py1:
        return None

    return (px1, py1, px2, py2)


def xyxy_to_yolo_cxcywh(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """Convert pixel xyxy bbox to normalized YOLO cxcywh format."""
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return (cx, cy, w, h)


def extract_label_studio_image_name(task: dict) -> str:
    """Extract the original NIR image filename from a Label Studio task."""
    candidates = [
        task.get("file_upload", ""),
        task.get("data", {}).get("image", ""),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        name = Path(candidate).name
        if "-" in name:
            return name.split("-", 1)[1]
        return name
    return ""


def read_image_size(path: Path) -> tuple[int, int] | None:
    """Read image dimensions as (width, height) without loading callers into PIL."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height


def label_studio_rect_to_xyxy(
    value: dict,
    image_w: int,
    image_h: int,
) -> tuple[float, float, float, float]:
    """Convert a Label Studio percentage rectangle to pixel xyxy coordinates."""
    x1 = value.get("x", 0.0) / 100.0 * image_w
    y1 = value.get("y", 0.0) / 100.0 * image_h
    x2 = (value.get("x", 0.0) + value.get("width", 0.0)) / 100.0 * image_w
    y2 = (value.get("y", 0.0) + value.get("height", 0.0)) / 100.0 * image_h
    return x1, y1, x2, y2


def main():
    parser = argparse.ArgumentParser(description="Convert NIR labels to YOLO via homography")
    parser.add_argument("--export", required=True, help="Label Studio export JSON")
    parser.add_argument("--homography", default="notebooks/matriz_homografia_aruco.npy")
    parser.add_argument("--output-dir", default="data/annotations/yolo/labels")
    parser.add_argument("--splits", default="data/annotations/yolo/labels",
                        help="Directory with existing train/val/test splits")
    parser.add_argument("--rgb-dir", default="data/cache/mango/rgb",
                        help="Directory with RGB images used for YOLO training")
    parser.add_argument("--nir-dir", default="data/cache/mango/nir",
                        help="Directory with NIR images annotated in Label Studio")
    args = parser.parse_args()

    # Load homography
    H = np.load(args.homography)
    H_inv = np.linalg.inv(H)

    # Load existing splits (to know which image goes where)
    splits_dir = Path(args.splits)
    image_to_split = {}
    for split in ["train", "val", "test"]:
        split_dir = splits_dir / split
        if split_dir.exists():
            for f in split_dir.glob("*.txt"):
                stem = f.stem.replace("_rgb", "")
                image_to_split[stem] = split

    # Load Label Studio export
    with open(args.export) as f:
        data = json.load(f)

    output_dir = Path(args.output_dir)
    rgb_dir = Path(args.rgb_dir)
    nir_dir = Path(args.nir_dir)
    updated = 0
    skipped = 0

    for task in data:
        img_file = extract_label_studio_image_name(task)
        if not img_file:
            logger.warning("Task without image filename, skipping")
            skipped += 1
            continue

        img_name = Path(img_file).stem  # e.g., "mango_nir_1780675906"
        rgb_stem = img_name.replace("_nir", "_rgb")
        rgb_path = rgb_dir / f"{rgb_stem}.jpg"
        nir_path = nir_dir / f"{img_name}.jpg"

        # Find which split this belongs to
        split = image_to_split.get(rgb_stem.replace("_rgb", ""))
        if split is None:
            # Try matching by nir stem
            for k, v in image_to_split.items():
                if k in img_name or img_name in k:
                    split = v
                    break

        if split is None:
            logger.warning(f"No split found for {img_name}, skipping")
            skipped += 1
            continue

        rgb_size = read_image_size(rgb_path)
        nir_size = read_image_size(nir_path)
        if rgb_size is None:
            logger.warning("RGB image not found or unreadable for %s: %s", rgb_stem, rgb_path)
            skipped += 1
            continue
        if nir_size is None:
            logger.warning("NIR image not found or unreadable for %s: %s", img_name, nir_path)
            skipped += 1
            continue
        rgb_w, rgb_h = rgb_size
        nir_w, nir_h = nir_size

        # Get annotations
        annotations = task.get("annotations", [])
        if not annotations:
            skipped += 1
            continue

        # Take the first (or last completed) annotation
        ann = annotations[0].get("result", [])

        damage_bboxes_rgb = []
        for item in ann:
            if item.get("type") == "rectanglelabels":
                value = item.get("value", {})
                nir_bbox = label_studio_rect_to_xyxy(value, nir_w, nir_h)
                rgb_bbox = project_nir_to_rgb(nir_bbox, H_inv, rgb_w, rgb_h)
                if rgb_bbox is not None:
                    damage_bboxes_rgb.append(rgb_bbox)

        if not damage_bboxes_rgb:
            skipped += 1
            continue

        # Read the existing YOLO label to get the mango bbox (class 0)
        source_label_path = splits_dir / split / f"{rgb_stem}.txt"
        label_path = output_dir / split / f"{rgb_stem}.txt"
        mango_line = None
        if source_label_path.exists():
            with open(source_label_path) as f:
                for line in f:
                    if line.startswith("0 "):
                        mango_line = line.strip()
                        break

        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, "w") as f:
            if mango_line:
                f.write(mango_line + "\n")
            for x1, y1, x2, y2 in damage_bboxes_rgb:
                cx, cy, w, h = xyxy_to_yolo_cxcywh(x1, y1, x2, y2, rgb_w, rgb_h)
                f.write(f"1 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        updated += 1
        logger.info(f"  {rgb_stem}: {len(damage_bboxes_rgb)} damage bboxes → {split}")

    print(f"\n{'='*50}")
    print(f"Updated: {updated} labels")
    print(f"Skipped: {skipped} tasks")
    print(f"{'='*50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
