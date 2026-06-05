#!/usr/bin/env python3
"""
Fix mango bounding boxes using Florence-2 zero-shot detection.

Replaces the HSV-based mango detection in the annotation pipeline with
Florence-2's "mango" detection, which is FAR more accurate.

Usage:
    python3 scripts/annotate_mango_florence.py \
        --rgb-dir data/cache/mango/rgb \
        --labels-dir data/annotations/yolo/labels \
        --model microsoft/Florence-2-base
"""
import argparse
import json
import logging
import re
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_florence_bboxes(text: str, image_w: int, image_h: int, target_class: str = "mango") -> list[dict]:
    """Parse Florence-2 OD output text to bbox list, filtering by target class.

    Florence-2 outputs: <OD>class1<loc_x1><loc_y1><loc_x2><loc_y2>class2<loc_...

    Returns:
        List of dicts with keys: x1, y1, x2, y2 (pixel coords).
    """
    # Pattern: word followed by 4 <loc_N> tokens
    pattern = r"([a-zA-Z_ ]+?)((?:<loc_\d+>){4})"
    matches = re.findall(pattern, text)

    if not matches:
        # Fallback: just extract all loc tokens
        loc_pattern = r"<loc_(\d+)>"
        locs = re.findall(loc_pattern, text)
        if locs:
            bboxes = []
            for i in range(0, len(locs), 4):
                if i + 3 >= len(locs):
                    break
                bboxes.append(parse_single_bbox(locs[i:i+4], image_w, image_h))
            return bboxes
        return []

    bboxes = []
    for class_name, loc_str in matches:
        class_name = class_name.strip().lower()
        if target_class and target_class.lower() not in class_name:
            continue
        locs = re.findall(r"<loc_(\d+)>", loc_str)
        if len(locs) == 4:
            bboxes.append(parse_single_bbox(locs, image_w, image_h))

    # If no mango found, take the largest bbox
    if not bboxes and target_class:
        # Try all detections
        for class_name, loc_str in matches:
            locs = re.findall(r"<loc_(\d+)>", loc_str)
            if len(locs) == 4:
                bboxes.append(parse_single_bbox(locs, image_w, image_h))

    return bboxes


def parse_single_bbox(locs: list, image_w: int, image_h: int) -> dict:
    """Parse 4 location tokens to a bbox dict."""
    x1_norm = int(locs[0]) / 1000.0
    y1_norm = int(locs[1]) / 1000.0
    x2_norm = int(locs[2]) / 1000.0
    y2_norm = int(locs[3]) / 1000.0
    return {
        "x1": int(x1_norm * image_w),
        "y1": int(y1_norm * image_h),
        "x2": int(x2_norm * image_w),
        "y2": int(y2_norm * image_h),
    }


def bbox_to_yolo_cxcywh(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> tuple:
    """Convert pixel xyxy bbox to YOLO cxcywh normalized format."""
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return (cx, cy, w, h)


def detect_mango_florence(
    model, processor, image: Image.Image, device: torch.device,
    task_prompt: str = "<OD>",
) -> str:
    """Run Florence-2 detection and return raw text output."""
    inputs = processor(
        text=task_prompt, images=image, return_tensors="pt",
    ).to(device)
    # Match model dtype (fp16)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            do_sample=False,
            use_cache=False,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return generated_text


def fix_labels_file(label_path: Path, mango_bbox_yolo: tuple, img_w: int, img_h: int):
    """Replace the mango bbox (class 0) in a YOLO label file, keep damage bboxes (class 1).

    Args:
        label_path: Path to .txt label file.
        mango_bbox_yolo: (cx, cy, w, h) normalized for the mango.
        img_w, img_h: Image dimensions (not used for YOLO format, kept for API consistency).
    """
    # Read existing labels
    damage_lines = []
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5 and int(parts[0]) == 1:
                    damage_lines.append(line.strip())

    # Write: mango bbox (class 0) + all damage bboxes (class 1)
    cx, cy, w, h = mango_bbox_yolo
    with open(label_path, "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        for dmg_line in damage_lines:
            f.write(dmg_line + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fix mango bboxes using Florence-2 zero-shot detection"
    )
    parser.add_argument("--rgb-dir", default="data/cache/mango/rgb",
                        help="Directory with RGB images")
    parser.add_argument("--labels-dir", default="data/annotations/yolo/labels",
                        help="Root directory with train/val/test YOLO labels")
    parser.add_argument("--model", default="microsoft/Florence-2-large",
                        help="Florence-2 model ID (base ~0.9GB, large ~3.4GB)")
    parser.add_argument("--device", default=None,
                        help="Device: cuda or cpu (auto-detect if not set)")
    parser.add_argument("--prompt", default="mango",
                        help="Object to detect (default: mango)")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Florence-2
    logger.info(f"Loading {args.model}...")
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    logger.info("Model loaded.")

    prompt = "<OD>"

    # Find all RGB images with existing labels (across all splits)
    rgb_dir = Path(args.rgb_dir)
    labels_root = Path(args.labels_dir)
    rgb_files = sorted(rgb_dir.glob("*.jpg"))

    total_updated = 0

    for split in ["train", "val", "test"]:
        split_labels_dir = labels_root / split
        if not split_labels_dir.exists():
            continue

        split_files = sorted(split_labels_dir.glob("*.txt"))
        if not split_files:
            continue

        logger.info(f"\n--- Processing {split} split ({len(split_files)} images) ---")

        for label_file in split_files:
            stem = label_file.stem  # e.g., "mango_rgb_1780675906"
            rgb_path = rgb_dir / f"{stem}.jpg"

            if not rgb_path.exists():
                logger.warning(f"  RGB not found for {stem}, skipping.")
                continue

            logger.info(f"  {stem}")

            try:
                # Load image
                image = Image.open(rgb_path).convert("RGB")
                img_w, img_h = image.size

                # Run Florence-2 detection
                raw_output = detect_mango_florence(model, processor, image, device, prompt)

                # Parse bboxes
                bboxes = parse_florence_bboxes(raw_output, img_w, img_h)

                if not bboxes:
                    logger.warning(f"    No mango detected by Florence-2!")
                    continue

                # Take the largest bbox (or the only one)
                bbox = max(bboxes, key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))

                # Convert to YOLO format
                cx, cy, w, h = bbox_to_yolo_cxcywh(
                    bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"], img_w, img_h
                )

                # Update YOLO label file (replace mango bbox, keep damage)
                fix_labels_file(label_file, (cx, cy, w, h), img_w, img_h)
                total_updated += 1

            except Exception as e:
                logger.error(f"    ERROR: {e}")
                continue

    print(f"\n{'='*50}")
    print(f"Total labels updated: {total_updated}")
    print(f"{'='*50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
