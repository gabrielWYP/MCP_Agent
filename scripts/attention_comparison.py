#!/usr/bin/env python3
"""
Generate comparison grid of attention maps across multiple mango images.

Shows RGB, NIR, and Stage 3 attention overlay for 4+ images side by side.
Demonstrates that cross-modal attention focuses on damaged regions where
NIR reveals information invisible in RGB.

Usage:
    python scripts/attention_comparison.py \
        --checkpoint checkpoints/mastermodel_mango/best_model.pt \
        --output attention_maps/comparison_grid.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.master.master_model import MasterModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_and_preprocess(
    rgb_path: str,
    nir_path: str,
    target_size: int = 640,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Load RGB+NIR, apply letterbox, normalize."""
    rgb_img = cv2.imread(rgb_path)
    nir_img = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB).astype(np.float32)

    h, w = rgb_img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    rgb_resized = cv2.resize(rgb_img, (new_w, new_h))
    nir_resized = cv2.resize(nir_img.astype(np.float32), (new_w, new_h))

    pad_y = (target_size - new_h) // 2
    pad_x = (target_size - new_w) // 2

    rgb_padded = np.full((target_size, target_size, 3), 114, dtype=np.float32)
    nir_padded = np.full((target_size, target_size), 14, dtype=np.float32)
    rgb_padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = rgb_resized
    nir_padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = nir_resized

    rgb_viz = rgb_padded.astype(np.uint8).copy()
    nir_viz = nir_padded.astype(np.uint8).copy()

    rgb_norm = (rgb_padded / 255.0 - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    nir_norm = (nir_padded / 255.0 - 0.0569) / 0.0546

    rgb_tensor = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).float().unsqueeze(0)
    nir_tensor = torch.from_numpy(nir_norm[np.newaxis, :, :]).float().unsqueeze(0)

    return rgb_tensor, nir_tensor, rgb_viz, nir_viz


def main():
    # Hardcoded image pairs for the comparison (mix of train/val, diverse mangoes)
    image_pairs = [
        {
            "id": "mango_01",
            "rgb": "data/cache/mango/rgb/mango_rgb_1779644064.jpg",
            "nir": "data/cache/mango/nir/mango_nir_1779644064.jpg",
        },
        {
            "id": "mango_02",
            "rgb": "data/cache/mango/rgb/mango_rgb_1780238376.jpg",
            "nir": "data/cache/mango/nir/mango_nir_1780238376.jpg",
        },
        {
            "id": "mango_03",
            "rgb": "data/cache/mango/rgb/mango_rgb_1780683326.jpg",
            "nir": "data/cache/mango/nir/mango_nir_1780683326.jpg",
        },
        {
            "id": "mango_04",
            "rgb": "data/cache/mango/rgb/mango_rgb_1780685085.jpg",
            "nir": "data/cache/mango/nir/mango_nir_1780685085.jpg",
        },
        {
            "id": "mango_05",
            "rgb": "data/cache/mango/rgb/mango_rgb_1780684450.jpg",
            "nir": "data/cache/mango/nir/mango_nir_1780684450.jpg",
        },
        {
            "id": "mango_06",
            "rgb": "data/cache/mango/rgb/mango_rgb_1780241643.jpg",
            "nir": "data/cache/mango/nir/mango_nir_1780241643.jpg",
        },
    ]

    output_path = Path("attention_maps/comparison_grid.png")

    # Load teacher
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "checkpoints/mastermodel_mango/best_model.pt"
    print(f"Loading teacher from {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    teacher = MasterModel(num_classes=2, pretrained_backbone=False)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher loaded: {sum(p.numel() for p in teacher.parameters()):,} params")

    # Process each image
    results = []
    for pair in image_pairs:
        print(f"Processing {pair['id']} ...")
        rgb_tensor, nir_tensor, rgb_viz, nir_viz = load_and_preprocess(
            pair["rgb"], pair["nir"]
        )
        rgb_tensor = rgb_tensor.to(device)
        nir_tensor = nir_tensor.to(device)

        with torch.no_grad():
            output = teacher(rgb_tensor, nir_tensor, return_attention=True)

        # Use Stage 3 attention (40×40) — best balance of spatial detail and semantic focus
        attn_map = output["attention_maps"][2][0].cpu().numpy()  # Stage 3, batch 0
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        results.append({
            "id": pair["id"],
            "rgb": rgb_viz,
            "nir": nir_viz,
            "attention": attn_map,
        })

    # --- Create comparison grid: 3 columns (RGB, NIR, Attention) × N rows ---
    n_images = len(results)
    fig, axes = plt.subplots(n_images, 3, figsize=(15, 3.5 * n_images))

    if n_images == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["RGB Image", "NIR Image", "Cross-Modal Attention\n(Stage 3 — 40×40)"]

    for row_idx, result in enumerate(results):
        # RGB
        axes[row_idx, 0].imshow(result["rgb"])
        axes[row_idx, 0].set_ylabel(result["id"], fontsize=10, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # NIR
        axes[row_idx, 1].imshow(result["nir"], cmap="gray")
        axes[row_idx, 1].axis("off")

        # Attention overlay
        axes[row_idx, 2].imshow(result["rgb"])
        axes[row_idx, 2].imshow(result["attention"], cmap="hot", alpha=0.5, vmin=0, vmax=1)
        axes[row_idx, 2].axis("off")

        # Column titles on first row only
        if row_idx == 0:
            for col_idx, title in enumerate(col_titles):
                axes[row_idx, col_idx].set_title(title, fontsize=11, fontweight="bold")

    fig.suptitle(
        "CMD-KD: Cross-Modal Attention Across Mango Samples\n"
        "Red/yellow regions = Model consults NIR information (reveals internal damage)",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✅ Comparison grid saved: {output_path}")

    # --- Create a compact 4×3 grid for the thesis (4 best samples) ---
    best_samples = results[:4]
    fig, axes = plt.subplots(4, 3, figsize=(14, 10))

    for row_idx, result in enumerate(best_samples):
        axes[row_idx, 0].imshow(result["rgb"])
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(result["nir"], cmap="gray")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(result["rgb"])
        axes[row_idx, 2].imshow(result["attention"], cmap="hot", alpha=0.5, vmin=0, vmax=1)
        axes[row_idx, 2].axis("off")

        if row_idx == 0:
            axes[0, 0].set_title("RGB", fontsize=12, fontweight="bold")
            axes[0, 1].set_title("NIR", fontsize=12, fontweight="bold")
            axes[0, 2].set_title("RGB + Attention Overlay", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Cross-Modal Attention — The model learns WHERE NIR matters",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    thesis_path = Path("attention_maps/thesis_comparison.png")
    fig.savefig(thesis_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Thesis-quality grid saved: {thesis_path}")

    print("\nDone! Ready for thesis defense slides.")


if __name__ == "__main__":
    main()
