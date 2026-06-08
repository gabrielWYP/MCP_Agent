#!/usr/bin/env python3
"""
Visualize Cross-Modal Attention Maps from MasterModel teacher.

Generates heatmap overlays showing WHERE the RGB stream attends to NIR
information during cross-modal fusion. Each of the 4 backbone stages produces
one attention map.

Usage:
    python scripts/visualize_attention.py \
        --checkpoint checkpoints/mastermodel_mango/best_model.pt \
        --rgb data/cache/mango/rgb/mango_rgb_1780685512.jpg \
        --nir data/cache/mango/nir/mango_nir_1780685512.jpg \
        --output attention_maps/

Output:
    attention_maps/
    ├── stage_1_attention.png    # 160×160 feature map attention
    ├── stage_2_attention.png    # 80×80
    ├── stage_3_attention.png    # 40×40
    ├── stage_4_attention.png    # 20×20 (most semantic)
    └── combined_grid.png        # All 4 stages + RGB + NIR in a grid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Ensure project root in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.master.master_model import MasterModel

# ImageNet normalization (must match dataset preprocessing)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Custom colormap for attention overlays: transparent → yellow → red
ATTENTION_CMAP = LinearSegmentedColormap.from_list(
    "attention",
    [(0, 0, 0, 0), (1, 1, 0, 0.6), (1, 0, 0, 0.9)],
    N=256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize cross-modal attention maps from MasterModel teacher"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/mastermodel_mango/best_model.pt",
        help="Path to teacher checkpoint",
    )
    parser.add_argument(
        "--rgb",
        type=str,
        required=True,
        help="Path to RGB image",
    )
    parser.add_argument(
        "--nir",
        type=str,
        required=True,
        help="Path to NIR image (grayscale, same scene)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="attention_maps",
        help="Output directory for visualization images",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Input image size (letterbox target)",
    )
    return parser.parse_args()


def load_and_preprocess(
    rgb_path: str,
    nir_path: str,
    target_size: int = 640,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Load RGB+NIR images, apply letterbox, normalize, return tensors + originals."""
    # Load images
    rgb_img = cv2.imread(rgb_path)
    nir_img = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)

    if rgb_img is None:
        raise FileNotFoundError(f"Cannot read RGB: {rgb_path}")
    if nir_img is None:
        raise FileNotFoundError(f"Cannot read NIR: {nir_path}")

    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Letterbox (maintain aspect ratio, pad to square)
    h, w = rgb_img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    rgb_resized = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    nir_resized = cv2.resize(nir_img.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target_size
    pad_y = (target_size - new_h) // 2
    pad_x = (target_size - new_w) // 2

    rgb_padded = np.full((target_size, target_size, 3), 114, dtype=np.float32)
    nir_padded = np.full((target_size, target_size), 14, dtype=np.float32)  # NIR mean ≈ 14

    rgb_padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = rgb_resized
    nir_padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = nir_resized

    # Keep uint8 copies for visualization
    rgb_viz = rgb_padded.astype(np.uint8).copy()
    nir_viz = nir_padded.astype(np.uint8).copy()

    # Normalize
    rgb_norm = (rgb_padded / 255.0 - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    nir_norm = (nir_padded / 255.0 - 0.0569) / 0.0546

    rgb_tensor = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).float().unsqueeze(0)
    nir_tensor = torch.from_numpy(nir_norm[np.newaxis, :, :]).float().unsqueeze(0)

    return rgb_tensor, nir_tensor, rgb_viz, nir_viz


def plot_attention_overlay(
    rgb_img: np.ndarray,
    attn_map: np.ndarray,
    title: str,
    ax: plt.Axes,
) -> None:
    """Plot RGB image with attention heatmap overlay on a given axis."""
    ax.imshow(rgb_img)
    ax.imshow(attn_map, cmap="hot", alpha=0.5, vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main():
    args = parse_args()

    # Load teacher
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading teacher from {args.checkpoint} ...")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    teacher = MasterModel(num_classes=2, pretrained_backbone=False)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.to(device)
    teacher.eval()

    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Teacher loaded: {sum(p.numel() for p in teacher.parameters()):,} params")

    # Load and preprocess images
    print(f"Loading RGB: {args.rgb}")
    print(f"Loading NIR: {args.nir}")
    rgb_tensor, nir_tensor, rgb_viz, nir_viz = load_and_preprocess(
        args.rgb, args.nir, args.image_size
    )
    rgb_tensor = rgb_tensor.to(device)
    nir_tensor = nir_tensor.to(device)

    # Forward with attention
    with torch.no_grad():
        output = teacher(rgb_tensor, nir_tensor, return_attention=True)

    attn_maps = output["attention_maps"]  # [M1, M2, M3, M4] each (1, H, W)
    stage_names = [
        "Stage 1 (160×160) — Low-level",
        "Stage 2 (80×80) — Textures",
        "Stage 3 (40×40) — Parts",
        "Stage 4 (20×20) — Semantics",
    ]

    # Create output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Individual stage plots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (attn, name, ax) in enumerate(zip(attn_maps, stage_names, axes)):
        attn_np = attn[0].cpu().numpy()  # (H, W)
        # Normalize to [0, 1]
        attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
        plot_attention_overlay(rgb_viz, attn_np, name, ax)

    fig.suptitle("Cross-Modal Attention Maps — Teacher (RGB attending to NIR)", fontsize=14, y=1.01)
    plt.tight_layout()

    stage_file = out_dir / "individual_stages.png"
    fig.savefig(stage_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {stage_file}")

    # --- Combined grid: RGB | NIR | Stage1 | Stage2 | Stage3 | Stage4 ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1: RGB, NIR, Stage 1
    axes[0, 0].imshow(rgb_viz)
    axes[0, 0].set_title("RGB Image", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(nir_viz, cmap="gray")
    axes[0, 1].set_title("NIR Image (grayscale)", fontsize=11)
    axes[0, 1].axis("off")

    attn0 = attn_maps[0][0].cpu().numpy()
    attn0 = (attn0 - attn0.min()) / (attn0.max() - attn0.min() + 1e-8)
    plot_attention_overlay(rgb_viz, attn0, "Stage 1 — Low-level features", axes[0, 2])

    # Row 2: Stage 2, Stage 3, Stage 4
    for i, (attn, name, ax) in enumerate(
        zip(attn_maps[1:], stage_names[1:], [axes[1, 0], axes[1, 1], axes[1, 2]])
    ):
        attn_np = attn[0].cpu().numpy()
        attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
        plot_attention_overlay(rgb_viz, attn_np, name, ax)

    fig.suptitle(
        "CMD-KD: Cross-Modal Attention — RGB learns WHERE to consult NIR",
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()

    grid_file = out_dir / "combined_grid.png"
    fig.savefig(grid_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {grid_file}")

    # --- Side-by-side: NIR vs Attention (Stage 3 — most informative) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(nir_viz, cmap="gray")
    ax1.set_title("NIR Image\n(Damage visible in NIR but not RGB)", fontsize=11)
    ax1.axis("off")

    attn3 = attn_maps[2][0].cpu().numpy()  # Stage 3
    attn3 = (attn3 - attn3.min()) / (attn3.max() - attn3.min() + 1e-8)
    plot_attention_overlay(rgb_viz, attn3, "Stage 3 Attention Map\n(High attention = model consults NIR here)", ax2)

    fig.suptitle("NIR vs Cross-Modal Attention — The model learns where NIR matters", fontsize=13)
    plt.tight_layout()

    nir_vs_file = out_dir / "nir_vs_attention.png"
    fig.savefig(nir_vs_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {nir_vs_file}")

    print(f"\n✅ Done! {3} visualizations saved to {out_dir}/")


if __name__ == "__main__":
    main()
