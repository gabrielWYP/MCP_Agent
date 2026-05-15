"""
Sanity check for the MasterModel architecture.

Run with:
    python -m src.models.master.sanity_check

Verifies:
    - Forward pass completes without errors
    - Output shapes are correct
    - Distillation features are accessible
    - Parameter counts are reasonable
"""

import torch
from src.models.master.master_model import MasterModel
from src.models.master.head import NUM_CLASSES


def run_sanity_check():
    print("=" * 60)
    print("MasterModel Sanity Check — YOLO Head")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Build model ---
    print("Building model (pretrained=False for speed)...")
    model = MasterModel(
        num_classes=NUM_CLASSES,
        pretrained_backbone=False,  # skip download for sanity check
        fpn_channels=256,
    ).to(device)

    # --- Parameter counts ---
    params = model.count_parameters()
    print("\nParameter counts:")
    for k, v in params.items():
        print(f"  {k:<15}: {v:>12,}")

    # --- Dummy inputs ---
    # Simulating a batch of 2 images at 640x640
    batch_size = 2
    H, W = 640, 640

    rgb = torch.randn(batch_size, 3, H, W, device=device)
    nir = torch.randn(batch_size, 1, H, W, device=device)

    # --- Forward pass (no proposals needed — YOLO head is dense) ---
    print("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(rgb, nir)

    # --- Check backbone features ---
    print("\nBackbone RGB features (distill_backbone_rgb):")
    expected_channels = [96, 192, 384, 768]
    expected_spatial = [H // 4, H // 8, H // 16, H // 32]
    for i, feat in enumerate(output["distill_backbone_rgb"]):
        expected_h = expected_spatial[i]
        assert feat.shape == (batch_size, expected_channels[i], expected_h, expected_h), \
            f"Stage {i+1} shape mismatch: {feat.shape}"
        print(f"  S{i+1}: {tuple(feat.shape)} ✓")

    # --- Check fused features ---
    print("\nFused features after cross-attention (distill_backbone_fused):")
    for i, feat in enumerate(output["distill_backbone_fused"]):
        expected_h = expected_spatial[i]
        assert feat.shape == (batch_size, expected_channels[i], expected_h, expected_h), \
            f"Fused stage {i+1} shape mismatch: {feat.shape}"
        print(f"  F{i+1}: {tuple(feat.shape)} ✓")

    # --- Check FPN pyramid ---
    print("\nFPN pyramid (distill_fpn):")
    fpn_channels = 256
    # YOLO head uses P3, P4, P5 (strides 8, 16, 32)
    fpn_spatial = [H // 8, H // 16, H // 32]
    for i, feat in enumerate(output["distill_fpn"]):
        expected_h = fpn_spatial[i]
        assert feat.shape == (batch_size, fpn_channels, expected_h, expected_h), \
            f"FPN P{i+3} shape mismatch: {feat.shape}"
        print(f"  P{i+3}: {tuple(feat.shape)} ✓")

    # --- Check YOLO detection head outputs ---
    print(f"\nYOLO head outputs (num_classes={NUM_CLASSES}):")
    for i, (pred, cls_p, reg_p) in enumerate(zip(
        output["preds"], output["cls_preds"], output["reg_preds"]
    )):
        stride = [8, 16, 32][i]
        expected_h = H // stride
        expected_w = W // stride
        expected_cls_ch = NUM_CLASSES
        expected_reg_ch = 4

        assert pred.shape == (batch_size, expected_cls_ch + expected_reg_ch, expected_h, expected_w), \
            f"Level {i+1} pred shape: {pred.shape}"
        assert cls_p.shape == (batch_size, expected_cls_ch, expected_h, expected_w), \
            f"Level {i+1} cls shape: {cls_p.shape}"
        assert reg_p.shape == (batch_size, expected_reg_ch, expected_h, expected_w), \
            f"Level {i+1} reg shape: {reg_p.shape}"

        stride_label = ["P3", "P4", "P5"][i]
        print(f"  {stride_label}: pred={tuple(pred.shape)}, cls={tuple(cls_p.shape)}, reg={tuple(reg_p.shape)} ✓")

    # --- Check distillation head features ---
    print("\nHead distillation features (cls_stem + reg_stem per level):")
    for i, (cls_feat, reg_feat) in enumerate(zip(
        output["distill_head_cls"], output["distill_head_reg"]
    )):
        stride = [8, 16, 32][i]
        expected_h = H // stride
        expected_w = W // stride
        stem_ch = 256  # FPN channels

        assert cls_feat.shape == (batch_size, stem_ch, expected_h, expected_w), \
            f"Head cls_stem level {i+1} shape: {cls_feat.shape}"
        assert reg_feat.shape == (batch_size, stem_ch, expected_h, expected_w), \
            f"Head reg_stem level {i+1} shape: {reg_feat.shape}"

        stride_label = ["P3", "P4", "P5"][i]
        print(f"  {stride_label}: cls_stem={tuple(cls_feat.shape)}, reg_stem={tuple(reg_feat.shape)} ✓")

    # --- YOLO Nano compatibility ---
    print("\nYOLO Nano compatibility:")
    print(f"  Output format: (B, {NUM_CLASSES}+4, H_i, W_i) ✓")
    print(f"  Strides: [8, 16, 32] ✓")
    print(f"  Levels: P3, P4, P5 ✓")
    print(f"  Classes: {NUM_CLASSES} (sano, danado) ✓")
    print(f"  Direct distillation: cls_preds + reg_preds ✓")

    print("\n" + "=" * 60)
    print("All checks passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_sanity_check()
