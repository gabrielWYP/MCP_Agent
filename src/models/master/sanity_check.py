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
from master_model import MasterModel


def run_sanity_check():
    print("=" * 60)
    print("MasterModel Sanity Check")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Build model ---
    print("Building model (pretrained=False for speed)...")
    model = MasterModel(
        num_classes=3,
        pretrained_backbone=False,  # skip download for sanity check
        fpn_channels=256,
        fc_channels=1024,
        roi_out_size=7,
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

    # Dummy proposals: 10 random boxes per image
    proposals = []
    for _ in range(batch_size):
        x1 = torch.randint(0, W // 2, (10,), dtype=torch.float32, device=device)
        y1 = torch.randint(0, H // 2, (10,), dtype=torch.float32, device=device)
        x2 = x1 + torch.randint(50, 200, (10,), dtype=torch.float32, device=device)
        y2 = y1 + torch.randint(50, 200, (10,), dtype=torch.float32, device=device)
        x2 = x2.clamp(max=W - 1)
        y2 = y2.clamp(max=H - 1)
        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        proposals.append(boxes)

    # --- Forward pass ---
    print("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(rgb, nir, proposals)

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
    fpn_spatial = [H // 4, H // 8, H // 16, H // 32]
    for i, feat in enumerate(output["distill_fpn"]):
        expected_h = fpn_spatial[i]
        assert feat.shape == (batch_size, fpn_channels, expected_h, expected_h), \
            f"FPN P{i+2} shape mismatch: {feat.shape}"
        print(f"  P{i+2}: {tuple(feat.shape)} ✓")

    # --- Check detection head outputs ---
    total_proposals = sum(p.shape[0] for p in proposals)
    print(f"\nCascade R-CNN head (total proposals: {total_proposals}):")
    for i, (cls, reg) in enumerate(zip(output["cls_scores"], output["bbox_deltas"])):
        assert cls.shape == (total_proposals, 3), f"Stage {i+1} cls shape: {cls.shape}"
        assert reg.shape == (total_proposals, 4), f"Stage {i+1} reg shape: {reg.shape}"
        print(f"  Stage {i+1}: cls={tuple(cls.shape)}, reg={tuple(reg.shape)} ✓")

    # --- Check distillation head features ---
    print("\nHead distillation features (FC2 per cascade stage):")
    for i, feat in enumerate(output["distill_head"]):
        assert feat.shape == (total_proposals, 1024), \
            f"Head distill stage {i+1} shape: {feat.shape}"
        print(f"  Stage {i+1}: {tuple(feat.shape)} ✓")

    print("\n" + "=" * 60)
    print("All checks passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_sanity_check()
