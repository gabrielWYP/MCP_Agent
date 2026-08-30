"""
Sanity check for the MasterModel architecture (fusion-redesign).

Run with:
    python -m src.models.master.sanity_check

Verifies:
    - Forward pass completes without errors (single 4-channel stream)
    - Output shapes are correct for the default [P2, P3, P4, P5] pyramid
    - Distillation features are accessible (7-key output dict)
    - Parameter counts are reasonable
"""

import torch
from src.models.master.master_model import MasterModel
from src.models.master.head import NUM_CLASSES
from src.training.strides import DEFAULT_HEAD_STRIDES


def run_sanity_check():
    print("=" * 60)
    print("MasterModel Sanity Check — Early Fusion, YOLO Head")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    head_strides = list(DEFAULT_HEAD_STRIDES)  # [4, 8, 16, 32] — includes P2

    # --- Build model ---
    print("Building model (pretrained=False for speed)...")
    model = MasterModel(
        num_classes=NUM_CLASSES,
        pretrained_backbone=False,  # skip download for sanity check
        fpn_channels=256,
        head_strides=head_strides,
    ).to(device)

    # --- Parameter counts ---
    params = model.count_parameters()
    print("\nParameter counts:")
    for k, v in params.items():
        print(f"  {k:<15}: {v:>12,}")

    # --- Dummy inputs ---
    batch_size = 2
    H, W = 640, 640

    rgb = torch.randn(batch_size, 3, H, W, device=device)
    nir = torch.randn(batch_size, 1, H, W, device=device)

    # --- Forward pass (no proposals needed — YOLO head is dense) ---
    print("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(rgb, nir)

    # --- Check output dict contract (7 keys) ---
    expected_keys = {
        "preds", "cls_preds", "reg_preds",
        "distill_backbone", "distill_fpn", "distill_head_cls", "distill_head_reg",
    }
    assert set(output.keys()) == expected_keys, (
        f"Output dict keys mismatch: {sorted(output.keys())} != {sorted(expected_keys)}"
    )
    print(f"\nOutput dict has exactly {len(expected_keys)} keys ✓")

    # --- Check backbone features ---
    print("\nBackbone features (distill_backbone):")
    expected_channels = [96, 192, 384, 768]
    expected_spatial = [H // 4, H // 8, H // 16, H // 32]
    for i, feat in enumerate(output["distill_backbone"]):
        expected_h = expected_spatial[i]
        assert feat.shape == (batch_size, expected_channels[i], expected_h, expected_h), \
            f"Stage {i+1} shape mismatch: {feat.shape}"
        print(f"  S{i+1}: {tuple(feat.shape)} ✓")

    # --- Check FPN pyramid (levels named by head_strides, finest-first) ---
    print(f"\nFPN pyramid (distill_fpn), head_strides={head_strides}:")
    fpn_channels = 256
    for i, stride in enumerate(head_strides):
        feat = output["distill_fpn"][i]
        expected_h = H // stride
        assert feat.shape == (batch_size, fpn_channels, expected_h, expected_h), \
            f"FPN level {i} (stride {stride}) shape mismatch: {feat.shape}"
        print(f"  stride={stride}: {tuple(feat.shape)} ✓")

    # --- Check YOLO detection head outputs ---
    print(f"\nYOLO head outputs (num_classes={NUM_CLASSES}):")
    for i, (pred, cls_p, reg_p) in enumerate(zip(
        output["preds"], output["cls_preds"], output["reg_preds"]
    )):
        stride = head_strides[i]
        expected_h = H // stride
        expected_w = W // stride
        expected_cls_ch = NUM_CLASSES
        expected_reg_ch = 4

        assert pred.shape == (batch_size, expected_cls_ch + expected_reg_ch, expected_h, expected_w), \
            f"Level {i} pred shape: {pred.shape}"
        assert cls_p.shape == (batch_size, expected_cls_ch, expected_h, expected_w), \
            f"Level {i} cls shape: {cls_p.shape}"
        assert reg_p.shape == (batch_size, expected_reg_ch, expected_h, expected_w), \
            f"Level {i} reg shape: {reg_p.shape}"

        print(f"  stride={stride}: pred={tuple(pred.shape)}, cls={tuple(cls_p.shape)}, reg={tuple(reg_p.shape)} ✓")

    # --- Check distillation head features ---
    print("\nHead distillation features (cls_stem + reg_stem per level):")
    for i, (cls_feat, reg_feat) in enumerate(zip(
        output["distill_head_cls"], output["distill_head_reg"]
    )):
        stride = head_strides[i]
        expected_h = H // stride
        expected_w = W // stride
        stem_ch = 256  # FPN channels

        assert cls_feat.shape == (batch_size, stem_ch, expected_h, expected_w), \
            f"Head cls_stem level {i} shape: {cls_feat.shape}"
        assert reg_feat.shape == (batch_size, stem_ch, expected_h, expected_w), \
            f"Head reg_stem level {i} shape: {reg_feat.shape}"

        print(f"  stride={stride}: cls_stem={tuple(cls_feat.shape)}, reg_stem={tuple(reg_feat.shape)} ✓")

    print("\n" + "=" * 60)
    print("All checks passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_sanity_check()
