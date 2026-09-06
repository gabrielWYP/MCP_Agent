#!/usr/bin/env python3
"""V0 (fusion-redesign): measure peak VRAM for one training step.

One forward + backward + optimizer step per configuration,
`torch.cuda.reset_peak_memory_stats()` then `torch.cuda.max_memory_allocated()`
— a real measurement, not the design document's hand-computed projection.

The pre-redesign `master_v1` variant (dual-stream fusion) was already
measured in `proposal.md` Round 2 (2.32 GB @ batch 1, 4.42 GB @ batch 2, OOM
@ batch 8, fp32), so it is not re-measured here. It is reachable again as
`fusion_mode="cross_attention"` (`DualConvNeXtBackbone`/`CrossModalFusion`/
`DualFPN`, restored as a selectable mode); this script measures the two
early-fusion variants:

    new_3lvl  — head_strides=[8, 16, 32]  (P2 ablated, matches the old level count)
    new_4lvl  — head_strides=[4, 8, 16, 32]  (the target config, P2 reconnected)

Both already include the W5 head-stem deduplication fix (it is not a toggle
in the shipped code — the duplicate stem computation was deleted outright,
not gated behind a flag), so a separate "preW5" measurement is not possible
without reverting that fix. The design's own §4 projection for "new, Nlvl,
pre-W5" is a hand-computed estimate, not something this script can reproduce
against shipped code.

Usage:
    ./.venv/bin/python scripts/measure_vram.py --image-size 640 --batches 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.master.master_model import MasterModel
from src.training.loss import YOLOv8Loss
from src.training.strides import DEFAULT_HEAD_STRIDES, STUDENT_STRIDES

VARIANTS = {
    "new_3lvl": list(STUDENT_STRIDES),
    "new_4lvl": list(DEFAULT_HEAD_STRIDES),
}


def _measure_one_step(head_strides: list[int], batch_size: int, image_size: int, device: torch.device) -> dict:
    model = MasterModel(
        num_classes=2, pretrained_backbone=False, backbone_variant="tiny",
        head_strides=head_strides,
    ).to(device)
    criterion = YOLOv8Loss(
        num_classes=2, box_weight=7.5, cls_weight=0.5,
        class_weights=[0.5, 1.5], strides=head_strides,
        assigner_level_ranges=[32.0, 64.0, 128.0][: len(head_strides) - 1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    rgb = torch.randn(batch_size, 3, image_size, image_size, device=device)
    nir = torch.randn(batch_size, 1, image_size, image_size, device=device)
    bboxes = [torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=device) for _ in range(batch_size)]
    labels = [torch.tensor([0], device=device) for _ in range(batch_size)]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(rgb, nir)
    loss, _ = criterion(out["preds"], {"bboxes": bboxes, "labels": labels})
    loss.backward()
    optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    else:
        peak_gb = None

    del model, criterion, optimizer, rgb, nir, out, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"peak_gb": peak_gb}


def main() -> int:
    parser = argparse.ArgumentParser(description="V0: measure peak VRAM for one training step.")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()), choices=list(VARIANTS.keys()))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")

    results = {}
    for variant in args.variants:
        head_strides = VARIANTS[variant]
        results[variant] = {}
        for batch_size in args.batches:
            try:
                measured = _measure_one_step(head_strides, batch_size, args.image_size, device)
                peak = measured["peak_gb"]
                results[variant][batch_size] = peak
                peak_str = f"{peak:.3f} GB" if peak is not None else "N/A (CPU)"
                print(f"[{variant}] batch={batch_size}: peak={peak_str}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    results[variant][batch_size] = "OOM"
                    print(f"[{variant}] batch={batch_size}: OOM")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise

    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
