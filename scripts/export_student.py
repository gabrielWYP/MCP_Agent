#!/usr/bin/env python3
"""Export the RGB-only StudentModel to an on-device (LiteRT / .tflite) model.

Loads a plain-student or KD-student checkpoint (legacy `kd_proj_*` adapter
keys are stripped, everything else loads strictly), wraps it so the graph
returns only raw head outputs, converts it with litert-torch (formerly
ai-edge-torch) to an FP32 .tflite with an NHWC input, and writes a metadata
sidecar JSON next to it describing preprocessing, output layout, anchors,
decode/NMS settings, source checkpoint and tool versions.

Runs in the export venv only (see requirements-export.txt):

    CUDA_VISIBLE_DEVICES="" .venv-export/bin/python scripts/export_student.py \\
        --checkpoint checkpoints/final_runs/<run>/destilado/<best>/best_model.pt \\
        --out-dir exports --verify --data-root data

See docs/deployment/mobile-export.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.export.parity import (
    DEFAULT_MAX_ABS_DIFF,
    DEFAULT_MIN_MATCH_RATE,
    run_parity,
)
from src.export.student_export import (
    SUPPORTED_PRECISIONS,
    StudentExportWrapper,
    build_metadata,
    convert_to_tflite,
    load_student_checkpoint,
    resolve_decode_settings,
    tflite_io_names,
    write_metadata,
)
from src.models.student.student_model import StudentModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the student detector for mobile (LiteRT).")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=str, help="Student or KD-student .pt checkpoint.")
    source.add_argument(
        "--random-init", action="store_true",
        help="Export a randomly initialised student (pipeline smoke test only).",
    )
    parser.add_argument("--out-dir", type=str, default="exports", help="Output directory (default: exports).")
    parser.add_argument("--name", type=str, default=None, help="Output basename (default: student_<precision>).")
    parser.add_argument("--format", type=str, choices=["tflite"], default="tflite")
    parser.add_argument(
        "--precision", type=str, choices=list(SUPPORTED_PRECISIONS), default="fp32",
        help="Numeric precision. Only fp32 is implemented; fp16/int8 are planned.",
    )
    parser.add_argument("--image-size", type=int, default=None, help="Default: checkpoint config, else 640.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for --random-init.")
    parser.add_argument("--verify", action="store_true", help="Run the parity check after export.")
    parser.add_argument("--data-root", type=str, default="data", help="Dataset root for --verify.")
    parser.add_argument("--split", type=str, default="val", help="Split for --verify (default: val).")
    parser.add_argument("--num-images", type=int, default=20, help="Images for --verify (default: 20).")
    parser.add_argument("--max-abs-diff", type=float, default=DEFAULT_MAX_ABS_DIFF)
    parser.add_argument("--min-match-rate", type=float, default=DEFAULT_MIN_MATCH_RATE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"student_{args.precision}"
    model_path = out_dir / f"{name}.{args.format}"
    metadata_path = out_dir / f"{name}.json"

    if args.checkpoint:
        loaded = load_student_checkpoint(args.checkpoint)
        student = loaded.model
        config = loaded.checkpoint_config
        checkpoint_meta = {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": loaded.checkpoint_sha256,
            "legacy_kd_adapter_keys_stripped": loaded.legacy_kd_keys_stripped,
            "model_type": config.get("model_type"),
            **loaded.checkpoint_info,
        }
        logger.info(
            "Loaded %s (stripped %d legacy kd_proj_* keys, epoch=%s, best_map50=%s)",
            args.checkpoint, loaded.legacy_kd_keys_stripped,
            loaded.checkpoint_info.get("epoch"), loaded.checkpoint_info.get("best_map50"),
        )
    else:
        torch.manual_seed(args.seed)
        student = StudentModel().eval()
        config = {}
        checkpoint_meta = {"path": None, "random_init_seed": args.seed}
        logger.warning("Exporting a RANDOM-INIT student (no checkpoint).")

    image_size = args.image_size or int(config.get("image_size", 640))
    letterbox_value = int(config.get("letterbox_value", 114))
    decode = resolve_decode_settings(config)
    wrapper = StudentExportWrapper(student).eval()

    logger.info("Converting to %s %s at %dx%d (NHWC input) ...", args.format, args.precision, image_size, image_size)
    convert_to_tflite(wrapper, model_path, image_size=image_size, precision=args.precision)
    size_mb = model_path.stat().st_size / (1024 * 1024)
    logger.info("Wrote %s (%.2f MiB)", model_path, size_mb)

    metadata = build_metadata(
        model_file=model_path,
        precision=args.precision,
        image_size=image_size,
        num_classes=student.num_classes,
        decode=decode,
        letterbox_value=letterbox_value,
        io_names=tflite_io_names(model_path),
        checkpoint=checkpoint_meta,
        num_parameters=sum(p.numel() for p in student.parameters()),
    )
    write_metadata(metadata, metadata_path)
    logger.info("Wrote %s", metadata_path)

    if not args.verify:
        return 0

    report = run_parity(
        model_path=model_path,
        metadata=metadata,
        wrapper=wrapper,
        data_root=Path(args.data_root),
        split=args.split,
        num_images=args.num_images,
        max_abs_diff=args.max_abs_diff,
        min_match_rate=args.min_match_rate,
    )
    result = report.to_dict()
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        logger.error("Parity FAILED: %s", "; ".join(result["failures"]))
        return 1
    logger.info("Parity PASSED on %d images.", result["images"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
