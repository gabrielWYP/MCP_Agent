#!/usr/bin/env python3
"""Evaluate grouped k-fold runs on their TEST folds and aggregate.

For every `<runs-root>/fold{k}/seed{s}/best_model.pt` this loads the
checkpoint exactly like `scripts/evaluate_checkpoint.py`, runs
`Trainer.collect_predictions()` on fold k's TEST stems (decoded at
`eval_conf_threshold`) and reports:

(a) per-fold metrics via `compute_map` — damage AP50, AP50-95 and
    precision/recall/F1 at the `conf_threshold` operating point — with
    mean +/- std across folds;
(b) POOLED out-of-fold metrics: one `compute_map` call over the concatenation
    of every fold's test predictions, so each labelled image is evaluated
    exactly once per seed. This is the headline metric: it is far less noisy
    than any single ~40-image test fold.

Outputs `<report-dir>/kfold_results.json` and `<report-dir>/kfold_results.md`
(default report dir: `reports/kfold/<experiment name>/`).

Usage:
    python scripts/aggregate_kfold.py --config configs/experiment/twostream.yaml \\
        --runs-root checkpoints/twostream_kfold --folds-dir data/annotations/yolo/folds
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_checkpoint import _apply_overrides, _load_model  # noqa: E402
from scripts.run_kfold import default_output_root, discover_folds, run_dir  # noqa: E402
from src.training.config import TrainingConfig  # noqa: E402
from src.training.dataset import YOLODataset, build_dataloader, collate_fn  # noqa: E402
from src.training.loop import Trainer  # noqa: E402
from src.training.metrics import compute_map  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DAMAGE_CLASS_ID = 1
PRED_KEYS = ("pred_boxes", "pred_scores", "pred_labels", "gt_boxes", "gt_labels")


def summarize_class(metrics: dict, cls_id: int = DAMAGE_CLASS_ID) -> dict:
    """Extract one class's AP/P/R/F1/counts from a `compute_map` result."""
    return {
        "ap50": float(metrics["per_class_ap_50"].get(cls_id, 0.0)),
        "ap50_95": float(metrics["per_class_ap_50_95"].get(cls_id, 0.0)),
        "precision": float(metrics["per_class_precision"].get(cls_id, 0.0)),
        "recall": float(metrics["per_class_recall"].get(cls_id, 0.0)),
        "f1": float(metrics["per_class_f1"].get(cls_id, 0.0)),
        "counts": {k: int(v) for k, v in metrics["per_class_counts"].get(cls_id, {}).items()},
        "map50": float(metrics["map50"]),
        "map_50_95": float(metrics["map_50_95"]),
    }


def _mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
        "n": len(values),
    }


def concat_predictions(per_fold: dict[int, dict[str, list]]) -> dict[str, list]:
    """Concatenate per-image prediction lists of several folds (fold order)."""
    pooled: dict[str, list] = {key: [] for key in PRED_KEYS}
    for fold in sorted(per_fold):
        for key in PRED_KEYS:
            pooled[key].extend(per_fold[fold][key])
    return pooled


def aggregate_seed(
    per_fold: dict[int, dict[str, list]],
    num_classes: int,
    score_threshold: float,
    cls_id: int = DAMAGE_CLASS_ID,
) -> dict:
    """Per-fold metrics, their mean/std, and pooled out-of-fold metrics for one seed."""
    folds = {}
    for fold in sorted(per_fold):
        preds = per_fold[fold]
        metrics = compute_map(**preds, num_classes=num_classes, score_threshold=score_threshold)
        folds[fold] = {**summarize_class(metrics, cls_id), "num_images": len(preds["gt_labels"])}

    pooled_preds = concat_predictions(per_fold)
    pooled_metrics = compute_map(**pooled_preds, num_classes=num_classes, score_threshold=score_threshold)
    pooled = {**summarize_class(pooled_metrics, cls_id), "num_images": len(pooled_preds["gt_labels"])}

    across = {
        name: _mean_std([f[name] for f in folds.values()])
        for name in ("ap50", "ap50_95", "precision", "recall", "f1")
    }
    return {"per_fold": folds, "per_fold_mean_std": across, "pooled": pooled}


def _to_cpu_lists(predictions: dict[str, list[torch.Tensor]]) -> dict[str, list[torch.Tensor]]:
    return {key: [t.detach().cpu() for t in predictions[key]] for key in PRED_KEYS}


def predict_fold(
    config_path: Path,
    checkpoint: Path,
    fold_manifest: Path,
    overrides: list[str],
    device: torch.device,
    eval_dir: Path,
) -> tuple[dict[str, list], TrainingConfig]:
    """Load one fold's best checkpoint and collect predictions on its TEST split."""
    config = TrainingConfig.from_yaml(config_path)
    _apply_overrides(config, [
        *overrides,
        f"split_manifest={fold_manifest}",
        "label_resolution=manifest",
    ])
    # Trainer creates output_dir (and a TensorBoard writer) on construction;
    # keep that away from the training run directory.
    config.output_dir = str(eval_dir)
    config.device = str(device)

    model = _load_model(config.model_type, str(checkpoint), config, device)
    if config.model_type == "master":
        config.head_strides = model.head_strides

    dataset = YOLODataset(
        rgb_dir=config.rgb_dir,
        nir_dir=config.nir_dir,
        labels_dir=config.labels_dir,
        split="test",
        image_size=config.image_size,
        nir_mean=config.nir_mean,
        nir_std=config.nir_std,
        letterbox_value=config.letterbox_value,
        manifest_path=config.split_manifest,
        label_resolution=config.label_resolution,
    )
    expected = len(json.loads(fold_manifest.read_text())["test"])
    if len(dataset) != expected:
        raise RuntimeError(f"{fold_manifest}: loaded {len(dataset)} test images, manifest lists {expected}.")

    loader = build_dataloader(
        dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=config.num_workers, pin_memory=True,
    )
    trainer = Trainer(model=model, config=config, train_loader=None, val_loader=loader)
    try:
        predictions = _to_cpu_lists(trainer.collect_predictions())
    finally:
        if trainer.writer is not None:
            trainer.writer.close()
    return predictions, config


def render_markdown(experiment: str, results: dict) -> str:
    """Short markdown table of per-fold and pooled damage metrics."""
    lines = [f"# Grouped k-fold results: {experiment}", ""]
    lines.append(
        f"Damage class (id {DAMAGE_CLASS_ID}); AP decoded at eval_conf_threshold="
        f"{results['eval_conf_threshold']}, P/R/F1 at conf_threshold={results['conf_threshold']}."
    )
    for seed, seed_res in results["seeds"].items():
        lines += ["", f"## Seed {seed}", "",
                  "| fold | images | AP50 | AP50-95 | P | R | F1 |",
                  "|---|---|---|---|---|---|---|"]
        for fold, m in seed_res["per_fold"].items():
            lines.append(
                f"| {fold} | {m['num_images']} | {m['ap50']:.4f} | {m['ap50_95']:.4f} | "
                f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |"
            )
        ms = seed_res["per_fold_mean_std"]
        lines.append(
            "| mean ± std | | " + " | ".join(
                f"{ms[k]['mean']:.4f} ± {ms[k]['std']:.4f}"
                for k in ("ap50", "ap50_95", "precision", "recall", "f1")
            ) + " |"
        )
        p = seed_res["pooled"]
        lines.append(
            f"| **pooled OOF** | {p['num_images']} | **{p['ap50']:.4f}** | {p['ap50_95']:.4f} | "
            f"{p['precision']:.4f} | {p['recall']:.4f} | {p['f1']:.4f} |"
        )
        if not seed_res["complete"]:
            lines.append("")
            lines.append(f"_Incomplete: missing folds {seed_res['missing_folds']}; pooled covers only "
                         "the evaluated folds._")
    if len(results["seeds"]) > 1:
        ps = results["pooled_ap50_across_seeds"]
        lines += ["", f"Pooled OOF damage AP50 across seeds: {ps['mean']:.4f} ± {ps['std']:.4f} (n={ps['n']})"]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate grouped k-fold test results.")
    parser.add_argument("--config", required=True, type=Path, help="Experiment YAML used for training.")
    parser.add_argument("--runs-root", type=Path, default=None,
                        help="Root holding fold{k}/seed{s}/best_model.pt "
                             "(default: <experiment output_dir>_kfold, as in run_kfold.py).")
    parser.add_argument("--folds-dir", type=Path, default=Path("data/annotations/yolo/folds"))
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--override", nargs="*", default=[],
                        help="Eval-time config overrides (same syntax as evaluate_checkpoint.py).")
    parser.add_argument("--report-dir", type=Path, default=None,
                        help="Default: reports/kfold/<experiment name>/")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto-detect).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment = args.config.stem
    runs_root = args.runs_root or default_output_root(args.config)
    report_dir = args.report_dir or Path("reports/kfold") / experiment
    report_dir.mkdir(parents=True, exist_ok=True)
    all_folds = discover_folds(args.folds_dir)
    folds = args.folds if args.folds else all_folds
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Test folds must be disjoint for the pooled metric to count each image once.
    seen: dict[str, int] = {}
    for fold in folds:
        for stem in json.loads((args.folds_dir / f"fold_{fold}.json").read_text())["test"]:
            if stem in seen:
                raise ValueError(f"Stem {stem} is in the test split of folds {seen[stem]} and {fold}.")
            seen[stem] = fold

    results: dict = {"experiment": experiment, "runs_root": str(runs_root),
                     "folds_dir": str(args.folds_dir), "seeds": {}}
    config = None
    for seed in args.seeds:
        per_fold: dict[int, dict[str, list]] = {}
        missing = []
        for fold in folds:
            checkpoint = run_dir(runs_root, fold, seed) / "best_model.pt"
            if not checkpoint.exists():
                logger.warning("Missing checkpoint %s, skipping fold %d seed %d.", checkpoint, fold, seed)
                missing.append(fold)
                continue
            logger.info("Evaluating fold %d seed %d (%s)", fold, seed, checkpoint)
            per_fold[fold], config = predict_fold(
                args.config, checkpoint, args.folds_dir / f"fold_{fold}.json", args.override,
                device, report_dir / "eval_logs" / f"fold{fold}_seed{seed}",
            )
        if not per_fold:
            logger.error("No checkpoints found for seed %d under %s.", seed, runs_root)
            continue
        seed_res = aggregate_seed(per_fold, config.num_classes, config.conf_threshold)
        missing += [f for f in all_folds if f not in folds]
        seed_res["missing_folds"] = sorted(set(missing))
        seed_res["complete"] = not seed_res["missing_folds"]
        results["seeds"][str(seed)] = seed_res
        p = seed_res["pooled"]
        logger.info("Seed %d pooled OOF damage AP50=%.4f AP50-95=%.4f over %d images%s",
                    seed, p["ap50"], p["ap50_95"], p["num_images"],
                    "" if seed_res["complete"] else f" (INCOMPLETE, missing folds {seed_res['missing_folds']})")

    if config is None:
        logger.error("Nothing evaluated.")
        return 1
    results["conf_threshold"] = config.conf_threshold
    results["eval_conf_threshold"] = config.eval_conf_threshold
    results["pooled_ap50_across_seeds"] = _mean_std([s["pooled"]["ap50"] for s in results["seeds"].values()])

    (report_dir / "kfold_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    markdown = render_markdown(experiment, results)
    (report_dir / "kfold_results.md").write_text(markdown)
    print(markdown)
    logger.info("Wrote %s/kfold_results.{json,md}", report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
