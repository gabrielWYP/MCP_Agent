#!/usr/bin/env python3
"""
Run the full data augmentation pipeline with homography alignment.

Usage:
    python3 scripts/run_pipeline.py [--config config/augmentation.yaml]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.config import PipelineConfig
from src.data_pipeline.pipeline import DataAugmentationPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run the data augmentation pipeline with homography alignment."
    )
    parser.add_argument(
        "--config",
        default="config/augmentation.yaml",
        help="Path to pipeline config YAML (default: config/augmentation.yaml)",
    )
    args = parser.parse_args()

    config = PipelineConfig.from_yaml(args.config)
    logger.info("Config loaded: matching=%s, homography=%s, aug_factor=%d",
                config.pair_matching, config.homography_path, config.augmentation_factor)

    # Clear old output for clean run
    output_dir = Path(config.output_dir)
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
        logger.info("Cleared old output: %s", output_dir)

    pipeline = DataAugmentationPipeline(config)
    stats = pipeline.run()

    print()
    print("=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Quick sanity checks
    expected = stats["downloaded"] * config.augmentation_factor
    if stats["augmented"] != expected:
        logger.warning("Expected %d augmented samples, got %d", expected, stats["augmented"])
    else:
        logger.info("Augmented count matches expected: %d", stats["augmented"])

    return 0 if stats["skipped"] == 0 and stats["augmented"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
