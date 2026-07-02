#!/usr/bin/env python3
"""
Download RGB+NIR image pairs from OCI to local cache.
No augmentation, no preprocessing — just listing, pairing, and downloading.

Usage:
    python3 scripts/download_oci.py [--config config/augmentation.yaml]
"""
import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.config import PipelineConfig
from src.data_pipeline.oci_client import OCIManager
from src.data_pipeline.pair_discovery import PairDiscovery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def _parse_since_date(value: str, timezone_name: str) -> int:
    """Convert a YYYY-MM-DD date into a Unix timestamp.

    Args:
        value: Date string in ISO format, e.g. ``2026-05-24``.
        timezone_name: IANA timezone used to interpret local midnight.

    Returns:
        Unix timestamp at the start of that local date.

    Raises:
        argparse.ArgumentTypeError: If the date or timezone is invalid.
    """
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
        tzinfo = ZoneInfo(timezone_name)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --since-date '{value}'. Expected YYYY-MM-DD."
        ) from exc
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --timezone '{timezone_name}'."
        ) from exc

    return int(parsed.replace(tzinfo=tzinfo).timestamp())


def main():
    parser = argparse.ArgumentParser(
        description="Download RGB+NIR pairs from OCI to local cache."
    )
    parser.add_argument(
        "--config",
        default="config/augmentation.yaml",
        help="Path to pipeline config YAML (default: config/augmentation.yaml)",
    )
    parser.add_argument(
        "--since-date",
        help="Only include timestamp-ID pairs captured on/after this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--timezone",
        default="UTC",
        help="Timezone for --since-date local midnight (default: UTC).",
    )
    parser.add_argument(
        "--min-timestamp",
        type=int,
        help="Override config min_timestamp directly with a Unix timestamp.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List and match OCI objects without downloading files.",
    )
    args = parser.parse_args()

    config = PipelineConfig.from_yaml(args.config).apply_env_overrides()
    if args.since_date and args.min_timestamp is not None:
        parser.error("Use either --since-date or --min-timestamp, not both.")
    if args.since_date:
        config.min_timestamp = _parse_since_date(args.since_date, args.timezone)
    elif args.min_timestamp is not None:
        config.min_timestamp = args.min_timestamp

    logger.info(
        "Config: bucket=%s, prefix=%s, matching=%s, min_timestamp=%d",
        config.oci_bucket,
        config.oci_prefix,
        config.pair_matching,
        config.min_timestamp,
    )

    # Ensure cache dir exists
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. List objects from OCI
    oci = OCIManager(config)
    logger.info("Listing objects from OCI: %s/%s", config.oci_bucket, config.oci_prefix)
    objects = oci.list_objects()
    logger.info("Found %d objects", len(objects))

    # 2. Discover pairs
    if config.pair_matching == "timestamp_id":
        matched_pairs, unmatched = PairDiscovery.match_by_timestamp_id(
            objects,
            class_name=config.default_class_name,
            min_timestamp=config.min_timestamp,
        )
    else:
        matched_pairs, unmatched = PairDiscovery.match_by_stem(objects)

    logger.info("Matched %d pairs, %d unmatched", len(matched_pairs), len(unmatched))

    if not matched_pairs:
        logger.warning("No matched pairs found. Check min_timestamp or prefix.")
        return 1

    if args.dry_run:
        print()
        print("=" * 50)
        print("DRY RUN COMPLETE")
        print("=" * 50)
        print(f"  Objects found:    {len(objects)}")
        print(f"  Pairs matched:    {len(matched_pairs)}")
        print(f"  Pairs unmatched:  {len(unmatched)}")
        print(f"  Min timestamp:    {config.min_timestamp}")
        if matched_pairs:
            first = matched_pairs[0]
            last = matched_pairs[-1]
            first_ts = int(first.stem.rsplit("_", 1)[-1])
            last_ts = int(last.stem.rsplit("_", 1)[-1])
            print(
                "  Timestamp range:  "
                f"{first_ts} ({datetime.fromtimestamp(first_ts, timezone.utc).isoformat()})"
                " -> "
                f"{last_ts} ({datetime.fromtimestamp(last_ts, timezone.utc).isoformat()})"
            )
            print("\n  Sample pairs:")
            for p in matched_pairs[:10]:
                print(f"    {p.stem}")
            if len(matched_pairs) > 10:
                print(f"    ... and {len(matched_pairs) - 10} more")
        return 0

    # 3. Convert to dicts for download
    pair_dicts = [
        {
            "class_name": p.class_name,
            "stem": p.stem,
            "rgb_key": p.rgb_key,
            "nir_key": p.nir_key,
        }
        for p in matched_pairs
    ]

    # 4. Download in parallel
    logger.info("Downloading %d pairs to %s (workers=%d)...", len(pair_dicts), cache_dir, config.max_workers)
    results = oci.download_pairs_parallel(pair_dicts, str(cache_dir))

    # 5. Summary
    print()
    print("=" * 50)
    print("DOWNLOAD COMPLETE")
    print("=" * 50)
    print(f"  Objects found:    {len(objects)}")
    print(f"  Pairs matched:    {len(matched_pairs)}")
    print(f"  Pairs unmatched:  {len(unmatched)}")
    print(f"  Downloaded:       {len(results)}")
    print(f"  Cache dir:        {cache_dir}")

    # Show sample of downloaded files
    if results:
        print(f"\n  Sample downloads:")
        for r in results[:5]:
            print(f"    {r['class_name']}/{r['stem']}")
            print(f"      rgb: {r['rgb_path']}")
            print(f"      nir: {r['nir_path']}")
        if len(results) > 5:
            print(f"    ... and {len(results) - 5} more")

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
