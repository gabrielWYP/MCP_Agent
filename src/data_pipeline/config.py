"""OCI download configuration with YAML loading and environment overrides."""

import os
import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    """Configuration for RGB/NIR cache downloads from OCI-compatible storage.

    Training-specific augmentation, normalization, and homography behavior lives
    in ``src.training``. This config only covers object listing, pair discovery,
    and download/cache behavior.
    """

    # OCI / S3 settings
    oci_endpoint: str = ""
    oci_bucket: str = ""
    oci_prefix: str = "news/"
    oci_region: str = "us-ashburn-1"
    oci_access_key: str = ""
    oci_secret_key: str = ""

    # Local paths
    cache_dir: str = "data/cache"

    # Pair discovery settings
    pair_matching: str = "stem"          # "stem" or "timestamp_id"
    min_timestamp: int = 0               # Filter pairs by timestamp ID >= this
    default_class_name: str = "mango"    # Class name when matching by timestamp_id

    # Download settings
    max_workers: int = 8
    retry_attempts: int = 3
    retry_base_delay: float = 1.0

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """Load config from YAML file, applying defaults for missing fields."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        return cls._from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """Create config from a dictionary."""
        return cls._from_dict(d)

    @classmethod
    def _from_dict(cls, raw: dict) -> "PipelineConfig":
        """Build PipelineConfig from a raw dict."""
        kwargs = {}

        # Map top-level keys
        simple_keys = [
            "oci_endpoint", "oci_bucket", "oci_prefix", "oci_region",
            "oci_access_key", "oci_secret_key",
            "cache_dir",
            "max_workers", "retry_attempts", "retry_base_delay",
            "pair_matching", "min_timestamp", "default_class_name",
        ]
        for key in simple_keys:
            if key in raw:
                kwargs[key] = raw[key]

        return cls(**kwargs)

    def apply_env_overrides(self) -> "PipelineConfig":
        """Override config values from environment variables if set."""
        env_map = {
            "S3_ENDPOINT_URL": "oci_endpoint",
            "BUCKET_NAME": "oci_bucket",
            "OCI_REGION": "oci_region",
            "S3_ACCESS_KEY": "oci_access_key",
            "S3_SECRET_KEY": "oci_secret_key",
            "CACHE_DIR": "cache_dir",
            "PAIR_MATCHING": "pair_matching",
            "MIN_TIMESTAMP": "min_timestamp",
            "DEFAULT_CLASS_NAME": "default_class_name",
        }
        int_attrs = {"min_timestamp"}
        for env_key, attr in env_map.items():
            val = os.getenv(env_key)
            if val:
                if attr in int_attrs:
                    setattr(self, attr, int(val))
                else:
                    setattr(self, attr, val)
        return self
