"""
Pipeline configuration with YAML loading and environment variable overrides.
"""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class NIRStats:
    """NIR channel normalization statistics."""
    mean: float = 0.45
    std: float = 0.22


@dataclass
class PipelineConfig:
    """
    Configuration for the data augmentation pipeline.

    All fields have sensible defaults and can be overridden via YAML or env vars.
    """

    # OCI / S3 settings
    oci_endpoint: str = ""
    oci_bucket: str = ""
    oci_prefix: str = "news/"
    oci_access_key: str = ""
    oci_secret_key: str = ""

    # Local paths
    cache_dir: str = "data/cache"
    output_dir: str = "data/augmented"
    homography_path: Optional[str] = None

    # Image dimensions
    target_size: tuple[int, int] = (640, 640)

    # Augmentation parameters
    augmentation_factor: int = 3
    spatial_intensity: float = 0.4
    photometric_intensity: float = 0.3

    # NIR normalization
    nir_stats: NIRStats = field(default_factory=NIRStats)

    # Download settings
    max_workers: int = 8
    retry_attempts: int = 3
    retry_base_delay: float = 1.0

    # ImageNet RGB normalization (fixed, not configurable)
    rgb_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    rgb_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

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
        """Build PipelineConfig from a raw dict, handling nested nir_stats."""
        kwargs = {}

        # Map top-level keys
        simple_keys = [
            "oci_endpoint", "oci_bucket", "oci_prefix",
            "oci_access_key", "oci_secret_key",
            "cache_dir", "output_dir", "homography_path",
            "augmentation_factor", "spatial_intensity", "photometric_intensity",
            "max_workers", "retry_attempts", "retry_base_delay",
            "rgb_mean", "rgb_std",
        ]
        for key in simple_keys:
            if key in raw:
                kwargs[key] = raw[key]

        # Handle target_size (may be list in YAML)
        if "target_size" in raw:
            ts = raw["target_size"]
            if isinstance(ts, (list, tuple)) and len(ts) == 2:
                kwargs["target_size"] = tuple(ts)
            elif isinstance(ts, int):
                kwargs["target_size"] = (ts, ts)

        # Handle nir_stats
        if "nir_stats" in raw:
            ns = raw["nir_stats"]
            if isinstance(ns, dict):
                kwargs["nir_stats"] = NIRStats(
                    mean=ns.get("mean", 0.45),
                    std=ns.get("std", 0.22),
                )

        return cls(**kwargs)

    def apply_env_overrides(self) -> "PipelineConfig":
        """Override config values from environment variables if set."""
        env_map = {
            "S3_ENDPOINT_URL": "oci_endpoint",
            "BUCKET_NAME": "oci_bucket",
            "S3_ACCESS_KEY": "oci_access_key",
            "S3_SECRET_KEY": "oci_secret_key",
            "CACHE_DIR": "cache_dir",
            "OUTPUT_DIR": "output_dir",
            "HOMOGRAPHY_PATH": "homography_path",
        }
        for env_key, attr in env_map.items():
            val = os.getenv(env_key)
            if val:
                setattr(self, attr, val)
        return self
