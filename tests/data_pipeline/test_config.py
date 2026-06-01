"""Tests for PipelineConfig loading and defaults."""

import os
import pytest
import yaml

from src.data_pipeline.config import PipelineConfig, NIRStats


class TestPipelineConfigDefaults:
    """Test default values are sensible."""

    def test_default_target_size(self):
        config = PipelineConfig()
        assert config.target_size == (640, 640)

    def test_default_augmentation_factor(self):
        config = PipelineConfig()
        assert config.augmentation_factor == 3

    def test_default_nir_stats(self):
        config = PipelineConfig()
        assert config.nir_stats.mean == 0.45
        assert config.nir_stats.std == 0.22

    def test_default_rgb_stats(self):
        config = PipelineConfig()
        assert config.rgb_mean == [0.485, 0.456, 0.406]
        assert config.rgb_std == [0.229, 0.224, 0.225]

    def test_default_max_workers(self):
        config = PipelineConfig()
        assert config.max_workers == 8


class TestPipelineConfigFromYAML:
    """Test YAML loading."""

    def test_load_yaml(self, tmp_path):
        yaml_content = {
            "oci_bucket": "test-bucket",
            "target_size": [416, 416],
            "augmentation_factor": 5,
            "nir_stats": {"mean": 0.42, "std": 0.19},
        }
        yaml_file = tmp_path / "test_config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = PipelineConfig.from_yaml(str(yaml_file))
        assert config.oci_bucket == "test-bucket"
        assert config.target_size == (416, 416)
        assert config.augmentation_factor == 5
        assert config.nir_stats.mean == 0.42
        assert config.nir_stats.std == 0.19

    def test_yaml_missing_fields_use_defaults(self, tmp_path):
        yaml_content = {"oci_bucket": "minimal-bucket"}
        yaml_file = tmp_path / "minimal.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = PipelineConfig.from_yaml(str(yaml_file))
        assert config.oci_bucket == "minimal-bucket"
        assert config.target_size == (640, 640)  # default
        assert config.augmentation_factor == 3     # default

    def test_yaml_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            PipelineConfig.from_yaml("/nonexistent/config.yaml")

    def test_target_size_int_becomes_tuple(self, tmp_path):
        yaml_content = {"target_size": 224}
        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = PipelineConfig.from_yaml(str(yaml_file))
        assert config.target_size == (224, 224)


class TestPipelineConfigFromDict:
    """Test dict-based construction."""

    def test_from_dict(self):
        d = {
            "oci_bucket": "dict-bucket",
            "augmentation_factor": 2,
            "spatial_intensity": 0.6,
        }
        config = PipelineConfig.from_dict(d)
        assert config.oci_bucket == "dict-bucket"
        assert config.augmentation_factor == 2
        assert config.spatial_intensity == 0.6


class TestEnvOverrides:
    """Test environment variable overrides."""

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("BUCKET_NAME", "env-bucket")
        monkeypatch.setenv("S3_ENDPOINT_URL", "https://env-endpoint.com")

        config = PipelineConfig()
        config.apply_env_overrides()
        assert config.oci_bucket == "env-bucket"
        assert config.oci_endpoint == "https://env-endpoint.com"
