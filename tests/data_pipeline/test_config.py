"""Tests for OCI download configuration loading and defaults."""

import pytest
import yaml

from src.data_pipeline.config import PipelineConfig


class TestPipelineConfigDefaults:
    """Test default values are sensible."""

    def test_default_cache_dir(self):
        config = PipelineConfig()
        assert config.cache_dir == "data/cache"

    def test_default_max_workers(self):
        config = PipelineConfig()
        assert config.max_workers == 8


class TestPipelineConfigFromYAML:
    """Test YAML loading."""

    def test_load_yaml(self, tmp_path):
        yaml_content = {
            "oci_bucket": "test-bucket",
            "pair_matching": "timestamp_id",
            "min_timestamp": 1780617600,
        }
        yaml_file = tmp_path / "test_config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = PipelineConfig.from_yaml(str(yaml_file))
        assert config.oci_bucket == "test-bucket"
        assert config.pair_matching == "timestamp_id"
        assert config.min_timestamp == 1780617600

    def test_yaml_missing_fields_use_defaults(self, tmp_path):
        yaml_content = {"oci_bucket": "minimal-bucket"}
        yaml_file = tmp_path / "minimal.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = PipelineConfig.from_yaml(str(yaml_file))
        assert config.oci_bucket == "minimal-bucket"
        assert config.cache_dir == "data/cache"

    def test_yaml_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            PipelineConfig.from_yaml("/nonexistent/config.yaml")

class TestPipelineConfigFromDict:
    """Test dict-based construction."""

    def test_from_dict(self):
        d = {
            "oci_bucket": "dict-bucket",
            "cache_dir": "data/cache/test",
        }
        config = PipelineConfig.from_dict(d)
        assert config.oci_bucket == "dict-bucket"
        assert config.cache_dir == "data/cache/test"


class TestEnvOverrides:
    """Test environment variable overrides."""

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("BUCKET_NAME", "env-bucket")
        monkeypatch.setenv("S3_ENDPOINT_URL", "https://env-endpoint.com")

        config = PipelineConfig()
        config.apply_env_overrides()
        assert config.oci_bucket == "env-bucket"
        assert config.oci_endpoint == "https://env-endpoint.com"
