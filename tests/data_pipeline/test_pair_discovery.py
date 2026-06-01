"""Tests for PairDiscovery stem matching."""

import pytest

from src.data_pipeline.pair_discovery import PairDiscovery, PairedSample


class TestPairDiscoveryAllMatched:
    """All RGB images have NIR pairs."""

    def test_all_matched(self):
        objects = [
            {"key": "bucket/sano/rgb/img001.jpg"},
            {"key": "bucket/sano/nir/img001.jpg"},
            {"key": "bucket/sano/rgb/img002.jpg"},
            {"key": "bucket/sano/nir/img002.jpg"},
        ]
        matched, unmatched = PairDiscovery.match_by_stem(objects)
        assert len(matched) == 2
        assert len(unmatched) == 0
        assert matched[0].class_name == "sano"
        assert matched[0].stem == "img001"

    def test_multi_class(self):
        objects = [
            {"key": "bucket/sano/rgb/img001.jpg"},
            {"key": "bucket/sano/nir/img001.jpg"},
            {"key": "bucket/danado/rgb/img002.jpg"},
            {"key": "bucket/danado/nir/img002.jpg"},
        ]
        matched, unmatched = PairDiscovery.match_by_stem(objects)
        assert len(matched) == 2
        assert len(unmatched) == 0
        classes = {m.class_name for m in matched}
        assert classes == {"sano", "danado"}


class TestPairDiscoveryMissingNIR:
    """Some RGB images lack NIR pairs."""

    def test_missing_nir(self):
        objects = [
            {"key": "bucket/sano/rgb/img001.jpg"},
            {"key": "bucket/sano/nir/img001.jpg"},
            {"key": "bucket/sano/rgb/img002.jpg"},
            # No NIR for img002
        ]
        matched, unmatched = PairDiscovery.match_by_stem(objects)
        assert len(matched) == 1
        assert len(unmatched) == 1
        assert "img002" in unmatched[0]

    def test_all_unmatched(self):
        objects = [
            {"key": "bucket/sano/rgb/img001.jpg"},
            {"key": "bucket/sano/rgb/img002.jpg"},
        ]
        matched, unmatched = PairDiscovery.match_by_stem(objects)
        assert len(matched) == 0
        assert len(unmatched) == 2


class TestPairDiscoveryEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_list(self):
        matched, unmatched = PairDiscovery.match_by_stem([])
        assert len(matched) == 0
        assert len(unmatched) == 0

    def test_non_image_files_ignored(self):
        objects = [
            {"key": "bucket/sano/rgb/img001.jpg"},
            {"key": "bucket/sano/nir/img001.jpg"},
            {"key": "bucket/sano/rgb/notes.txt"},
            {"key": "bucket/sano/rgb/.DS_Store"},
        ]
        matched, unmatched = PairDiscovery.match_by_stem(objects)
        assert len(matched) == 1

    def test_different_extensions_matched(self):
        objects = [
            {"key": "bucket/sano/rgb/img001.jpg"},
            {"key": "bucket/sano/nir/img001.png"},
        ]
        matched, unmatched = PairDiscovery.match_by_stem(objects)
        assert len(matched) == 1
        assert matched[0].stem == "img001"

    def test_string_keys_accepted(self):
        """Objects can be plain strings instead of dicts."""
        objects = [
            "bucket/sano/rgb/img001.jpg",
            "bucket/sano/nir/img001.jpg",
        ]
        matched, unmatched = PairDiscovery.match_by_stem(objects)
        assert len(matched) == 1
