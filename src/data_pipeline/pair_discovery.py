"""
RGB/NIR pair discovery by filename stem matching.

Matches RGB and NIR images by their filename stems (excluding directory
and extension) following the convention: {class}/{rgb|nir}/{stem}.{ext}
"""

import logging
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class PairedSample(NamedTuple):
    """A matched RGB+NIR image pair."""
    class_name: str
    stem: str
    rgb_key: str
    nir_key: str


class PairDiscovery:
    """
    Discovers paired RGB+NIR images from object listings.

    Uses filename stem matching across rgb/ and nir/ subdirectories
    within each class folder.
    """

    @staticmethod
    def match_by_stem(objects_list: list[dict]) -> tuple[list[PairedSample], list[str]]:
        """
        Match RGB and NIR images by filename stem.

        Args:
            objects_list: List of dicts with 'key' field (full object key path).
                Expected structure: {prefix}/{class}/{rgb|nir}/{stem}.{ext}

        Returns:
            Tuple of (matched_pairs, unmatched_rgb):
                - matched_pairs: List of PairedSample with both rgb_key and nir_key
                - unmatched_rgb: List of RGB keys with no corresponding NIR match
        """
        # Index by class -> modality -> {stem: full_key}
        rgb_index: dict[str, dict[str, str]] = {}
        nir_index: dict[str, dict[str, str]] = {}

        for obj in objects_list:
            key = obj.get("key", obj) if isinstance(obj, dict) else obj
            if not isinstance(key, str):
                continue

            parts = Path(key).parts
            if len(parts) < 3:
                continue

            # Find the modality segment (rgb or nir)
            # Convention: .../class_name/rgb/stem.ext or .../class_name/nir/stem.ext
            modality = None
            class_name = None
            for i, part in enumerate(parts):
                if part in ("rgb", "nir") and i > 0:
                    modality = part
                    class_name = parts[i - 1]
                    break

            if modality is None or class_name is None:
                continue

            stem = Path(key).stem
            ext = Path(key).suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".tiff", ".tif"}:
                continue

            if modality == "rgb":
                rgb_index.setdefault(class_name, {})[stem] = key
            else:
                nir_index.setdefault(class_name, {})[stem] = key

        # Match pairs
        matched_pairs: list[PairedSample] = []
        unmatched_rgb: list[str] = []

        all_classes = set(rgb_index.keys()) | set(nir_index.keys())

        for class_name in sorted(all_classes):
            rgb_stems = rgb_index.get(class_name, {})
            nir_stems = nir_index.get(class_name, {})

            for stem in sorted(rgb_stems.keys()):
                if stem in nir_stems:
                    matched_pairs.append(PairedSample(
                        class_name=class_name,
                        stem=stem,
                        rgb_key=rgb_stems[stem],
                        nir_key=nir_stems[stem],
                    ))
                else:
                    unmatched_rgb.append(rgb_stems[stem])
                    logger.warning(
                        "No NIR match for RGB image: %s/%s", class_name, stem,
                    )

        logger.info(
            "Pair discovery: %d matched, %d unmatched RGB",
            len(matched_pairs), len(unmatched_rgb),
        )

        return matched_pairs, unmatched_rgb
