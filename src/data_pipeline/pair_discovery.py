"""
RGB/NIR pair discovery by filename stem matching or timestamp ID matching.

Stem matching: {class}/{rgb|nir}/{stem}.{ext}
Timestamp ID matching: {prefix}_rgb_{id}.jpg / {prefix}_nir_{id}.jpg
"""

import logging
import re
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

    @staticmethod
    def match_by_timestamp_id(
        objects_list: list[dict],
        class_name: str = "mango",
        min_timestamp: int = 0,
    ) -> tuple[list[PairedSample], list[str]]:
        """
        Match RGB and NIR images by timestamp ID embedded in filename.

        Expected pattern: {prefix}_rgb_{id}.{ext} / {prefix}_nir_{id}.{ext}

        Args:
            objects_list: List of dicts with 'key' field (full object key path).
            class_name: Class name to assign to all pairs.
            min_timestamp: Only include pairs with timestamp ID >= this value.

        Returns:
            Tuple of (matched_pairs, unmatched_entries).
        """
        # Pattern: captures modality (rgb/nir) and numeric ID
        FILENAME_RE = re.compile(
            r'(?:.*/)?(\w+)_(rgb|nir)_(\d+)\.(jpg|jpeg|png|tiff|tif)$',
            re.IGNORECASE,
        )

        # Index: {id: {"rgb": key, "nir": key, "prefix": str}}
        id_index: dict[int, dict[str, str]] = {}

        for obj in objects_list:
            key = obj.get("key", obj) if isinstance(obj, dict) else obj
            if not isinstance(key, str):
                continue

            m = FILENAME_RE.search(key)
            if not m:
                continue

            prefix = m.group(1)
            modality = m.group(2).lower()
            timestamp_id = int(m.group(3))

            if timestamp_id < min_timestamp:
                continue

            entry = id_index.setdefault(timestamp_id, {})
            entry["prefix"] = prefix
            entry[f"{modality}_key"] = key

        # Match pairs — need both rgb_key and nir_key for the same id
        matched_pairs: list[PairedSample] = []
        unmatched: list[str] = []

        for tid in sorted(id_index.keys()):
            entry = id_index[tid]
            rgb_key = entry.get("rgb_key")
            nir_key = entry.get("nir_key")

            if rgb_key and nir_key:
                matched_pairs.append(PairedSample(
                    class_name=class_name,
                    stem=f"{entry['prefix']}_{tid}",
                    rgb_key=rgb_key,
                    nir_key=nir_key,
                ))
            else:
                missing = entry.get("rgb_key") or entry.get("nir_key")
                unmatched.append(missing)
                which = "NIR" if rgb_key else "RGB"
                logger.warning(
                    "No %s match for timestamp %d (%s)", which, tid, missing,
                )

        logger.info(
            "Timestamp-ID pair discovery: %d matched, %d unmatched",
            len(matched_pairs), len(unmatched),
        )

        return matched_pairs, unmatched
