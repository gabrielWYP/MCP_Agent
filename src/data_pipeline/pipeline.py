"""
Pipeline orchestrator: chains all components end-to-end.

OCI list -> pair discovery -> download -> preprocess -> augment -> save.
Streams one pair at a time to manage memory efficiently.
"""

import logging
from pathlib import Path
from typing import Optional

from src.data_pipeline.augmentor import Augmentor
from src.data_pipeline.config import PipelineConfig
from src.data_pipeline.oci_client import OCIManager
from src.data_pipeline.pair_discovery import PairDiscovery
from src.data_pipeline.preprocessor import Preprocessor
from src.data_pipeline.vector_store import VectorStore

logger = logging.getLogger(__name__)


class DataAugmentationPipeline:
    """
    End-to-end data augmentation pipeline.

    Chains: OCI listing -> pair discovery -> parallel download ->
    preprocess -> augment -> vector store -> manifest.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.oci = OCIManager(config)
        self.preprocessor = Preprocessor(config)
        self.augmentor = Augmentor(config)
        self.vector_store = VectorStore()

    def run(
        self,
        local_pairs: Optional[list[dict]] = None,
    ) -> dict:
        """
        Run the full pipeline.

        Args:
            local_pairs: Optional list of local pairs (for testing without OCI).
                Each dict: {class_name, stem, rgb_path, nir_path}
                If None, downloads from OCI.

        Returns:
            Stats dict: {downloaded, processed, augmented, skipped, unmatched}
        """
        stats = {
            "downloaded": 0,
            "processed": 0,
            "augmented": 0,
            "skipped": 0,
            "unmatched": 0,
        }

        all_manifest_entries = []

        if local_pairs is not None:
            # Use pre-downloaded local pairs (testing mode)
            pairs = local_pairs
            stats["downloaded"] = len(pairs)
        else:
            # OCI mode: list, discover, download
            logger.info("Listing objects from OCI: %s/%s", self.config.oci_bucket, self.config.oci_prefix)
            objects = self.oci.list_objects()
            logger.info("Found %d objects", len(objects))

            matched_pairs, unmatched_rgb = (
                PairDiscovery.match_by_timestamp_id(
                    objects,
                    class_name=self.config.default_class_name,
                    min_timestamp=self.config.min_timestamp,
                )
                if self.config.pair_matching == "timestamp_id"
                else PairDiscovery.match_by_stem(objects)
            )
            stats["unmatched"] = len(unmatched_rgb)
            logger.info("Matched %d pairs, %d unmatched RGB", len(matched_pairs), len(unmatched_rgb))

            if not matched_pairs:
                logger.warning("No matched pairs found. Exiting pipeline.")
                return stats

            # Convert PairedSample namedtuples to dicts for download
            pair_dicts = [
                {
                    "class_name": p.class_name,
                    "stem": p.stem,
                    "rgb_key": p.rgb_key,
                    "nir_key": p.nir_key,
                }
                for p in matched_pairs
            ]

            # Parallel download
            pairs = self.oci.download_pairs_parallel(pair_dicts, self.config.cache_dir)
            stats["downloaded"] = len(pairs)

        # Ensure output directory exists
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Stream one pair at a time
        for pair in pairs:
            class_name = pair["class_name"]
            stem = pair["stem"]
            rgb_path = pair["rgb_path"]
            nir_path = pair["nir_path"]

            # Assign label based on class name (sorted for determinism)
            label = self._class_to_label(class_name)

            # Preprocess
            result = self.preprocessor.process_pair(rgb_path, nir_path)
            if result is None:
                logger.warning("Skipping pair %s/%s: preprocessing failed", class_name, stem)
                stats["skipped"] += 1
                continue

            stats["processed"] += 1

            # Augment
            augmented_samples = self.augmentor.augment_pair(
                rgb_tensor=result["rgb"],
                nir_tensor=result["nir"],
                label=label,
                class_name=class_name,
                stem=stem,
            )

            # Save each augmented sample
            for aug_sample in augmented_samples:
                meta = {
                    **result["metadata"],
                    **aug_sample.metadata,
                    "source_oci_key": pair.get("rgb_key", ""),
                }

                entry = self.vector_store.save_sample(
                    rgb=aug_sample.rgb,
                    nir=aug_sample.nir,
                    label=aug_sample.label,
                    class_name=aug_sample.class_name,
                    stem=stem,
                    aug_id=aug_sample.metadata.get("aug_id", 0),
                    output_dir=str(output_dir),
                    metadata=meta,
                )
                all_manifest_entries.append(entry)
                stats["augmented"] += 1

        # Write final manifest
        if all_manifest_entries:
            self.vector_store.write_manifest(all_manifest_entries, str(output_dir))

        logger.info(
            "Pipeline complete: %d downloaded, %d processed, %d augmented, %d skipped, %d unmatched",
            stats["downloaded"], stats["processed"], stats["augmented"],
            stats["skipped"], stats["unmatched"],
        )

        return stats

    def _class_to_label(self, class_name: str) -> int:
        """Map class name to integer label (deterministic sort)."""
        # Use a simple sorted mapping for determinism
        known_classes = ["danado", "sano"]
        if class_name in known_classes:
            return known_classes.index(class_name)
        # Fallback: hash-based for unknown classes
        return hash(class_name) % 1000
