"""
OCI object storage client with retry logic and parallel download.

Wraps s3fs with exponential backoff retry, paginated listing,
and ThreadPoolExecutor-based parallel downloads.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from src.data_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class OCIManager:
    """
    Manages OCI/S3 object storage operations.

    Provides paginated listing, exponential backoff retry,
    and parallel download with local caching.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._s3 = None

    @property
    def s3(self):
        """Lazy-init s3fs client."""
        if self._s3 is None:
            import s3fs
            self._s3 = s3fs.S3FileSystem(
                client_kwargs={
                    "endpoint_url": self.config.oci_endpoint,
                    "region_name": self.config.oci_region,
                },
                config_kwargs={"signature_version": "s3v4"},
                key=self.config.oci_access_key,
                secret=self.config.oci_secret_key,
            )
        return self._s3

    def _retry(self, fn, *args, **kwargs):
        """Execute fn with exponential backoff retry."""
        attempts = self.config.retry_attempts
        base_delay = self.config.retry_base_delay
        last_exc = None

        for attempt in range(attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_exc = e
                if attempt < attempts - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                        attempt + 1, attempts, e, delay,
                    )
                    time.sleep(delay)

        raise ConnectionError(
            f"All {attempts} attempts failed. Last error: {last_exc}"
        )

    def list_objects(self, prefix: Optional[str] = None) -> list[dict]:
        """
        List objects under the configured bucket + prefix.

        Returns list of dicts with keys: key, size, last_modified.
        Supports pagination for buckets exceeding 1000 objects.
        """
        full_prefix = f"{self.config.oci_bucket}/{prefix or self.config.oci_prefix}"

        def _list():
            results = []
            # Use find() for recursive listing — ls() with delimiter="/"
            # only shows immediate "subdirectories", not nested files
            keys = self.s3.find(full_prefix, maxdepth=None)
            for key in keys:
                if isinstance(key, str) and not key.endswith("/"):
                    results.append({"key": key, "size": 0, "last_modified": None})
            return results

        return self._retry(_list)

    def download_pair(
        self,
        rgb_key: str,
        nir_key: str,
        cache_dir: str,
        class_name: str,
    ) -> tuple[str, str]:
        """
        Download an RGB+NIR pair to local cache.

        Returns (local_rgb_path, local_nir_path).
        Uses size-based dedup to skip already-cached files.
        """
        rgb_ext = Path(rgb_key).suffix
        nir_ext = Path(nir_key).suffix
        rgb_stem = Path(rgb_key).stem
        nir_stem = Path(nir_key).stem

        local_rgb = Path(cache_dir) / class_name / "rgb" / f"{rgb_stem}{rgb_ext}"
        local_nir = Path(cache_dir) / class_name / "nir" / f"{nir_stem}{nir_ext}"

        self._download_single(rgb_key, str(local_rgb))
        self._download_single(nir_key, str(local_nir))

        return str(local_rgb), str(local_nir)

    def _download_single(self, key: str, local_path: str) -> bool:
        """Download a single object with cache check."""
        local = Path(local_path)

        # Size-based cache check
        if local.exists():
            try:
                info = self.s3.info(key)
                remote_size = info.get("size", info.get("Size", -1))
                if remote_size == local.stat().st_size:
                    return False  # cache hit
            except Exception:
                pass

        local.parent.mkdir(parents=True, exist_ok=True)

        # Atomic download via temp file
        tmp_path = local.with_suffix(local.suffix + ".tmp")

        def _download():
            self.s3.get(key, str(tmp_path))

        try:
            self._retry(_download)
            tmp_path.rename(local)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        return True

    def download_pairs_parallel(
        self,
        pairs: list[dict],
        cache_dir: str,
    ) -> list[dict]:
        """
        Download multiple pairs in parallel using ThreadPoolExecutor.

        Args:
            pairs: List of dicts with keys: class_name, stem, rgb_key, nir_key
            cache_dir: Local cache directory

        Returns:
            List of dicts with: class_name, stem, rgb_path, nir_path (local paths)
        """
        results = []
        max_workers = self.config.max_workers

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for pair in pairs:
                future = executor.submit(
                    self.download_pair,
                    pair["rgb_key"],
                    pair["nir_key"],
                    cache_dir,
                    pair["class_name"],
                )
                futures[future] = pair

            for i, future in enumerate(as_completed(futures), 1):
                pair = futures[future]
                try:
                    rgb_path, nir_path = future.result()
                    results.append({
                        "class_name": pair["class_name"],
                        "stem": pair["stem"],
                        "rgb_path": rgb_path,
                        "nir_path": nir_path,
                    })
                except Exception as e:
                    logger.error(
                        "Failed to download pair %s/%s: %s",
                        pair["class_name"], pair["stem"], e,
                    )

                if i % 100 == 0:
                    logger.info("Downloaded %d/%d pairs", i, len(pairs))

        return results
