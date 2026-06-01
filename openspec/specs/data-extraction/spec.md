# Data Extraction Specification

## Purpose

Extract paired RGB+NIR mango images from OCI object storage, discover filename-matched pairs, and cache downloads locally for the preprocessing pipeline.

## Requirements

### Requirement: OCI Bucket Listing

The system SHALL list objects in a configurable OCI bucket and prefix, returning file metadata (key, size, last_modified).

- The client MUST use s3fs/S3-compatible API with configurable endpoint, credentials, and bucket.
- Listing MUST support paginated responses for buckets exceeding 1000 objects.
- Connection failures MUST be retried with exponential backoff (3 attempts, 2^n second delays).

#### Scenario: Successful bucket listing

- GIVEN valid OCI credentials and a bucket with objects
- WHEN the client lists objects under prefix "mango-2024/"
- THEN a list of object metadata entries is returned with key, size, and last_modified fields

#### Scenario: OCI connectivity failure

- GIVEN OCI endpoint is unreachable
- WHEN the client attempts to list objects
- THEN the client SHALL retry 3 times with exponential backoff
- AND raise a ConnectionError with descriptive message after all retries exhausted

### Requirement: RGB/NIR Pair Discovery

The system SHALL discover paired RGB and NIR images by matching filename stems (excluding directory and extension).

- If an RGB image has stem "img001" and an NIR image has stem "img001" in the corresponding `rgb/` and `nir/` subdirectories, they form a pair.
- The path convention MUST follow: `{class}/{rgb|nir}/{stem}.{ext}`.
- Pair discovery MUST return both matched and unmatched entries.

#### Scenario: All RGB images have NIR pairs

- GIVEN a bucket with structure `sano/rgb/img001.jpg` and `sano/nir/img001.jpg`
- WHEN pair discovery runs
- THEN result contains one matched pair with rgb_key and nir_key pointing to the respective objects

#### Scenario: Missing NIR pair

- GIVEN a bucket with `sano/rgb/img002.jpg` but no `sano/nir/img002.*`
- WHEN pair discovery runs
- THEN img002 is reported as unmatched_rgb with a warning logged
- AND processing SHALL continue with matched pairs only

### Requirement: Parallel Download with Caching

The system SHALL download discovered pairs to a local cache directory with deduplication.

- Downloaded files MUST be stored under `{cache_dir}/{class}/{rgb|nir}/{stem}.{ext}` mirroring OCI structure.
- If a file already exists locally with matching size, the download SHALL be skipped.
- Downloads MUST use parallel execution (up to `max_workers` configurable, default 8).
- Partial downloads on failure MUST be cleaned up (no corrupt files left in cache).

#### Scenario: Cache hit skips download

- GIVEN a file exists locally with size matching the OCI object metadata
- WHEN download is requested
- THEN the file is skipped and not re-downloaded

#### Scenario: Large batch memory management

- GIVEN a bucket listing of 5000+ image pairs
- WHEN download is triggered with max_workers=8
- THEN memory usage MUST NOT exceed 2× the largest single image size per worker
- AND download progress is logged every 100 pairs