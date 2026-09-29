"""Single source of truth for how a split's images and label files are resolved.

Kept free of torch/cv2 imports so `config.py` can validate `label_resolution`
without pulling in the dataset module.

  "split_dir": the split is whatever `labels_dir/<split>/*.txt` contains
               (historical behaviour; the manifest, if given, is only a
               fail-closed consistency guard).
  "manifest":  the split is exactly `manifest[split]`; each stem's label is
               resolved by searching every split subdirectory of
               `labels_dir`. Lets alternative manifests (e.g. grouped k-fold
               manifests from `scripts/make_grouped_folds.py`) reuse the
               canonical label files without copying them.
"""

from __future__ import annotations

LABEL_RESOLUTION_SPLIT_DIR = "split_dir"
LABEL_RESOLUTION_MANIFEST = "manifest"
LABEL_RESOLUTIONS: tuple[str, ...] = (LABEL_RESOLUTION_SPLIT_DIR, LABEL_RESOLUTION_MANIFEST)
