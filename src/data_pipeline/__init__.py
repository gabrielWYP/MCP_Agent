"""
Data Augmentation Pipeline for MasterModel.

Bridges OCI object storage to MasterModel-ready dual-stream tensors.
Pipeline: OCI download -> pair discovery -> preprocess -> augment -> vector store.

Spatial sync strategy: 4-channel temp stack -> albumentations -> split back
to RGB(3,H,W) + NIR(1,H,W) for MasterModel.forward(rgb, nir).
"""

from src.data_pipeline.config import PipelineConfig
from src.data_pipeline.pipeline import DataAugmentationPipeline

__all__ = [
    "PipelineConfig",
    "DataAugmentationPipeline",
]
