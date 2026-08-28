"""Segmentation backends shared by the mapping pipeline."""

from .interfaces import SegmentationBackend
from .models import SegmentationOutput
from .dino import DINOFeaturesExtractor
from .yoloe import YOLOESegmenter
from .sam3_precomputed import SAM3PrecomputedSegmenter

__all__ = [
    "SegmentationBackend",
    "SegmentationOutput",
    "YOLOESegmenter",
    "SAM3PrecomputedSegmenter",
    "DINOFeaturesExtractor",
]
