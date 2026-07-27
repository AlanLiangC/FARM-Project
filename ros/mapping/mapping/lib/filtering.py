"""ROS-facing aliases for the shared segmentation filtering utilities."""

from __future__ import annotations

from scene_graph.map_update.filtering import (
    filter_detections_by_distance,
    filter_detections_by_num_pixels,
    filter_detections_duplicates_iou,
    filter_detections_touching_image_border,
    filter_duplicate_masks_iou,
    filter_uninformative_yoloe_labels,
    mask_seg_outputs,
    normalize_seg_outputs,
)

__all__ = [
    "filter_detections_by_distance",
    "filter_detections_by_num_pixels",
    "filter_detections_duplicates_iou",
    "filter_detections_touching_image_border",
    "filter_duplicate_masks_iou",
    "filter_uninformative_yoloe_labels",
    "mask_seg_outputs",
    "normalize_seg_outputs",
]
