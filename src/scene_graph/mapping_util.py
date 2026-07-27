"""Backward-compatibility shim — imports have moved to their canonical modules.

Geometry helpers      → scene_graph.utils.geometry
ImageRecord           → scene_graph.storage.models
initialize_scene_*    → scene_graph.map_update.models
"""

from __future__ import annotations

from scene_graph.map_update.models import (  # noqa: F401
    BITS_PER_BLOCK,
    DEFAULT_COVISIBILITY_MAX_OBJECTS,
    initialize_scene_graph_state,
)
from scene_graph.storage.models import ImageRecord  # noqa: F401
from scene_graph.utils.geometry import (  # noqa: F401
    cov6_to_matrix,
    invert_pose,
    matrix_to_cov6,
    relative_pose,
    transform_segmentation_to_world,
)


def iter_batches(dataset, batch_size: int):
    total = len(dataset)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        idxs = list(range(start, end))
        colors, depths, intrinsics, poses = [], [], [], []
        for idx in idxs:
            color, depth, intrinsics_i, pose = dataset[idx]
            colors.append(color)
            depths.append(depth)
            intrinsics.append(intrinsics_i)
            poses.append(pose)
        yield idxs, colors, depths, intrinsics, poses
