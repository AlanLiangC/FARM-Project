"""Typed models for scene graph state.

SceneState is a TypedDict documenting every field of the scene_state dict
used throughout the pipeline. Using TypedDict (rather than a dataclass)
preserves full backward compatibility with existing dict access patterns.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TypedDict

import torch


class SceneState(TypedDict, total=False):
    # -- Gaussian state (core per-object tensors, device-resident) --
    count: torch.Tensor               # (N,) int32 — detection count per object
    means: torch.Tensor               # (N, 3) float32 — 3D centroid
    cov6: torch.Tensor                # (N, 6) float32 — packed covariance
    features: torch.Tensor            # (N, F) float32 — semantic embeddings
    active: torch.Tensor              # (N,) bool — active flag
    class_ids: torch.Tensor           # (N,) int64 — semantic class id
    object_id: torch.Tensor           # (N,) int64 — canonical external id

    # -- Identity / merge tracking --
    id_redirect: Dict[int, int]
    loser_object_ids: List[Set[int]]
    cannot_link_object_ids: Dict[int, Set[int]]

    # -- Captions (current, per-object) --
    object_caption: List[str]
    object_caption_decision: List[str]
    object_caption_embedding: List[list]
    object_siglip2_embedding: List[list]
    object_qwen3_vl_embedding: List[list]

    # -- Caption history (per-object) --
    object_caption_history: List[List[str]]
    object_caption_embedding_history: List[List[list]]
    object_siglip2_embedding_history: List[List[list]]
    object_qwen3_vl_embedding_history: List[List[list]]

    # -- Categories & attributes --
    object_category: List[str]
    object_supercategory: List[str]
    object_category_candidates: List[List[str]]
    object_key_attributes: List[List[str]]
    object_detection_category_conf: List[Dict[str, float]]

    # -- High-quality captioning --
    high_quality_captioning: List[bool]
    high_quality_views: List[list]

    # -- Observation data --
    rgb_observations: List[list]
    view_means: List[list]
    view_cov6: List[list]
    object_image_ids: List[List[int]]
    viewpoint_image_ids: List[List[int]]
    object_mask_observations: List[List[Dict[str, Any]]]
    images: List[Any]
    image_positions: List[torch.Tensor]

    # -- Per-object sparse voxel point cloud (CSR-flat layout) --
    # object_voxel_keys_flat[off[i]:off[i+1]] = packed int64 voxel keys for
    # object i, all at the object's current level. Effective voxel size is
    # base_v * 2**level. Used for tight-AABB at retrieval time + retrieval-time
    # SOR/DBSCAN filtering. See scene_graph.utils.geometry voxel helpers.
    object_voxel_keys_flat: torch.Tensor       # (M_total,) int64
    object_voxel_keys_offsets: torch.Tensor    # (N+1,) int64
    object_voxel_levels: torch.Tensor          # (N,) int8

    # -- Locking --
    is_locked: List[bool]

    # -- Covisibility --
    covisibility_max_objects: int
    covisibility_blocks: int
    covisibility_adj_u64: torch.Tensor
    covisibility_active_u64: torch.Tensor
    covisibility_weights: Dict[int, Dict[int, float]]

    # -- Region hierarchy --
    region_ids: List[int]                    # (N,) per-object region index, -1 = unassigned
    region_labels: List[str]                 # per-region label
    region_object_lists: List[List[int]]     # per-region list of object indices
    region_centroids: List[List[float]]      # per-region [x, y, z] centroid
    region_label_confidence: List[float]     # per-region LLM confidence (0-1)
    region_version: int                      # monotonic counter, bumped on each update

    # -- Misc runtime --
    viewpoint_map: dict
    current_robot_position: Optional[torch.Tensor]


BITS_PER_BLOCK = 64
DEFAULT_COVISIBILITY_MAX_OBJECTS = 1000


def _covisibility_blocks(max_objects: int) -> int:
    max_objects = max(1, int(max_objects))
    return (max_objects + BITS_PER_BLOCK - 1) // BITS_PER_BLOCK


def initialize_scene_graph_state(
    feature_dim: int,
    device: torch.device | str,
    *,
    covisibility_max_objects: int = DEFAULT_COVISIBILITY_MAX_OBJECTS,
) -> dict[str, Any]:
    """Allocate the empty tensors/containers for the scene-graph database."""
    torch_device = torch.device(device)
    covis_max_objects = max(1, int(covisibility_max_objects))
    covis_blocks = _covisibility_blocks(covis_max_objects)

    def _zeros(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device=torch_device)

    return {
        "count": _zeros((0,), torch.int32),
        "means": _zeros((0, 3), torch.float32),
        "cov6": _zeros((0, 6), torch.float32),
        "features": _zeros((0, feature_dim), torch.float32),
        "object_caption": [],
        "object_caption_decision": [],
        "object_category": [],
        "object_supercategory": [],
        "object_category_candidates": [],
        "object_key_attributes": [],
        "object_caption_embedding": [],
        "object_siglip2_embedding": [],
        "object_qwen3_vl_embedding": [],
        "object_caption_history": [],
        "object_caption_embedding_history": [],
        "object_siglip2_embedding_history": [],
        "object_qwen3_vl_embedding_history": [],
        "caption_error_counts": {},
        "caption_results_total": 0,
        "caption_results_empty_total": 0,
        "caption_results_keep_total": 0,
        "caption_results_drop_total": 0,
        "object_detection_category_conf": [],
        "active": _zeros((0,), torch.bool),
        "object_id": torch.zeros((0,), dtype=torch.int64),
        "id_redirect": {},
        "loser_object_ids": [],
        "cannot_link_object_ids": {},
        "is_locked": [],
        "viewpoint_map": {},
        "class_ids": _zeros((0,), torch.long),
        "high_quality_captioning": [],
        "high_quality_views": [],
        "rgb_observations": [],
        "object_image_ids": [],
        "viewpoint_image_ids": [],
        "object_mask_observations": [],
        "view_means": [],
        "view_cov6": [],
        "images": [],
        "image_positions": [],
        "current_robot_position": None,
        "covisibility_max_objects": covis_max_objects,
        "covisibility_blocks": covis_blocks,
        "covisibility_adj_u64": _zeros((covis_max_objects, covis_blocks), torch.int64),
        "covisibility_active_u64": _zeros((covis_blocks,), torch.int64),
        "covisibility_weights": {},
        "region_ids": [],
        "region_labels": [],
        "region_object_lists": [],
        "region_centroids": [],
        "region_label_confidence": [],
        "region_version": 0,
        "object_voxel_keys_flat": _zeros((0,), torch.int64),
        "object_voxel_keys_offsets": _zeros((1,), torch.int64),
        "object_voxel_levels": _zeros((0,), torch.int8),
    }
