import contextlib
import logging
import math
import os
import random
from typing import Any, Dict, List, Optional

import torch

from scene_graph.utils.geometry import (
    VOXEL_BASE_V,
    VOXEL_MAX_LEVEL,
    init_voxel_level,
    merge_voxel_buffers,
    promote_voxel_keys,
    voxel_keys_to_world,
    voxelize_points,
)

from .cannot_link import cannot_link_index_pairs
from .covisibility import (
    ensure_covisibility_state,
    merge_covisibility_loser_into_winner,
    update_covisibility_active_bitset,
)
from .mask_observations import DEFAULT_MAX_MASK_OBSERVATIONS_PER_OBJECT

logger = logging.getLogger(__name__)


# ---- Per-object sparse voxel-cloud configuration ----
# Voxel-cloud cap per object (paired with adaptive level promotion).
# Once an object's unique voxel count exceeds this, the level is bumped
# (effective spacing doubles) until the count fits again.
VOXEL_CAP_PER_OBJECT = 1000
# Target number of voxels along the longest axis when initializing a brand-new
# object's level from its first detection. Leaves headroom for later views.
VOXEL_K_INIT = 32

# Smallest eigenvalue we tolerate for an active object's covariance. Any
# fused cov whose minimum eigenvalue is already ≥ this value is left alone;
# otherwise it gets shifted by ``floor − λ_min`` along I so λ_min lands
# exactly at the floor. Conditional shifting (vs. unconditional ridge) is
# important: an unconditional ridge added every batch compounds across
# updates, eventually making cov_eff ≈ N·ε after N updates. Override via
# env var SCENE_GRAPH_FUSED_COV_RIDGE. With fp64 moment accumulation the
# floor only really fires when the underlying point geometry is genuinely
# degenerate (planar surfaces, single-frame slivers).
DEFAULT_FUSED_COV_RIDGE = float(os.environ.get("SCENE_GRAPH_FUSED_COV_RIDGE", "1e-6"))  # m²

# Maximum world-space distance allowed between the loser's and the winner's
# Gaussian centres for a union-find merge to go through. The candidate-
# generation stage of get_neighbors uses Hellinger distance which can admit
# loose merges between objects metres apart when their cov ellipsoids
# overlap (large-cov "wall" / "floor"-like fragments). This guard refuses
# such cross-room merges before they corrupt the fused stats.  ``inf``
# disables the guard. Override with env var SCENE_GRAPH_MAX_MERGE_DISTANCE_M.
DEFAULT_MAX_MERGE_DISTANCE_M = float(os.environ.get("SCENE_GRAPH_MAX_MERGE_DISTANCE_M", "1.0"))  # m


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


# After an object-object merge, the voxel buffer is the tightest available 3D
# support. Recompute the compact Gaussian from that support so future Hellinger
# matching does not keep using the pre-merge winner geometry.
DEFAULT_RECOMPUTE_MERGED_GAUSSIAN_FROM_VOXELS = _env_bool(
    "SCENE_GRAPH_RECOMPUTE_MERGED_GAUSSIAN_FROM_VOXELS",
    True,
)

# Scale-normalized safety guard for object-object merges. This avoids a fixed
# center-distance cutoff while still refusing merges whose hypothetical merged
# support becomes much larger than the members being merged.
DEFAULT_MERGE_VOXEL_GEOMETRY_GUARD = _env_bool("SCENE_GRAPH_MERGE_VOXEL_GEOMETRY_GUARD", True)
DEFAULT_MERGE_VOXEL_MAX_AABB_GROWTH = _env_float("SCENE_GRAPH_MERGE_VOXEL_MAX_AABB_GROWTH", 2.75)
DEFAULT_MERGE_VOXEL_MAX_COV_EIG_GROWTH = _env_float("SCENE_GRAPH_MERGE_VOXEL_MAX_COV_EIG_GROWTH", 8.0)
DEFAULT_MERGE_VOXEL_MIN_KEYS_FOR_GUARD = _env_int("SCENE_GRAPH_MERGE_VOXEL_MIN_KEYS_FOR_GUARD", 8)


def _symmetrize_and_eigfloor_cov33(cov: torch.Tensor, floor: float) -> torch.Tensor:
    """Symmetrize ``cov`` and *conditionally* shift it so λ_min ≥ ``floor``.

    For each (..., 3, 3) covariance, we add ``max(0, floor − λ_min) · I``.
    Well-conditioned covs (λ_min already ≥ floor) get only the symmetrize
    pass — no inflation. Ill-conditioned covs get shifted by exactly the
    minimum amount needed to satisfy SPD, so the perturbation doesn't
    compound across batches: cov_after = cov_before whenever cov_before
    is already SPD.

    Cheap: 3×3 symmetric eigvalsh has a closed form and PyTorch dispatches
    a batched LAPACK call.
    """
    if cov is None or not isinstance(cov, torch.Tensor) or cov.numel() == 0:
        return cov
    cov = 0.5 * (cov + cov.transpose(-2, -1))
    if floor <= 0.0:
        return cov
    try:
        w_min = torch.linalg.eigvalsh(cov)[..., 0]
    except Exception:
        # If eigvalsh fails for some reason, fall back to unconditional ridge.
        eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
        return cov + float(floor) * eye
    shift = (float(floor) - w_min).clamp(min=0.0)
    if not bool((shift > 0).any()):
        return cov
    eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
    return cov + shift[..., None, None] * eye


def _eigfloor_cov6(cov6: torch.Tensor, floor: float) -> torch.Tensor:
    """Conditional ridge for a packed (..., 6) covariance vector.

    Unpacks to 3×3, runs ``_symmetrize_and_eigfloor_cov33``, repacks.
    """
    if cov6 is None or not isinstance(cov6, torch.Tensor) or cov6.numel() == 0 or floor <= 0.0:
        return cov6
    cov33 = _unpack_cov6(cov6)
    cov33 = _symmetrize_and_eigfloor_cov33(cov33, floor)
    return _pack_cov6(cov33)


def _read_voxel_buffer(
    state: Dict[str, Any],
    n_objects: int,
    device: torch.device,
) -> tuple[List[torch.Tensor], List[int]]:
    """Read the CSR-flat voxel buffer into per-object lists for in-place edits.

    Pads to ``n_objects`` if the persisted buffer is shorter. Returned tensors
    share storage with ``object_voxel_keys_flat`` (so we don't copy a 100k-int
    buffer just to slice it); callers must not mutate slices in place.
    """
    keys_flat = state.get("object_voxel_keys_flat")
    offsets = state.get("object_voxel_keys_offsets")
    levels = state.get("object_voxel_levels")
    key_lists: List[torch.Tensor] = []
    if isinstance(keys_flat, torch.Tensor) and isinstance(offsets, torch.Tensor) and offsets.numel() >= 1:
        keys_flat = keys_flat.to(device=device)
        offsets_cpu = offsets.detach().to("cpu", dtype=torch.int64).tolist()
        n_persisted = max(0, len(offsets_cpu) - 1)
        for i in range(n_persisted):
            s, e = int(offsets_cpu[i]), int(offsets_cpu[i + 1])
            key_lists.append(keys_flat[s:e])
    while len(key_lists) < n_objects:
        key_lists.append(torch.empty((0,), dtype=torch.int64, device=device))

    if isinstance(levels, torch.Tensor) and levels.numel() > 0:
        level_list = [int(x) for x in levels.detach().to("cpu", dtype=torch.int64).tolist()]
    else:
        level_list = []
    while len(level_list) < n_objects:
        level_list.append(0)
    return key_lists, level_list


def _write_voxel_buffer(
    state: Dict[str, Any],
    key_lists: List[torch.Tensor],
    level_list: List[int],
    device: torch.device,
) -> None:
    """Re-pack per-object lists back into CSR-flat tensors on ``state``."""
    n = len(key_lists)
    counts = [int(k.numel()) for k in key_lists]
    offsets = torch.zeros((n + 1,), dtype=torch.int64, device=device)
    if n > 0:
        offsets[1:] = torch.tensor(counts, dtype=torch.int64, device=device).cumsum(0)
    if any(counts):
        # Move any CPU tensors to device (rare; keys_lists are normally on device).
        flat = torch.cat([k.to(device=device, dtype=torch.int64) for k in key_lists], dim=0)
    else:
        flat = torch.empty((0,), dtype=torch.int64, device=device)
    levels_t = torch.tensor(level_list[:n], dtype=torch.int8, device=device) if n > 0 else torch.empty(
        (0,), dtype=torch.int8, device=device
    )
    state["object_voxel_keys_flat"] = flat
    state["object_voxel_keys_offsets"] = offsets
    state["object_voxel_levels"] = levels_t


def _ingest_points_into_object(
    key_lists: List[torch.Tensor],
    level_list: List[int],
    obj_idx: int,
    pts: torch.Tensor,
    *,
    cap: int = VOXEL_CAP_PER_OBJECT,
    base_v: float = VOXEL_BASE_V,
    max_level: int = VOXEL_MAX_LEVEL,
) -> None:
    """Voxelize ``pts`` (world frame) into object ``obj_idx``'s buffer.

    For new objects (empty buffer), choose the initial level from the points.
    Otherwise voxelize at the object's current level, append + unique, and
    promote until the cap is satisfied.
    """
    if pts.numel() == 0:
        return
    cur_keys = key_lists[obj_idx]
    cur_level = int(level_list[obj_idx])

    if cur_keys.numel() == 0:
        # Brand-new buffer: pick the level from the first observation.
        new_level = init_voxel_level(pts, cap=cap, k_init=VOXEL_K_INIT, base_v=base_v, max_level=max_level)
        new_keys = voxelize_points(pts, new_level, base_v=base_v, dedup=True)
        # init_voxel_level already guarantees keys.numel() <= cap (or max_level).
        key_lists[obj_idx] = new_keys
        level_list[obj_idx] = int(new_level)
        return

    new_keys = voxelize_points(pts, cur_level, base_v=base_v, dedup=True)
    if new_keys.numel() == 0:
        return
    merged = torch.unique(torch.cat([cur_keys, new_keys], dim=0))
    level = cur_level
    while merged.numel() > int(cap) and level < int(max_level):
        merged = promote_voxel_keys(merged, levels=1)
        level += 1
    key_lists[obj_idx] = merged
    level_list[obj_idx] = int(level)


def _slice_det_points(
    det_points_flat: Optional[torch.Tensor],
    det_points_offsets: Optional[torch.Tensor],
    det_idx: int,
) -> torch.Tensor:
    """Return the (n,3) point cloud for detection ``det_idx`` from CSR storage."""
    if det_points_flat is None or det_points_offsets is None:
        return torch.empty((0, 3), dtype=torch.float32)
    if det_idx < 0 or det_idx + 1 >= det_points_offsets.numel():
        return torch.empty((0, 3), dtype=torch.float32, device=det_points_flat.device)
    s = int(det_points_offsets[det_idx].item())
    e = int(det_points_offsets[det_idx + 1].item())
    if e <= s:
        return torch.empty((0, 3), dtype=det_points_flat.dtype, device=det_points_flat.device)
    return det_points_flat[s:e]


def _unpack_cov6(cov6: torch.Tensor) -> torch.Tensor:
    """
    cov6: (..., 6) -> (..., 3, 3) symmetric covariance matrix
    Order: [xx, xy, xz, yy, yz, zz] (matches segmentation output)
    """
    xx, xy, xz, yy, yz, zz = cov6.unbind(-1)
    row0 = torch.stack([xx, xy, xz], dim=-1)
    row1 = torch.stack([xy, yy, yz], dim=-1)
    row2 = torch.stack([xz, yz, zz], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _pack_cov6(cov: torch.Tensor) -> torch.Tensor:
    """
    cov: (..., 3, 3) -> (..., 6) in [xx, xy, xz, yy, yz, zz] format
    (keeps ordering consistent with segmentation and get_neighbors)
    """
    xx = cov[..., 0, 0]
    xy = cov[..., 0, 1]
    xz = cov[..., 0, 2]
    yy = cov[..., 1, 1]
    yz = cov[..., 1, 2]
    zz = cov[..., 2, 2]
    return torch.stack([xx, xy, xz, yy, yz, zz], dim=-1)


def _voxel_geometry_stats(
    keys: torch.Tensor,
    level: int,
    *,
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """Return mean/cov/AABB stats for an object's sparse voxel support."""
    if keys is None or not isinstance(keys, torch.Tensor) or keys.numel() == 0:
        return None
    pts = voxel_keys_to_world(
        keys.to(device=device, dtype=torch.int64),
        int(level),
        base_v=VOXEL_BASE_V,
    ).to(device=device, dtype=torch.float32)
    if pts.numel() == 0:
        return None
    finite = torch.isfinite(pts).all(dim=1)
    if not bool(finite.any().item()):
        return None
    pts = pts[finite]
    mean = pts.mean(dim=0)
    centered = pts - mean.unsqueeze(0)
    denom = max(1, int(pts.shape[0]))
    cov = centered.transpose(0, 1).matmul(centered) / float(denom)
    cov = _symmetrize_and_eigfloor_cov33(cov, DEFAULT_FUSED_COV_RIDGE)
    mins = pts.min(dim=0).values
    maxs = pts.max(dim=0).values
    voxel_size = float(VOXEL_BASE_V) * float(1 << int(level))
    dims = (maxs - mins).clamp(min=0.0) + float(voxel_size)
    try:
        eig_max = torch.linalg.eigvalsh(cov)[-1].clamp(min=float(DEFAULT_FUSED_COV_RIDGE))
    except Exception:
        eig_max = torch.diagonal(cov).max().clamp(min=float(DEFAULT_FUSED_COV_RIDGE))
    return {
        "mean": mean,
        "cov": cov,
        "aabb_dims": dims,
        "aabb_diag": torch.linalg.norm(dims).clamp(min=1e-6),
        "aabb_volume": torch.prod(dims.clamp(min=1e-6)),
        "eig_max": eig_max,
        "n": torch.tensor(int(pts.shape[0]), device=device, dtype=torch.int64),
    }


def _merge_voxel_group(
    key_lists: List[torch.Tensor],
    level_list: List[int],
    indices: List[int],
    *,
    device: torch.device,
) -> tuple[Optional[torch.Tensor], int]:
    merged_keys: Optional[torch.Tensor] = None
    merged_level = 0
    for idx in indices:
        if idx < 0 or idx >= len(key_lists):
            continue
        keys = key_lists[idx]
        if keys is None or not isinstance(keys, torch.Tensor) or keys.numel() == 0:
            continue
        level = int(level_list[idx]) if idx < len(level_list) else 0
        keys = keys.to(device=device, dtype=torch.int64)
        if merged_keys is None:
            merged_keys = keys
            merged_level = level
            continue
        merged_keys, merged_level = merge_voxel_buffers(
            merged_keys,
            int(merged_level),
            keys,
            level,
            cap=VOXEL_CAP_PER_OBJECT,
        )
    return merged_keys, int(merged_level)


def _recompute_object_gaussian_from_voxels(
    state: Dict[str, Any],
    key_lists: List[torch.Tensor],
    level_list: List[int],
    obj_idx: int,
    *,
    device: torch.device,
) -> bool:
    if obj_idx < 0 or obj_idx >= len(key_lists):
        return False
    stats = _voxel_geometry_stats(
        key_lists[obj_idx],
        int(level_list[obj_idx]) if obj_idx < len(level_list) else 0,
        device=device,
    )
    if stats is None:
        return False
    means = state.get("means")
    cov6 = state.get("cov6")
    if not isinstance(means, torch.Tensor) or not isinstance(cov6, torch.Tensor):
        return False
    if obj_idx >= int(means.shape[0]) or obj_idx >= int(cov6.shape[0]):
        return False
    means[obj_idx] = stats["mean"].to(device=device, dtype=means.dtype)
    cov6[obj_idx] = _pack_cov6(stats["cov"].view(1, 3, 3))[0].to(device=device, dtype=cov6.dtype)
    return True


def _voxel_merge_geometry_allowed(
    key_lists: List[torch.Tensor],
    level_list: List[int],
    group: List[int],
    *,
    device: torch.device,
    max_aabb_growth: float,
    max_cov_eig_growth: float,
    min_keys: int,
) -> bool:
    member_stats: List[Dict[str, torch.Tensor]] = []
    for idx in group:
        if idx < 0 or idx >= len(key_lists):
            continue
        keys = key_lists[idx]
        if keys is None or not isinstance(keys, torch.Tensor) or int(keys.numel()) < int(min_keys):
            continue
        stats = _voxel_geometry_stats(
            keys,
            int(level_list[idx]) if idx < len(level_list) else 0,
            device=device,
        )
        if stats is not None:
            member_stats.append(stats)

    # If voxel evidence is missing or too sparse, do not make a hard decision
    # from this guard. Hellinger + cannot-link still apply.
    if len(member_stats) < 2:
        return True

    merged_keys, merged_level = _merge_voxel_group(key_lists, level_list, group, device=device)
    if merged_keys is None or int(merged_keys.numel()) < int(min_keys):
        return True
    merged_stats = _voxel_geometry_stats(merged_keys, int(merged_level), device=device)
    if merged_stats is None:
        return True

    max_member_diag = torch.stack([s["aabb_diag"] for s in member_stats]).max().clamp(min=1e-6)
    max_member_eig = torch.stack([s["eig_max"] for s in member_stats]).max().clamp(min=float(DEFAULT_FUSED_COV_RIDGE))
    aabb_growth = float((merged_stats["aabb_diag"] / max_member_diag).detach().item())
    eig_growth = float((merged_stats["eig_max"] / max_member_eig).detach().item())
    return aabb_growth <= float(max_aabb_growth) and eig_growth <= float(max_cov_eig_growth)


def _collect_view_positions(
    image_ids: List[int],
    image_positions: object,
    images_meta: object,
) -> tuple[List[int], Optional[torch.Tensor]]:
    valid_ids: List[int] = []
    positions: List[torch.Tensor] = []
    positions_list = image_positions if isinstance(image_positions, list) else None
    images_list = images_meta if isinstance(images_meta, list) else None

    for image_id in image_ids:
        try:
            image_id_int = int(image_id)
        except Exception:
            continue
        pos = None
        if positions_list is not None and 0 <= image_id_int < len(positions_list):
            pos = positions_list[image_id_int]
        if pos is None and images_list is not None and 0 <= image_id_int < len(images_list):
            pose = getattr(images_list[image_id_int], "pose", None)
            if pose is None:
                pos = None
            elif isinstance(pose, torch.Tensor):
                pos = pose[:3, 3]
            else:
                try:
                    pos = torch.as_tensor(pose, dtype=torch.float32)[:3, 3]
                except Exception:
                    pos = None
        if pos is None:
            continue
        if not isinstance(pos, torch.Tensor):
            try:
                pos = torch.as_tensor(pos, dtype=torch.float32)
            except Exception:
                continue
        if pos.numel() != 3:
            continue
        pos = pos.view(-1)[:3]
        if not torch.isfinite(pos).all():
            continue
        positions.append(pos)
        valid_ids.append(image_id_int)

    if not positions:
        return [], None
    return valid_ids, torch.stack(positions, dim=0)


def _update_viewpoint_image_ids(state: Dict[str, Any], object_indices: List[int]) -> None:
    means = state.get("means")
    if means is None or not isinstance(means, torch.Tensor):
        return
    object_image_ids = state.get("object_image_ids", [])
    viewpoint_image_ids = state.get("viewpoint_image_ids", [])
    image_positions = state.get("image_positions")
    images_meta = state.get("images")
    high_quality_views = state.get("high_quality_views", [])
    rgb_observations = state.get("rgb_observations", [])

    device = means.device
    dtype = means.dtype

    for obj_idx in object_indices:
        if obj_idx < 0 or obj_idx >= len(object_image_ids):
            continue
        image_ids = object_image_ids[obj_idx]
        if not image_ids:
            if obj_idx < len(viewpoint_image_ids):
                viewpoint_image_ids[obj_idx] = []
            continue
        valid_ids, positions = _collect_view_positions(image_ids, image_positions, images_meta)
        if positions is None or not valid_ids:
            if obj_idx < len(viewpoint_image_ids):
                viewpoint_image_ids[obj_idx] = []
            continue
        mean_vec = means[obj_idx]
        if torch.isnan(mean_vec).any():
            if obj_idx < len(viewpoint_image_ids):
                viewpoint_image_ids[obj_idx] = []
            continue

        positions = positions.to(device=device, dtype=dtype)
        mean_vec = mean_vec.to(device=device, dtype=dtype)

        diffs = positions - mean_vec.unsqueeze(0)
        distances = torch.linalg.norm(diffs, dim=1)
        sorted_idx = torch.argsort(distances)
        closest_idx = int(sorted_idx[0].item())
        second_closest_idx = int(sorted_idx[1].item()) if sorted_idx.numel() > 1 else closest_idx

        selected_ids: List[int] = [
            int(valid_ids[closest_idx]),
            int(valid_ids[second_closest_idx]),
        ]

        hq_candidates: List[int] = []
        hq_row = (
            high_quality_views[obj_idx]
            if isinstance(high_quality_views, list) and obj_idx < len(high_quality_views)
            else []
        )
        obs_row = (
            rgb_observations[obj_idx] if isinstance(rgb_observations, list) and obj_idx < len(rgb_observations) else []
        )
        if isinstance(hq_row, list) and isinstance(obs_row, list):
            common = min(len(hq_row), len(obs_row))
            for view_idx in range(common):
                if hq_row[view_idx] is None:
                    continue
                obs = obs_row[view_idx]
                if not isinstance(obs, dict):
                    continue
                with contextlib.suppress(Exception):
                    image_id = int(obs.get("image_id"))
                    if image_id >= 0:
                        hq_candidates.append(image_id)
        hq_candidates = list(dict.fromkeys(hq_candidates))

        if len(hq_candidates) >= 2:
            selected_ids.extend(random.sample(hq_candidates, 2))
        elif len(hq_candidates) == 1:
            selected_ids.extend([hq_candidates[0], hq_candidates[0]])
        else:
            selected_ids.extend([selected_ids[0], selected_ids[1]])

        viewpoint_image_ids[obj_idx] = selected_ids


def update_scene_graph_state(
    state: Dict[str, Any],
    mu_d: torch.Tensor,  # (D, 3)
    cov6_d: torch.Tensor,  # (D, 6)
    feat_d: torch.Tensor,  # (D, F)
    captions_d: List[str],  # len D (can be empty strings)
    det_winner_idx: torch.Tensor,  # (D,) long, -1 => no neighbor
    obj_winner_idx: torch.Tensor,  # (N,) long, union-find result for objects
    rgb_observations: Optional[List[Any]] = None,  # Optional per-detection RGB debug info
    detection_image_ids: Optional[List[Optional[int]]] = None,
    *,
    allow_new_objects: bool = True,
    det_points_flat: Optional[torch.Tensor] = None,        # (P, 3) world-frame
    det_points_offsets: Optional[torch.Tensor] = None,     # (D+1,) int64 CSR
    class_ids_d: Optional[torch.Tensor] = None,             # (D,) semantic class ids
    max_merge_distance_m: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Update `state` in-place using detections and winner indices.

    * Existing objects are merged according to obj_winner_idx (union-find).
    * Detections either merge into their winner object or become new objects.
    * All Gaussian stats + features updated in batched GPU ops.
    * Non-winner objects become inactive; id_redirect is updated on CPU.
    Returns a small dict of update metadata (e.g., indices of new objects).
    """

    device = state["means"].device
    N = state["means"].shape[0]
    D = mu_d.shape[0]

    # Keep scene-state numeric tensors in canonical float32 so batched updates
    # and concatenations do not fail under mixed-precision checkpoints/runtimes.
    state["means"] = state["means"].to(device=device, dtype=torch.float32)
    state["cov6"] = state["cov6"].to(device=device, dtype=torch.float32)
    state["features"] = state["features"].to(device=device, dtype=torch.float32)

    mu_d = mu_d.to(device=device, dtype=state["means"].dtype)
    cov6_d = cov6_d.to(device=device, dtype=state["cov6"].dtype)
    feat_d = feat_d.to(device=device, dtype=state["features"].dtype)
    det_winner_idx = det_winner_idx.to(device)
    obj_winner_idx = obj_winner_idx.to(device)
    if class_ids_d is not None and isinstance(class_ids_d, torch.Tensor):
        class_ids_d = class_ids_d.to(device=device, dtype=state["class_ids"].dtype).view(-1)
        if class_ids_d.numel() != D:
            class_ids_d = None

    # NOTE: we deliberately do NOT add an unconditional ridge to detection
    # covs at the function entry. Doing so would compound across batches
    # (the prior batch's ridge gets baked into M2_old and propagates through
    # to the next fusion). Instead, we apply a *conditional* eigenvalue floor
    # only at the points where SPD must hold for downstream consumers:
    #  - the post-fusion cov_t (before storing in state["cov6"])
    #  - new-object covs lifted directly from cov6_d (before append)
    # See ``_symmetrize_and_eigfloor_cov33`` for the math.

    # ---------- 0) Existing sizes ----------
    assert det_winner_idx.shape[0] == D
    assert obj_winner_idx.shape[0] == N

    ensure_covisibility_state(state, needed_objects=N)

    is_locked_list: List[bool] = state.get("is_locked") or []
    while len(is_locked_list) < N:
        is_locked_list.append(False)
    state["is_locked"] = is_locked_list

    def _is_locked(obj_idx: int) -> bool:
        return 0 <= obj_idx < len(is_locked_list) and bool(is_locked_list[obj_idx])

    rgb_obs_state: List[List[Any]] = state.get("rgb_observations", [])
    object_image_ids: List[List[int]] = state.get("object_image_ids", [])
    object_mask_observations: List[List[Dict[str, Any]]] = state.get("object_mask_observations", [])
    view_means_state: List[List[torch.Tensor]] = state.get("view_means", [])
    view_cov6_state: List[List[torch.Tensor]] = state.get("view_cov6", [])
    viewpoint_image_ids: List[List[int]] = state.get("viewpoint_image_ids", [])
    high_quality_captioning_state: List[bool] = state.get("high_quality_captioning", [])
    high_quality_views_state: List[List[torch.Tensor]] = state.get("high_quality_views", [])
    loser_object_ids_state = state.get("loser_object_ids")
    loser_object_ids_need_backfill = False
    if not isinstance(loser_object_ids_state, list):
        loser_object_ids_state = []
        state["loser_object_ids"] = loser_object_ids_state
        loser_object_ids_need_backfill = True
    if not isinstance(high_quality_captioning_state, list):
        high_quality_captioning_state = []
    if not isinstance(high_quality_views_state, list):
        high_quality_views_state = []
    state["rgb_observations"] = rgb_obs_state
    state["object_image_ids"] = object_image_ids
    if not isinstance(object_mask_observations, list):
        object_mask_observations = []
    state["object_mask_observations"] = object_mask_observations
    state["view_means"] = view_means_state
    state["view_cov6"] = view_cov6_state
    state["viewpoint_image_ids"] = viewpoint_image_ids
    state["high_quality_captioning"] = high_quality_captioning_state
    state["high_quality_views"] = high_quality_views_state
    # Keep per-object view buffers bounded and aligned.
    # IMPORTANT: `rgb_observations`, `view_means`, `view_cov6`, and `high_quality_views`
    # must stay index-aligned
    # because captioning selects a `view_idx` from cov6/means and then indexes rgb_observations
    # with the same `view_idx`.
    MAX_VIEWS_PER_OBJECT = 256
    HQ_ANGLE_DEG = 30.0
    HQ_ANGLE_COS_THRESHOLD = math.cos(math.radians(HQ_ANGLE_DEG))
    HQ_REPLACE_DISTANCE_RATIO = 0.75
    EPS = 1e-6

    def _ensure_obs_len(min_len: int) -> None:
        while len(rgb_obs_state) < min_len:
            rgb_obs_state.append([])

    def _ensure_img_len(min_len: int) -> None:
        while len(object_image_ids) < min_len:
            object_image_ids.append([])

    def _ensure_mask_obs_len(min_len: int) -> None:
        while len(object_mask_observations) < min_len:
            object_mask_observations.append([])

    def _append_image_id(obj_idx: int, image_id: Optional[int]) -> None:
        if image_id is None:
            return
        _ensure_img_len(obj_idx + 1)
        if image_id not in object_image_ids[obj_idx]:
            object_image_ids[obj_idx].append(image_id)

    def _ensure_view_len(min_len: int) -> None:
        while len(view_means_state) < min_len:
            view_means_state.append([])
        while len(view_cov6_state) < min_len:
            view_cov6_state.append([])

    def _ensure_hq_len(min_len: int) -> None:
        while len(high_quality_views_state) < min_len:
            high_quality_views_state.append([])
        while len(high_quality_captioning_state) < min_len:
            high_quality_captioning_state.append(False)

    def _align_object_views(obj_idx: int) -> None:
        _ensure_obs_len(obj_idx + 1)
        _ensure_view_len(obj_idx + 1)
        _ensure_hq_len(obj_idx + 1)
        rgb_list = rgb_obs_state[obj_idx]
        mean_list = view_means_state[obj_idx]
        cov_list = view_cov6_state[obj_idx]
        hq_list = high_quality_views_state[obj_idx]
        target_common = min(len(rgb_list), len(mean_list), len(cov_list))
        if len(hq_list) < target_common:
            hq_list = list(hq_list) + [None] * (target_common - len(hq_list))
            high_quality_views_state[obj_idx] = hq_list
        common = min(len(rgb_list), len(mean_list), len(cov_list), len(high_quality_views_state[obj_idx]))
        if len(rgb_list) != common:
            rgb_obs_state[obj_idx] = rgb_list[:common]
        if len(mean_list) != common:
            view_means_state[obj_idx] = mean_list[:common]
        if len(cov_list) != common:
            view_cov6_state[obj_idx] = cov_list[:common]
        if len(high_quality_views_state[obj_idx]) != common:
            high_quality_views_state[obj_idx] = high_quality_views_state[obj_idx][:common]

    def _cap_object_views(obj_idx: int) -> None:
        _align_object_views(obj_idx)
        if len(rgb_obs_state[obj_idx]) <= MAX_VIEWS_PER_OBJECT:
            return
        rgb_obs_state[obj_idx] = rgb_obs_state[obj_idx][-MAX_VIEWS_PER_OBJECT:]
        view_means_state[obj_idx] = view_means_state[obj_idx][-MAX_VIEWS_PER_OBJECT:]
        view_cov6_state[obj_idx] = view_cov6_state[obj_idx][-MAX_VIEWS_PER_OBJECT:]
        high_quality_views_state[obj_idx] = high_quality_views_state[obj_idx][-MAX_VIEWS_PER_OBJECT:]

    def _ensure_viewpoint_len(min_len: int) -> None:
        while len(viewpoint_image_ids) < min_len:
            viewpoint_image_ids.append([])

    def _ensure_loser_len(min_len: int) -> None:
        while len(loser_object_ids_state) < min_len:
            loser_object_ids_state.append(set())

    def _ensure_loser_entry(obj_idx: int) -> set[int]:
        _ensure_loser_len(obj_idx + 1)
        entry = loser_object_ids_state[obj_idx]
        if isinstance(entry, set):
            return entry
        if entry is None:
            loser_object_ids_state[obj_idx] = set()
            return loser_object_ids_state[obj_idx]
        if isinstance(entry, (list, tuple)):
            loser_object_ids_state[obj_idx] = {int(x) for x in entry if x is not None}
            return loser_object_ids_state[obj_idx]
        loser_object_ids_state[obj_idx] = set()
        return loser_object_ids_state[obj_idx]

    def _obs_has_image(obs: Any) -> bool:
        if obs is None:
            return False
        if isinstance(obs, dict):
            return obs.get("image_caption") is not None or obs.get("image") is not None
        return True

    def _to_vec3_cpu(value: Any) -> Optional[torch.Tensor]:
        if value is None:
            return None
        vec = value
        if not isinstance(vec, torch.Tensor):
            with contextlib.suppress(Exception):
                vec = torch.as_tensor(vec, dtype=torch.float32)
            if not isinstance(vec, torch.Tensor):
                return None
        if vec.numel() < 3:
            return None
        vec = vec.detach().to("cpu", dtype=torch.float32, copy=False).view(-1)[:3]
        if not torch.isfinite(vec).all():
            return None
        return vec

    def _object_center_cpu(obj_idx: int) -> Optional[torch.Tensor]:
        if obj_idx < 0 or obj_idx >= state["means"].shape[0]:
            return None
        return _to_vec3_cpu(state["means"][obj_idx])

    image_positions_list = state.get("image_positions", []) if isinstance(state.get("image_positions"), list) else []
    images_meta_list = state.get("images", []) if isinstance(state.get("images"), list) else []
    image_viewpoint_cache: Dict[int, Optional[torch.Tensor]] = {}

    def _resolve_image_viewpoint(image_id: Optional[int]) -> Optional[torch.Tensor]:
        if image_id is None:
            return None
        try:
            image_id_int = int(image_id)
        except Exception:
            return None
        if image_id_int in image_viewpoint_cache:
            return image_viewpoint_cache[image_id_int]

        pos = None
        if 0 <= image_id_int < len(image_positions_list):
            pos = image_positions_list[image_id_int]
        if pos is None and 0 <= image_id_int < len(images_meta_list):
            pose = getattr(images_meta_list[image_id_int], "pose", None)
            if isinstance(pose, torch.Tensor):
                pos = pose[:3, 3]
            elif pose is not None:
                with contextlib.suppress(Exception):
                    pos = torch.as_tensor(pose, dtype=torch.float32)[:3, 3]

        out = _to_vec3_cpu(pos)
        image_viewpoint_cache[image_id_int] = out
        return out

    def _set_high_quality_captioning(obj_idx: int, value: bool) -> None:
        _ensure_hq_len(obj_idx + 1)
        high_quality_captioning_state[obj_idx] = bool(value)

    def _append_or_replace_high_quality_view(
        obj_idx: int,
        *,
        viewpoint_pos: Optional[torch.Tensor],
        obs: Any,
        mu_view: torch.Tensor,
        cov6_view: torch.Tensor,
    ) -> bool:
        if obj_idx < 0:
            return False
        if viewpoint_pos is None or not _obs_has_image(obs):
            return False
        obj_center = _object_center_cpu(obj_idx)
        if obj_center is None:
            return False
        vp_new = _to_vec3_cpu(viewpoint_pos)
        if vp_new is None:
            return False

        vec_new = vp_new - obj_center
        dist_new = float(torch.linalg.norm(vec_new).item())
        if not math.isfinite(dist_new) or dist_new <= EPS:
            return False
        dir_new = vec_new / dist_new

        _ensure_obs_len(obj_idx + 1)
        _ensure_view_len(obj_idx + 1)
        _ensure_hq_len(obj_idx + 1)
        _align_object_views(obj_idx)

        mu_view_cpu = mu_view.detach().to("cpu", dtype=torch.float32)
        cov6_view_cpu = cov6_view.detach().to("cpu", dtype=torch.float32)
        hq_row = high_quality_views_state[obj_idx]
        rgb_row = rgb_obs_state[obj_idx]
        mean_row = view_means_state[obj_idx]
        cov_row = view_cov6_state[obj_idx]

        if not hq_row:
            rgb_row.append(obs)
            mean_row.append(mu_view_cpu)
            cov_row.append(cov6_view_cpu)
            hq_row.append(vp_new)
            _cap_object_views(obj_idx)
            return True

        valid_indices: List[int] = []
        dirs: List[torch.Tensor] = []
        dists: List[float] = []
        for idx, vp_existing in enumerate(hq_row):
            vp_vec = _to_vec3_cpu(vp_existing)
            if vp_vec is None:
                continue
            vec_existing = vp_vec - obj_center
            dist_existing = float(torch.linalg.norm(vec_existing).item())
            if not math.isfinite(dist_existing) or dist_existing <= EPS:
                continue
            valid_indices.append(int(idx))
            dirs.append(vec_existing / dist_existing)
            dists.append(dist_existing)

        if not valid_indices:
            rgb_row.append(obs)
            mean_row.append(mu_view_cpu)
            cov_row.append(cov6_view_cpu)
            hq_row.append(vp_new)
            _cap_object_views(obj_idx)
            return True

        dirs_tensor = torch.stack(dirs, dim=0)  # (V, 3)
        dot_vals = torch.sum(dirs_tensor * dir_new.unsqueeze(0), dim=1)
        best_rel_idx = int(torch.argmax(dot_vals).item())
        best_dot = float(dot_vals[best_rel_idx].item())

        # "Diverse enough": minimum angle > 30deg  <=>  max cosine < cos(30deg)
        if best_dot < HQ_ANGLE_COS_THRESHOLD:
            rgb_row.append(obs)
            mean_row.append(mu_view_cpu)
            cov_row.append(cov6_view_cpu)
            hq_row.append(vp_new)
            _cap_object_views(obj_idx)
            return True

        nearest_idx = int(valid_indices[best_rel_idx])
        nearest_dist = float(dists[best_rel_idx])
        # Similar direction but much closer to the object center -> replace weaker prior view.
        if nearest_dist > EPS and dist_new < (HQ_REPLACE_DISTANCE_RATIO * nearest_dist):
            if nearest_idx < len(rgb_row):
                rgb_row[nearest_idx] = obs
            if nearest_idx < len(mean_row):
                mean_row[nearest_idx] = mu_view_cpu
            if nearest_idx < len(cov_row):
                cov_row[nearest_idx] = cov6_view_cpu
            if nearest_idx < len(hq_row):
                hq_row[nearest_idx] = vp_new
            _cap_object_views(obj_idx)
            return True
        return False

    def _collect_object_hq_records(obj_idx: int) -> List[tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor]]:
        if obj_idx < 0:
            return []
        _ensure_obs_len(obj_idx + 1)
        _ensure_view_len(obj_idx + 1)
        _ensure_hq_len(obj_idx + 1)
        _align_object_views(obj_idx)
        hq_row = high_quality_views_state[obj_idx]
        rgb_row = rgb_obs_state[obj_idx]
        mean_row = view_means_state[obj_idx]
        cov_row = view_cov6_state[obj_idx]
        records: List[tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor]] = []
        common = min(len(hq_row), len(rgb_row), len(mean_row), len(cov_row))
        for idx in range(common):
            vp = _to_vec3_cpu(hq_row[idx])
            if vp is None:
                continue
            records.append((vp, rgb_row[idx], mean_row[idx], cov_row[idx]))
        return records

    def _clear_object_hq_records(obj_idx: int) -> None:
        if obj_idx < 0:
            return
        _ensure_obs_len(obj_idx + 1)
        _ensure_view_len(obj_idx + 1)
        _ensure_hq_len(obj_idx + 1)
        rgb_obs_state[obj_idx] = []
        view_means_state[obj_idx] = []
        view_cov6_state[obj_idx] = []
        high_quality_views_state[obj_idx] = []
        high_quality_captioning_state[obj_idx] = False

    _ensure_obs_len(N)
    _ensure_img_len(N)
    _ensure_mask_obs_len(N)
    _ensure_view_len(N)
    _ensure_viewpoint_len(N)
    _ensure_hq_len(N)
    if len(loser_object_ids_state) < N:
        loser_object_ids_need_backfill = True
    _ensure_loser_len(N)
    new_object_indices: List[int] = []
    viewpoint_update_targets: set[int] = set()

    # ---- Voxel buffer (CSR -> per-object lists for in-place edits) ----
    voxel_key_lists, voxel_level_list = _read_voxel_buffer(state, N, device)
    voxel_points_available = (
        det_points_flat is not None
        and det_points_offsets is not None
        and det_points_flat.numel() > 0
        and det_points_offsets.numel() == D + 1
    )
    if voxel_points_available:
        det_points_flat = det_points_flat.to(device=device)
        det_points_offsets = det_points_offsets.to(device=device, dtype=torch.int64)

    # ---------- 1) Mark non-winner objects inactive + build id_redirect ----------
    # Locked objects cannot be merged (as loser or winner). Force them to stay as their own winner.
    obj_winner_idx = obj_winner_idx.clone()
    for i in range(N):
        if _is_locked(i):
            obj_winner_idx[i] = i
        else:
            winner_idx = int(obj_winner_idx[i].item())
            if winner_idx != i and _is_locked(winner_idx):
                obj_winner_idx[i] = i

    obj_indices = torch.arange(N, device=device)

    # Vectorized "refuse far merges" guard: union-find can chain through a
    # large-cov detection and propose a merge between two objects metres
    # apart. Compute |mu[i] - mu[winner[i]]| in a single GPU op and revert
    # winner = i wherever the proposed merge exceeds the floor.
    n_far_merges_blocked = 0
    max_merge_distance = (
        float(DEFAULT_MAX_MERGE_DISTANCE_M)
        if max_merge_distance_m is None
        else float(max_merge_distance_m)
    )
    if max_merge_distance > 0.0 and math.isfinite(max_merge_distance) and N > 0:
        proposes_merge = obj_winner_idx != obj_indices  # (N,) on device
        if bool(proposes_merge.any().item()):
            mu_self = state["means"]  # (N, 3) — already on `device`
            mu_winner = mu_self[obj_winner_idx]
            merge_dist = (mu_self - mu_winner).norm(dim=1)  # (N,)
            too_far = proposes_merge & (merge_dist > max_merge_distance)
            n_far_merges_blocked = int(too_far.sum().item())
            if n_far_merges_blocked > 0:
                obj_winner_idx = torch.where(too_far, obj_indices, obj_winner_idx)

    # Refuse object merges that violate same-frame cannot-link constraints.
    n_cannot_link_merges_blocked = 0
    blocked_pairs = cannot_link_index_pairs(state, n_objects=N)
    if blocked_pairs and N > 0:
        obj_winner_idx = obj_winner_idx.clone()
        blocked_by_idx: Dict[int, set[int]] = {}
        for a, b in blocked_pairs:
            blocked_by_idx.setdefault(int(a), set()).add(int(b))
            blocked_by_idx.setdefault(int(b), set()).add(int(a))
        groups: Dict[int, List[int]] = {}
        for i in range(N):
            winner_idx = int(obj_winner_idx[i].item())
            if winner_idx < 0 or winner_idx >= N:
                continue
            groups.setdefault(winner_idx, []).append(int(i))
        for winner_idx, group in groups.items():
            if len(group) < 2:
                continue
            group_set = set(group)
            violates = any(bool(blocked_by_idx.get(a, set()) & group_set) for a in group)
            if not violates:
                continue
            for idx in group:
                if idx != winner_idx and int(obj_winner_idx[idx].item()) != idx:
                    obj_winner_idx[idx] = idx
                    n_cannot_link_merges_blocked += 1

    # Refuse merges whose merged voxel support becomes scale-inconsistently
    # large. This is deliberately ratio-based, not an absolute center-distance
    # cutoff, so large objects are not penalized just for being large.
    n_voxel_geometry_merges_blocked = 0
    if DEFAULT_MERGE_VOXEL_GEOMETRY_GUARD and N > 0:
        groups: Dict[int, List[int]] = {}
        for i in range(N):
            winner_idx = int(obj_winner_idx[i].item())
            if winner_idx < 0 or winner_idx >= N:
                continue
            groups.setdefault(winner_idx, []).append(int(i))
        for winner_idx, group in groups.items():
            if len(group) < 2:
                continue
            if _voxel_merge_geometry_allowed(
                voxel_key_lists,
                voxel_level_list,
                group,
                device=device,
                max_aabb_growth=DEFAULT_MERGE_VOXEL_MAX_AABB_GROWTH,
                max_cov_eig_growth=DEFAULT_MERGE_VOXEL_MAX_COV_EIG_GROWTH,
                min_keys=DEFAULT_MERGE_VOXEL_MIN_KEYS_FOR_GUARD,
            ):
                continue
            for idx in group:
                if idx != winner_idx and int(obj_winner_idx[idx].item()) != idx:
                    obj_winner_idx[idx] = idx
                    n_voxel_geometry_merges_blocked += 1

    is_obj_winner = obj_winner_idx == obj_indices

    # Update active flags (GPU)
    state["active"] = state["active"] & is_obj_winner

    # Update id_redirect (CPU, tiny)
    object_id_cpu = state["object_id"].cpu()
    id_redirect = state.get("id_redirect") or {}
    if (
        isinstance(id_redirect, dict)
        and id_redirect
        and (loser_object_ids_need_backfill or not any(bool(entry) for entry in loser_object_ids_state[:N]))
    ):
        # Best-effort backfill for runs where loser_object_ids_state wasn't previously tracked.
        # This keeps alias sets consistent with already-populated id_redirect chains.
        id_to_idx: dict[int, int] = {}
        for idx in range(N):
            with contextlib.suppress(Exception):
                id_to_idx[int(object_id_cpu[idx].item())] = int(idx)

        def _resolve_canonical_id(object_id: int) -> int:
            cur = int(object_id)
            seen: set[int] = set()
            while True:
                if cur in seen:
                    break
                seen.add(cur)
                nxt = id_redirect.get(cur)
                if nxt is None:
                    break
                try:
                    nxt_int = int(nxt)
                except Exception:
                    break
                if nxt_int == cur:
                    break
                cur = nxt_int
            return cur

        for loser_id, winner_id in list(id_redirect.items()):
            with contextlib.suppress(Exception):
                loser_int = int(loser_id)
                winner_int = _resolve_canonical_id(int(winner_id))
                if loser_int == winner_int:
                    continue
                winner_idx = id_to_idx.get(winner_int)
                if winner_idx is None:
                    continue
                _ensure_loser_entry(int(winner_idx)).add(loser_int)
    merged_objects: List[Dict[str, Any]] = []
    voxel_gaussian_recompute_targets: set[int] = set()
    for i in range(N):
        winner_idx = int(obj_winner_idx[i].item())
        if winner_idx == i:
            continue
        if winner_idx < 0 or winner_idx >= N:
            # No winner -> keep object as-is
            obj_winner_idx[i] = i
            continue

        loser_ext = int(object_id_cpu[i].item())
        winner_ext = int(object_id_cpu[winner_idx].item())

        # Collect caption info for logging
        captions = state.get("object_caption") or []
        loser_caption = ""
        winner_caption = ""
        if i < len(captions) and captions[i] is not None:
            loser_caption = str(captions[i]).strip()
        if winner_idx < len(captions) and captions[winner_idx] is not None:
            winner_caption = str(captions[winner_idx]).strip()

        # Collect position info for logging
        means = state.get("means")
        loser_pos = None
        winner_pos = None
        if means is not None and i < means.shape[0]:
            loser_pos = means[i].detach().cpu().tolist()
        if means is not None and winner_idx < means.shape[0]:
            winner_pos = means[winner_idx].detach().cpu().tolist()

        # Merge lightweight support statistics before clearing the loser. The
        # Gaussian itself is recomputed from merged voxels below when possible.
        if (
            isinstance(state.get("count"), torch.Tensor)
            and isinstance(state.get("features"), torch.Tensor)
            and 0 <= i < int(state["count"].shape[0])
            and 0 <= winner_idx < int(state["count"].shape[0])
            and i < int(state["features"].shape[0])
            and winner_idx < int(state["features"].shape[0])
        ):
            with contextlib.suppress(Exception):
                count_w = state["count"][winner_idx].to(dtype=torch.float64)
                count_l = state["count"][i].to(dtype=torch.float64)
                count_total = count_w + count_l
                if bool((count_total > 0).item()):
                    feat_dtype = state["features"].dtype
                    feat_w = state["features"][winner_idx].to(dtype=torch.float64)
                    feat_l = state["features"][i].to(dtype=torch.float64)
                    state["features"][winner_idx] = (
                        (count_w * feat_w + count_l * feat_l) / count_total.clamp(min=1.0)
                    ).to(dtype=feat_dtype)
                    state["count"][winner_idx] = count_total.to(dtype=state["count"].dtype)

        state["active"][i] = False
        merge_covisibility_loser_into_winner(state, i, winner_idx, num_objects=N)

        state["id_redirect"][loser_ext] = winner_ext

        # Log the merge
        merged_objects.append({
            "loser_idx": i,
            "loser_id": loser_ext,
            "loser_caption": loser_caption,
            "loser_pos": loser_pos,
            "winner_idx": winner_idx,
            "winner_id": winner_ext,
            "winner_caption": winner_caption,
            "winner_pos": winner_pos,
        })
        if loser_ext != winner_ext:
            winner_set = _ensure_loser_entry(int(winner_idx))
            winner_set.add(int(loser_ext))
            if i < len(loser_object_ids_state):
                with contextlib.suppress(Exception):
                    winner_set.update(set(_ensure_loser_entry(int(i))))
                with contextlib.suppress(Exception):
                    _ensure_loser_entry(int(i)).clear()
        loser_hq_records = _collect_object_hq_records(i)
        for vp_pos, obs, mu_view, cov6_view in loser_hq_records:
            changed = _append_or_replace_high_quality_view(
                int(winner_idx),
                viewpoint_pos=vp_pos,
                obs=obs,
                mu_view=mu_view,
                cov6_view=cov6_view,
            )
            if changed:
                _set_high_quality_captioning(int(winner_idx), True)
        _clear_object_hq_records(i)

        # Fold the loser's voxel buffer into the winner.
        if i < len(voxel_key_lists) and winner_idx < len(voxel_key_lists):
            merged_keys, merged_level = merge_voxel_buffers(
                voxel_key_lists[winner_idx],
                int(voxel_level_list[winner_idx]),
                voxel_key_lists[i],
                int(voxel_level_list[i]),
                cap=VOXEL_CAP_PER_OBJECT,
            )
            voxel_key_lists[winner_idx] = merged_keys
            voxel_level_list[winner_idx] = int(merged_level)
            voxel_key_lists[i] = torch.empty((0,), dtype=torch.int64, device=device)
            voxel_level_list[i] = 0
            voxel_gaussian_recompute_targets.add(int(winner_idx))
        if i < len(object_image_ids):
            _ensure_img_len(winner_idx + 1)
            winner_images = object_image_ids[winner_idx]
            for image_id in object_image_ids[i]:
                if image_id not in winner_images:
                    winner_images.append(image_id)
            object_image_ids[i] = []
            viewpoint_update_targets.add(int(winner_idx))
        if i < len(object_mask_observations):
            _ensure_mask_obs_len(winner_idx + 1)
            winner_masks = object_mask_observations[winner_idx]
            loser_masks = object_mask_observations[i]
            if isinstance(winner_masks, list) and isinstance(loser_masks, list) and loser_masks:
                winner_masks.extend(loser_masks)
                cap = max(1, int(DEFAULT_MAX_MASK_OBSERVATIONS_PER_OBJECT))
                if len(winner_masks) > cap:
                    del winner_masks[: len(winner_masks) - cap]
            object_mask_observations[i] = []
        if i < len(viewpoint_image_ids):
            viewpoint_image_ids[i] = []
        # Ensure view buffers remain bounded and index-aligned after merges.
        if winner_idx >= 0:
            _ensure_obs_len(winner_idx + 1)
            _ensure_view_len(winner_idx + 1)
            _ensure_hq_len(winner_idx + 1)
            _cap_object_views(winner_idx)

    n_voxel_gaussian_recomputed = 0
    if DEFAULT_RECOMPUTE_MERGED_GAUSSIAN_FROM_VOXELS and voxel_gaussian_recompute_targets:
        for winner_idx in sorted(voxel_gaussian_recompute_targets):
            if _recompute_object_gaussian_from_voxels(
                state,
                voxel_key_lists,
                voxel_level_list,
                int(winner_idx),
                device=device,
            ):
                n_voxel_gaussian_recomputed += 1

    # For attaching detections, always map via the object winners
    # so that detections go straight into canonical objects.
    # det_winner_idx: detection -> object index (possibly stale)
    # Locked objects cannot receive new detections.
    valid_det_mask = det_winner_idx >= 0
    if valid_det_mask.any():
        det_targets_raw = det_winner_idx[valid_det_mask]  # (M,)
        det_targets = obj_winner_idx[det_targets_raw]  # (M,) canonical object indices
        valid_det_indices = torch.nonzero(valid_det_mask, as_tuple=False).view(-1)
        for j in range(valid_det_indices.shape[0]):
            obj_idx = int(det_targets[j].item())
            if _is_locked(obj_idx):
                valid_det_mask[int(valid_det_indices[j].item())] = False
        det_targets_raw = det_winner_idx[valid_det_mask]
        det_targets = obj_winner_idx[det_targets_raw]
    else:
        det_targets = torch.empty(0, dtype=torch.long, device=device)

    # ---------- 2) Batched Gaussian + feature merge for detections with neighbors ----------

    if valid_det_mask.any():
        # The moment-based variance update cov = E[xx^T] - E[x]E[x]^T loses
        # 6+ decimal digits when |mu|^2 dominates the cov scale (HM3D scenes
        # have origin 10-20 m away → mu^2 ~10^2 with cov scale ~10^-3, so the
        # subtraction in fp32 leaves ~1e-5 noise — bigger than any reasonable
        # SPD ridge). Run the entire moment accumulation in fp64; cast the
        # final mu / cov6 back to fp32 for storage.
        F64 = torch.float64
        # Old object stats
        count_old = state["count"].to(device=device, dtype=F64)  # (N,)
        mu_old = state["means"].to(F64)  # (N, 3)
        cov6_old = state["cov6"].to(F64)  # (N, 6)
        feat_old = state["features"]  # (N, F)  — features stay in their native dtype

        cov_old = _unpack_cov6(cov6_old)  # (N, 3, 3)

        # Sufficient stats for old objects:
        # M1_old = w * mu
        # M2_old = w * (Sigma + mu mu^T)
        w_old = count_old  # (N,)
        M1_old = w_old.unsqueeze(-1) * mu_old  # (N, 3)
        mu_outer_old = torch.einsum("ni,nj->nij", mu_old, mu_old)
        M2_old = w_old.view(-1, 1, 1) * (cov_old + mu_outer_old)  # (N, 3, 3)

        # Sufficient stats from detections that have neighbors
        mu_valid = mu_d[valid_det_mask].to(F64)  # (M, 3)
        cov_valid = _unpack_cov6(cov6_d[valid_det_mask].to(F64))  # (M, 3, 3)
        feat_valid = feat_d[valid_det_mask]  # (M, F)

        w_det = torch.ones(mu_valid.shape[0], device=device, dtype=F64)  # (M,)
        M1_det = mu_valid  # (M, 3)   (w=1)
        mu_outer_det = torch.einsum("mi,mj->mij", mu_valid, mu_valid)
        M2_det = cov_valid + mu_outer_det  # (M, 3, 3)

        # Per-object accumulators for detection contributions
        extra_w = torch.zeros(N, device=device, dtype=F64)
        extra_M1 = torch.zeros(N, 3, device=device, dtype=F64)
        extra_M2 = torch.zeros(N, 3, 3, device=device, dtype=F64)
        extra_feat_num = torch.zeros(N, device=device, dtype=F64)
        extra_feat_sum = torch.zeros(N, feat_old.shape[1], device=device, dtype=feat_valid.dtype)

        # Scatter-add detections into their object winners
        extra_w.index_add_(0, det_targets, w_det)
        extra_M1.index_add_(0, det_targets, M1_det)
        extra_M2.index_add_(0, det_targets, M2_det)
        extra_feat_num.index_add_(0, det_targets, torch.ones_like(w_det))
        extra_feat_sum.index_add_(0, det_targets, feat_valid)

        # Combine old + new sufficient stats (fp64)
        W_total = w_old + extra_w  # (N,)
        # Only update objects that got any detections this batch
        touched_mask = extra_w > 0

        if touched_mask.any():
            M1_total = M1_old + extra_M1  # (N, 3)
            M2_total = M2_old + extra_M2  # (N, 3, 3)

            mu_new = mu_old.clone()
            cov_new = cov_old.clone()

            # Only recompute where needed to avoid dividing by zero
            W_t = W_total[touched_mask]  # (T,)
            M1_t = M1_total[touched_mask]  # (T, 3)
            M2_t = M2_total[touched_mask]  # (T, 3, 3)
            touched_indices = torch.nonzero(touched_mask, as_tuple=False).view(-1)

            mu_t = M1_t / W_t.unsqueeze(-1)  # (T, 3) — fp64
            cov_t = M2_t / W_t.view(-1, 1, 1) - torch.einsum("ti,tj->tij", mu_t, mu_t)  # fp64

            # Conditional SPD enforcement: symmetrize and floor λ_min only
            # for cov entries that need it. Well-conditioned covs are passed
            # through unchanged so the ridge doesn't compound across batches.
            if DEFAULT_FUSED_COV_RIDGE > 0.0:
                cov_t = _symmetrize_and_eigfloor_cov33(cov_t, DEFAULT_FUSED_COV_RIDGE)

            # Skip updates that would introduce NaNs; keep old values instead
            valid_updates = ~(torch.isnan(mu_t).any(dim=1) | torch.isnan(cov_t).any(dim=(1, 2)))
            if valid_updates.any():
                valid_indices = touched_indices[valid_updates]
                mu_new[valid_indices] = mu_t[valid_updates]
                cov_new[valid_indices] = cov_t[valid_updates]

            # Update features as weighted average (w_old vs num_detections).
            # Features stay in their native dtype (fp32) — only the moment
            # math needed fp64 for cancellation safety.
            feat_dtype = feat_old.dtype
            feat_new = feat_old.clone()
            feat_num_t = extra_feat_num[touched_mask].unsqueeze(-1).to(feat_dtype)  # (T,1)
            feat_sum_t = extra_feat_sum[touched_mask].to(feat_dtype)  # (T,F)
            w_old_feat = count_old[touched_mask].unsqueeze(-1).to(feat_dtype)  # (T,1)
            feat_old_t = feat_old[touched_mask]  # (T,F)

            # New feature = (w_old * feat_old + sum feat_det) / (w_old + num_det)
            feat_total_num = w_old_feat + feat_num_t  # (T,1)
            feat_total_sum = w_old_feat * feat_old_t + feat_sum_t  # (T,F)
            feat_t = feat_total_sum / feat_total_num.clamp(min=1.0)

            if valid_updates.any():
                feat_new[valid_indices] = feat_t[valid_updates]

            # Push back to state, casting moment buffers back to fp32 to keep
            # the on-disk schema unchanged.
            state["count"] = W_total.to(state["count"].dtype)
            state["means"] = mu_new.to(torch.float32)
            state["cov6"] = _pack_cov6(cov_new).to(torch.float32)
            state["features"] = feat_new

            # Captions: keep most recent per object that got detections.
            for det_idx, obj_idx in zip(
                torch.nonzero(valid_det_mask, as_tuple=False).view(-1).tolist(),
                det_targets.tolist(),
            ):
                obs = None
                if rgb_observations is not None and det_idx < len(rgb_observations):
                    obs = rgb_observations[det_idx]
                image_id = None
                if detection_image_ids is not None and det_idx < len(detection_image_ids):
                    image_id = detection_image_ids[det_idx]
                    _append_image_id(obj_idx, image_id)
                mu_view = mu_d[det_idx].detach().cpu()
                cov6_view = cov6_d[det_idx].detach().cpu()
                viewpoint_pos = _resolve_image_viewpoint(image_id)
                changed = _append_or_replace_high_quality_view(
                    int(obj_idx),
                    viewpoint_pos=viewpoint_pos,
                    obs=obs,
                    mu_view=mu_view,
                    cov6_view=cov6_view,
                )
                if changed:
                    _set_high_quality_captioning(int(obj_idx), True)
                cap_det = captions_d[det_idx]
                if not cap_det or str(cap_det).strip().lower() == "na":
                    continue
                state["object_caption"][obj_idx] = cap_det
            if det_targets.numel() > 0:
                for obj_idx in torch.unique(det_targets).tolist():
                    viewpoint_update_targets.add(int(obj_idx))

            # Preserve a lightweight class id for future correspondence gates.
            # Keep established object labels stable; only fill unknown (-1)
            # slots from matched detections.
            if class_ids_d is not None and class_ids_d.numel() == D:
                class_state = state.get("class_ids")
                if isinstance(class_state, torch.Tensor) and class_state.numel() >= N:
                    det_indices_for_class = torch.nonzero(valid_det_mask, as_tuple=False).view(-1)
                    for det_idx_int, obj_idx_int in zip(det_indices_for_class.tolist(), det_targets.tolist()):
                        if int(obj_idx_int) < 0 or int(obj_idx_int) >= int(class_state.numel()):
                            continue
                        det_class = int(class_ids_d[int(det_idx_int)].item())
                        if det_class < 0:
                            continue
                        try:
                            cur_class = int(class_state[int(obj_idx_int)].item())
                        except Exception:
                            cur_class = -1
                        if cur_class < 0:
                            class_state[int(obj_idx_int)] = det_class

            # Voxel-cloud accumulation: group valid detections by their target
            # object, concat the points, voxelize once per touched object at
            # the object's current level, then merge + promote on cap.
            if voxel_points_available and det_targets.numel() > 0:
                valid_det_indices_list = (
                    torch.nonzero(valid_det_mask, as_tuple=False).view(-1).tolist()
                )
                det_targets_list = det_targets.tolist()
                obj_to_dets: Dict[int, List[int]] = {}
                for det_idx_int, obj_idx_int in zip(valid_det_indices_list, det_targets_list):
                    obj_to_dets.setdefault(int(obj_idx_int), []).append(int(det_idx_int))
                for obj_idx_int, det_idx_list in obj_to_dets.items():
                    if obj_idx_int < 0 or obj_idx_int >= len(voxel_key_lists):
                        continue
                    pts_list = [
                        _slice_det_points(det_points_flat, det_points_offsets, d)
                        for d in det_idx_list
                    ]
                    pts_list = [p for p in pts_list if p.numel() > 0]
                    if not pts_list:
                        continue
                    pts = torch.cat(pts_list, dim=0)
                    _ingest_points_into_object(
                        voxel_key_lists, voxel_level_list, obj_idx_int, pts
                    )

    # ---------- 3) Detections that become brand-new objects ----------
    if allow_new_objects:
        new_mask = det_winner_idx < 0
    else:
        new_mask = det_winner_idx.new_zeros(det_winner_idx.shape, dtype=torch.bool)

    if new_mask.any():
        mu_new = mu_d[new_mask]  # (K, 3)
        cov6_new = cov6_d[new_mask]  # (K, 6)
        # Conditional SPD floor on the brand-new covs (which are lifted
        # directly from the segmenter without going through fusion). This is
        # idempotent: covs already satisfying λ_min ≥ floor pass through.
        if DEFAULT_FUSED_COV_RIDGE > 0.0 and cov6_new.numel() > 0:
            cov6_new = _eigfloor_cov6(cov6_new, DEFAULT_FUSED_COV_RIDGE)
        feat_new = feat_d[new_mask]  # (K, F)
        K_new = mu_new.shape[0]
        new_object_indices = list(range(N, N + K_new))

        # GPU tensors
        count_new = torch.ones(K_new, dtype=state["count"].dtype, device=device)
        active_new = torch.ones(K_new, dtype=torch.bool, device=device)
        class_ids_new = torch.full((K_new,), -1, dtype=state["class_ids"].dtype, device=device)
        if class_ids_d is not None and class_ids_d.numel() == D:
            new_det_indices_for_class = torch.nonzero(new_mask, as_tuple=False).view(-1)
            if new_det_indices_for_class.numel() == K_new:
                class_ids_new = class_ids_d[new_det_indices_for_class].to(device=device, dtype=state["class_ids"].dtype)

        # CPU side: object_id + captions
        if state["object_id"].numel() == 0:
            next_id_start = 0
        else:
            next_id_start = int(state["object_id"].max().item()) + 1
        new_object_ids = torch.arange(next_id_start, next_id_start + K_new, dtype=state["object_id"].dtype)

        # Append tensors
        state["count"] = torch.cat([state["count"].to(device), count_new], dim=0)
        state["means"] = torch.cat([state["means"], mu_new], dim=0)
        state["cov6"] = torch.cat([state["cov6"], cov6_new], dim=0)
        state["features"] = torch.cat([state["features"], feat_new], dim=0)
        state["active"] = torch.cat([state["active"], active_new], dim=0)
        state["class_ids"] = torch.cat([state["class_ids"], class_ids_new], dim=0)
        state["object_id"] = torch.cat([state["object_id"], new_object_ids.cpu()], dim=0)
        _ensure_obs_len(N + K_new)
        _ensure_view_len(N + K_new)
        _ensure_viewpoint_len(N + K_new)
        _ensure_hq_len(N + K_new)
        _ensure_loser_len(N + K_new)
        _ensure_mask_obs_len(N + K_new)
        is_locked_list.extend([False] * K_new)

        # Append captions
        new_caps = []
        for i in torch.nonzero(new_mask, as_tuple=False).view(-1).tolist():
            cap = captions_d[i]
            if not cap or str(cap).strip().lower() == "na":
                cap = ""
            new_caps.append(cap)
        state["object_caption"].extend(new_caps)
        for key, fill_value in (
            ("object_caption_decision", ""),
            ("object_category", ""),
            ("object_supercategory", ""),
            ("object_category_candidates", []),
            ("object_key_attributes", []),
        ):
            rows = state.get(key)
            if not isinstance(rows, list):
                rows = []
            while len(rows) < N:
                rows.append([] if isinstance(fill_value, list) else fill_value)
            for _ in range(K_new):
                rows.append([] if isinstance(fill_value, list) else fill_value)
            state[key] = rows

        new_det_indices = torch.nonzero(new_mask, as_tuple=False).view(-1).tolist()
        for offset in range(K_new):
            target_idx = N + offset
            state["rgb_observations"][target_idx] = []
            view_means_state[target_idx] = []
            view_cov6_state[target_idx] = []
            object_mask_observations[target_idx] = []
            high_quality_views_state[target_idx] = []
            high_quality_captioning_state[target_idx] = False

        _ensure_img_len(N + K_new)
        for offset, det_idx in enumerate(new_det_indices):
            target_idx = N + offset
            image_id = None
            if detection_image_ids is not None and det_idx < len(detection_image_ids):
                image_id = detection_image_ids[det_idx]
            if image_id is None:
                object_image_ids[target_idx] = []
            else:
                object_image_ids[target_idx] = [image_id]
            obs = None
            if rgb_observations is not None and det_idx < len(rgb_observations):
                obs = rgb_observations[det_idx]
            mu_view = mu_d[det_idx].detach().cpu()
            cov6_view = cov6_d[det_idx].detach().cpu()
            viewpoint_pos = _resolve_image_viewpoint(image_id)
            changed = _append_or_replace_high_quality_view(
                int(target_idx),
                viewpoint_pos=viewpoint_pos,
                obs=obs,
                mu_view=mu_view,
                cov6_view=cov6_view,
            )
            if changed:
                _set_high_quality_captioning(int(target_idx), True)

        for offset in range(len(new_det_indices), K_new):
            target_idx = N + offset
            object_image_ids[target_idx] = []

        # Corresponding viewpoint update
        for new_idx in new_object_indices:
            viewpoint_update_targets.add(int(new_idx))

        # Allocate voxel buffers for the K_new newly-introduced objects and
        # ingest the originating detection's points (if available). For new
        # objects the voxel level is chosen from the first detection's spatial
        # extent (init_voxel_level) to leave headroom for later views.
        while len(voxel_key_lists) < N + K_new:
            voxel_key_lists.append(torch.empty((0,), dtype=torch.int64, device=device))
        while len(voxel_level_list) < N + K_new:
            voxel_level_list.append(0)
        if voxel_points_available:
            for offset, det_idx in enumerate(new_det_indices):
                target_idx = N + offset
                pts = _slice_det_points(det_points_flat, det_points_offsets, det_idx)
                if pts.numel() == 0:
                    continue
                _ingest_points_into_object(
                    voxel_key_lists, voxel_level_list, target_idx, pts
                )

    # Done: state is updated in-place
    if viewpoint_update_targets:
        _update_viewpoint_image_ids(state, sorted(viewpoint_update_targets))
    update_covisibility_active_bitset(state, num_objects=state["means"].shape[0])

    # Reflatten the per-object voxel buffer back into CSR storage.
    n_total_objects = state["means"].shape[0]
    while len(voxel_key_lists) < n_total_objects:
        voxel_key_lists.append(torch.empty((0,), dtype=torch.int64, device=device))
    while len(voxel_level_list) < n_total_objects:
        voxel_level_list.append(0)
    _write_voxel_buffer(
        state,
        voxel_key_lists[:n_total_objects],
        voxel_level_list[:n_total_objects],
        device,
    )

    # Log merged objects (objects marked inactive due to merging)
    if merged_objects:
        for merge_info in merged_objects:
            logger.info(
                "Object marked INACTIVE due to MERGE: "
                f"loser_idx={merge_info['loser_idx']} loser_id={merge_info['loser_id']} "
                f"loser_caption={repr(merge_info['loser_caption'])} "
                f"loser_pos={merge_info['loser_pos']} -> "
                f"winner_idx={merge_info['winner_idx']} winner_id={merge_info['winner_id']} "
                f"winner_caption={repr(merge_info['winner_caption'])} "
                f"winner_pos={merge_info['winner_pos']}"
            )

    return {
        "new_object_indices": new_object_indices,
        "merged_objects": merged_objects,
        # How many union-find merges this batch were blocked because the
        # loser/winner Gaussian centers were further apart than
        # DEFAULT_MAX_MERGE_DISTANCE_M. Sustained non-zero values usually
        # indicate the Hellinger threshold is too lax for the scene scale.
        "far_merges_blocked": int(n_far_merges_blocked),
        "cannot_link_merges_blocked": int(n_cannot_link_merges_blocked),
        "voxel_geometry_merges_blocked": int(n_voxel_geometry_merges_blocked),
        "voxel_gaussian_recomputed": int(n_voxel_gaussian_recomputed),
    }
