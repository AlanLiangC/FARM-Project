"""SE(3) pose math and segmentation-to-world transforms."""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch

# ---- Voxel-hash conventions (kept identical across all callers) ----
# Per-axis bits; with 21 bits signed we cover ±2^20 voxels = ±~5 km at base_v=5 mm.
VOXEL_BITS_PER_AXIS = 21
VOXEL_AXIS_MASK = (1 << VOXEL_BITS_PER_AXIS) - 1
VOXEL_AXIS_BIAS = 1 << (VOXEL_BITS_PER_AXIS - 1)
VOXEL_INVALID_KEY = torch.iinfo(torch.int64).min  # sentinel for "no voxel"
VOXEL_BASE_V = 0.005  # meters; level 0 grid spacing
VOXEL_MAX_LEVEL = 15


def invert_pose(pose: torch.Tensor) -> torch.Tensor:
    """Return the SE(3) inverse of a 4x4 pose matrix."""
    R = pose[:3, :3]
    t = pose[:3, 3]
    R_inv = R.transpose(0, 1)
    t_inv = -R_inv @ t
    eye = torch.eye(4, device=pose.device, dtype=pose.dtype)
    eye[:3, :3] = R_inv
    eye[:3, 3] = t_inv
    return eye


def relative_pose(origin: torch.Tensor | None, pose: torch.Tensor) -> torch.Tensor:
    """Express ``pose`` in the coordinate frame defined by ``origin``."""
    if origin is None:
        return pose.clone()
    origin_inv = invert_pose(origin)
    return origin_inv @ pose


def cov6_to_matrix(cov6: torch.Tensor) -> torch.Tensor:
    """Convert packed 6D covariance to (N,3,3) matrices."""
    S = torch.zeros(cov6.shape[0], 3, 3, device=cov6.device, dtype=cov6.dtype)
    S[:, 0, 0] = cov6[:, 0]
    S[:, 0, 1] = S[:, 1, 0] = cov6[:, 1]
    S[:, 0, 2] = S[:, 2, 0] = cov6[:, 2]
    S[:, 1, 1] = cov6[:, 3]
    S[:, 1, 2] = S[:, 2, 1] = cov6[:, 4]
    S[:, 2, 2] = cov6[:, 5]
    return S


def matrix_to_cov6(cov: torch.Tensor) -> torch.Tensor:
    """Pack (N,3,3) covariance matrices back into 6D representation."""
    xx = cov[:, 0, 0]
    xy = cov[:, 0, 1]
    xz = cov[:, 0, 2]
    yy = cov[:, 1, 1]
    yz = cov[:, 1, 2]
    zz = cov[:, 2, 2]
    return torch.stack([xx, xy, xz, yy, yz, zz], dim=-1)


def transform_segmentation_to_world(
    seg_outputs: dict[str, Any],
    poses_world: Sequence[torch.Tensor],
) -> None:
    """Convert detection means/covariances from camera coordinates into the shared world frame."""
    means = seg_outputs.get("means")
    cov6 = seg_outputs.get("cov6")
    batch_ids = seg_outputs.get("batch_ids")
    if means is None or cov6 is None or batch_ids is None:
        return
    if means.numel() == 0 or len(poses_world) == 0:
        return

    device = means.device
    dtype = means.dtype
    batch_ids_long = batch_ids.long()

    means_world = means.clone()
    cov6_world = cov6.clone()

    # Optional per-detection point cloud (camera frame). Transformed in lockstep
    # with means/cov6 so downstream voxelization sees world-frame points.
    det_points_flat = seg_outputs.get("det_points_flat")
    det_points_offsets = seg_outputs.get("det_points_offsets")
    has_points = (
        isinstance(det_points_flat, torch.Tensor)
        and isinstance(det_points_offsets, torch.Tensor)
        and det_points_flat.numel() > 0
        and det_points_offsets.numel() == batch_ids_long.numel() + 1
    )
    if has_points:
        # batch_id per point: repeat each detection's batch_id by its run length.
        counts = det_points_offsets[1:] - det_points_offsets[:-1]
        batch_ids_per_point = torch.repeat_interleave(batch_ids_long, counts.long())
        points_world = det_points_flat.clone()
    else:
        batch_ids_per_point = None
        points_world = None

    unique_batches = torch.unique(batch_ids_long).tolist()
    for batch_idx in unique_batches:
        if batch_idx < 0 or batch_idx >= len(poses_world):
            continue
        mask = batch_ids_long == batch_idx
        if not torch.any(mask):
            continue

        pose = poses_world[batch_idx].to(device=device, dtype=dtype)
        R = pose[:3, :3]
        t = pose[:3, 3]

        means_subset = means[mask]
        if means_subset.numel() > 0:
            means_world[mask] = means_subset @ R.transpose(0, 1) + t

        cov_subset = cov6[mask]
        if cov_subset.numel() > 0:
            cov_mats = cov6_to_matrix(cov_subset)
            rot = R.unsqueeze(0).expand(cov_mats.shape[0], -1, -1)
            cov_world = rot @ cov_mats @ rot.transpose(1, 2)
            cov6_world[mask] = matrix_to_cov6(cov_world)

        if has_points and points_world is not None and batch_ids_per_point is not None:
            point_mask = batch_ids_per_point == batch_idx
            if torch.any(point_mask):
                pts = det_points_flat[point_mask]
                points_world[point_mask] = pts.to(dtype=dtype) @ R.transpose(0, 1) + t

    seg_outputs["means"] = means_world
    seg_outputs["cov6"] = cov6_world
    if has_points and points_world is not None:
        seg_outputs["det_points_flat"] = points_world


# =====================================================================
# Voxel-hash helpers
# =====================================================================
#
# Per-object sparse point cloud is stored as int64 bit-packed voxel keys.
# Each object owns its own voxel level L (effective spacing = base_v * 2**L);
# levels are monotonically non-decreasing — once we coarsen, we stay coarse.
# The keys themselves are always in the **object's current level grid**, so
# packing/unpacking does not require a separate level lookup at decode time
# (the level just tells you the world-space spacing).


def pack_voxel_keys(qxyz: torch.Tensor) -> torch.Tensor:
    """Bit-pack signed integer voxel indices ``(N, 3)`` into ``(N,) int64`` keys.

    21 bits per axis with a sign-bias offset, so packed keys remain comparable
    via int64 ordering / unique. Caller is responsible for ensuring the values
    fit in ±2^20 along each axis (~5 km at the finest level — never an issue
    for indoor scenes).
    """
    if qxyz.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=qxyz.device)
    q = qxyz.to(torch.int64)
    qx = (q[..., 0] + VOXEL_AXIS_BIAS) & VOXEL_AXIS_MASK
    qy = (q[..., 1] + VOXEL_AXIS_BIAS) & VOXEL_AXIS_MASK
    qz = (q[..., 2] + VOXEL_AXIS_BIAS) & VOXEL_AXIS_MASK
    return (qx << (2 * VOXEL_BITS_PER_AXIS)) | (qy << VOXEL_BITS_PER_AXIS) | qz


def unpack_voxel_keys(keys: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_voxel_keys`. Returns ``(N, 3) int64``."""
    if keys.numel() == 0:
        return torch.empty((0, 3), dtype=torch.int64, device=keys.device)
    qz = (keys & VOXEL_AXIS_MASK) - VOXEL_AXIS_BIAS
    qy = ((keys >> VOXEL_BITS_PER_AXIS) & VOXEL_AXIS_MASK) - VOXEL_AXIS_BIAS
    qx = ((keys >> (2 * VOXEL_BITS_PER_AXIS)) & VOXEL_AXIS_MASK) - VOXEL_AXIS_BIAS
    return torch.stack([qx, qy, qz], dim=-1)


def voxelize_points(
    points: torch.Tensor,
    level: int,
    *,
    base_v: float = VOXEL_BASE_V,
    dedup: bool = True,
) -> torch.Tensor:
    """Quantize world-frame ``(N, 3)`` points into packed voxel keys at ``level``.

    Returns int64 keys, optionally deduplicated. Points containing NaN / Inf
    are dropped before quantization.
    """
    if points.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=points.device)
    finite = torch.isfinite(points).all(dim=-1)
    if not torch.any(finite):
        return torch.empty((0,), dtype=torch.int64, device=points.device)
    pts = points[finite] if not finite.all() else points
    v = float(base_v) * float(1 << int(level))
    qxyz = torch.floor(pts / v).to(torch.int64)
    keys = pack_voxel_keys(qxyz)
    if dedup and keys.numel() > 0:
        keys = torch.unique(keys)
    return keys


def promote_voxel_keys(keys: torch.Tensor, *, levels: int = 1) -> torch.Tensor:
    """Coarsen packed keys by ``levels`` (each level halves spacing per axis).

    Decode → arithmetic shift right → repack → unique. Single GPU pass.
    ``levels`` defaults to 1 (one promotion); for multi-step promote pass a
    larger value to skip intermediates.
    """
    if levels <= 0 or keys.numel() == 0:
        return keys
    qxyz = unpack_voxel_keys(keys)
    qxyz = qxyz >> int(levels)  # arithmetic shift on int64 preserves sign
    new_keys = pack_voxel_keys(qxyz)
    return torch.unique(new_keys)


def init_voxel_level(
    points: torch.Tensor,
    *,
    cap: int,
    k_init: int = 32,
    base_v: float = VOXEL_BASE_V,
    max_level: int = VOXEL_MAX_LEVEL,
) -> int:
    """Pick the starting voxel level for a brand-new object.

    First finds the smallest ``L`` such that the longest axis of the first
    detection's AABB spans roughly ``k_init`` voxels (so future views have
    headroom before tripping the cap). Then bumps ``L`` further if the first
    detection alone would exceed ``cap`` unique voxels at that resolution.
    """
    if points.numel() == 0:
        return 0
    finite = torch.isfinite(points).all(dim=-1)
    pts = points[finite]
    if pts.numel() == 0:
        return 0
    ext = pts.amax(dim=0) - pts.amin(dim=0)
    longest = float(ext.amax().item())
    if not math.isfinite(longest) or longest <= 0.0:
        return 0
    # k_init voxels along the longest axis -> v_target = longest / k_init.
    v_target = longest / max(1, int(k_init))
    L = max(0, round(math.log2(v_target / base_v)))
    L = min(int(L), int(max_level))

    # If the first detection alone overflows cap at this level, coarsen.
    while L < int(max_level):
        keys = voxelize_points(pts, L, base_v=base_v, dedup=True)
        if keys.numel() <= int(cap):
            return L
        L += 1
    return L


def merge_voxel_buffers(
    keys_a: torch.Tensor,
    level_a: int,
    keys_b: torch.Tensor,
    level_b: int,
    *,
    cap: int,
    max_level: int = VOXEL_MAX_LEVEL,
) -> Tuple[torch.Tensor, int]:
    """Merge two voxel buffers, promoting both to ``max(level_a, level_b)``.

    Returns ``(merged_keys, merged_level)``. If the merged buffer exceeds the
    cap, promotes further until it fits or ``max_level`` is reached.
    """
    target_level = max(int(level_a), int(level_b))
    a = promote_voxel_keys(keys_a, levels=target_level - int(level_a))
    b = promote_voxel_keys(keys_b, levels=target_level - int(level_b))
    if a.numel() == 0 and b.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=keys_a.device), target_level
    cat = torch.cat([a, b]) if a.numel() and b.numel() else (a if a.numel() else b)
    keys = torch.unique(cat)
    while keys.numel() > int(cap) and target_level < int(max_level):
        keys = promote_voxel_keys(keys, levels=1)
        target_level += 1
    return keys, target_level


def voxel_keys_to_world(
    keys: torch.Tensor,
    level: int,
    *,
    base_v: float = VOXEL_BASE_V,
) -> torch.Tensor:
    """Decode packed keys to world-frame voxel-center coordinates ``(N, 3) float``."""
    if keys.numel() == 0:
        return torch.empty((0, 3), dtype=torch.float32, device=keys.device)
    qxyz = unpack_voxel_keys(keys).to(torch.float32)
    v = float(base_v) * float(1 << int(level))
    return qxyz * v + (v * 0.5)


def decode_voxel_keys_numpy(
    keys: np.ndarray,
    level: int,
    *,
    base_v: float = VOXEL_BASE_V,
) -> np.ndarray:
    """Numpy counterpart of :func:`voxel_keys_to_world`, for CPU-side callers
    (e.g. viser visualization) that don't otherwise need a torch round-trip."""
    keys = np.asarray(keys, dtype=np.int64).reshape(-1)
    if keys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    qz = (keys & VOXEL_AXIS_MASK) - VOXEL_AXIS_BIAS
    qy = ((keys >> VOXEL_BITS_PER_AXIS) & VOXEL_AXIS_MASK) - VOXEL_AXIS_BIAS
    qx = ((keys >> (2 * VOXEL_BITS_PER_AXIS)) & VOXEL_AXIS_MASK) - VOXEL_AXIS_BIAS
    v = float(base_v) * float(1 << int(level))
    pts = np.stack([qx, qy, qz], axis=-1).astype(np.float64) * v + (0.5 * v)
    return pts.astype(np.float32)


def voxel_cloud_aabb(
    voxel_keys: np.ndarray,
    level: int,
    *,
    base_v: float = VOXEL_BASE_V,
    sor_k: int = 0,
    sor_alpha: float = 2.0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Tight AABB from a sparse per-object voxel cloud.

    Decodes the bit-packed int64 keys back to world-frame voxel-center
    coordinates and returns ``(min, max)`` along each axis. ``level`` is the
    object's voxel level (effective spacing = ``base_v * 2**level``). When
    ``sor_k > 0``, removes outlier voxels whose mean distance to the ``sor_k``
    nearest neighbors exceeds ``mean + sor_alpha * std`` (cheap statistical
    outlier removal). Returns ``None`` if the buffer is empty.
    """
    pts = decode_voxel_keys_numpy(voxel_keys, level, base_v=base_v)
    if pts.shape[0] == 0:
        return None

    if sor_k > 0 and pts.shape[0] > sor_k + 1:
        # Cheap k-NN distance via brute force; voxel clouds are tiny (<=1k pts).
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        # k+1 to skip the self-distance at index 0.
        part = np.partition(d2, sor_k, axis=1)[:, : sor_k + 1]
        mean_d = np.sqrt(np.maximum(part[:, 1:], 0.0)).mean(axis=1)
        thresh = float(mean_d.mean() + sor_alpha * mean_d.std())
        keep_mask = mean_d <= thresh
        if keep_mask.sum() >= 4:
            pts = pts[keep_mask]

    v = float(base_v) * float(1 << int(level))
    bbox_min = pts.min(axis=0).astype(np.float32)
    bbox_max = pts.max(axis=0).astype(np.float32)
    # Pad by half a voxel so the box covers the *outer* edge of edge voxels.
    half = np.float32(v * 0.5)
    return bbox_min - half, bbox_max + half
