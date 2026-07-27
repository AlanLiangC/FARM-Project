"""Predicted-object → GT-instance matching for ReferIt3D evaluation.

We extract an axis-aligned bounding box from each predicted object's stored
3D Gaussian (``mean`` + packed ``cov6``), then compute 3D IoU against the GT
instance bbox loaded from :mod:`scene_graph.eval.referit3d.scannet_gt`.

The cov6 packing convention used elsewhere in the repo is upper-triangular,
row-major: ``[xx, xy, xz, yy, yz, zz]`` (see
``scene_graph.utils.geometry.cov6_to_matrix``). Marginal variances along world
axes are therefore ``[cov6[0], cov6[3], cov6[5]]``, so the k-σ AABB of the
ellipsoid is simply ``mean ± k_sigma · sqrt(diag)`` — no eigendecomposition
needed (the projection of a 3D Gaussian onto axis i has variance C_ii).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .scannet_gt import GTInstance


def gaussian_aabb(
    mean: np.ndarray,
    cov6: np.ndarray,
    *,
    k_sigma: float = 2.5,
    floor: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box of the k-σ ellipsoid of a 3D Gaussian.

    ``mean`` is shape (3,); ``cov6`` is shape (6,) packed [xx, xy, xz, yy, yz, zz].
    ``floor`` clamps the diagonal variance from below so degenerate (single-
    observation) Gaussians still have a non-zero box. Returns (mins, maxs).
    """
    mean = np.asarray(mean, dtype=np.float64).reshape(3)
    cov6 = np.asarray(cov6, dtype=np.float64).reshape(6)
    diag = np.array([cov6[0], cov6[3], cov6[5]], dtype=np.float64)
    diag = np.clip(diag, floor, None)
    half = k_sigma * np.sqrt(diag)
    return (mean - half).astype(np.float32), (mean + half).astype(np.float32)


def voxel_cloud_aabb(
    voxel_keys: np.ndarray,
    level: int,
    *,
    base_v: float = 0.005,
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

    Bit-pack scheme matches :func:`scene_graph.utils.geometry.pack_voxel_keys`:
    21 bits per axis with sign-bias offset.
    """
    keys = np.asarray(voxel_keys, dtype=np.int64).reshape(-1)
    if keys.size == 0:
        return None

    BITS = 21
    AXIS_MASK = (1 << BITS) - 1
    AXIS_BIAS = 1 << (BITS - 1)

    qz = (keys & AXIS_MASK) - AXIS_BIAS
    qy = ((keys >> BITS) & AXIS_MASK) - AXIS_BIAS
    qx = ((keys >> (2 * BITS)) & AXIS_MASK) - AXIS_BIAS

    v = float(base_v) * float(1 << int(level))
    pts = (np.stack([qx, qy, qz], axis=-1).astype(np.float64) * v) + (v * 0.5)

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

    bbox_min = pts.min(axis=0).astype(np.float32)
    bbox_max = pts.max(axis=0).astype(np.float32)
    # Pad by half a voxel so the box covers the *outer* edge of edge voxels.
    half = np.float32(v * 0.5)
    return bbox_min - half, bbox_max + half


def aabb_volume(mins: np.ndarray, maxs: np.ndarray) -> float:
    extent = np.clip(maxs - mins, 0.0, None)
    return float(extent[0] * extent[1] * extent[2])


def iou_3d_aabb(
    a_min: np.ndarray,
    a_max: np.ndarray,
    b_min: np.ndarray,
    b_max: np.ndarray,
) -> float:
    """Axis-aligned 3D IoU. Returns 0 for degenerate / disjoint boxes."""
    a_min = np.asarray(a_min, dtype=np.float64)
    a_max = np.asarray(a_max, dtype=np.float64)
    b_min = np.asarray(b_min, dtype=np.float64)
    b_max = np.asarray(b_max, dtype=np.float64)
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter_extent = np.clip(inter_max - inter_min, 0.0, None)
    inter_vol = float(inter_extent[0] * inter_extent[1] * inter_extent[2])
    if inter_vol == 0.0:
        return 0.0
    union_vol = aabb_volume(a_min, a_max) + aabb_volume(b_min, b_max) - inter_vol
    if union_vol <= 0.0:
        return 0.0
    return inter_vol / union_vol


@dataclass(frozen=True)
class PredictedObject:
    """Minimal record we need from a ranked retrieval result."""

    object_id: int
    score: float
    bbox_min: np.ndarray  # (3,)
    bbox_max: np.ndarray  # (3,)
    label: Optional[str] = None
    caption: Optional[str] = None
    region_label: Optional[str] = None


@dataclass(frozen=True)
class MatchResult:
    """Per-utterance evaluation outcome."""

    target_iou_per_rank: List[float]                 # len(ranked)
    distractor_iou_per_rank: List[List[float]]       # len(ranked) × len(distractors)
    first_hit_rank_at: dict                          # {0.25: int|None, 0.5: int|None}
    top1_target_iou: float
    top1_max_distractor_iou: float

    def hit_at_k(self, k: int, threshold: float) -> bool:
        for r, iou in enumerate(self.target_iou_per_rank[:k]):
            if iou >= threshold:
                return True
        return False

    def reciprocal_rank(self, threshold: float = 0.25) -> float:
        for r, iou in enumerate(self.target_iou_per_rank, start=1):
            if iou >= threshold:
                return 1.0 / r
        return 0.0

    def first_correct_rank(self, threshold: float = 0.25) -> Optional[int]:
        for r, iou in enumerate(self.target_iou_per_rank, start=1):
            if iou >= threshold:
                return r
        return None


def match_predictions_to_target(
    ranked: Sequence[PredictedObject],
    gt_target: GTInstance,
    gt_distractors: Iterable[GTInstance] = (),
    *,
    iou_thresholds: Tuple[float, ...] = (0.25, 0.5),
) -> MatchResult:
    """Score a ranked prediction list against one GT target + its distractors."""
    distractors = list(gt_distractors)
    target_iou_per_rank: List[float] = []
    distractor_iou_per_rank: List[List[float]] = []

    for pred in ranked:
        ti = iou_3d_aabb(pred.bbox_min, pred.bbox_max, gt_target.bbox_min, gt_target.bbox_max)
        target_iou_per_rank.append(ti)
        distractor_iou_per_rank.append(
            [iou_3d_aabb(pred.bbox_min, pred.bbox_max, d.bbox_min, d.bbox_max) for d in distractors]
        )

    first_hit_rank_at: dict = {}
    for thr in iou_thresholds:
        hit_rank: Optional[int] = None
        for r, iou in enumerate(target_iou_per_rank, start=1):
            if iou >= thr:
                hit_rank = r
                break
        first_hit_rank_at[float(thr)] = hit_rank

    top1_target_iou = target_iou_per_rank[0] if target_iou_per_rank else 0.0
    top1_max_distractor_iou = (
        max(distractor_iou_per_rank[0]) if distractor_iou_per_rank and distractor_iou_per_rank[0] else 0.0
    )

    return MatchResult(
        target_iou_per_rank=target_iou_per_rank,
        distractor_iou_per_rank=distractor_iou_per_rank,
        first_hit_rank_at=first_hit_rank_at,
        top1_target_iou=top1_target_iou,
        top1_max_distractor_iou=top1_max_distractor_iou,
    )
