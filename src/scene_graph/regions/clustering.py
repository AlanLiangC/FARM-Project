"""Partition active scene-graph objects into spatial regions.

Uses union-find over covisibility + Euclidean distance edges to produce
non-overlapping clusters suitable for room/region labeling.

Post-processes with a max-diameter constraint to split oversized clusters
that arise from single-linkage transitivity (e.g., a camera path chaining
objects through doorways into one mega-cluster).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

BITS_PER_BLOCK = 64


def _check_covisible(adj_u64: torch.Tensor, i: int, j: int) -> bool:
    block = j // BITS_PER_BLOCK
    bit = j % BITS_PER_BLOCK
    if i >= adj_u64.shape[0] or block >= adj_u64.shape[1]:
        return False
    return bool(int(adj_u64[i, block].item()) & (1 << bit))


class _UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _cluster_diameter(means_np: np.ndarray, indices: List[int]) -> float:
    """Max pairwise distance within a cluster (the diameter)."""
    if len(indices) <= 1:
        return 0.0
    pts = means_np[indices]
    # Use max distance from centroid * 2 as fast approximation for large clusters
    if len(indices) > 50:
        centroid = pts.mean(axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        return float(dists.max()) * 2.0
    # Exact for small clusters
    from scipy.spatial.distance import pdist
    return float(pdist(pts).max()) if len(indices) > 1 else 0.0


def _split_cluster_bisection(means_np: np.ndarray, indices: List[int]) -> tuple[List[int], List[int]]:
    """Split a cluster into two halves using k-means bisection on positions."""
    pts = means_np[indices]
    n = len(indices)

    # Simple 2-means: initialize with the two farthest points from centroid
    centroid = pts.mean(axis=0)
    dists_to_center = np.linalg.norm(pts - centroid, axis=1)
    i0 = int(np.argmax(dists_to_center))
    dists_to_i0 = np.linalg.norm(pts - pts[i0], axis=1)
    i1 = int(np.argmax(dists_to_i0))

    c0, c1 = pts[i0].copy(), pts[i1].copy()
    for _ in range(20):
        d0 = np.linalg.norm(pts - c0, axis=1)
        d1 = np.linalg.norm(pts - c1, axis=1)
        assignments = (d1 < d0).astype(int)  # 0 = closer to c0, 1 = closer to c1
        mask0 = assignments == 0
        mask1 = assignments == 1
        if not mask0.any() or not mask1.any():
            # Degenerate — just split in half by index
            mid = n // 2
            return [indices[i] for i in range(mid)], [indices[i] for i in range(mid, n)]
        new_c0 = pts[mask0].mean(axis=0)
        new_c1 = pts[mask1].mean(axis=0)
        if np.allclose(new_c0, c0) and np.allclose(new_c1, c1):
            break
        c0, c1 = new_c0, new_c1

    group_a = [indices[i] for i in range(n) if assignments[i] == 0]
    group_b = [indices[i] for i in range(n) if assignments[i] == 1]
    return group_a, group_b


def _split_oversized(
    means_np: np.ndarray,
    clusters: List[List[int]],
    max_diameter_m: float,
    min_cluster_size: int,
) -> List[List[int]]:
    """Recursively split clusters that exceed max_diameter_m."""
    result: List[List[int]] = []
    stack = list(clusters)

    while stack:
        cluster = stack.pop()
        if len(cluster) < min_cluster_size:
            continue
        diam = _cluster_diameter(means_np, cluster)
        if diam <= max_diameter_m:
            result.append(cluster)
            continue
        # Split and recurse
        a, b = _split_cluster_bisection(means_np, cluster)
        if len(a) >= min_cluster_size:
            stack.append(a)
        if len(b) >= min_cluster_size:
            stack.append(b)

    return result


def compute_regions(
    scene_state: Dict[str, Any],
    *,
    distance_threshold_m: float = 2.0,
    min_cluster_size: int = 2,
    covisibility_required: bool = True,
    min_covisibility_weight: float = 0.0,
    max_region_diameter_m: float = 5.0,
) -> List[List[int]]:
    """Partition active captioned objects into spatial regions.

    Returns a list of clusters, each a sorted list of object indices (into the
    scene_state parallel arrays). Tiny clusters are absorbed into the nearest
    large cluster when possible; truly isolated objects are omitted.

    Parameters
    ----------
    distance_threshold_m : float
        Max Euclidean distance to form an edge between two objects.
    min_cluster_size : int
        Clusters smaller than this are absorbed or discarded.
    covisibility_required : bool
        If True, edges require both distance AND covisibility.
    min_covisibility_weight : float
        Minimum covisibility weight (from covisibility_weights dict) for an
        edge. 0.0 means any binary covisibility bit is sufficient.
    max_region_diameter_m : float
        Maximum spatial diameter of a region. Clusters exceeding this are
        recursively split via bisection. Set to inf to disable.
    """
    means = scene_state.get("means")
    if not isinstance(means, torch.Tensor) or means.shape[0] == 0:
        return []

    active = scene_state.get("active")
    captions = scene_state.get("object_caption", [])
    n = int(means.shape[0])

    eligible: List[int] = []
    for i in range(n):
        if isinstance(active, torch.Tensor) and i < active.shape[0] and not bool(active[i]):
            continue
        cap = captions[i] if i < len(captions) else ""
        if not cap:
            continue
        eligible.append(i)

    if len(eligible) < min_cluster_size:
        return []

    means_np = means.detach().cpu().numpy().astype(np.float32)
    adj_u64 = scene_state.get("covisibility_adj_u64")
    has_covis = isinstance(adj_u64, torch.Tensor) and adj_u64.ndim == 2
    if has_covis:
        adj_u64_cpu = adj_u64.detach().cpu()

    # Covisibility weights (continuous strength, not just binary)
    covis_weights: Optional[Dict[int, Dict[int, float]]] = None
    if min_covisibility_weight > 0:
        covis_weights = scene_state.get("covisibility_weights")
        if not isinstance(covis_weights, dict):
            covis_weights = None

    local_to_global = eligible
    m = len(eligible)
    uf = _UnionFind(m)

    threshold_sq = distance_threshold_m * distance_threshold_m
    for li in range(m):
        gi = local_to_global[li]
        for lj in range(li + 1, m):
            gj = local_to_global[lj]
            diff = means_np[gi] - means_np[gj]
            dist_sq = float(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2)
            if dist_sq > threshold_sq:
                continue
            if covisibility_required and has_covis:
                if not (_check_covisible(adj_u64_cpu, gi, gj) or _check_covisible(adj_u64_cpu, gj, gi)):
                    continue
                # Check minimum weight if available
                if covis_weights is not None and min_covisibility_weight > 0:
                    w_ij = covis_weights.get(gi, {}).get(gj, 0.0)
                    w_ji = covis_weights.get(gj, {}).get(gi, 0.0)
                    if max(w_ij, w_ji) < min_covisibility_weight:
                        continue
            uf.union(li, lj)

    components: Dict[int, List[int]] = {}
    for li in range(m):
        root = uf.find(li)
        components.setdefault(root, []).append(local_to_global[li])

    large: List[List[int]] = []
    small: List[List[int]] = []
    for comp in components.values():
        if len(comp) >= min_cluster_size:
            large.append(sorted(comp))
        else:
            small.append(comp)

    if not large and small:
        all_sorted = sorted([idx for comp in small for idx in comp])
        return [all_sorted] if len(all_sorted) >= min_cluster_size else []

    # Split oversized clusters (breaks single-linkage chains)
    if max_region_diameter_m < float("inf"):
        large = _split_oversized(means_np, large, max_region_diameter_m, min_cluster_size)

    # Absorb small clusters into nearest large cluster
    large_centroids = []
    for cluster in large:
        pts = means_np[cluster]
        large_centroids.append(pts.mean(axis=0))

    absorb_threshold_sq = (1.5 * distance_threshold_m) ** 2
    for comp in small:
        centroid = means_np[comp].mean(axis=0)
        best_dist_sq = float("inf")
        best_idx = -1
        for ci, lc in enumerate(large_centroids):
            d = centroid - lc
            dsq = float(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
            if dsq < best_dist_sq:
                best_dist_sq = dsq
                best_idx = ci
        if best_idx >= 0 and best_dist_sq <= absorb_threshold_sq:
            large[best_idx] = sorted(large[best_idx] + comp)

    return [sorted(c) for c in large if len(c) >= min_cluster_size]
