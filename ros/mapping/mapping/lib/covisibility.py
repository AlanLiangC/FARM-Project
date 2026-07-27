"""Covisibility graph maintenance utilities for the streaming mapper node.

All functions are pure (no ROS node state). Callers pass in the scene-state
dict, threading lock, and config values rather than a ``self`` reference.
Logger callables (info/warn) are optional; pass ``node.get_logger().info`` etc.
to route messages through the ROS logger.
"""

from __future__ import annotations

import contextlib
import math
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from scene_graph.map_update.covisibility import (
    update_covisibility_active_bitset,
    update_covisibility_from_visible_indices,
)
from scene_graph.storage.models import ImageRecord

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None


# ---------------------------------------------------------------------------
# Gaussian distance metric
# ---------------------------------------------------------------------------


def hellinger2_pairs(
    mu1: torch.Tensor,
    cov6_1: torch.Tensor,
    mu2: torch.Tensor,
    cov6_2: torch.Tensor,
) -> torch.Tensor:
    """Compute Hellinger^2 distance between matched Gaussian pairs.

    Args:
        mu1, mu2: (K, 3) mean vectors.
        cov6_1, cov6_2: (K, 6) upper-triangular covariance in
            ``[s00, s01, s02, s11, s12, s22]`` order.

    Returns:
        (K,) float32 tensor.  ``1.0`` marks invalid / failed pairs.
    """
    if mu1.numel() == 0:
        return torch.empty(0, device=mu1.device, dtype=torch.float32)

    device = mu1.device
    mu1_f = mu1.to(dtype=torch.float32)
    mu2_f = mu2.to(dtype=torch.float32)
    cov6_1_f = cov6_1.to(dtype=torch.float32)
    cov6_2_f = cov6_2.to(dtype=torch.float32)

    K = int(mu1_f.shape[0])
    cov1 = torch.zeros((K, 3, 3), device=device, dtype=torch.float32)
    cov2 = torch.zeros((K, 3, 3), device=device, dtype=torch.float32)

    cov1[:, 0, 0] = cov6_1_f[:, 0]
    cov1[:, 0, 1] = cov1[:, 1, 0] = cov6_1_f[:, 1]
    cov1[:, 0, 2] = cov1[:, 2, 0] = cov6_1_f[:, 2]
    cov1[:, 1, 1] = cov6_1_f[:, 3]
    cov1[:, 1, 2] = cov1[:, 2, 1] = cov6_1_f[:, 4]
    cov1[:, 2, 2] = cov6_1_f[:, 5]

    cov2[:, 0, 0] = cov6_2_f[:, 0]
    cov2[:, 0, 1] = cov2[:, 1, 0] = cov6_2_f[:, 1]
    cov2[:, 0, 2] = cov2[:, 2, 0] = cov6_2_f[:, 2]
    cov2[:, 1, 1] = cov6_2_f[:, 3]
    cov2[:, 1, 2] = cov2[:, 2, 1] = cov6_2_f[:, 4]
    cov2[:, 2, 2] = cov6_2_f[:, 5]

    eps = 1e-4
    eye = torch.eye(3, device=device, dtype=torch.float32).expand(K, 3, 3)
    cov1 = 0.5 * (cov1 + cov1.transpose(-1, -2)) + eps * eye
    cov2 = 0.5 * (cov2 + cov2.transpose(-1, -2)) + eps * eye
    sigma = 0.5 * (cov1 + cov2)

    H2 = torch.ones((K,), device=device, dtype=torch.float32)

    with torch.cuda.amp.autocast(enabled=False):
        if hasattr(torch.linalg, "cholesky_ex"):
            L1, info1 = torch.linalg.cholesky_ex(cov1)
            L2, info2 = torch.linalg.cholesky_ex(cov2)
            Ls, infos = torch.linalg.cholesky_ex(sigma)
            valid = (info1 == 0) & (info2 == 0) & (infos == 0)
        else:
            try:
                L1 = torch.linalg.cholesky(cov1)
                L2 = torch.linalg.cholesky(cov2)
                Ls = torch.linalg.cholesky(sigma)
                valid = torch.ones((K,), device=device, dtype=torch.bool)
            except Exception:
                return H2

        if not bool(valid.any()):
            return H2

        idx = torch.nonzero(valid, as_tuple=False).view(-1)
        L1_v = L1.index_select(0, idx)
        L2_v = L2.index_select(0, idx)
        Ls_v = Ls.index_select(0, idx)
        mu1_v = mu1_f.index_select(0, idx)
        mu2_v = mu2_f.index_select(0, idx)

        logdet1 = 2.0 * torch.log(torch.diagonal(L1_v, dim1=-2, dim2=-1)).sum(-1)
        logdet2 = 2.0 * torch.log(torch.diagonal(L2_v, dim1=-2, dim2=-1)).sum(-1)
        logdets = 2.0 * torch.log(torch.diagonal(Ls_v, dim1=-2, dim2=-1)).sum(-1)

        delta = (mu1_v - mu2_v).unsqueeze(-1)
        y = torch.cholesky_solve(delta, Ls_v)
        quad = (delta.squeeze(-1) * y.squeeze(-1)).sum(-1)

        logBC = 0.25 * (logdet1 + logdet2) - 0.5 * logdets - 0.125 * quad
        BC = torch.exp(logBC)
        H2_valid = torch.clamp(1.0 - BC, min=0.0, max=1.0)
        H2.index_copy_(0, idx, H2_valid)

    return H2


# ---------------------------------------------------------------------------
# Batch covisibility update
# ---------------------------------------------------------------------------


def update_covisibility_from_batch(
    *,
    state: dict,
    det_idx: torch.Tensor,
    obj_idx: torch.Tensor,
    detection_image_ids: Sequence[Optional[int]],
    prev_object_count: int,
    allow_new_objects: bool,
    camera_neighbors: Dict[str, Tuple[str, ...]],
    history_batches: int,
    prev_per_camera: Dict[str, Set[int]],
    prev_unknown_visible: Set[int],
    logger_info: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Set[int]], Set[int]]:
    """Update the covisibility graph from one decoded batch.

    Returns the updated ``(prev_per_camera, prev_unknown_visible)`` pair so
    the caller can store them for the next window (when ``history_batches >= 2``).
    """
    if det_idx is None or not hasattr(det_idx, "numel"):
        return prev_per_camera, prev_unknown_visible
    if not detection_image_ids:
        return prev_per_camera, prev_unknown_visible

    import time
    t0 = time.perf_counter()

    num_det = min(int(det_idx.numel()), len(detection_image_ids))
    if num_det <= 0:
        return prev_per_camera, prev_unknown_visible

    current_objects = int(getattr(state.get("means"), "shape", [0])[0])
    if current_objects <= 1:
        return prev_per_camera, prev_unknown_visible

    try:
        det_idx_cpu = det_idx[:num_det].detach().to("cpu", copy=False).to(torch.long)
    except Exception:
        return prev_per_camera, prev_unknown_visible
    try:
        obj_idx_cpu = obj_idx.detach().to("cpu", copy=False).to(torch.long)
    except Exception:
        obj_idx_cpu = torch.empty(0, dtype=torch.long)

    new_det_map: dict[int, int] = {}
    if allow_new_objects and current_objects > prev_object_count:
        new_mask = det_idx_cpu < 0
        new_det_indices = torch.nonzero(new_mask, as_tuple=False).view(-1).tolist()
        for offset, det_i in enumerate(new_det_indices):
            new_det_map[int(det_i)] = int(prev_object_count + offset)

    per_image: dict[int, set[int]] = {}
    for det_i in range(num_det):
        image_id_raw = detection_image_ids[det_i]
        if image_id_raw is None:
            continue
        try:
            image_id = int(image_id_raw)
        except Exception:
            continue
        det_target = int(det_idx_cpu[det_i].item())
        obj_index: Optional[int] = None
        if det_target >= 0 and det_target < obj_idx_cpu.numel():
            obj_index = int(obj_idx_cpu[det_target].item())
        elif det_target < 0 and allow_new_objects:
            obj_index = new_det_map.get(det_i)
        if obj_index is None or obj_index < 0 or obj_index >= current_objects:
            continue
        obj_set = per_image.setdefault(image_id, set())
        obj_set.add(obj_index)

    images_meta_raw = state.get("images", [])
    images_meta: List[ImageRecord] = images_meta_raw if isinstance(images_meta_raw, list) else []
    per_camera: dict[str, set[int]] = {}
    unknown_visible: set[int] = set()
    for image_id, obj_set in per_image.items():
        if image_id < 0 or image_id >= len(images_meta):
            unknown_visible.update(obj_set)
            continue
        cam = str(getattr(images_meta[image_id], "camera_id", "") or "").strip()
        if not cam:
            unknown_visible.update(obj_set)
            continue
        per_camera.setdefault(cam, set()).update(obj_set)

    visible_objects_total: set[int] = set()
    for obj_set in per_camera.values():
        visible_objects_total.update(obj_set)
    visible_objects_total.update(unknown_visible)

    updated_groups = 0
    use_prev_per_camera = prev_per_camera if history_batches >= 2 else {}
    use_prev_unknown = prev_unknown_visible if history_batches >= 2 else set()

    for cam_name, cam_visible in per_camera.items():
        neighbors = camera_neighbors.get(cam_name, (cam_name,))
        local_visible: set[int] = set()
        for neigh in neighbors:
            local_visible.update(per_camera.get(neigh, set()))
            local_visible.update(use_prev_per_camera.get(neigh, set()))
        if local_visible:
            local_visible.update(cam_visible)
            local_visible.update(use_prev_per_camera.get(cam_name, set()))
        if len(local_visible) < 2:
            continue
        update_covisibility_from_visible_indices(
            state,
            local_visible,
            num_objects=current_objects,
            k=10,
            soft_mean_scale=2.0,
        )
        updated_groups += 1

    unknown_union = set(unknown_visible)
    unknown_union.update(use_prev_unknown)
    if updated_groups == 0 and len(unknown_union) >= 2:
        update_covisibility_from_visible_indices(
            state,
            unknown_union,
            num_objects=current_objects,
            k=10,
            soft_mean_scale=2.0,
        )
        updated_groups = 1

    new_prev_per_camera = {k: set(v) for k, v in per_camera.items()} if history_batches >= 2 else prev_per_camera
    new_prev_unknown = set(unknown_visible) if history_batches >= 2 else prev_unknown_visible

    state["covisibility_last_updated_images"] = int(updated_groups)
    state["covisibility_last_seen_images"] = int(len(per_image))

    t1 = time.perf_counter()
    if updated_groups > 0 and logger_info is not None:
        logger_info(
            f"Co-visibility update (local, history={history_batches}, groups={updated_groups}, "
            "knn=10, soft_mean_scale=2.0): "
            f"visible_objects={len(visible_objects_total)} images={len(per_image)} "
            f"dt_ms={(t1 - t0) * 1000.0:.2f}"
        )

    return new_prev_per_camera, new_prev_unknown


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def save_covisibility_visualization(
    *,
    state: dict,
    viz_path: Path,
    logger_info: Optional[Callable[[str], None]] = None,
    logger_warn: Optional[Callable[[str], None]] = None,
) -> None:
    """Render the covisibility adjacency matrix to a PNG and save to *viz_path*."""
    if plt is None:
        return

    active = state.get("active")
    means = state.get("means")
    adj = state.get("covisibility_adj_u64")
    blocks = int(state.get("covisibility_blocks") or 0)
    if not isinstance(active, torch.Tensor) or not isinstance(means, torch.Tensor):
        return
    if not isinstance(adj, torch.Tensor) or blocks <= 0:
        return

    import time
    t0 = time.perf_counter()

    num_objects = int(means.shape[0])
    if num_objects <= 1:
        return
    try:
        active_u64 = update_covisibility_active_bitset(state, num_objects=num_objects)
    except Exception:
        active_u64 = state.get("covisibility_active_u64")
    if not isinstance(active_u64, torch.Tensor):
        return
    try:
        active_cpu = active[:num_objects].detach().to("cpu", copy=False).to(torch.bool)
        active_ids = active_cpu.nonzero(as_tuple=False).view(-1).tolist()
    except Exception:
        return
    if len(active_ids) < 2:
        return
    try:
        adj_cpu = adj[:num_objects, :blocks].detach().to("cpu", copy=False).numpy().astype(np.uint64, copy=False)
        active_mask = active_u64[:blocks].detach().to("cpu", copy=False).numpy().astype(np.uint64, copy=False)
    except Exception:
        return

    active_arr = np.asarray(active_ids, dtype=np.int64)
    active_blocks = active_arr // 64
    active_masks = np.left_shift(np.uint64(1), (active_arr % 64).astype(np.uint64))
    matrix = np.zeros((len(active_ids), len(active_ids)), dtype=np.uint8)

    for row_i, obj_idx in enumerate(active_ids):
        row_bits = adj_cpu[obj_idx, :blocks] & active_mask
        matrix[row_i, :] = (row_bits[active_blocks] & active_masks) != 0
        matrix[row_i, row_i] = 0

    edges = int(matrix.sum())
    possible = max(1, len(active_ids) * (len(active_ids) - 1))
    density = float(edges) / float(possible)
    if edges == 0 and logger_warn is not None:
        last_updated = int(state.get("covisibility_last_updated_images") or 0)
        last_seen = int(state.get("covisibility_last_seen_images") or 0)
        logger_warn(
            f"Co-visibility graph has no edges (active={len(active_ids)}). "
            f"(last_update_groups={last_updated}/{last_seen}) "
            "This usually means each local group sees <2 mapped objects, "
            "`detection_image_ids` are missing/None, or the kNN search produced no finite Hellinger distances."
        )

    with contextlib.suppress(Exception):
        viz_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig = plt.figure(figsize=(6.5, 6.0), dpi=200)
        ax = fig.add_subplot(111)
        ax.imshow(matrix, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
        ax.set_facecolor("white")
        ax.set_title(f"Co-visibility graph (active={len(active_ids)} edges={edges} density={density:.4f})")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(viz_path)
        plt.close(fig)
        t1 = time.perf_counter()
        if logger_info is not None:
            logger_info(f"Saved covisibility graph visualization to {viz_path} (dt_ms={(t1 - t0) * 1000.0:.2f})")
    except Exception as exc:
        if logger_warn is not None:
            logger_warn(f"Failed to save covisibility graph visualization to {viz_path}: {exc}")


# ---------------------------------------------------------------------------
# Filtered neighbor computation
# ---------------------------------------------------------------------------


def compute_covisibility_filtered_neighbors_indices(
    *,
    state: dict,
    covisibility_lock: threading.Lock,
    include_idx: List[int],
    include_mask: torch.Tensor,
    means_cpu: torch.Tensor,
    cov6_cpu: torch.Tensor,
    limit: int,
    hellinger_thresh: float,
) -> Dict[int, List[int]]:
    """Return Hellinger-filtered neighbor indices per object.

    The returned neighbor lists are directed (per-row) and ordered by the
    composite score used for ``SceneGraphSnapshotSimple`` publishing.

    Args:
        state: scene-state dict (read-only for covisibility tensors).
        covisibility_lock: lock guarding covisibility weights in *state*.
        include_idx: object indices to compute neighbors for.
        include_mask: boolean mask of shape ``(limit,)`` marking valid objects.
        means_cpu: ``(limit, 3)`` object mean positions on CPU.
        cov6_cpu: ``(limit, 6)`` object covariances on CPU.
        limit: maximum valid object index (exclusive).
        hellinger_thresh: Hellinger^2 threshold for the neighbor caption filter.
    """
    neighbors_by_obj: Dict[int, List[int]] = {int(i): [] for i in include_idx}

    weights_rows: Dict[int, Dict[int, float]] = {}
    weights_is_dict = False
    with covisibility_lock:
        weights = state.get("covisibility_weights")
        weights_is_dict = isinstance(weights, dict)
        if weights_is_dict:
            for obj_i in include_idx:
                row = weights.get(obj_i)
                if isinstance(row, dict) and row:
                    weights_rows[int(obj_i)] = dict(row)

    if weights_is_dict:
        strength: Dict[int, float] = {}
        for obj_i in include_idx:
            row = weights_rows.get(obj_i)
            if not isinstance(row, dict):
                strength[int(obj_i)] = 0.0
                continue
            total = 0.0
            for neighbor_idx, weight in row.items():
                try:
                    neighbor_idx_int = int(neighbor_idx)
                    weight_val = float(weight)
                except Exception:
                    continue
                if neighbor_idx_int == obj_i or neighbor_idx_int < 0 or neighbor_idx_int >= limit:
                    continue
                if not bool(include_mask[neighbor_idx_int].item()):
                    continue
                if weight_val > 0.0:
                    total += weight_val
            strength[int(obj_i)] = total

        edges_by_obj: Dict[int, Tuple[List[int], List[float], List[float], Dict[int, float]]] = {}

        for obj_i in include_idx:
            obj_i_int = int(obj_i)
            row = weights_rows.get(obj_i_int)
            if not isinstance(row, dict):
                continue

            neighbor_indices: List[int] = []
            neighbor_weights: List[float] = []
            for neighbor_idx, weight in row.items():
                try:
                    neighbor_idx_int = int(neighbor_idx)
                    weight_val = float(weight)
                except Exception:
                    continue
                if neighbor_idx_int == obj_i_int or neighbor_idx_int < 0 or neighbor_idx_int >= limit:
                    continue
                if not bool(include_mask[neighbor_idx_int].item()):
                    continue
                if weight_val <= 0.0:
                    continue
                neighbor_indices.append(neighbor_idx_int)
                neighbor_weights.append(weight_val)

            if not neighbor_indices:
                continue

            try:
                mu_i = means_cpu[obj_i_int].to(torch.float32).view(1, 3).expand(len(neighbor_indices), 3)
                cov_i = cov6_cpu[obj_i_int].to(torch.float32).view(1, 6).expand(len(neighbor_indices), 6)
                mu_j = means_cpu[neighbor_indices].to(torch.float32)
                cov_j = cov6_cpu[neighbor_indices].to(torch.float32)
                d = hellinger2_pairs(mu_i, cov_i, mu_j, cov_j).detach().cpu().to(torch.float32)
            except Exception:
                d = torch.ones((len(neighbor_indices),), dtype=torch.float32)

            distances = [float(x) for x in d.tolist()]
            dist_by_neighbor = {neighbor_indices[i]: distances[i] for i in range(len(neighbor_indices))}
            edges_by_obj[obj_i_int] = (neighbor_indices, neighbor_weights, distances, dist_by_neighbor)

        median_h_by_obj: Dict[int, float] = {}
        for obj_i_int, (_nbrs, _wts, distances, _dist_by_nbr) in edges_by_obj.items():
            valid_d2 = [d2 for d2 in distances if math.isfinite(d2) and d2 < 0.999]
            if not valid_d2:
                continue
            median_d2 = float(np.median(np.asarray(valid_d2, dtype=np.float32)))
            median_h_by_obj[obj_i_int] = math.sqrt(max(0.0, median_d2))

        for obj_i in include_idx:
            obj_i_int = int(obj_i)
            packed = edges_by_obj.get(obj_i_int)
            if packed is None:
                continue
            neighbor_indices, neighbor_weights, distances, dist_by_neighbor = packed

            num_edges = len(neighbor_indices)
            all_by_dist = sorted(zip(distances, neighbor_indices), key=lambda x: x[0])
            keep_base = {nbr for _dist, nbr in all_by_dist[: min(2, num_edges)]}

            keep_thresh: Set[int] = set()
            median_h_i = median_h_by_obj.get(obj_i_int)
            if median_h_i is not None and num_edges > 0:
                hi_scaled = float(median_h_i) * 1.01
                for d2, nbr in all_by_dist:
                    if not math.isfinite(d2):
                        continue
                    median_h_j = median_h_by_obj.get(int(nbr))
                    hj = float(median_h_j) if median_h_j is not None else 0.0
                    thresh_h = 0.5 * (hi_scaled + hj)
                    if d2 <= thresh_h:
                        keep_thresh.add(int(nbr))

            keep_all = keep_base | keep_thresh

            top5 = sorted(zip(neighbor_weights, neighbor_indices), key=lambda x: x[0], reverse=True)[:5]
            top5_by_dist = sorted(top5, key=lambda x: dist_by_neighbor.get(x[1], float("inf")))
            keep_top5 = {nbr for _w, nbr in top5_by_dist[: min(2, len(top5_by_dist))]}

            keep_neighbors = keep_all | keep_top5
            if not keep_neighbors:
                continue

            s_i = strength.get(obj_i_int, 0.0)
            candidates: List[Tuple[float, int]] = []
            if s_i > 0.0:
                for idx, nbr in enumerate(neighbor_indices):
                    if nbr not in keep_neighbors:
                        continue
                    s_j = strength.get(nbr, 0.0)
                    if s_j <= 0.0:
                        continue
                    denom = math.sqrt(float(s_i) * float(s_j))
                    if denom <= 0.0:
                        continue
                    score = float(neighbor_weights[idx]) / denom
                    candidates.append((score, nbr))

            candidates.sort(key=lambda x: x[0], reverse=True)
            if candidates:
                neighbors_by_obj[obj_i_int] = [nbr for _score, nbr in candidates]
        return neighbors_by_obj

    # Fallback: use the stored bitset adjacency, restricted to the included subset.
    adj = state.get("covisibility_adj_u64")
    blocks = int(state.get("covisibility_blocks") or 0)
    if isinstance(adj, torch.Tensor) and blocks > 0 and adj.ndim == 2:
        try:
            adj_cpu = adj[:limit, :blocks].detach().to("cpu", copy=False).numpy().astype(np.uint64, copy=False)
        except Exception:
            return neighbors_by_obj

        included_mask = np.zeros((blocks,), dtype=np.uint64)
        for idx in include_idx:
            idx_int = int(idx)
            if idx_int < 0 or idx_int >= limit:
                continue
            included_mask[idx_int // 64] |= np.uint64(1) << np.uint64(idx_int % 64)

        for obj_i in include_idx:
            obj_i_int = int(obj_i)
            if obj_i_int < 0 or obj_i_int >= limit:
                continue
            row_bits = adj_cpu[obj_i_int, :blocks] & included_mask
            row_neighbors: List[int] = []
            for block_i in range(blocks):
                bits = int(row_bits[block_i])
                base = block_i * 64
                while bits:
                    lsb = bits & -bits
                    bit_pos = lsb.bit_length() - 1
                    neighbor_idx = base + bit_pos
                    bits -= lsb
                    if neighbor_idx == obj_i_int or neighbor_idx < 0 or neighbor_idx >= limit:
                        continue
                    if not bool(include_mask[neighbor_idx].item()):
                        continue
                    row_neighbors.append(int(neighbor_idx))
            neighbors_by_obj[obj_i_int] = row_neighbors

    return neighbors_by_obj


# ---------------------------------------------------------------------------
# Filtered adjacency update
# ---------------------------------------------------------------------------


def update_covisibility_filtered_adjacency(
    *,
    state: dict,
    covisibility_lock: threading.Lock,
    limit: int,
    hellinger_thresh: float,
) -> None:
    """Populate *state* with a pruned covisibility adjacency for visualization/publishing."""
    if limit <= 0:
        return

    active = state.get("active")
    means = state.get("means")
    cov6 = state.get("cov6")
    if (
        not isinstance(active, torch.Tensor)
        or not isinstance(means, torch.Tensor)
        or not isinstance(cov6, torch.Tensor)
    ):
        return

    try:
        active_cpu = active[:limit].detach().to("cpu", copy=False).to(torch.bool)
        means_cpu = means[:limit].detach().to("cpu", copy=False)
        cov6_cpu = cov6[:limit].detach().to("cpu", copy=False)
    except Exception:
        return

    finite_mask = torch.isfinite(means_cpu).all(dim=1) if means_cpu.ndim == 2 and means_cpu.shape[1] == 3 else None
    if finite_mask is None or finite_mask.numel() != active_cpu.numel():
        return
    include_mask = active_cpu & finite_mask
    include_idx = torch.nonzero(include_mask, as_tuple=False).view(-1).tolist()
    if len(include_idx) < 2:
        state["covisibility_filtered_blocks"] = int((limit + 63) // 64)
        state["covisibility_filtered_adj_u64"] = torch.zeros(
            (limit, state["covisibility_filtered_blocks"]), dtype=torch.int64
        )
        return

    neighbors_by_obj = compute_covisibility_filtered_neighbors_indices(
        state=state,
        covisibility_lock=covisibility_lock,
        include_idx=[int(x) for x in include_idx],
        include_mask=include_mask,
        means_cpu=means_cpu,
        cov6_cpu=cov6_cpu,
        limit=limit,
        hellinger_thresh=hellinger_thresh,
    )

    blocks = int((limit + 63) // 64)
    adj_u = np.zeros((limit, blocks), dtype=np.uint64)

    for obj_i in include_idx:
        obj_i_int = int(obj_i)
        for nbr in neighbors_by_obj.get(obj_i_int, []):
            nbr_int = int(nbr)
            if nbr_int == obj_i_int or nbr_int < 0 or nbr_int >= limit:
                continue
            a, b = (obj_i_int, nbr_int) if obj_i_int < nbr_int else (nbr_int, obj_i_int)
            adj_u[a, b // 64] |= np.uint64(1) << np.uint64(b % 64)

    state["covisibility_filtered_blocks"] = int(blocks)
    state["covisibility_filtered_adj_u64"] = torch.from_numpy(adj_u.view(np.int64)).contiguous()
