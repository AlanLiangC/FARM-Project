"""Pure stat functions feeding the JSONL tracer.

All functions here are *side-effect free*: they accept already-computed
pipeline values (numpy- or torch-friendly) and return JSON-ready dicts. The
goal is a structured, human-readable per-frame summary that captures:

  - segmentation funnel (raw, after each filter step)
  - neighbor lookup distribution
  - correspondence outcomes (new vs matched vs merged)
  - Gaussian fusion numerical health (NaNs, eigenvalue extremes, position
    jumps, condition numbers, drift)
  - voxel-cloud level + count distribution

These are computed defensively (no exceptions surface to the caller) so
tracing never destabilizes the recon run.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch


# ------------------------------------------------------------------------- helpers


def _percentiles(values: np.ndarray, ps=(1, 10, 50, 90, 99)) -> Dict[str, float]:
    if values is None or len(values) == 0:
        return {f"p{int(p)}": None for p in ps}
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {f"p{int(p)}": None for p in ps}
    qs = np.percentile(finite, list(ps))
    return {f"p{int(p)}": float(q) for p, q in zip(ps, qs)}


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _to_numpy_1d(t) -> np.ndarray:
    if t is None:
        return np.zeros(0, dtype=np.float64)
    if isinstance(t, torch.Tensor):
        return t.detach().to("cpu", dtype=torch.float32).numpy().reshape(-1)
    return np.asarray(t).reshape(-1)


def _to_numpy(t) -> np.ndarray:
    if t is None:
        return np.zeros(0, dtype=np.float64)
    if isinstance(t, torch.Tensor):
        return t.detach().to("cpu", dtype=torch.float32).numpy()
    return np.asarray(t)


# ------------------------------------------------------------------------- segmentation


def summarize_segmentation(seg: dict, *, names: Optional[Sequence[str]] = None, top_k: int = 8) -> Dict[str, Any]:
    """Per-frame raw segmentation summary."""
    if not isinstance(seg, dict):
        return {"n_raw": 0}
    means = seg.get("means")
    n_raw = int(getattr(means, "shape", [0])[0]) if means is not None else 0
    out: Dict[str, Any] = {"n_raw": n_raw}
    if n_raw == 0:
        return out

    class_ids = seg.get("class_ids")
    if isinstance(class_ids, torch.Tensor) and class_ids.numel() > 0:
        cids = class_ids.detach().to("cpu", dtype=torch.int64).numpy().tolist()
        hist: Dict[int, int] = {}
        for c in cids:
            hist[int(c)] = hist.get(int(c), 0) + 1
        ranked = sorted(hist.items(), key=lambda kv: -kv[1])[:top_k]
        out["class_id_top"] = [
            {
                "class_id": int(cid),
                "label": (str(names[cid]) if names is not None and 0 <= cid < len(names) else None),
                "n": int(n),
            }
            for cid, n in ranked
        ]

    scores = seg.get("scores")
    if isinstance(scores, torch.Tensor) and scores.numel() > 0:
        s = scores.detach().to("cpu", dtype=torch.float32).numpy().reshape(-1)
        out["score_pct"] = _percentiles(s)

    num_pixels = seg.get("num_pixels")
    if isinstance(num_pixels, torch.Tensor) and num_pixels.numel() > 0:
        np_arr = num_pixels.detach().to("cpu", dtype=torch.float32).numpy().reshape(-1)
        out["num_pixels_pct"] = _percentiles(np_arr)

    # Mean position spread tells you the world-frame coverage of this batch.
    if isinstance(means, torch.Tensor) and means.numel() > 0:
        m = means.detach().to("cpu", dtype=torch.float32).numpy()
        if m.ndim == 2 and m.shape[1] >= 3:
            out["means_world_minmax"] = {
                "min": [float(x) for x in m[:, :3].min(axis=0)],
                "max": [float(x) for x in m[:, :3].max(axis=0)],
            }
    return out


# ------------------------------------------------------------------------- filtering


def summarize_filtering(filter_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cumulative funnel summary across filter events.

    ``filter_events`` is a list with entries like::

        {"name": "border", "n_in": 47, "n_out": 35, "config": {...},
         "dropped_class_top": [{"class_id": 4, "label": "wall", "n": 2}, ...]}
    """
    out: Dict[str, Any] = {"steps": list(filter_events or [])}
    if filter_events:
        out["n_in"] = int(filter_events[0].get("n_in", 0))
        out["n_out"] = int(filter_events[-1].get("n_out", 0))
        out["n_dropped_total"] = max(0, out["n_in"] - out["n_out"])
    else:
        out["n_in"] = 0
        out["n_out"] = 0
        out["n_dropped_total"] = 0
    return out


# ------------------------------------------------------------------------- neighbors


def summarize_neighbors(
    neighbors: List[torch.Tensor],
    seg_outputs: dict,
    state: dict,
    *,
    feature_sim_thresh: float = 0.5,
    hellinger_thresh: float = 0.8,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Aggregate stats about the neighbor-lookup output.

    Reports neighbor-count histogram and recomputes the closest-Hellinger /
    highest-feature-similarity for each detection so we can see distributions
    near the threshold (useful for tuning).
    """
    n_dets = len(neighbors) if neighbors is not None else 0
    state_means = state.get("means") if isinstance(state, dict) else None
    n_state_objects = int(getattr(state_means, "shape", [0])[0]) if state_means is not None else 0

    counts = np.zeros(n_dets, dtype=np.int64)
    for i, nbr in enumerate(neighbors or []):
        if isinstance(nbr, torch.Tensor):
            counts[i] = int(nbr.numel())
        elif nbr is not None:
            try:
                counts[i] = len(nbr)
            except Exception:
                counts[i] = 0
    hist = {
        "0": int((counts == 0).sum()),
        "1": int((counts == 1).sum()),
        "2": int((counts == 2).sum()),
        "3+": int((counts >= 3).sum()),
    }

    out: Dict[str, Any] = {
        "n_detections": n_dets,
        "n_state_objects": n_state_objects,
        "n_with_at_least_one": int((counts > 0).sum()),
        "neighbors_per_det_hist": hist,
        "neighbors_per_det_pct": _percentiles(counts.astype(np.float64), ps=(50, 90, 99)),
        "thresholds": {
            "feature_sim_thresh": float(feature_sim_thresh),
            "hellinger_thresh": float(hellinger_thresh),
        },
    }

    # Optional: closest-hellinger / best-cosine-sim per detection. We avoid
    # recomputing for every det × every state-object (could be expensive); we
    # compute only for detections that actually have ≥1 neighbor (small subset).
    det_means = seg_outputs.get("means") if isinstance(seg_outputs, dict) else None
    det_cov6 = seg_outputs.get("cov6") if isinstance(seg_outputs, dict) else None
    det_feats = seg_outputs.get("features") if isinstance(seg_outputs, dict) else None
    state_cov6 = state.get("cov6") if isinstance(state, dict) else None
    state_feats = state.get("features") if isinstance(state, dict) else None
    if (
        isinstance(det_means, torch.Tensor)
        and isinstance(state_means, torch.Tensor)
        and isinstance(det_cov6, torch.Tensor)
        and isinstance(state_cov6, torch.Tensor)
        and isinstance(det_feats, torch.Tensor)
        and isinstance(state_feats, torch.Tensor)
        and det_means.numel() > 0
        and state_means.numel() > 0
        and out["n_with_at_least_one"] > 0
    ):
        try:
            from scene_graph.map_update.get_neighbors import _cov6_to_matrix, _hellinger_distance

            det_cov = _cov6_to_matrix(det_cov6.detach().to("cpu", dtype=torch.float32))
            state_cov = _cov6_to_matrix(state_cov6.detach().to("cpu", dtype=torch.float32))
            det_means_cpu = det_means.detach().to("cpu", dtype=torch.float32)
            state_means_cpu = state_means.detach().to("cpu", dtype=torch.float32)
            det_feats_cpu = det_feats.detach().to("cpu", dtype=torch.float32)
            state_feats_cpu = state_feats.detach().to("cpu", dtype=torch.float32)
            df = torch.nn.functional.normalize(det_feats_cpu, dim=1, eps=1e-12)
            sf = torch.nn.functional.normalize(state_feats_cpu, dim=1, eps=1e-12)

            best_h = []
            best_sim = []
            for d_i, nbr in enumerate(neighbors):
                if not isinstance(nbr, torch.Tensor) or nbr.numel() == 0:
                    continue
                idx = nbr.detach().to("cpu", dtype=torch.long).numpy().reshape(-1)
                idx = idx[(idx >= 0) & (idx < state_means_cpu.shape[0])]
                if idx.size == 0:
                    continue
                idx_t = torch.as_tensor(idx, dtype=torch.long)
                mu1 = det_means_cpu[d_i].unsqueeze(0).expand(len(idx), -1)
                cov1 = det_cov[d_i].unsqueeze(0).expand(len(idx), -1, -1)
                mu2 = state_means_cpu[idx_t]
                cov2 = state_cov[idx_t]
                h2 = _hellinger_distance(mu1, cov1, mu2, cov2)
                best_h.append(float(h2.min().item()))
                sim = (df[d_i].unsqueeze(0) @ sf[idx_t].t()).reshape(-1)
                best_sim.append(float(sim.max().item()))
            if best_h:
                out["min_hellinger_per_det_pct"] = _percentiles(np.asarray(best_h, dtype=np.float64))
                out["max_feat_sim_per_det_pct"] = _percentiles(np.asarray(best_sim, dtype=np.float64))
        except Exception as exc:  # noqa: BLE001
            out["neighbor_recompute_error"] = repr(exc)

    return out


# ------------------------------------------------------------------------- correspondence


def summarize_correspondence(
    det_idx: torch.Tensor,
    obj_idx: torch.Tensor,
    *,
    prev_object_count: int,
    update_info: Optional[dict],
    top_k_merges: int = 8,
) -> Dict[str, Any]:
    """Map raw correspondence output to readable counts + a few sample merges."""
    out: Dict[str, Any] = {
        "prev_object_count": int(prev_object_count),
    }
    if isinstance(det_idx, torch.Tensor) and det_idx.numel() > 0:
        di = det_idx.detach().to("cpu", dtype=torch.long).numpy()
        out["n_detections"] = int(di.shape[0])
        out["n_new_detections"] = int((di < 0).sum())
        out["n_matched_detections"] = int((di >= 0).sum())
    else:
        out["n_detections"] = 0
        out["n_new_detections"] = 0
        out["n_matched_detections"] = 0

    n_merged = 0
    if isinstance(obj_idx, torch.Tensor) and obj_idx.numel() > 0:
        oi = obj_idx.detach().to("cpu", dtype=torch.long).numpy()
        # an object i is merged-as-loser iff obj_idx[i] != i
        idx_arr = np.arange(oi.shape[0])
        n_merged = int((oi != idx_arr).sum())
    out["n_objects_merged"] = n_merged

    merges = []
    if isinstance(update_info, dict):
        ms = update_info.get("merged_objects") or []
        for m in ms[:top_k_merges]:
            d = {k: v for k, v in m.items() if k in ("loser_idx", "loser_id", "winner_idx", "winner_id", "loser_caption", "winner_caption")}
            lp, wp = m.get("loser_pos"), m.get("winner_pos")
            try:
                if lp is not None and wp is not None:
                    arr_l = np.asarray(lp, dtype=np.float64).reshape(-1)
                    arr_w = np.asarray(wp, dtype=np.float64).reshape(-1)
                    if arr_l.size >= 3 and arr_w.size >= 3:
                        d["delta_m"] = float(np.linalg.norm(arr_l[:3] - arr_w[:3]))
                        d["loser_pos"] = arr_l[:3].tolist()
                        d["winner_pos"] = arr_w[:3].tolist()
            except Exception:
                pass
            merges.append(d)
        out["merges_top"] = merges
        ni = update_info.get("new_object_indices") or []
        out["n_new_objects_appended"] = int(len(ni))
        if "far_merges_blocked" in update_info:
            out["n_far_merges_blocked"] = int(update_info.get("far_merges_blocked") or 0)
        if "cannot_link_merges_blocked" in update_info:
            out["n_cannot_link_merges_blocked"] = int(update_info.get("cannot_link_merges_blocked") or 0)
        if "same_frame_cannot_links_added" in update_info:
            out["n_same_frame_cannot_links_added"] = int(update_info.get("same_frame_cannot_links_added") or 0)
    return out


# ------------------------------------------------------------------------- Gaussian numerical health


def _cov6_to_3x3(cov6: np.ndarray) -> np.ndarray:
    """(N,6) -> (N,3,3) using the same packing convention as the algorithm."""
    if cov6.ndim != 2 or cov6.shape[1] < 6:
        return np.zeros((0, 3, 3), dtype=np.float64)
    n = cov6.shape[0]
    out = np.zeros((n, 3, 3), dtype=np.float64)
    out[:, 0, 0] = cov6[:, 0]
    out[:, 0, 1] = out[:, 1, 0] = cov6[:, 1]
    out[:, 0, 2] = out[:, 2, 0] = cov6[:, 2]
    out[:, 1, 1] = cov6[:, 3]
    out[:, 1, 2] = out[:, 2, 1] = cov6[:, 4]
    out[:, 2, 2] = cov6[:, 5]
    return out


def summarize_gaussian_update(
    *,
    means_before: Optional[np.ndarray],
    cov6_before: Optional[np.ndarray],
    state_after: dict,
    update_info: Optional[dict],
    matched_object_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Snapshot-based numerical health.

    Compares a saved (pre-update) snapshot of state["means"]/state["cov6"]
    against the post-update tensors, then computes:
      - count of NaN / Inf in post-update means/cov6
      - max position jump for matched objects (delta in m)
      - covariance condition number percentiles (post-update, active subset)
      - min eigenvalue of post-update active covariances
      - count of negative-definite (or near-zero) covariances
      - count of newly created objects (from update_info)
    """
    out: Dict[str, Any] = {}
    means_after = state_after.get("means") if isinstance(state_after, dict) else None
    cov6_after = state_after.get("cov6") if isinstance(state_after, dict) else None
    active = state_after.get("active") if isinstance(state_after, dict) else None
    if not isinstance(means_after, torch.Tensor) or not isinstance(cov6_after, torch.Tensor):
        return {"n_objects": 0}

    means_after_np = means_after.detach().to("cpu", dtype=torch.float32).numpy()
    cov6_after_np = cov6_after.detach().to("cpu", dtype=torch.float32).numpy()
    out["n_objects"] = int(means_after_np.shape[0])

    # NaN / Inf detection on the full state
    out["n_nan_means"] = int(np.isnan(means_after_np).any(axis=1).sum()) if means_after_np.size > 0 else 0
    out["n_inf_means"] = int(np.isinf(means_after_np).any(axis=1).sum()) if means_after_np.size > 0 else 0
    out["n_nan_cov6"] = int(np.isnan(cov6_after_np).any(axis=1).sum()) if cov6_after_np.size > 0 else 0
    out["n_inf_cov6"] = int(np.isinf(cov6_after_np).any(axis=1).sum()) if cov6_after_np.size > 0 else 0

    # Position jumps: only meaningful for objects that existed pre-update AND
    # were touched (matched). We compare same indices in before/after.
    if means_before is not None and cov6_before is not None and means_before.size > 0:
        n_before = means_before.shape[0]
        # post-update may have more (new objects appended); compare overlap.
        n_overlap = min(n_before, means_after_np.shape[0])
        if n_overlap > 0:
            mb = means_before[:n_overlap]
            ma = means_after_np[:n_overlap]
            valid = np.isfinite(mb).all(axis=1) & np.isfinite(ma).all(axis=1)
            if matched_object_indices is not None and len(matched_object_indices) > 0:
                idx = np.asarray([i for i in matched_object_indices if 0 <= i < n_overlap], dtype=np.int64)
                if idx.size > 0:
                    valid_idx = idx[valid[idx]]
                    if valid_idx.size > 0:
                        dist = np.linalg.norm(ma[valid_idx] - mb[valid_idx], axis=1)
                        out["pos_jump_m_pct_matched"] = _percentiles(dist)
                        out["max_pos_jump_m"] = float(dist.max())
                        out["n_pos_jump_gt_0p3m"] = int((dist > 0.3).sum())
                        out["n_pos_jump_gt_1p0m"] = int((dist > 1.0).sum())
            # Also report overall pre→post drift (active only) regardless of match.
            if isinstance(active, torch.Tensor):
                act = active.detach().to("cpu").numpy().astype(bool)
                act_overlap = act[:n_overlap]
                idx = np.where(valid & act_overlap)[0]
                if idx.size > 0:
                    dist = np.linalg.norm(ma[idx] - mb[idx], axis=1)
                    out["pos_drift_m_pct_active"] = _percentiles(dist)

    # Eigenvalue / condition-number stats on currently-active covariances.
    ill_cov_top: List[Dict[str, Any]] = []
    if cov6_after_np.size > 0:
        if isinstance(active, torch.Tensor):
            act = active.detach().to("cpu").numpy().astype(bool)
        else:
            act = np.ones(cov6_after_np.shape[0], dtype=bool)
        idx = np.where(act)[0]
        if idx.size > 0:
            cov33 = _cov6_to_3x3(cov6_after_np[idx])
            try:
                # Use eigvalsh for symmetric matrices; faster + more stable.
                w = np.linalg.eigvalsh(cov33)  # (N, 3)
                w_min = w[:, 0]
                w_max = w[:, -1]
                cond = np.where(w_min > 0, w_max / np.maximum(w_min, 1e-30), np.inf)
                log_cond = np.where(np.isfinite(cond), np.log10(np.maximum(cond, 1.0)), np.nan)
                out["eig_min_pct_active"] = _percentiles(w_min)
                out["eig_max_pct_active"] = _percentiles(w_max)
                out["log10_cond_pct_active"] = _percentiles(log_cond)
                # Indefinite (truly broken) and "below half the SPD floor"
                # (genuine outliers — covs at the conditional ridge floor of
                # ~1e-6 sit slightly under 1e-6 due to fp32 round-trip and
                # would otherwise look alarming).
                out["n_cov_indef_active"] = int((w_min <= 0).sum())
                out["n_cov_eig_lt_5e-7_active"] = int((w_min < 5e-7).sum())
                # Top-K worst-conditioned active objects, for the per-frame
                # ill-cov scatter visualization.
                worst_local = np.argsort(w_min)[: min(16, idx.size)]
                for li in worst_local:
                    if w_min[li] < 1e-6:
                        oi = int(idx[int(li)])
                        ill_cov_top.append({
                            "object_idx": oi,
                            "eig_min": float(w_min[int(li)]),
                            "eig_max": float(w_max[int(li)]),
                            "cov_diag": [float(x) for x in cov6_after_np[oi, [0, 3, 5]].tolist()],
                            "mean": [float(x) for x in means_after_np[oi].tolist()],
                        })
            except Exception as exc:  # noqa: BLE001
                out["eig_compute_error"] = repr(exc)
    if ill_cov_top:
        out["ill_cov_top"] = ill_cov_top

    # Top-K matched objects with the largest position jumps this batch — used
    # to visualize "where did the Gaussian center jump" events.
    if (
        means_before is not None
        and means_before.size > 0
        and matched_object_indices is not None
        and len(matched_object_indices) > 0
    ):
        try:
            n_overlap = min(int(means_before.shape[0]), int(means_after_np.shape[0]))
            mb = means_before[:n_overlap]
            ma = means_after_np[:n_overlap]
            valid = np.isfinite(mb).all(axis=1) & np.isfinite(ma).all(axis=1)
            mset = np.asarray([i for i in matched_object_indices if 0 <= i < n_overlap], dtype=np.int64)
            if mset.size > 0:
                v_idx = mset[valid[mset]]
                if v_idx.size > 0:
                    dist = np.linalg.norm(ma[v_idx] - mb[v_idx], axis=1)
                    order = np.argsort(-dist)[:8]
                    jumps = []
                    for k in order:
                        oi = int(v_idx[int(k)])
                        jumps.append({
                            "object_idx": oi,
                            "delta_m": float(dist[int(k)]),
                            "before_pos": [float(x) for x in mb[oi].tolist()],
                            "after_pos": [float(x) for x in ma[oi].tolist()],
                        })
                    if jumps:
                        out["jumped_objects_top"] = jumps
        except Exception:
            pass

    if isinstance(update_info, dict):
        out["n_new_objects"] = int(len(update_info.get("new_object_indices") or []))
        out["n_merged_objects"] = int(len(update_info.get("merged_objects") or []))

    if isinstance(active, torch.Tensor):
        out["n_active"] = int(active.detach().to("cpu").to(torch.bool).sum().item())

    return out


# ------------------------------------------------------------------------- voxel cloud


def summarize_voxel_cloud(state: dict) -> Dict[str, Any]:
    """Voxel-cloud level + size distribution from the post-update state."""
    if not isinstance(state, dict):
        return {"available": False}
    levels = state.get("object_voxel_levels")
    offsets = state.get("object_voxel_keys_offsets")
    flat = state.get("object_voxel_keys_flat")
    out: Dict[str, Any] = {"available": False}
    if not (isinstance(levels, torch.Tensor) and isinstance(offsets, torch.Tensor)):
        return out
    out["available"] = True
    levels_np = levels.detach().to("cpu", dtype=torch.int64).numpy()
    offsets_np = offsets.detach().to("cpu", dtype=torch.int64).numpy()
    counts = np.diff(offsets_np) if offsets_np.size >= 2 else np.zeros(0, dtype=np.int64)
    out["n_objects_with_voxel_buffer"] = int((counts > 0).sum())
    out["voxels_per_object_pct"] = _percentiles(counts.astype(np.float64), ps=(50, 90, 99))
    if counts.size > 0:
        out["voxels_per_object_max"] = int(counts.max())
        out["voxels_total"] = int(int(flat.numel()) if isinstance(flat, torch.Tensor) else int(counts.sum()))
    # Level histogram only over objects with nonzero buffer to avoid bias.
    if levels_np.size > 0 and counts.size > 0:
        n_overlap = min(levels_np.size, counts.size)
        eligible = counts[:n_overlap] > 0
        lv = levels_np[:n_overlap][eligible]
        if lv.size > 0:
            unique, ct = np.unique(lv, return_counts=True)
            out["level_histogram"] = {int(k): int(v) for k, v in zip(unique.tolist(), ct.tolist())}
    return out


# ------------------------------------------------------------------------- state digest


def state_digest(state: dict) -> Dict[str, Any]:
    """Compact 'how big is the scene graph right now' header."""
    if not isinstance(state, dict):
        return {}
    means = state.get("means")
    active = state.get("active")
    n_total = int(getattr(means, "shape", [0])[0]) if isinstance(means, torch.Tensor) else 0
    n_active = (
        int(active.detach().to("cpu").to(torch.bool).sum().item())
        if isinstance(active, torch.Tensor)
        else 0
    )
    return {"n_total": n_total, "n_active": n_active}
