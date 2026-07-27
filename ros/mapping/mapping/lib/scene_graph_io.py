"""Scene graph I/O utilities for the streaming mapper node.

All functions are pure (no ROS node state). Callers supply scene-state
dicts, threading locks, configuration values, and optional logger callables
instead of a ``self`` reference.
"""

from __future__ import annotations

import contextlib
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scene_graph.storage.models import ImageRecord
from mapping_msgs.msg import DetectedObject, DetectedObjects

try:
    import h5py
except Exception:  # pragma: no cover - optional dependency
    h5py = None

from mapping.lib.covisibility import compute_covisibility_filtered_neighbors_indices
from mapping.lib.embedding_io import (
    bbox_area_xyxy_clamped,
    observation_to_hwc_uint8,
    pack_embedding_history_matrix,
    pack_embedding_matrix,
    pack_text_history,
)
from mapping.lib.geometric_transforms import xyz_quat_from_matrix


# ---------------------------------------------------------------------------
# Detected-objects message builder
# ---------------------------------------------------------------------------


def build_detected_objects(seg_outputs: dict) -> Optional[DetectedObjects]:
    """Build a DetectedObjects ROS message from segmentation outputs."""
    means = seg_outputs.get("means")
    cov6s = seg_outputs.get("cov6")
    scores = seg_outputs.get("scores")
    if means is None or cov6s is None or scores is None:
        return None

    try:
        means_list = means.detach().cpu().tolist()
        cov6s_list = cov6s.detach().cpu().tolist()
        scores_list = scores.detach().cpu().tolist()
    except Exception:
        return None

    msg = DetectedObjects()
    for m, c, s in zip(means_list, cov6s_list, scores_list):
        obj = DetectedObject()
        obj.mean = m
        obj.cov6 = c
        obj.score = s
        msg.objects.append(obj)
    return msg


# ---------------------------------------------------------------------------
# Timed-persist checkpoint builder
# ---------------------------------------------------------------------------


def build_timed_persist_checkpoints(max_time_sec: float) -> List[Dict[str, object]]:
    """Build one-shot checkpoint triggers keyed off remaining mission time.

    Fraction checkpoints are derived from ``max_time_sec``:
    - 0.25 * max_time fires when remaining <= 0.75 * max_time, etc.

    Absolute checkpoints are keyed directly off remaining time (seconds):
    - remaining <= 3000, 2100, 1500, 720, 300, 200, 100, 20.
    """
    candidate_pairs_remaining: List[Tuple[float, str]] = [
        (3000.0, "remaining_le_3000s"),
        (2100.0, "remaining_le_2100s"),
        (1500.0, "remaining_le_1500s"),
        (720.0, "remaining_le_720s"),
        (300.0, "remaining_le_300s"),
        (200.0, "remaining_le_200s"),
        (100.0, "remaining_le_100s"),
        (20.0, "remaining_le_20s"),
    ]

    if math.isfinite(max_time_sec) and max_time_sec > 0.0:
        candidate_pairs_remaining.extend([
            (0.75 * max_time_sec, "0.25x_max_time"),
            (0.5 * max_time_sec, "0.50x_max_time"),
            (0.25 * max_time_sec, "0.75x_max_time"),
        ])

    grouped: Dict[int, Dict[str, object]] = {}
    for remaining_sec, label in candidate_pairs_remaining:
        if not math.isfinite(remaining_sec) or remaining_sec <= 0.0:
            continue
        remaining_ms = int(round(remaining_sec * 1000.0))
        checkpoint = grouped.get(remaining_ms)
        if checkpoint is None:
            grouped[remaining_ms] = {
                "remaining_sec": float(remaining_ms) / 1000.0,
                "labels": [label],
                "fired": False,
            }
        else:
            labels = checkpoint.get("labels", [])
            if isinstance(labels, list):
                labels.append(label)

    return [grouped[key] for key in sorted(grouped.keys(), reverse=True)]


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------


def maybe_save_published_scene_graph_json(
    json_payload: str,
    *,
    save_enabled: bool,
    save_path: str,
    logger_info: Optional[Callable[[str], None]] = None,
    logger_warn: Optional[Callable[[str], None]] = None,
) -> bool:
    """Write scene-graph JSON to disk. Returns True on success."""
    if not save_enabled:
        return False
    raw = str(save_path or "").strip()
    if not raw:
        return False
    try:
        path = Path(raw).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json_payload if json_payload.endswith("\n") else (json_payload + "\n")
        path.write_text(payload, encoding="utf-8")
        if logger_info:
            logger_info(f"✓ Saved scene graph JSON to: {path}")
        return True
    except Exception as exc:
        if logger_warn:
            with contextlib.suppress(Exception):
                logger_warn(f"✗ Failed to save published scene graph JSON to {raw}: {exc}")
        return False


# ---------------------------------------------------------------------------
# In-memory snapshot (JSON-serialisable + tensor context)
# ---------------------------------------------------------------------------


def take_scene_graph_snapshot(
    scene_state: dict,
    *,
    logger_info: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Dict[str, object]], Optional[Dict[str, object]]]:
    """Extract a JSON-serialisable object list and snapshot context from scene state.

    Returns:
        ``(objects, snapshot_ctx)`` where ``objects`` is ready for JSON
        serialisation and ``snapshot_ctx`` carries tensors/arrays needed by
        :func:`write_scene_graph_snapshot`.  Both are empty/None when the
        state has no active captioned objects.
    """
    t0 = time.perf_counter()
    state = scene_state
    active = state.get("active")
    means = state.get("means")
    obj_ids = state.get("object_id")
    cov6 = state.get("cov6")
    captions: List[str] = state.get("object_caption", [])
    rgb_observations = state.get("rgb_observations", []) or []
    object_image_ids = state.get("object_image_ids", [])
    viewpoint_image_ids = state.get("viewpoint_image_ids", [])
    images_meta: List[ImageRecord] = state.get("images", [])
    caption_embeddings: List[List[float]] = state.get("object_caption_embedding", []) or []
    siglip2_embeddings: List[List[float]] = state.get("object_siglip2_embedding", []) or []
    qwen3_vl_embeddings: List[List[float]] = state.get("object_qwen3_vl_embedding", []) or []
    caption_history: List[List[str]] = state.get("object_caption_history", []) or []
    caption_embedding_history: List[List[List[float]]] = state.get("object_caption_embedding_history", []) or []
    siglip2_embedding_history: List[List[List[float]]] = state.get("object_siglip2_embedding_history", []) or []
    qwen3_vl_embedding_history: List[List[List[float]]] = state.get("object_qwen3_vl_embedding_history", []) or []

    if active is None or means is None or obj_ids is None:
        return [], None
    if not hasattr(active, "numel") or active.numel() == 0:
        return [], None

    try:
        active_cpu = active.detach().to("cpu", copy=False).to(torch.bool)
        means_cpu = means.detach().to("cpu", copy=False)
        obj_ids_cpu = obj_ids.detach().to("cpu", copy=False)
        cov6_cpu = cov6.detach().to("cpu", copy=False) if isinstance(cov6, torch.Tensor) else None
    except Exception:
        return [], None

    nan_mask = torch.isnan(means_cpu).any(dim=1)
    nan_count = int(nan_mask.sum().item())

    if logger_info:
        logger_info(f"NaN means count: {nan_count}")
        logger_info(f"Total means count: {means_cpu.shape[0] - nan_count}")

    limit = min(active_cpu.numel(), means_cpu.shape[0], obj_ids_cpu.shape[0])
    if limit == 0:
        return [], None
    active_count = int(active_cpu[:limit].sum().item())
    no_caption_count = 0
    for idx in range(limit):
        if not bool(active_cpu[idx]):
            continue
        caption_raw = captions[idx] if idx < len(captions) else None
        caption = str(caption_raw).strip() if caption_raw is not None else ""
        if not caption:
            no_caption_count += 1
    if logger_info:
        logger_info(f"Active objects count: {active_count}")
        logger_info(f"Active objects without caption: {no_caption_count}")

    objects: List[Dict[str, object]] = []
    include_idx: List[int] = []
    for idx in range(limit):
        if nan_mask[idx] or not bool(active_cpu[idx]):
            continue
        mean_vec = means_cpu[idx]
        mean_list = mean_vec.tolist() if hasattr(mean_vec, "tolist") else []
        caption_raw = captions[idx] if idx < len(captions) else None
        caption = str(caption_raw).strip() if caption_raw is not None else ""
        if not caption:
            continue
        viewpoints_payload: List[List[float]] = []
        viewpoint_camera_ids: List[str] = []
        if idx < len(viewpoint_image_ids):
            for image_id in viewpoint_image_ids[idx]:
                try:
                    image_id_int = int(image_id)
                except Exception:
                    continue
                if image_id_int < 0 or image_id_int >= len(images_meta):
                    continue
                pose = getattr(images_meta[image_id_int], "pose", None)
                if pose is None:
                    continue
                try:
                    pose_np = (
                        pose.detach().to("cpu", copy=False).numpy()
                        if isinstance(pose, torch.Tensor)
                        else np.asarray(pose)
                    )
                except Exception:
                    continue
                if pose_np.shape != (4, 4):
                    continue
                payload = xyz_quat_from_matrix(pose_np)
                if payload is None:
                    continue
                xyz, quat = payload
                viewpoints_payload.append(xyz + quat)
                viewpoint_camera_ids.append(getattr(images_meta[image_id_int], "camera_id", "") or "")
        objects.append({
            "object_id": int(obj_ids_cpu[idx].item()),
            "mean": mean_list,
            "object_caption": caption,
            "viewpoints": viewpoints_payload,
            "viewpoint_camera_ids": viewpoint_camera_ids,
        })
        include_idx.append(int(idx))

    t1 = time.perf_counter()
    if logger_info:
        logger_info(f"Scene graph objects (active + captioned): {len(objects)}")
        logger_info(f"scene graph to json time: {t1 - t0}")

    snapshot_ctx = {
        "include_idx": include_idx,
        "limit": limit,
        "active_cpu": active_cpu,
        "means_cpu": means_cpu,
        "cov6_cpu": cov6_cpu,
        "obj_ids_cpu": obj_ids_cpu,
        "captions": captions,
        "rgb_observations": rgb_observations,
        "object_image_ids": object_image_ids,
        "viewpoint_image_ids": viewpoint_image_ids,
        "images_meta": images_meta,
        "caption_embeddings": caption_embeddings,
        "siglip2_embeddings": siglip2_embeddings,
        "qwen3_vl_embeddings": qwen3_vl_embeddings,
        "caption_history": caption_history,
        "caption_embedding_history": caption_embedding_history,
        "siglip2_embedding_history": siglip2_embedding_history,
        "qwen3_vl_embedding_history": qwen3_vl_embedding_history,
    }
    return objects, snapshot_ctx


# ---------------------------------------------------------------------------
# HDF5 snapshot writer
# ---------------------------------------------------------------------------


def write_scene_graph_snapshot(
    *,
    snapshot_dir: Path,
    graph_version: int,
    lock: threading.Lock,
    save_enabled: bool,
    include_idx: Sequence[int],
    limit: int,
    active_cpu: torch.Tensor,
    means_cpu: torch.Tensor,
    cov6_cpu: Optional[torch.Tensor],
    obj_ids_cpu: torch.Tensor,
    rgb_observations: Sequence[Any],
    object_image_ids: Sequence[Sequence[int]],
    viewpoint_image_ids: Sequence[Sequence[int]],
    images_meta: Sequence[ImageRecord],
    caption_embeddings: Sequence[Sequence[float]],
    siglip2_embeddings: Sequence[Sequence[float]],
    qwen3_vl_embeddings: Sequence[Sequence[float]],
    caption_history: Sequence[Sequence[str]],
    caption_embedding_history: Sequence[Sequence[Sequence[float]]],
    siglip2_embedding_history: Sequence[Sequence[Sequence[float]]],
    qwen3_vl_embedding_history: Sequence[Sequence[Sequence[float]]],
    object_captions: Sequence[str],
    scene_state: dict,
    covisibility_enabled: bool,
    covisibility_lock: threading.Lock,
    hellinger_thresh: float,
    stamp_sec: int,
    stamp_nanosec: int,
    logger_info: Optional[Callable[[str], None]] = None,
    logger_warn: Optional[Callable[[str], None]] = None,
) -> Tuple[int, Optional[Dict[str, object]]]:
    """Write a versioned HDF5 scene graph snapshot to disk.

    Returns:
        ``(new_graph_version, result_dict)`` where ``new_graph_version`` is
        one past the version written and ``result_dict`` is ``None`` on failure.
        The caller is responsible for persisting ``new_graph_version`` back to
        node state (e.g. ``self._scene_graph_snapshot_version = new_graph_version``).
    """
    if not save_enabled:
        return graph_version, None
    if snapshot_dir is None:
        return graph_version, None
    if h5py is None:
        if logger_warn:
            with contextlib.suppress(Exception):
                logger_warn("h5py unavailable; skipping scene graph snapshot save.")
        return graph_version, None
    if not include_idx:
        return graph_version, None
    if (
        not isinstance(active_cpu, torch.Tensor)
        or not isinstance(means_cpu, torch.Tensor)
        or not isinstance(obj_ids_cpu, torch.Tensor)
    ):
        return graph_version, None

    if not lock.acquire(blocking=False):
        return graph_version, None

    try:
        version_dir = snapshot_dir / f"v{graph_version:06d}"
        while version_dir.exists():
            graph_version += 1
            version_dir = snapshot_dir / f"v{graph_version:06d}"
        new_version = graph_version + 1

        version_dir.mkdir(parents=True, exist_ok=False)

        object_count = len(include_idx)
        image_count = len(images_meta)

        obj_ids = obj_ids_cpu[:limit].view(-1).tolist()
        obj_ids_selected = [int(obj_ids[idx]) for idx in include_idx if 0 <= idx < len(obj_ids)]

        caption_mat, caption_dim = pack_embedding_matrix(caption_embeddings, include_idx)
        siglip2_mat, siglip2_dim = pack_embedding_matrix(siglip2_embeddings, include_idx)
        qwen3_vl_mat, qwen3_vl_dim = pack_embedding_matrix(qwen3_vl_embeddings, include_idx)
        caption_hist_row_ptr, caption_hist_mat, caption_hist_dim = pack_embedding_history_matrix(
            caption_embedding_history, include_idx, caption_embeddings,
        )
        siglip2_hist_row_ptr, siglip2_hist_mat, siglip2_hist_dim = pack_embedding_history_matrix(
            siglip2_embedding_history, include_idx, siglip2_embeddings,
        )
        qwen3_vl_hist_row_ptr, qwen3_vl_hist_mat, qwen3_vl_hist_dim = pack_embedding_history_matrix(
            qwen3_vl_embedding_history, include_idx, qwen3_vl_embeddings,
        )
        caption_text_hist_row_ptr, caption_text_hist = pack_text_history(
            caption_history,
            include_idx,
            [str(x or "") for x in object_captions],
        )

        # Legacy single crop datasets (for backward compatibility).
        crop_row_ptr: List[int] = [0]
        crop_data_parts: List[np.ndarray] = []
        crop_hw = np.zeros((object_count, 2), dtype=np.int32)
        crop_channels = np.zeros((object_count,), dtype=np.int32)
        crop_image_ids = np.full((object_count,), -1, dtype=np.int64)
        crop_bbox_xyxy = np.full((object_count, 4), np.nan, dtype=np.float32)
        crop_encoding: List[str] = [""] * object_count
        # New list crop datasets: object -> [crop items].
        crop_item_row_ptr: List[int] = [0]
        crop_item_flat_row_ptr: List[int] = [0]
        crop_item_data_parts: List[np.ndarray] = []
        crop_item_hw_rows: List[List[int]] = []
        crop_item_channels_rows: List[int] = []
        crop_item_image_ids_rows: List[int] = []
        crop_item_bbox_rows: List[List[float]] = []
        crop_item_encoding_rows: List[str] = []

        for row_idx, obj_idx in enumerate(include_idx):
            obs_list = rgb_observations[obj_idx] if obj_idx < len(rgb_observations) else []
            selected_img: Optional[np.ndarray] = None
            selected_image_id = -1
            selected_bbox = np.full((4,), np.nan, dtype=np.float32)
            selected_encoding = ""

            obs_candidates = list(obs_list) if isinstance(obs_list, (list, tuple)) else [obs_list]
            best_area = -1.0
            best_entry: Optional[Tuple[np.ndarray, int, np.ndarray, str]] = None
            best_shape: Optional[Tuple[int, int, int]] = None
            for obs_entry in obs_candidates:
                image_arr, obs_image_id, bbox_xyxy, obs_encoding = observation_to_hwc_uint8(obs_entry)
                if image_arr is None:
                    continue
                height_item = int(image_arr.shape[0])
                width_item = int(image_arr.shape[1])
                channels_item = int(image_arr.shape[2]) if image_arr.ndim == 3 else 1
                area = bbox_area_xyxy_clamped(bbox_xyxy, width_item, height_item) or 0.0
                if area > best_area:
                    best_area = float(area)
                    best_entry = (image_arr, int(obs_image_id), bbox_xyxy, str(obs_encoding or ""))
                    best_shape = (height_item, width_item, channels_item)

            if best_entry is not None and best_shape is not None:
                image_arr, obs_image_id, bbox_xyxy, obs_encoding = best_entry
                height_item, width_item, channels_item = best_shape
                flat_item = image_arr.reshape(-1).astype(np.uint8, copy=False)
                crop_item_data_parts.append(flat_item)
                crop_item_flat_row_ptr.append(crop_item_flat_row_ptr[-1] + int(flat_item.size))
                crop_item_hw_rows.append([height_item, width_item])
                crop_item_channels_rows.append(channels_item)
                crop_item_image_ids_rows.append(int(obs_image_id))
                crop_item_bbox_rows.append([float(x) for x in np.asarray(bbox_xyxy, dtype=np.float32).tolist()])
                crop_item_encoding_rows.append(str(obs_encoding or ""))
                selected_img = image_arr
                selected_image_id = int(obs_image_id)
                selected_bbox = bbox_xyxy
                selected_encoding = obs_encoding

            crop_item_row_ptr.append(crop_item_row_ptr[-1] + (1 if best_entry is not None else 0))

            if selected_img is None:
                crop_row_ptr.append(crop_row_ptr[-1])
                continue

            height, width = int(selected_img.shape[0]), int(selected_img.shape[1])
            channels = int(selected_img.shape[2]) if selected_img.ndim == 3 else 1
            flat = selected_img.reshape(-1).astype(np.uint8, copy=False)
            crop_data_parts.append(flat)
            crop_hw[row_idx, 0] = height
            crop_hw[row_idx, 1] = width
            crop_channels[row_idx] = channels
            crop_image_ids[row_idx] = selected_image_id
            crop_bbox_xyxy[row_idx, :] = selected_bbox
            crop_encoding[row_idx] = selected_encoding
            crop_row_ptr.append(crop_row_ptr[-1] + int(flat.size))

        crop_data = np.concatenate(crop_data_parts, axis=0) if crop_data_parts else np.empty((0,), dtype=np.uint8)
        crop_item_data = (
            np.concatenate(crop_item_data_parts, axis=0) if crop_item_data_parts else np.empty((0,), dtype=np.uint8)
        )
        crop_item_hw = (
            np.asarray(crop_item_hw_rows, dtype=np.int32) if crop_item_hw_rows else np.zeros((0, 2), dtype=np.int32)
        )
        crop_item_channels = np.asarray(crop_item_channels_rows, dtype=np.int32)
        crop_item_image_ids = np.asarray(crop_item_image_ids_rows, dtype=np.int64)
        crop_item_bbox_xyxy = (
            np.asarray(crop_item_bbox_rows, dtype=np.float32)
            if crop_item_bbox_rows
            else np.zeros((0, 4), dtype=np.float32)
        )

        vp_row_ptr: List[int] = [0]
        vp_image_ids: List[int] = []
        obj_img_row_ptr: List[int] = [0]
        obj_img_ids: List[int] = []
        for idx in include_idx:
            object_row = object_image_ids[idx] if idx < len(object_image_ids) else []
            row = viewpoint_image_ids[idx] if idx < len(viewpoint_image_ids) else []
            if not isinstance(object_row, Sequence):
                object_row = []
            if not isinstance(row, Sequence):
                row = []
            obj_added = 0
            added = 0
            for image_id in object_row:
                try:
                    obj_img_ids.append(int(image_id))
                    obj_added += 1
                except Exception:
                    continue
            for image_id in row:
                try:
                    vp_image_ids.append(int(image_id))
                    added += 1
                except Exception:
                    continue
            obj_img_row_ptr.append(obj_img_row_ptr[-1] + obj_added)
            vp_row_ptr.append(vp_row_ptr[-1] + added)
        object_image_count = len(obj_img_ids)
        viewpoint_count = len(vp_image_ids)

        image_ids_list: List[int] = []
        image_poses = np.full((image_count, 4, 4), np.nan, dtype=np.float32)
        image_camera_ids: List[str] = []
        image_storage_paths: List[str] = []
        for idx, record in enumerate(images_meta):
            image_ids_list.append(int(getattr(record, "image_id", idx)))
            pose = getattr(record, "pose", None)
            if pose is not None:
                try:
                    pose_np = (
                        pose.detach().to("cpu", copy=False).numpy()
                        if isinstance(pose, torch.Tensor)
                        else np.asarray(pose)
                    )
                    if pose_np.shape == (4, 4):
                        image_poses[idx] = pose_np.astype(np.float32, copy=False)
                except Exception:
                    pass
            image_camera_ids.append(str(getattr(record, "camera_id", "") or ""))
            image_storage_paths.append(str(getattr(record, "storage_path", "") or ""))

        covis_blocks = 0
        covis_adj = np.zeros((object_count, 0), dtype=np.uint64)
        if covisibility_enabled and cov6_cpu is not None:
            finite_mask = (
                torch.isfinite(means_cpu).all(dim=1)
                if means_cpu.ndim == 2 and means_cpu.shape[1] == 3
                else None
            )
            if finite_mask is not None and finite_mask.numel() == active_cpu.numel():
                include_mask = active_cpu & finite_mask
                neighbors_by_obj = compute_covisibility_filtered_neighbors_indices(
                    state=scene_state,
                    covisibility_lock=covisibility_lock,
                    include_idx=[int(x) for x in include_idx],
                    include_mask=include_mask,
                    means_cpu=means_cpu,
                    cov6_cpu=cov6_cpu,
                    limit=limit,
                    hellinger_thresh=hellinger_thresh,
                )
                covis_blocks = int((object_count + 63) // 64)
                covis_adj = np.zeros((object_count, covis_blocks), dtype=np.uint64)
                index_map = {int(full_idx): row for row, full_idx in enumerate(include_idx)}
                for full_i, neighbors in neighbors_by_obj.items():
                    row_i = index_map.get(int(full_i))
                    if row_i is None:
                        continue
                    for full_j in neighbors:
                        row_j = index_map.get(int(full_j))
                        if row_j is None or row_j == row_i:
                            continue
                        a, b = (row_i, row_j) if row_i < row_j else (row_j, row_i)
                        covis_adj[a, b // 64] |= np.uint64(1) << np.uint64(b % 64)

        state_tmp = version_dir / "state.h5.tmp"
        state_path = version_dir / "state.h5"
        with h5py.File(state_tmp, "w", libver="latest") as handle:

            def _create_numeric(name: str, data: np.ndarray) -> None:
                if isinstance(data, np.ndarray) and data.size == 0:
                    handle.create_dataset(name, data=data)
                else:
                    handle.create_dataset(name, data=data, compression="lzf", chunks=True)

            handle.create_dataset("object_id", data=np.asarray(obj_ids_selected, dtype=np.int64))
            _create_numeric("caption_embedding", caption_mat)
            _create_numeric("siglip2_embedding", siglip2_mat)
            _create_numeric("qwen3_vl_embedding", qwen3_vl_mat)
            handle.create_dataset("caption_embedding_hist_row_ptr", data=caption_hist_row_ptr)
            _create_numeric("caption_embedding_hist", caption_hist_mat)
            handle.create_dataset("siglip2_embedding_hist_row_ptr", data=siglip2_hist_row_ptr)
            _create_numeric("siglip2_embedding_hist", siglip2_hist_mat)
            handle.create_dataset("qwen3_vl_embedding_hist_row_ptr", data=qwen3_vl_hist_row_ptr)
            _create_numeric("qwen3_vl_embedding_hist", qwen3_vl_hist_mat)
            handle.create_dataset("caption_crop_row_ptr", data=np.asarray(crop_row_ptr, dtype=np.int64))
            _create_numeric("caption_crop_data", crop_data)
            _create_numeric("caption_crop_hw", crop_hw)
            _create_numeric("caption_crop_channels", crop_channels)
            _create_numeric("caption_crop_image_id", crop_image_ids)
            _create_numeric("caption_crop_bbox_xyxy", crop_bbox_xyxy)
            handle.create_dataset(
                "caption_crop_item_row_ptr", data=np.asarray(crop_item_row_ptr, dtype=np.int64)
            )
            handle.create_dataset(
                "caption_crop_item_flat_row_ptr", data=np.asarray(crop_item_flat_row_ptr, dtype=np.int64)
            )
            _create_numeric("caption_crop_item_data", crop_item_data)
            _create_numeric("caption_crop_item_hw", crop_item_hw)
            _create_numeric("caption_crop_item_channels", crop_item_channels)
            _create_numeric("caption_crop_item_image_id", crop_item_image_ids)
            _create_numeric("caption_crop_item_bbox_xyxy", crop_item_bbox_xyxy)
            handle.create_dataset("object_image_row_ptr", data=np.asarray(obj_img_row_ptr, dtype=np.int64))
            handle.create_dataset("object_image_ids", data=np.asarray(obj_img_ids, dtype=np.int64))
            handle.create_dataset("viewpoint_row_ptr", data=np.asarray(vp_row_ptr, dtype=np.int64))
            handle.create_dataset("viewpoint_image_ids", data=np.asarray(vp_image_ids, dtype=np.int64))
            handle.create_dataset("image_ids", data=np.asarray(image_ids_list, dtype=np.int64))
            _create_numeric("image_poses", image_poses)
            str_dtype = h5py.string_dtype(encoding="utf-8")
            handle.create_dataset("caption_text_hist_row_ptr", data=caption_text_hist_row_ptr)
            handle.create_dataset("caption_text_hist", data=caption_text_hist, dtype=str_dtype)
            handle.create_dataset("caption_crop_encoding", data=crop_encoding, dtype=str_dtype)
            handle.create_dataset("caption_crop_item_encoding", data=crop_item_encoding_rows, dtype=str_dtype)
            handle.create_dataset("image_camera_ids", data=image_camera_ids, dtype=str_dtype)
            handle.create_dataset("image_storage_paths", data=image_storage_paths, dtype=str_dtype)
            handle.create_dataset("covisibility_blocks", data=np.asarray([covis_blocks], dtype=np.int64))
            handle.create_dataset("covisibility_adj_u64", data=covis_adj.astype(np.uint64, copy=False))

            # Per-object sparse voxel cloud (CSR-flat layout). Sliced by
            # include_idx so on-disk indices align with the rest of the
            # snapshot. Empty arrays when the buffer is absent.
            voxel_keys_t = scene_state.get("object_voxel_keys_flat")
            voxel_offsets_t = scene_state.get("object_voxel_keys_offsets")
            voxel_levels_t = scene_state.get("object_voxel_levels")
            sliced_voxel_keys: List[np.ndarray] = []
            sliced_voxel_levels: List[int] = []
            if (
                isinstance(voxel_keys_t, torch.Tensor)
                and isinstance(voxel_offsets_t, torch.Tensor)
                and isinstance(voxel_levels_t, torch.Tensor)
                and voxel_offsets_t.numel() > 1
            ):
                keys_np = voxel_keys_t.detach().cpu().numpy().astype(np.int64, copy=False)
                offsets_np = voxel_offsets_t.detach().cpu().numpy().astype(np.int64, copy=False)
                levels_np = voxel_levels_t.detach().cpu().numpy().astype(np.int8, copy=False)
                n_persisted = max(0, offsets_np.shape[0] - 1)
                for full_idx in include_idx:
                    fi = int(full_idx)
                    if 0 <= fi < n_persisted:
                        s, e = int(offsets_np[fi]), int(offsets_np[fi + 1])
                        sliced_voxel_keys.append(keys_np[s:e])
                        sliced_voxel_levels.append(int(levels_np[fi]) if fi < levels_np.shape[0] else 0)
                    else:
                        sliced_voxel_keys.append(np.empty((0,), dtype=np.int64))
                        sliced_voxel_levels.append(0)
            voxel_counts = np.asarray([k.size for k in sliced_voxel_keys], dtype=np.int64)
            voxel_row_ptr = np.zeros((len(sliced_voxel_keys) + 1,), dtype=np.int64)
            if voxel_counts.size:
                voxel_row_ptr[1:] = np.cumsum(voxel_counts)
            voxel_keys_flat = (
                np.concatenate(sliced_voxel_keys, axis=0)
                if sliced_voxel_keys
                else np.empty((0,), dtype=np.int64)
            )
            voxel_levels_arr = (
                np.asarray(sliced_voxel_levels, dtype=np.int8)
                if sliced_voxel_levels
                else np.empty((0,), dtype=np.int8)
            )
            handle.create_dataset("object_voxel_row_ptr", data=voxel_row_ptr)
            _create_numeric("object_voxel_keys", voxel_keys_flat)
            handle.create_dataset("object_voxel_levels", data=voxel_levels_arr)
        state_tmp.replace(state_path)

        manifest = {
            "graph_version": int(graph_version),
            "created_unix_s": float(time.time()),
            "object_count": int(object_count),
            "image_count": int(image_count),
            "object_image_count": int(object_image_count),
            "viewpoint_count": int(viewpoint_count),
            "caption_embedding_dim": int(caption_dim),
            "siglip2_embedding_dim": int(siglip2_dim),
            "qwen3_vl_embedding_dim": int(qwen3_vl_dim),
            "caption_embedding_hist_dim": int(caption_hist_dim),
            "siglip2_embedding_hist_dim": int(siglip2_hist_dim),
            "qwen3_vl_embedding_hist_dim": int(qwen3_vl_hist_dim),
            "covisibility_blocks": int(covis_blocks),
            "datasets": {
                "object_id": {"shape": [object_count], "dtype": "int64"},
                "caption_embedding": {"shape": [object_count, caption_dim], "dtype": "float32"},
                "siglip2_embedding": {"shape": [object_count, siglip2_dim], "dtype": "float32"},
                "qwen3_vl_embedding": {"shape": [object_count, qwen3_vl_dim], "dtype": "float32"},
                "caption_embedding_hist_row_ptr": {"shape": [object_count + 1], "dtype": "int64"},
                "caption_embedding_hist": {
                    "shape": [int(caption_hist_mat.shape[0]), caption_hist_dim],
                    "dtype": "float32",
                },
                "siglip2_embedding_hist_row_ptr": {"shape": [object_count + 1], "dtype": "int64"},
                "siglip2_embedding_hist": {
                    "shape": [int(siglip2_hist_mat.shape[0]), siglip2_hist_dim],
                    "dtype": "float32",
                },
                "qwen3_vl_embedding_hist_row_ptr": {"shape": [object_count + 1], "dtype": "int64"},
                "qwen3_vl_embedding_hist": {
                    "shape": [int(qwen3_vl_hist_mat.shape[0]), qwen3_vl_hist_dim],
                    "dtype": "float32",
                },
                "caption_crop_row_ptr": {"shape": [object_count + 1], "dtype": "int64"},
                "caption_crop_data": {"shape": [int(crop_data.shape[0])], "dtype": "uint8"},
                "caption_crop_hw": {"shape": [object_count, 2], "dtype": "int32"},
                "caption_crop_channels": {"shape": [object_count], "dtype": "int32"},
                "caption_crop_image_id": {"shape": [object_count], "dtype": "int64"},
                "caption_crop_bbox_xyxy": {"shape": [object_count, 4], "dtype": "float32"},
                "caption_crop_encoding": {"shape": [object_count], "dtype": "string"},
                "caption_crop_item_row_ptr": {"shape": [object_count + 1], "dtype": "int64"},
                "caption_crop_item_flat_row_ptr": {
                    "shape": [len(crop_item_flat_row_ptr)],
                    "dtype": "int64",
                },
                "caption_crop_item_data": {"shape": [int(crop_item_data.shape[0])], "dtype": "uint8"},
                "caption_crop_item_hw": {"shape": [int(crop_item_hw.shape[0]), 2], "dtype": "int32"},
                "caption_crop_item_channels": {
                    "shape": [int(crop_item_channels.shape[0])],
                    "dtype": "int32",
                },
                "caption_crop_item_image_id": {
                    "shape": [int(crop_item_image_ids.shape[0])],
                    "dtype": "int64",
                },
                "caption_crop_item_bbox_xyxy": {
                    "shape": [int(crop_item_bbox_xyxy.shape[0]), 4],
                    "dtype": "float32",
                },
                "caption_crop_item_encoding": {
                    "shape": [len(crop_item_encoding_rows)],
                    "dtype": "string",
                },
                "caption_text_hist_row_ptr": {"shape": [object_count + 1], "dtype": "int64"},
                "caption_text_hist": {"shape": [len(caption_text_hist)], "dtype": "string"},
                "object_image_row_ptr": {"shape": [object_count + 1], "dtype": "int64"},
                "object_image_ids": {"shape": [object_image_count], "dtype": "int64"},
                "viewpoint_row_ptr": {"shape": [object_count + 1], "dtype": "int64"},
                "viewpoint_image_ids": {"shape": [viewpoint_count], "dtype": "int64"},
                "image_ids": {"shape": [image_count], "dtype": "int64"},
                "image_poses": {"shape": [image_count, 4, 4], "dtype": "float32"},
                "image_camera_ids": {"shape": [image_count], "dtype": "string"},
                "image_storage_paths": {"shape": [image_count], "dtype": "string"},
                "covisibility_blocks": {"shape": [1], "dtype": "int64"},
                "covisibility_adj_u64": {"shape": [object_count, covis_blocks], "dtype": "uint64"},
            },
            "paths": {
                "state_h5": str(state_path),
            },
        }

        manifest_tmp = version_dir / "manifest.json.tmp"
        manifest_path = version_dir / "manifest.json"
        manifest_tmp.write_text(json.dumps(manifest), encoding="utf-8")
        manifest_tmp.replace(manifest_path)

        latest = {
            "graph_version": int(graph_version),
            "manifest_path": str(manifest_path),
            "state_path": str(state_path),
            "snapshot_dir": str(version_dir),
            "stamp": {"sec": int(stamp_sec), "nanosec": int(stamp_nanosec)},
            "created_unix_s": float(time.time()),
        }
        latest_tmp = snapshot_dir / "latest.json.tmp"
        latest_path = snapshot_dir / "latest.json"
        latest_tmp.write_text(json.dumps(latest), encoding="utf-8")
        latest_tmp.replace(latest_path)

        if logger_info:
            logger_info(
                f"✓ Wrote scene graph snapshot to: {version_dir} "
                f"(manifest: {manifest_path}, state: {state_path})"
            )
        return new_version, {
            "graph_version": int(graph_version),
            "manifest_path": str(manifest_path),
            "state_path": str(state_path),
            "latest_path": str(latest_path),
            "snapshot_dir": str(version_dir),
            "stamp": {"sec": int(stamp_sec), "nanosec": int(stamp_nanosec)},
        }
    except Exception as exc:
        if logger_warn:
            with contextlib.suppress(Exception):
                logger_warn(f"✗ Failed to save scene graph snapshot to {snapshot_dir}: {exc}")
        return graph_version, None
    finally:
        with contextlib.suppress(Exception):
            lock.release()
