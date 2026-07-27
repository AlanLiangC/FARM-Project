"""Captioning utilities for the streaming mapper node.

All functions are pure (no ROS node state). Callers pass in the scene-state
dict, threading constructs, and configuration values rather than a ``self``
reference. Logger callables are optional; pass ``node.get_logger().info`` etc.
to route messages through the ROS logger.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Build helpers — read from scene_state
# ---------------------------------------------------------------------------


def build_local_caption_texts(scene_state: dict, det_idx: object, count: int) -> List[str]:
    """Return per-detection caption strings looked up from *scene_state*."""
    if count <= 0:
        return []
    captions_state = scene_state.get("object_caption", []) or []
    texts = [""] * count

    det_idx_list: List[int] = []
    if det_idx is None:
        return texts
    if isinstance(det_idx, torch.Tensor):
        try:
            det_idx_list = det_idx.detach().to("cpu", copy=False).tolist()
        except Exception:
            det_idx_list = []
    else:
        try:
            det_idx_list = list(det_idx)
        except Exception:
            det_idx_list = []

    limit = min(count, len(det_idx_list))
    for det_i in range(limit):
        try:
            obj_idx = int(det_idx_list[det_i])
        except Exception:
            continue
        if obj_idx < 0 or obj_idx >= len(captions_state):
            continue
        caption_raw = captions_state[obj_idx]
        caption = str(caption_raw).strip() if caption_raw is not None else ""
        if caption:
            texts[det_i] = caption
    return texts


def build_local_caption_embeddings(scene_state: dict, det_idx: object, count: int) -> List[List[float]]:
    """Return per-detection caption embedding vectors looked up from *scene_state*."""
    if count <= 0:
        return []

    embeddings_state = scene_state.get("object_caption_embedding", []) or []
    out: List[List[float]] = [[] for _ in range(count)]

    det_idx_list: List[int] = []
    if det_idx is None:
        return out
    if isinstance(det_idx, torch.Tensor):
        try:
            det_idx_list = det_idx.detach().to("cpu", copy=False).tolist()
        except Exception:
            det_idx_list = []
    else:
        try:
            det_idx_list = list(det_idx)
        except Exception:
            det_idx_list = []

    limit = min(count, len(det_idx_list))
    for det_i in range(limit):
        try:
            obj_idx = int(det_idx_list[det_i])
        except Exception:
            continue
        if obj_idx < 0 or obj_idx >= len(embeddings_state):
            continue

        emb_raw = embeddings_state[obj_idx]
        if emb_raw is None:
            continue
        if isinstance(emb_raw, torch.Tensor):
            with contextlib.suppress(Exception):
                out[det_i] = (
                    emb_raw.detach().to("cpu", copy=False).to(torch.float32).view(-1).tolist()
                    if emb_raw.numel()
                    else []
                )
            continue
        if isinstance(emb_raw, np.ndarray):
            with contextlib.suppress(Exception):
                out[det_i] = emb_raw.astype(np.float32, copy=False).reshape(-1).tolist()
            continue
        if isinstance(emb_raw, Sequence) and not isinstance(emb_raw, (str, bytes)):
            with contextlib.suppress(Exception):
                out[det_i] = [float(x) for x in emb_raw]
            continue
    return out


def build_local_caption_object_ids(scene_state: dict, det_idx: object, count: int) -> List[int]:
    """Return per-detection matched object IDs looked up from *scene_state*."""
    if count <= 0:
        return []

    object_ids_state = scene_state.get("object_id")
    if object_ids_state is None:
        return [-1] * count

    if isinstance(object_ids_state, torch.Tensor):
        try:
            object_ids_list = object_ids_state.detach().to("cpu", copy=False).view(-1).tolist()
        except Exception:
            object_ids_list = []
    else:
        try:
            object_ids_list = list(object_ids_state)
        except Exception:
            object_ids_list = []

    obj_ids_by_det = [-1] * count
    if det_idx is None:
        return obj_ids_by_det

    if isinstance(det_idx, torch.Tensor):
        try:
            det_idx_list = det_idx.detach().to("cpu", copy=False).view(-1).tolist()
        except Exception:
            det_idx_list = []
    else:
        try:
            det_idx_list = list(det_idx)
        except Exception:
            det_idx_list = []

    limit = min(count, len(det_idx_list))
    for det_i in range(limit):
        try:
            obj_idx = int(det_idx_list[det_i])
        except Exception:
            continue
        if obj_idx < 0 or obj_idx >= len(object_ids_list):
            continue
        try:
            obj_ids_by_det[det_i] = int(object_ids_list[obj_idx])
        except Exception:
            continue
    return obj_ids_by_det


def build_local_caption_loser_ids(scene_state: dict, det_idx: object, count: int) -> List[List[int]]:
    """Return per-detection loser object ID lists looked up from *scene_state*."""
    if count <= 0:
        return []

    loser_state = scene_state.get("loser_object_ids") or []
    out: List[List[int]] = [[] for _ in range(count)]
    if det_idx is None:
        return out

    det_idx_list: List[int] = []
    if isinstance(det_idx, torch.Tensor):
        try:
            det_idx_list = det_idx.detach().to("cpu", copy=False).view(-1).tolist()
        except Exception:
            det_idx_list = []
    else:
        try:
            det_idx_list = list(det_idx)
        except Exception:
            det_idx_list = []

    limit = min(count, len(det_idx_list))
    for det_i in range(limit):
        try:
            obj_idx = int(det_idx_list[det_i])
        except Exception:
            continue
        if obj_idx < 0 or obj_idx >= len(loser_state):
            continue
        entry = loser_state[obj_idx]
        if not entry:
            continue
        if isinstance(entry, set):
            out[det_i] = sorted(int(x) for x in entry)
        elif isinstance(entry, (list, tuple)) and not isinstance(entry, (str, bytes)):
            with contextlib.suppress(Exception):
                out[det_i] = sorted({int(x) for x in entry if x is not None})
    return out


# ---------------------------------------------------------------------------
# Caption manager drain
# ---------------------------------------------------------------------------


def maybe_process_captions(
    *,
    caption_manager: Any,
    auto_caption_enabled: bool,
    step_index: int,
    caption_start_step: int,
    caption_step_interval: int,
    pending_caption_indices: list,
) -> None:
    """Enqueue pending objects into *caption_manager* and drain completed results.

    Args:
        caption_manager: the ``CaptionManager`` instance (or ``None``).
        auto_caption_enabled: if ``False`` the function returns immediately.
        step_index: current mapping step counter.
        caption_start_step: first step at which auto-captioning is active.
        caption_step_interval: run captioning every N steps after *caption_start_step*.
        pending_caption_indices: mutable list; objects are enqueued then cleared in-place.
    """
    if not auto_caption_enabled:
        return
    if step_index < caption_start_step:
        return
    if (step_index - caption_start_step) % caption_step_interval != 0:
        return
    if pending_caption_indices:
        caption_manager.enqueue_objects(pending_caption_indices)
        pending_caption_indices.clear()
    caption_manager.drain_results()
