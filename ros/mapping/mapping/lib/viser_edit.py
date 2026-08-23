"""Pure-function helpers for Viser interactive editing callbacks."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from mapping.lib.geometric_transforms import matrix_from_xyz_quat
from mapping.lib.viser_integration import resolve_object_index_by_id
from scene_graph.map_update.covisibility import update_covisibility_active_bitset
from scene_graph.storage.models import ImageRecord


def edit_caption(
    scene_state: dict,
    object_id: int,
    new_caption: str,
    *,
    request_embeddings: Callable[[List[str]], Tuple[bool, str, list]],
) -> Tuple[bool, str]:
    """Update an object's caption and recompute embeddings in *scene_state*.

    *request_embeddings* signature: (texts) -> (ok, error_msg, vecs_list).
    Returns (success, error_message).
    """
    obj_idx = resolve_object_index_by_id(scene_state, object_id)
    if obj_idx is None:
        return False, f"Object {object_id} not found in scene state."

    is_locked = scene_state.get("is_locked") or []
    if obj_idx < len(is_locked) and bool(is_locked[obj_idx]):
        return False, f"Object {object_id} is locked and cannot be edited."

    captions = scene_state.get("object_caption")
    if captions is None or obj_idx >= len(captions):
        return False, "Scene state captions not properly initialized."
    captions[obj_idx] = str(new_caption or "")

    try:
        ok, emb_error, vecs = request_embeddings([str(new_caption or "")])
        if not ok:
            return False, f"Caption embedding failed: {emb_error}"
        if not vecs or not vecs[0]:
            return False, "Caption embedding failed: empty_vector"
        embeddings = scene_state.get("object_caption_embedding")
        if embeddings is not None and obj_idx < len(embeddings):
            embeddings[obj_idx] = vecs[0]
    except Exception as exc:
        return False, f"Caption embedding request exception: {exc}"

    cap_hist = scene_state.get("object_caption_history")
    if cap_hist is not None and obj_idx < len(cap_hist):
        if not isinstance(cap_hist[obj_idx], list):
            cap_hist[obj_idx] = []
        cap_hist[obj_idx].append(str(new_caption or ""))

    emb_hist = scene_state.get("object_caption_embedding_history")
    if emb_hist is not None and obj_idx < len(emb_hist):
        if not isinstance(emb_hist[obj_idx], list):
            emb_hist[obj_idx] = []
        emb_hist[obj_idx].append(vecs[0])

    return True, ""


def delete_object(
    scene_state: dict,
    object_id: int,
    *,
    step_index: int = -1,
    logger_info: Callable[..., None] = lambda *a, **kw: None,
) -> Tuple[bool, str]:
    """Mark an object as inactive in *scene_state*. Returns (success, error)."""
    obj_idx = resolve_object_index_by_id(scene_state, object_id)
    if obj_idx is None:
        return False, f"Object {object_id} not found in scene state."

    active = scene_state.get("active")
    if active is None or obj_idx >= len(active):
        return False, "Scene state active flags not properly initialized."

    captions = scene_state.get("object_caption") or []
    means = scene_state.get("means")
    caption = ""
    pos = None
    if obj_idx < len(captions) and captions[obj_idx] is not None:
        caption = str(captions[obj_idx]).strip()
    if means is not None and obj_idx < means.shape[0]:
        with contextlib.suppress(Exception):
            pos = means[obj_idx].detach().cpu().tolist()

    active[obj_idx] = False

    logger_info(
        "Object marked INACTIVE due to MANUAL DELETION (Viser UI): "
        f"idx={obj_idx} object_id={object_id} "
        f"caption={repr(caption)} position={pos} "
        f"(step={step_index})"
    )
    return True, ""


def toggle_lock(
    scene_state: dict,
    object_id: int,
    *,
    logger_info: Callable[..., None] = lambda *a, **kw: None,
) -> Tuple[bool, str]:
    """Toggle the lock state of an object. Returns (success, message)."""
    obj_idx = resolve_object_index_by_id(scene_state, object_id)
    if obj_idx is None:
        return False, f"Object {object_id} not found in scene state."

    is_locked = scene_state.get("is_locked")
    if not isinstance(is_locked, list):
        is_locked = []
        scene_state["is_locked"] = is_locked
    while len(is_locked) <= obj_idx:
        is_locked.append(False)

    is_locked[obj_idx] = not is_locked[obj_idx]
    new_state = "locked" if is_locked[obj_idx] else "unlocked"

    logger_info(f"Object {object_id} {new_state} (idx={obj_idx}) via Viser UI")
    return True, f"Object {object_id} {new_state}."


def validate_add_object_inputs(
    caption: str,
    location_xyz: Any,
    views: Any,
    image_path: Optional[str] = None,
) -> Tuple[Optional[str], Optional[np.ndarray], str, List[List[float]], Optional[np.ndarray]]:
    """Validate and parse inputs for add_object.

    Returns (error_or_None, location_arr, caption_text, parsed_views, uploaded_image_rgb).
    On error, the first element is a non-None string describing the problem.
    """
    caption_text = str(caption or "").strip()
    if not caption_text:
        return "Caption is empty.", None, "", [], None

    if not isinstance(location_xyz, Sequence) or isinstance(location_xyz, (str, bytes, bytearray)):
        return "Location must be a numeric sequence.", None, "", [], None
    if len(location_xyz) != 3:
        return "Location must have 3 values [x,y,z].", None, "", [], None
    try:
        location_values = [float(x) for x in location_xyz]
    except Exception:
        return "Location contains non-numeric values.", None, "", [], None
    location_arr = np.asarray(location_values, dtype=np.float64)
    if not np.all(np.isfinite(location_arr)):
        return "Location contains NaN or Inf.", None, "", [], None

    if not isinstance(views, Sequence) or len(views) != 3:
        return "Expected exactly 3 views.", None, "", [], None

    parsed_views: List[List[float]] = []
    for idx, raw in enumerate(views, start=1):
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            return f"View {idx} is not a numeric sequence.", None, "", [], None
        if len(raw) != 7:
            return f"View {idx} must have 7 values.", None, "", [], None
        try:
            values = [float(x) for x in raw]
        except Exception:
            return f"View {idx} contains non-numeric values.", None, "", [], None
        if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
            return f"View {idx} contains NaN or Inf.", None, "", [], None
        q = np.asarray(values[3:7], dtype=np.float64)
        q_norm = float(np.linalg.norm(q))
        if q_norm <= 1e-12:
            return f"View {idx} quaternion norm is zero.", None, "", [], None
        values[3:7] = (q / q_norm).astype(np.float64).tolist()
        parsed_views.append(values)

    uploaded_image_rgb: Optional[np.ndarray] = None
    image_path_text = str(image_path or "").strip()
    if image_path_text:
        image_candidate = Path(image_path_text).expanduser()
        if not image_candidate.is_absolute():
            image_candidate = (Path.cwd() / image_candidate).resolve()
        if not image_candidate.exists() or not image_candidate.is_file():
            return f"Image path does not exist: {image_candidate}", None, "", [], None
        loaded_bgr = cv2.imread(str(image_candidate), cv2.IMREAD_COLOR)
        if loaded_bgr is None:
            return f"Failed to decode image file: {image_candidate}", None, "", [], None
        uploaded_image_rgb = cv2.cvtColor(loaded_bgr, cv2.COLOR_BGR2RGB)

    return None, location_arr, caption_text, parsed_views, uploaded_image_rgb


def add_object_to_scene_state(
    scene_state: dict,
    *,
    caption_text: str,
    location_arr: np.ndarray,
    parsed_views: List[List[float]],
    uploaded_image_rgb: Optional[np.ndarray],
    uploaded_image_resolved_path: str,
    caption_embedding: List[float],
    embedding_error: Optional[str],
) -> Tuple[int, int, List[int]]:
    """Append a new object to *scene_state* tensors and lists.

    Returns (n_old, next_object_id, manual_image_ids).
    """
    means = scene_state["means"]
    cov6 = scene_state["cov6"]
    features = scene_state["features"]
    count = scene_state["count"]
    active = scene_state["active"]
    class_ids = scene_state["class_ids"]
    object_ids = scene_state["object_id"]

    n_old = int(means.shape[0])
    mean_xyz = location_arr.astype(np.float32)

    default_var = np.float32(0.05)
    new_cov6 = np.asarray([default_var, 0.0, 0.0, default_var, 0.0, default_var], dtype=np.float32)

    count_new = torch.ones((1,), dtype=count.dtype, device=count.device)
    means_new = torch.as_tensor(mean_xyz, dtype=means.dtype, device=means.device).view(1, 3)
    cov6_new = torch.as_tensor(new_cov6, dtype=cov6.dtype, device=means.device).view(1, 6)
    feat_new = torch.zeros((1, int(features.shape[1])), dtype=features.dtype, device=features.device)
    active_new = torch.ones((1,), dtype=torch.bool, device=active.device)
    class_ids_new = torch.full((1,), -1, dtype=class_ids.dtype, device=class_ids.device)

    next_object_id = 0 if object_ids.numel() == 0 else int(object_ids.max().item()) + 1
    object_ids_new = torch.tensor([next_object_id], dtype=object_ids.dtype)

    scene_state["count"] = torch.cat([count.to(count.device), count_new], dim=0)
    scene_state["means"] = torch.cat([means, means_new], dim=0)
    scene_state["cov6"] = torch.cat([cov6, cov6_new], dim=0)
    scene_state["features"] = torch.cat([features, feat_new], dim=0)
    scene_state["active"] = torch.cat([active, active_new], dim=0)
    scene_state["class_ids"] = torch.cat([class_ids, class_ids_new], dim=0)
    scene_state["object_id"] = torch.cat([object_ids, object_ids_new.cpu()], dim=0)

    def _ensure_list(name: str, fill_value):
        arr = scene_state.get(name)
        if not isinstance(arr, list):
            arr = []
        while len(arr) < n_old:
            arr.append(fill_value() if callable(fill_value) else fill_value)
        scene_state[name] = arr
        return arr

    captions = _ensure_list("object_caption", "")
    captions.append(caption_text)

    cap_embeds = _ensure_list("object_caption_embedding", list)
    cap_embeds.append(list(caption_embedding))

    _ensure_list("object_siglip2_embedding", list).append([])
    _ensure_list("object_qwen3_vl_embedding", list).append([])

    _ensure_list("object_caption_history", list).append([caption_text])

    cap_emb_hist = _ensure_list("object_caption_embedding_history", list)
    cap_emb_hist.append([list(caption_embedding)] if caption_embedding else [])

    _ensure_list("object_siglip2_embedding_history", list).append([])
    _ensure_list("object_qwen3_vl_embedding_history", list).append([])
    _ensure_list("object_detection_category", "").append("")
    _ensure_list("object_detection_category_conf", dict).append({})
    _ensure_list("loser_object_ids", set).append(set())

    rgb_observations = _ensure_list("rgb_observations", list)
    if uploaded_image_rgb is not None:
        h_img, w_img = int(uploaded_image_rgb.shape[0]), int(uploaded_image_rgb.shape[1])
        manual_obs = {
            "image": uploaded_image_rgb,
            "image_id": -1,
            "bbox": [0.0, 0.0, float(w_img), float(h_img)],
            "encoding": "manual_upload",
        }
        rgb_observations.append([manual_obs])
    else:
        rgb_observations.append([])

    _ensure_list("view_means", list).append([])
    _ensure_list("view_cov6", list).append([])
    _ensure_list("high_quality_views", list).append([])
    _ensure_list("high_quality_captioning", False).append(False)

    images_meta = scene_state.get("images")
    if not isinstance(images_meta, list):
        images_meta = []
    image_positions = scene_state.get("image_positions")
    if not isinstance(image_positions, list):
        image_positions = []

    base_image_id = len(images_meta)
    manual_image_ids: List[int] = []
    for offset, view in enumerate(parsed_views):
        xyz = view[:3]
        quat = view[3:7]
        image_id = base_image_id + offset
        pose_np = matrix_from_xyz_quat(xyz, quat).astype(np.float32)
        storage_path = uploaded_image_resolved_path if (offset == 0 and uploaded_image_resolved_path) else ""
        images_meta.append(
            ImageRecord(
                image_id=int(image_id),
                pose=torch.from_numpy(pose_np),
                camera_id="right",
                storage_path=storage_path,
            )
        )
        image_positions.append(torch.as_tensor(xyz, dtype=torch.float32))
        manual_image_ids.append(int(image_id))

    if uploaded_image_rgb is not None and manual_image_ids and rgb_observations:
        try:
            last_row = rgb_observations[-1]
            if isinstance(last_row, list) and last_row and isinstance(last_row[0], dict):
                last_row[0]["image_id"] = int(manual_image_ids[0])
        except Exception:
            pass

    scene_state["images"] = images_meta
    scene_state["image_positions"] = image_positions

    _ensure_list("object_image_ids", list).append(list(manual_image_ids))
    _ensure_list("viewpoint_image_ids", list).append(list(manual_image_ids))

    update_covisibility_active_bitset(scene_state, num_objects=int(scene_state["means"].shape[0]))

    return n_old, next_object_id, manual_image_ids
