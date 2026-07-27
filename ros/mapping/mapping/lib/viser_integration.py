"""Viser integration utilities for the streaming mapper node.

Pure helper functions for image encoding, detection captions, and scene-state
lookups triggered by Viser UI actions.  All functions receive explicit
parameters instead of ``self``; no ROS node state is captured.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Header


# ---------------------------------------------------------------------------
# Image encoding helpers
# ---------------------------------------------------------------------------


def encode_np_image_to_compressed(image: object) -> Optional[CompressedImage]:
    """Encode a numpy RGB image (or dict with ``image`` key) to a JPEG CompressedImage."""
    if image is None:
        return None
    encoding = ""
    if isinstance(image, dict):
        encoding = str(image.get("encoding", "") or image.get("color_encoding", "") or "").strip().lower()
        image = image.get("image")
        if image is None:
            return None
    if hasattr(image, "detach"):
        try:
            tensor = image.detach()
            if hasattr(tensor, "to"):
                tensor = tensor.to("cpu")
            image = tensor.numpy() if hasattr(tensor, "numpy") else tensor
        except Exception:
            return None
    try:
        arr = np.asarray(image)
    except Exception:
        return None
    if arr.ndim not in (2, 3):
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    elif arr.shape[2] == 3:
        pass
    else:
        return None
    arr = np.ascontiguousarray(arr)
    if encoding in {"bgr8", "bgr"}:
        bgr = arr
    else:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(".jpg", bgr)
    if not success:
        return None
    msg = CompressedImage()
    msg.format = "jpeg"
    msg.data = buf.tobytes()
    return msg


def depth_image_from_array(depth_f32: np.ndarray, header: Header) -> Image:
    """Build a 32FC1 ROS Image message from a 2-D float32 depth array."""
    depth_msg = Image()
    depth_msg.header = header
    depth_msg.height = int(depth_f32.shape[0])
    depth_msg.width = int(depth_f32.shape[1])
    depth_msg.encoding = "32FC1"
    depth_msg.is_bigendian = 0
    depth_msg.step = int(depth_f32.shape[1] * 4)
    depth_msg.data = depth_f32.astype(np.float32, copy=False).tobytes()
    return depth_msg


def rgb_image_from_array(rgb: np.ndarray, header: Header) -> Optional[Image]:
    """Build an rgb8 ROS Image message from a numpy array."""
    if rgb is None:
        return None
    try:
        arr = np.asarray(rgb)
    except Exception:
        return None
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3:
        return None
    if arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    if arr.shape[2] != 3:
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = np.ascontiguousarray(arr)

    msg = Image()
    msg.header = header
    msg.height = int(arr.shape[0])
    msg.width = int(arr.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = int(arr.shape[1] * 3)
    msg.data = arr.tobytes()
    return msg


# ---------------------------------------------------------------------------
# Detection caption / label helpers
# ---------------------------------------------------------------------------


def label_from_id(names: Any, cls_id: int) -> str:
    """Return the human-readable label for *cls_id* given *names*."""
    if (isinstance(names, dict) and cls_id in names) or (
        isinstance(names, Sequence)
        and not isinstance(names, (str, bytes))
        and 0 <= cls_id < len(names)
    ):
        return str(names[cls_id])
    return str(cls_id)


def build_detection_captions(
    class_ids: Any,
    scores: Any,
    names: Optional[Sequence[str]],
    count: int,
) -> List[str]:
    """Return a list of ``"label (score)"`` strings for detected objects."""
    captions: List[str] = []
    ids_list: List[int] = []
    scores_list: List[Any] = []
    if isinstance(class_ids, torch.Tensor):
        try:
            ids_list = class_ids.detach().cpu().tolist()
        except Exception:
            ids_list = []
    elif isinstance(class_ids, Sequence):
        ids_list = list(class_ids)

    if isinstance(scores, torch.Tensor):
        try:
            scores_list = scores.detach().cpu().tolist()
        except Exception:
            scores_list = []
    elif isinstance(scores, Sequence):
        scores_list = list(scores)

    for idx in range(count):
        cls_id = ids_list[idx] if idx < len(ids_list) else -1
        score = scores_list[idx] if idx < len(scores_list) else None
        if names and isinstance(cls_id, int) and 0 <= cls_id < len(names):
            cls_name = names[cls_id]
        else:
            cls_name = f"id={cls_id}"
        if score is None:
            captions.append(str(cls_name))
        else:
            try:
                captions.append(f"{cls_name} ({float(score):.2f})")
            except Exception:
                captions.append(str(cls_name))
    return captions


def format_detection_summary(
    det_map: Optional[Dict[str, float]],
    *,
    max_items: int = 5,
) -> str:
    """Format a detection-category confidence dict into a human-readable string."""
    if not det_map:
        return "Detections: N/A"
    try:
        max_items_int = max(1, int(max_items))
    except Exception:
        max_items_int = 5
    items = []
    for k, v in det_map.items():
        if k is None:
            continue
        try:
            key = str(k)
        except Exception:
            continue
        if not key:
            continue
        try:
            score = float(v)
        except Exception:
            continue
        if not math.isfinite(score):
            continue
        items.append((key, score))
    if not items:
        return "Detections: N/A"
    items.sort(key=lambda kv: kv[1], reverse=True)
    shown = items[:max_items_int]
    parts = [f"{cat} ({conf:.2f})" for cat, conf in shown]
    extra = max(0, len(items) - len(shown))
    suffix = f", ... (+{extra})" if extra > 0 else ""
    return f"Detections: {', '.join(parts)}{suffix}"


def flatten_neighbors(
    neighbors: Optional[Sequence[Any]],
    expected: int,
) -> Tuple[List[int], List[int]]:
    """Flatten a ragged sequence of neighbor index lists into a CSR-style pair.

    Returns:
        ``(indices, splits)`` where ``splits[i]`` gives the start offset of
        detection ``i``'s neighbors in ``indices``.
    """
    if neighbors is None:
        return [], [0] * (expected + 1)

    indices: List[int] = []
    splits: List[int] = [0]
    limited_neighbors = list(neighbors)[:expected]
    for neighbor in limited_neighbors:
        if neighbor is None:
            splits.append(len(indices))
            continue
        if isinstance(neighbor, torch.Tensor):
            try:
                neighbor_list = neighbor.detach().cpu().tolist()
            except Exception:
                neighbor_list = []
        else:
            neighbor_list = list(neighbor)
        indices.extend(int(x) for x in neighbor_list)
        splits.append(len(indices))

    while len(splits) < expected + 1:
        splits.append(len(indices))

    return indices, splits


# ---------------------------------------------------------------------------
# Scene-state lookup
# ---------------------------------------------------------------------------


def resolve_object_index_by_id(
    scene_state: Optional[dict],
    object_id: int,
) -> Optional[int]:
    """Map *object_id* to its row index in *scene_state*.

    Returns ``None`` when *scene_state* is ``None`` or the ID is not found.
    """
    if scene_state is None:
        return None
    object_ids = scene_state.get("object_id")
    if object_ids is None:
        return None
    try:
        ids_np = np.asarray(object_ids, dtype=int)
        matches = np.where(ids_np == object_id)[0]
        if matches.size > 0:
            return int(matches[0])
    except Exception:
        pass
    return None

