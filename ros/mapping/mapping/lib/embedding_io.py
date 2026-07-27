"""Embedding matrix packing and observation serialization utilities."""

from __future__ import annotations

import contextlib
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch


def pack_embedding_matrix(
    embeddings_state: Sequence[Sequence[float]],
    include_idx: Sequence[int],
) -> Tuple[np.ndarray, int]:
    dim = 0
    for idx in include_idx:
        if idx < 0 or idx >= len(embeddings_state):
            continue
        vec = embeddings_state[idx]
        if isinstance(vec, (list, tuple)) and vec:
            dim = len(vec)
            break
    if dim <= 0:
        return np.zeros((len(include_idx), 0), dtype=np.float32), 0
    mat = np.zeros((len(include_idx), dim), dtype=np.float32)
    for row, idx in enumerate(include_idx):
        if idx < 0 or idx >= len(embeddings_state):
            continue
        vec = embeddings_state[idx]
        if isinstance(vec, (list, tuple)) and len(vec) == dim:
            mat[row, :] = np.asarray(vec, dtype=np.float32)
    return mat, dim


def pack_embedding_history_matrix(
    history_state: Sequence[Sequence[Sequence[float]]],
    include_idx: Sequence[int],
    fallback_state: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray, int]:
    dim = 0
    for idx in include_idx:
        if idx < 0:
            continue
        history_row = history_state[idx] if idx < len(history_state) else []
        if isinstance(history_row, Sequence) and not isinstance(history_row, (str, bytes, bytearray)):
            for vec in history_row:
                if isinstance(vec, (list, tuple)) and vec:
                    dim = len(vec)
                    break
        if dim > 0:
            break
        vec_fallback = fallback_state[idx] if idx < len(fallback_state) else []
        if isinstance(vec_fallback, (list, tuple)) and vec_fallback:
            dim = len(vec_fallback)
            break

    row_ptr: List[int] = [0]
    rows: List[np.ndarray] = []
    if dim <= 0:
        for _ in include_idx:
            row_ptr.append(row_ptr[-1])
        return (
            np.asarray(row_ptr, dtype=np.int64),
            np.zeros((0, 0), dtype=np.float32),
            0,
        )

    for idx in include_idx:
        count_before = len(rows)
        history_row = history_state[idx] if idx < len(history_state) else []
        if isinstance(history_row, Sequence) and not isinstance(history_row, (str, bytes, bytearray)):
            for vec in history_row:
                if isinstance(vec, (list, tuple)) and len(vec) == dim:
                    rows.append(np.asarray(vec, dtype=np.float32))
        if len(rows) == count_before:
            vec_fallback = fallback_state[idx] if idx < len(fallback_state) else []
            if isinstance(vec_fallback, (list, tuple)) and len(vec_fallback) == dim:
                rows.append(np.asarray(vec_fallback, dtype=np.float32))
        row_ptr.append(len(rows))

    mat = np.vstack(rows).astype(np.float32, copy=False) if rows else np.zeros((0, dim), dtype=np.float32)
    return np.asarray(row_ptr, dtype=np.int64), mat, int(dim)


def pack_text_history(
    history_state: Sequence[Sequence[str]],
    include_idx: Sequence[int],
    fallback_state: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    row_ptr: List[int] = [0]
    values: List[str] = []
    for idx in include_idx:
        before = len(values)
        history_row = history_state[idx] if idx < len(history_state) else []
        if isinstance(history_row, Sequence) and not isinstance(history_row, (str, bytes, bytearray)):
            for text in history_row:
                text_norm = str(text or "").strip()
                if text_norm:
                    values.append(text_norm)
        if len(values) == before:
            fallback = str(fallback_state[idx] or "").strip() if idx < len(fallback_state) else ""
            if fallback:
                values.append(fallback)
        row_ptr.append(len(values))
    return np.asarray(row_ptr, dtype=np.int64), values


def observation_to_hwc_uint8(
    obs: Any,
) -> Tuple[Optional[np.ndarray], int, np.ndarray, str]:
    """
    Convert one observation entry to HWC uint8 image + lightweight metadata.
    Returns (image_hwc_uint8_or_none, image_id, bbox_xyxy, encoding).
    """
    image_id = -1
    bbox = np.full((4,), np.nan, dtype=np.float32)
    encoding = ""

    image_payload = obs
    used_caption_image = False
    if isinstance(obs, dict):
        image_payload = obs.get("image_caption")
        used_caption_image = image_payload is not None
        if image_payload is None:
            image_payload = obs.get("image")
        try:
            image_id = int(obs.get("image_id", -1))
        except Exception:
            image_id = -1
        encoding = str(obs.get("encoding", "") or "")
        raw_bbox = obs.get("bbox")
        if used_caption_image:
            raw_bbox = obs.get("bbox_caption") or raw_bbox
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
            try:
                bbox = np.asarray(raw_bbox[:4], dtype=np.float32)
            except Exception:
                bbox = np.full((4,), np.nan, dtype=np.float32)

    if image_payload is None:
        return None, image_id, bbox, encoding
    try:
        arr = (
            image_payload.detach().to("cpu", copy=False).numpy()
            if isinstance(image_payload, torch.Tensor)
            else np.asarray(image_payload)
        )
    except Exception:
        return None, image_id, bbox, encoding
    if arr is None:
        return None, image_id, bbox, encoding
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        return None, image_id, bbox, encoding

    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = np.ascontiguousarray(arr)
    return arr, image_id, bbox, encoding


def bbox_area_xyxy(raw_bbox: Any) -> Optional[float]:
    if not isinstance(raw_bbox, (list, tuple, np.ndarray)) or len(raw_bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = [float(raw_bbox[i]) for i in range(4)]
    except Exception:
        return None
    if not np.isfinite([x0, y0, x1, y1]).all():
        return None
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    area = width * height
    return float(area) if np.isfinite(area) and area > 0.0 else None


def bbox_area_xyxy_clamped(raw_bbox: Any, width: int, height: int) -> Optional[float]:
    if not isinstance(raw_bbox, (list, tuple, np.ndarray)) or len(raw_bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = [float(raw_bbox[i]) for i in range(4)]
    except Exception:
        return None
    if not np.isfinite([x0, y0, x1, y1]).all():
        return None
    w_img = float(max(int(width), 0))
    h_img = float(max(int(height), 0))
    if w_img <= 0.0 or h_img <= 0.0:
        return None
    width_box = max(0.0, min(w_img, x1) - max(0.0, x0))
    height_box = max(0.0, min(h_img, y1) - max(0.0, y0))
    area = width_box * height_box
    return float(area) if np.isfinite(area) and area > 0.0 else None
