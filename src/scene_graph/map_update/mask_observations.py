"""Persistence helpers for per-object 2D mask observations.

These records are evaluation evidence, not reconstruction geometry.  They keep
the detector's image-space masks attached to the object that consumed the
detection, so offline metrics can compare predicted objects in the image plane
without re-projecting the object's 3D voxel support.

Optionally, the same sidecar also embeds a JPEG-encoded padded RGB crop
(reusing the per-detection crop already computed for the captioner). This
gives later VL-reranking passes the visual context for each observation
without a second pass over the original frame source.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

try:  # Pillow is already a runtime dep of the imaging pipeline.
    from PIL import Image
except Exception:  # pragma: no cover - defensive import
    Image = None  # type: ignore[assignment]


DEFAULT_MAX_MASK_OBSERVATIONS_PER_OBJECT = 256
DEFAULT_CROP_JPEG_QUALITY = 85


def _encode_padded_crop_jpeg(
    obs: Any,
    quality: int = DEFAULT_CROP_JPEG_QUALITY,
) -> Optional[Dict[str, Any]]:
    """Pull the padded RGB crop out of a caption-observation entry and JPEG-encode it.

    Returns ``{"crop_jpeg_bytes": <uint8 ndarray>, "crop_bbox_xyxy": <int32[4]>,
    "crop_shape": <int32[2]>}`` or ``None`` if no usable crop is present.
    """
    if obs is None or Image is None:
        return None
    image = obs.get("image") if isinstance(obs, Mapping) else None
    bbox = obs.get("bbox") if isinstance(obs, Mapping) else None
    if image is None or bbox is None:
        return None
    if isinstance(image, torch.Tensor):
        arr = image.detach()
        if arr.is_cuda:
            arr = arr.cpu()
        arr = arr.numpy()
    else:
        arr = np.asarray(image)
    # Accept HWC uint8; if CHW float, transpose+cast.
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        return None
    buf = io.BytesIO()
    try:
        Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=int(quality), optimize=True)
    except Exception:
        return None
    return {
        "crop_jpeg_bytes": np.frombuffer(buf.getvalue(), dtype=np.uint8),
        "crop_bbox_xyxy": np.asarray(
            [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])], dtype=np.int32
        ),
        "crop_shape": np.asarray([int(arr.shape[0]), int(arr.shape[1])], dtype=np.int32),
    }


def _as_numpy_bool(mask: Any) -> Optional[np.ndarray]:
    if mask is None:
        return None
    value = mask
    if isinstance(value, torch.Tensor):
        value = value.detach().to("cpu")
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        value = value.numpy()
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        return None
    return arr.astype(bool, copy=False)


def _mask_bbox_xyxy(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if ys.size == 0 or xs.size == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return [x0, y0, x1, y1]


def _pack_crop(mask: np.ndarray, bbox_xyxy: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    crop = np.asarray(mask[y0:y1, x0:x1], dtype=np.uint8)
    packed = np.packbits(crop.reshape(-1), bitorder="little")
    shape = np.asarray([int(crop.shape[0]), int(crop.shape[1])], dtype=np.int32)
    return packed, shape


def _score_at(scores: Any, det_i: int) -> Optional[float]:
    if scores is None:
        return None
    with contextlib.suppress(Exception):
        if isinstance(scores, torch.Tensor):
            return float(scores.detach().to("cpu").view(-1)[int(det_i)].item())
        return float(np.asarray(scores).reshape(-1)[int(det_i)])
    return None


def _int_at(values: Any, det_i: int) -> Optional[int]:
    if values is None:
        return None
    with contextlib.suppress(Exception):
        if isinstance(values, torch.Tensor):
            return int(values.detach().to("cpu").view(-1)[int(det_i)].item())
        return int(np.asarray(values).reshape(-1)[int(det_i)])
    return None


def ensure_object_mask_observation_rows(state: Dict[str, Any], min_len: int) -> List[List[Dict[str, Any]]]:
    rows = state.get("object_mask_observations")
    if not isinstance(rows, list):
        rows = []
    while len(rows) < int(min_len):
        rows.append([])
    state["object_mask_observations"] = rows
    return rows


def register_detection_mask_observations(
    state: Dict[str, Any],
    seg_outputs: Mapping[str, Any],
    detection_image_ids: Sequence[Optional[int]],
    det_to_obj: Sequence[Optional[int]],
    mask_storage_dir: Path,
    *,
    max_per_object: int = DEFAULT_MAX_MASK_OBSERVATIONS_PER_OBJECT,
    detection_rgb_observations: Optional[Sequence[Any]] = None,
    save_crops: bool = False,
    crop_jpeg_quality: int = DEFAULT_CROP_JPEG_QUALITY,
) -> int:
    """Save detector masks and attach observation records to scene-state rows.

    Args:
        state: Mutable scene graph state.
        seg_outputs: Filtered segmentation outputs for the current batch.
        detection_image_ids: Image id per detection.
        det_to_obj: Canonical object index per detection after state update.
        mask_storage_dir: Directory where compact ``.npz`` sidecars are stored.
        max_per_object: Keep at most this many records referenced per object.
        detection_rgb_observations: Optional list aligned with detections,
            holding the padded RGB crops already prepared for the captioner
            (each entry is the dict from
            :func:`scene_graph.captioning.crop_util.compute_caption_observations`
            with ``image`` HWC uint8 + ``bbox`` xyxy in original frame).
        save_crops: If True and ``detection_rgb_observations`` is provided,
            JPEG-encode the padded crop into the same ``.npz`` sidecar as the
            mask, so downstream rerank passes get visual context without
            reloading the source frame.
        crop_jpeg_quality: JPEG quality for the saved crop (default 85).

    Returns:
        Number of observation records appended to the state.
    """

    raw_masks = list(seg_outputs.get("masks") or [])
    if not raw_masks:
        return 0
    inlier_masks = list(seg_outputs.get("masks_inlier") or [])
    scores = seg_outputs.get("scores")
    class_ids = seg_outputs.get("class_ids")
    batch_ids = seg_outputs.get("batch_ids")
    object_ids = state.get("object_id")
    n_objects = int(object_ids.shape[0]) if isinstance(object_ids, torch.Tensor) else len(det_to_obj)
    rows = ensure_object_mask_observation_rows(state, n_objects)
    mask_storage_dir = Path(mask_storage_dir)
    mask_storage_dir.mkdir(parents=True, exist_ok=True)

    added = 0
    cap = max(1, int(max_per_object))
    for det_i, obj_idx_raw in enumerate(det_to_obj):
        if obj_idx_raw is None:
            continue
        try:
            obj_idx = int(obj_idx_raw)
        except Exception:
            continue
        if obj_idx < 0:
            continue
        if det_i >= len(raw_masks) or det_i >= len(detection_image_ids):
            continue
        image_id_raw = detection_image_ids[det_i]
        if image_id_raw is None:
            continue
        try:
            image_id = int(image_id_raw)
        except Exception:
            continue
        raw = _as_numpy_bool(raw_masks[det_i])
        if raw is None:
            continue
        raw_bbox = _mask_bbox_xyxy(raw)
        if raw_bbox is None:
            continue

        inlier = None
        inlier_bbox = None
        if det_i < len(inlier_masks):
            inlier = _as_numpy_bool(inlier_masks[det_i])
            if inlier is not None and inlier.shape == raw.shape:
                inlier_bbox = _mask_bbox_xyxy(inlier)
            else:
                inlier = None

        ensure_object_mask_observation_rows(state, obj_idx + 1)
        rows = state["object_mask_observations"]
        object_id = obj_idx
        if isinstance(object_ids, torch.Tensor) and obj_idx < int(object_ids.shape[0]):
            with contextlib.suppress(Exception):
                object_id = int(object_ids[obj_idx].detach().to("cpu").item())

        object_dir = mask_storage_dir / f"object_{int(object_id):06d}"
        object_dir.mkdir(parents=True, exist_ok=True)
        path = object_dir / f"img_{image_id:06d}_det_{int(det_i):04d}.npz"

        raw_bits, raw_shape = _pack_crop(raw, raw_bbox)
        payload: Dict[str, Any] = {
            "image_shape": np.asarray([int(raw.shape[0]), int(raw.shape[1])], dtype=np.int32),
            "raw_bits": raw_bits,
            "raw_shape": raw_shape,
            "raw_bbox_xyxy": np.asarray(raw_bbox, dtype=np.int32),
        }
        mask_kinds = ["raw"]
        if inlier is not None and inlier_bbox is not None:
            inlier_bits, inlier_shape = _pack_crop(inlier, inlier_bbox)
            payload.update(
                {
                    "inlier_bits": inlier_bits,
                    "inlier_shape": inlier_shape,
                    "inlier_bbox_xyxy": np.asarray(inlier_bbox, dtype=np.int32),
                }
            )
            mask_kinds.append("inlier")

        crop_record_extra: Dict[str, Any] = {}
        if save_crops and detection_rgb_observations is not None:
            obs_entry = (
                detection_rgb_observations[det_i]
                if det_i < len(detection_rgb_observations)
                else None
            )
            crop_payload = _encode_padded_crop_jpeg(obs_entry, quality=crop_jpeg_quality)
            if crop_payload is not None:
                payload.update(crop_payload)
                mask_kinds.append("crop")
                crop_record_extra = {
                    "crop_jpeg_bytes_len": int(crop_payload["crop_jpeg_bytes"].size),
                    "crop_bbox_xyxy": [int(v) for v in crop_payload["crop_bbox_xyxy"].tolist()],
                    "crop_shape": [
                        int(crop_payload["crop_shape"][0]),
                        int(crop_payload["crop_shape"][1]),
                    ],
                }

        np.savez_compressed(path, **payload)

        record: Dict[str, Any] = {
            "image_id": int(image_id),
            "path": str(path),
            "image_shape": [int(raw.shape[0]), int(raw.shape[1])],
            "detection_idx": int(det_i),
            "object_idx": int(obj_idx),
            "object_id": int(object_id),
            "mask_kinds": mask_kinds,
            "raw_pixels": int(raw.sum()),
        }
        if inlier is not None:
            record["inlier_pixels"] = int(inlier.sum())
        record.update(crop_record_extra)
        score = _score_at(scores, det_i)
        if score is not None:
            record["score"] = float(score)
        class_id = _int_at(class_ids, det_i)
        if class_id is not None:
            record["class_id"] = int(class_id)
        batch_id = _int_at(batch_ids, det_i)
        if batch_id is not None:
            record["batch_id"] = int(batch_id)

        row = rows[obj_idx]
        if not isinstance(row, list):
            row = []
            rows[obj_idx] = row
        row.append(record)
        if len(row) > cap:
            del row[: len(row) - cap]
        added += 1
    return int(added)


def load_mask_observation(
    observation: Mapping[str, Any],
    *,
    kind: str = "raw",
    root: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Load a persisted mask observation as a full image-shaped bool array."""

    path_raw = str(observation.get("path") or observation.get("mask_path") or "")
    if not path_raw:
        return None
    path = Path(path_raw)
    if not path.is_absolute() and root is not None:
        path = Path(root) / path
    elif path.is_absolute() and root is not None:
        # Some archived scene states store absolute sidecar paths from the
        # original artifact tree.  If the caller loaded the state from a local
        # mirror, prefer the matching mask sidecar under that mirror.
        parts = path.parts
        for idx, part in enumerate(parts):
            if part.endswith("_masks"):
                local_path = Path(root).joinpath(*parts[idx:])
                if local_path.exists():
                    path = local_path
                break
    if not path.exists():
        return None
    desired = str(kind or "raw").strip().lower()
    if desired not in {"raw", "inlier"}:
        desired = "raw"

    try:
        with np.load(str(path), allow_pickle=False) as data:
            selected = desired
            if f"{selected}_bits" not in data.files:
                selected = "raw"
            if f"{selected}_bits" not in data.files:
                return None
            bits = np.asarray(data[f"{selected}_bits"], dtype=np.uint8)
            crop_shape = np.asarray(data[f"{selected}_shape"], dtype=np.int32).reshape(2)
            bbox = np.asarray(data[f"{selected}_bbox_xyxy"], dtype=np.int32).reshape(4)
            image_shape = np.asarray(data["image_shape"], dtype=np.int32).reshape(2)
            crop_h, crop_w = int(crop_shape[0]), int(crop_shape[1])
            if crop_h <= 0 or crop_w <= 0:
                return None
            flat = np.unpackbits(bits, bitorder="little")[: crop_h * crop_w]
            crop = flat.reshape(crop_h, crop_w).astype(bool, copy=False)
            h, w = int(image_shape[0]), int(image_shape[1])
            if h <= 0 or w <= 0:
                return None
            x0, y0, x1, y1 = [int(v) for v in bbox.tolist()]
            x0 = max(0, min(w, x0))
            x1 = max(0, min(w, x1))
            y0 = max(0, min(h, y0))
            y1 = max(0, min(h, y1))
            if x1 <= x0 or y1 <= y0:
                return None
            out = np.zeros((h, w), dtype=bool)
            out[y0:y1, x0:x1] = crop[: y1 - y0, : x1 - x0]
            return out
    except Exception:
        return None


def load_mask_observation_crop(
    observation: Mapping[str, Any],
    *,
    root: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Load the JPEG-encoded padded RGB crop from a mask observation sidecar.

    Returns the crop as an ``HxWx3`` uint8 array, or ``None`` if the sidecar
    has no crop payload (e.g., older runs that only saved masks).
    """
    if Image is None:
        return None
    path_raw = str(observation.get("path") or observation.get("mask_path") or "")
    if not path_raw:
        return None
    path = Path(path_raw)
    if not path.is_absolute() and root is not None:
        path = Path(root) / path
    if not path.exists():
        return None
    try:
        with np.load(str(path), allow_pickle=False) as data:
            if "crop_jpeg_bytes" not in data.files:
                return None
            blob = np.asarray(data["crop_jpeg_bytes"], dtype=np.uint8)
            img = Image.open(io.BytesIO(blob.tobytes())).convert("RGB")
            return np.array(img, dtype=np.uint8)
    except Exception:
        return None
