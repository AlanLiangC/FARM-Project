from __future__ import annotations

import time
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .mapping_util import ImageRecord, initialize_scene_graph_state

SCENE_STATE_FORMAT_VERSION = 1


def load_scene_state_image(storage_path: str) -> np.ndarray:
    """Decode a storage / source reference back to an RGB array.

    Supported forms:

    * ``/path/to/<scene>.sens#frame=<N>`` — pulls frame N from a ScanNet .sens file.
    * ``/path/to/<chunk>.npz#frame=<N>`` — pulls local frame N from an NPZ render chunk.
    * ``/path/to/frame.jpg`` or ``.png`` — read via imageio.
    * ``/path/to/frame.h5`` — read via scene_graph.storage.load_image_from_hdf5.
    """
    import imageio.v2 as imageio

    if not storage_path:
        raise ValueError("Empty storage_path")

    if "#frame=" in storage_path:
        base, _, suffix = storage_path.rpartition("#frame=")
        frame_idx = int(suffix)
        ext = Path(base).suffix.lower()
        if ext == ".sens":
            from scene_graph.offline.frame_sources.sens import read_sens_frame_color

            return read_sens_frame_color(base, frame_idx)
        if ext == ".npz":
            with np.load(base) as data:
                arr = np.asarray(data["images"][frame_idx], dtype=np.uint8)
                return arr.copy()
        raise ValueError(f"Unsupported #frame= reference: {storage_path!r}")

    suffix = Path(storage_path).suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        from scene_graph.storage import load_image_from_hdf5

        return np.asarray(load_image_from_hdf5(Path(storage_path)))
    return np.asarray(imageio.imread(storage_path))


def resolve_image_record(record: "ImageRecord | Dict[str, Any]") -> np.ndarray:
    """Load the RGB for an ImageRecord (or dict-like), preferring a saved path.

    Falls back to ``source_ref`` when ``storage_path`` is empty — lets the
    viser eval viewer recover RGB frames even when the recon ran without
    ``--image-saving``, as long as the source data is still on disk at the
    path captured at registration time.
    """
    if isinstance(record, dict):
        storage = str(record.get("storage_path") or "")
        source = str(record.get("source_ref") or "")
    else:
        storage = str(getattr(record, "storage_path", "") or "")
        source = str(getattr(record, "source_ref", "") or "")
    ref = storage or source
    if not ref:
        raise ValueError("ImageRecord has neither storage_path nor source_ref")
    return load_scene_state_image(ref)


def _image_record_to_dict(record: object) -> Dict[str, Any]:
    if isinstance(record, ImageRecord):
        return {
            "image_id": int(record.image_id),
            "pose": record.pose,
            "camera_id": str(record.camera_id or ""),
            "storage_path": str(record.storage_path or ""),
            "source_ref": str(getattr(record, "source_ref", "") or ""),
        }
    if isinstance(record, dict):
        return dict(record)
    if is_dataclass(record):
        try:
            return dict(record.__dict__)
        except Exception:
            return {}
    return {}


def _dict_to_image_record(payload: object) -> ImageRecord:
    if isinstance(payload, ImageRecord):
        return payload
    data = payload if isinstance(payload, dict) else {}
    image_id = int(data.get("image_id", 0) or 0)
    pose = data.get("pose")
    camera_id = str(data.get("camera_id") or "")
    storage_path = str(data.get("storage_path") or "")
    source_ref = str(data.get("source_ref") or "")
    if isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu()
    return ImageRecord(
        image_id=image_id,
        pose=pose,
        camera_id=camera_id,
        storage_path=storage_path,
        source_ref=source_ref,
    )


def _to_cpu(obj: object) -> object:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    if isinstance(obj, set):
        return {_to_cpu(v) for v in obj}
    return obj


def make_serializable_scene_state(
    state: Dict[str, Any],
    *,
    include_observations: bool = False,
    observation_view_limit: Optional[int] = None,
) -> Dict[str, Any]:
    drop_keys = set()
    if not include_observations:
        drop_keys.update({"rgb_observations", "view_means", "view_cov6", "high_quality_views"})
    else:
        try:
            if observation_view_limit is not None:
                observation_view_limit = int(observation_view_limit)
        except Exception:
            observation_view_limit = None

        # Negative means "no limit".
        if observation_view_limit is not None and observation_view_limit < 0:
            observation_view_limit = None

    out: Dict[str, Any] = {}
    for key, value in (state or {}).items():
        if key in drop_keys:
            continue
        if (
            include_observations
            and observation_view_limit is not None
            and key in {"rgb_observations", "view_means", "view_cov6", "high_quality_views"}
        ):
            # These are list-of-lists, aligned by object index. Limit per-object views to reduce save size.
            rows = value if isinstance(value, list) else []
            limited_rows = []
            for row in rows:
                if not isinstance(row, list) or not row:
                    limited_rows.append([])
                    continue
                limited_rows.append(row[:observation_view_limit])
            out[key] = _to_cpu(limited_rows)
            continue
        if key == "images":
            images = value if isinstance(value, list) else []
            out[key] = [_to_cpu(_image_record_to_dict(r)) for r in images]
            continue
        if key == "loser_object_ids":
            loser = value if isinstance(value, list) else []
            normalized: List[List[int]] = []
            for entry in loser:
                if entry is None:
                    normalized.append([])
                    continue
                if isinstance(entry, set):
                    normalized.append(sorted(int(x) for x in entry))
                    continue
                if isinstance(entry, (list, tuple)):
                    normalized.append(sorted(int(x) for x in entry if x is not None))
                    continue
                normalized.append([])
            out[key] = normalized
            continue
        if key == "id_redirect":
            redirect = value if isinstance(value, dict) else {}
            normalized_redirect: Dict[int, int] = {}
            for k, v in redirect.items():
                try:
                    normalized_redirect[int(k)] = int(v)
                except Exception:
                    continue
            out[key] = normalized_redirect
            continue
        if key == "is_locked":
            locked = value if isinstance(value, list) else []
            out[key] = [bool(x) for x in locked]
            continue
        out[key] = _to_cpu(value)
    return out


def save_scene_state(
    path: str | Path,
    state: Dict[str, Any],
    *,
    feature_dim: int,
    include_observations: bool = False,
    observation_view_limit: Optional[int] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "format_version": SCENE_STATE_FORMAT_VERSION,
        "saved_unix_s": time.time(),
        "feature_dim": int(feature_dim),
        "state": make_serializable_scene_state(
            state,
            include_observations=include_observations,
            observation_view_limit=observation_view_limit,
        ),
    }
    if extra_metadata:
        payload["meta"] = dict(extra_metadata)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(out_path)
    return out_path


def _ensure_list_len(lst: list, length: int, fill_value: object) -> list:
    if lst is None or not isinstance(lst, list):
        lst = []
    if len(lst) > length:
        return lst[:length]
    while len(lst) < length:
        lst.append(fill_value() if callable(fill_value) else fill_value)
    return lst


def load_scene_state(
    path: str | Path,
    *,
    feature_dim: int,
    device: torch.device | str,
) -> Dict[str, Any]:
    load_path = Path(path).expanduser()
    # Scene-state files are generated by this pipeline and may contain numpy
    # arrays / Python containers in addition to tensors. PyTorch 2.6 defaults
    # torch.load(..., weights_only=True), which rejects those trusted payloads.
    payload = torch.load(load_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state" in payload:
        loaded = payload.get("state") or {}
        loaded_feature_dim = payload.get("feature_dim")
        if loaded_feature_dim is not None:
            try:
                loaded_feature_dim = int(loaded_feature_dim)
            except Exception:
                loaded_feature_dim = None
        if loaded_feature_dim is not None and int(loaded_feature_dim) != int(feature_dim):
            raise ValueError(f"scene_state feature_dim={loaded_feature_dim} != current feature_dim={feature_dim}")
    else:
        loaded = payload if isinstance(payload, dict) else {}

    restored = initialize_scene_graph_state(int(feature_dim), device)

    for key in restored.keys():
        if key not in loaded:
            continue
        restored[key] = loaded[key]

    means = restored.get("means")
    if not isinstance(means, torch.Tensor):
        return restored
    N = int(means.shape[0])

    torch_device = torch.device(device)
    for tensor_key in (
        "count",
        "means",
        "cov6",
        "features",
        "active",
        "class_ids",
        "covisibility_adj_u64",
        "covisibility_active_u64",
        "object_voxel_keys_flat",
        "object_voxel_keys_offsets",
        "object_voxel_levels",
    ):
        value = restored.get(tensor_key)
        if isinstance(value, torch.Tensor):
            restored[tensor_key] = value.to(device=torch_device)

    object_id = restored.get("object_id")
    if isinstance(object_id, torch.Tensor):
        restored["object_id"] = object_id.detach().cpu()

    # Pad voxel buffer to align with N if loaded from an older .pt without it.
    voxel_offsets = restored.get("object_voxel_keys_offsets")
    voxel_levels = restored.get("object_voxel_levels")
    expected_offset_len = N + 1
    if isinstance(voxel_offsets, torch.Tensor) and voxel_offsets.numel() != expected_offset_len:
        # Re-initialize to all-empty for this object count; flat stays empty.
        restored["object_voxel_keys_flat"] = torch.empty(
            (0,), dtype=torch.int64, device=torch_device
        )
        restored["object_voxel_keys_offsets"] = torch.zeros(
            (expected_offset_len,), dtype=torch.int64, device=torch_device
        )
        restored["object_voxel_levels"] = torch.zeros(
            (N,), dtype=torch.int8, device=torch_device
        )
    elif isinstance(voxel_levels, torch.Tensor) and voxel_levels.numel() != N:
        # Offsets aligned but levels missing/wrong length: zero-init levels.
        restored["object_voxel_levels"] = torch.zeros(
            (N,), dtype=torch.int8, device=torch_device
        )

    images = restored.get("images")
    if isinstance(images, list):
        restored["images"] = [_dict_to_image_record(entry) for entry in images]
    else:
        restored["images"] = []

    restored["object_caption"] = _ensure_list_len(restored.get("object_caption", []), N, "")
    restored["object_caption_decision"] = _ensure_list_len(restored.get("object_caption_decision", []), N, "")
    restored["object_category"] = _ensure_list_len(restored.get("object_category", []), N, "")
    restored["object_supercategory"] = _ensure_list_len(restored.get("object_supercategory", []), N, "")
    restored["object_category_candidates"] = _ensure_list_len(
        restored.get("object_category_candidates", []), N, list
    )
    restored["object_key_attributes"] = _ensure_list_len(restored.get("object_key_attributes", []), N, list)
    restored["object_caption_embedding"] = _ensure_list_len(restored.get("object_caption_embedding", []), N, list)
    restored["object_siglip2_embedding"] = _ensure_list_len(restored.get("object_siglip2_embedding", []), N, list)
    restored["object_qwen3_vl_embedding"] = _ensure_list_len(restored.get("object_qwen3_vl_embedding", []), N, list)
    restored["object_caption_history"] = _ensure_list_len(restored.get("object_caption_history", []), N, list)
    restored["object_caption_embedding_history"] = _ensure_list_len(
        restored.get("object_caption_embedding_history", []), N, list
    )
    restored["object_siglip2_embedding_history"] = _ensure_list_len(
        restored.get("object_siglip2_embedding_history", []), N, list
    )
    restored["object_qwen3_vl_embedding_history"] = _ensure_list_len(
        restored.get("object_qwen3_vl_embedding_history", []), N, list
    )
    restored["rgb_observations"] = _ensure_list_len(restored.get("rgb_observations", []), N, list)
    restored["object_image_ids"] = _ensure_list_len(restored.get("object_image_ids", []), N, list)
    restored["viewpoint_image_ids"] = _ensure_list_len(restored.get("viewpoint_image_ids", []), N, list)
    restored["object_mask_observations"] = _ensure_list_len(
        restored.get("object_mask_observations", []), N, list
    )
    restored["view_means"] = _ensure_list_len(restored.get("view_means", []), N, list)
    restored["view_cov6"] = _ensure_list_len(restored.get("view_cov6", []), N, list)
    restored["high_quality_captioning"] = _ensure_list_len(restored.get("high_quality_captioning", []), N, False)
    restored["high_quality_views"] = _ensure_list_len(restored.get("high_quality_views", []), N, list)
    normalized_hq_flags: List[bool] = []
    for idx in range(N):
        flag = bool(restored["high_quality_captioning"][idx])
        views_row = restored["high_quality_views"][idx]
        if not isinstance(views_row, list) or not views_row:
            flag = False
        normalized_hq_flags.append(flag)
    restored["high_quality_captioning"] = normalized_hq_flags

    det_state = restored.get("object_detection_category_conf")
    det_list: list = det_state if isinstance(det_state, list) else []
    det_list = _ensure_list_len(det_list, N, dict)
    normalized_det: List[Dict[str, float]] = []
    for entry in det_list[:N]:
        if not isinstance(entry, dict):
            normalized_det.append({})
            continue
        out: Dict[str, float] = {}
        for k, v in entry.items():
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
            out[key] = score
        normalized_det.append(out)
    restored["object_detection_category_conf"] = normalized_det

    loser = restored.get("loser_object_ids")
    loser_list: list = loser if isinstance(loser, list) else []
    loser_list = _ensure_list_len(loser_list, N, set)
    normalized_loser: List[set[int]] = []
    for entry in loser_list[:N]:
        if entry is None:
            normalized_loser.append(set())
        elif isinstance(entry, (set, list, tuple)):
            normalized_loser.append({int(x) for x in entry if x is not None})
        else:
            normalized_loser.append(set())
    restored["loser_object_ids"] = normalized_loser

    is_locked_raw = restored.get("is_locked")
    is_locked_list: list = is_locked_raw if isinstance(is_locked_raw, list) else []
    is_locked_list = _ensure_list_len(is_locked_list, N, False)
    normalized_locked: List[bool] = []
    for idx in range(N):
        entry = is_locked_list[idx] if idx < len(is_locked_list) else False
        normalized_locked.append(bool(entry))
    restored["is_locked"] = normalized_locked

    # Region fields (optional — absent in pre-region saves)
    region_ids_raw = restored.get("region_ids")
    if isinstance(region_ids_raw, list) and len(region_ids_raw) == N:
        restored["region_ids"] = [int(x) for x in region_ids_raw]
    else:
        restored["region_ids"] = [-1] * N
    restored["region_labels"] = restored.get("region_labels") if isinstance(restored.get("region_labels"), list) else []
    restored["region_object_lists"] = (
        restored.get("region_object_lists") if isinstance(restored.get("region_object_lists"), list) else []
    )
    restored["region_centroids"] = (
        restored.get("region_centroids") if isinstance(restored.get("region_centroids"), list) else []
    )
    restored["region_label_confidence"] = (
        restored.get("region_label_confidence") if isinstance(restored.get("region_label_confidence"), list) else []
    )
    restored["region_version"] = int(restored.get("region_version", 0) or 0)

    return restored
