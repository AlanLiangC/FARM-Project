from __future__ import annotations

import contextlib
import functools
import json
import logging
import math
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:  # torch is optional for the JSON+H5 load path; required by from_scene_state.
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

try:  # h5py is optional in some environments.
    import h5py
except Exception:  # pragma: no cover - optional dependency
    h5py = None

try:  # ROS2 client path is optional for offline use.
    import rclpy
except Exception:  # pragma: no cover - optional dependency
    rclpy = None

try:  # Plotting is optional; debug visualizations are skipped if unavailable.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None

try:  # Optional at import time for tests that don't build interfaces.
    from mapping_msgs.srv import Siglip2TextEmbed
except Exception:  # pragma: no cover - optional dependency
    Siglip2TextEmbed = None

LOGGER = logging.getLogger(__name__)
TIMING_LOG_LEVEL = logging.INFO


def _log_runtime(func=None, *, level: int | None = None, label: str | None = None):
    """Log wall runtime for instrumented functions.

    Uses `LOGGER.isEnabledFor(...)` to avoid timing overhead when disabled.
    """

    def decorate(target):
        name = label or getattr(target, "__qualname__", getattr(target, "__name__", "function"))
        log_level = TIMING_LOG_LEVEL if level is None else int(level)

        @functools.wraps(target)
        def wrapper(*args, **kwargs):
            if not LOGGER.isEnabledFor(log_level):
                return target(*args, **kwargs)
            t0 = time.perf_counter()
            try:
                return target(*args, **kwargs)
            finally:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                LOGGER.log(log_level, "[SceneGraphProcessing] %s dt_ms=%.2f", name, dt_ms)

        return wrapper

    if func is None:
        return decorate
    return decorate(func)


"""
Key functions to be used:
1. build_covisibility_graph(): build a covisibility graph based on shared viewpoint images/cameras and
   spatial proximity
2. get_covisible_neighbors(): given an object_id, return a list of covisible objects sorted by proximity
   (using the covisibility graph and means_xyz)
3. retrieve_by_caption(): given a text query, retrieve relevant objects based on caption embeddings and
   optional reranking
4. retrieve_by_siglip2(): given a text query, retrieve relevant objects based on siglip2 embeddings
5. retrieve_by_qwen3_vl(): given a text query, retrieve relevant objects based on Qwen3-VL crop embeddings
6. cluster_objects(): given a list of object_ids, cluster them based on covisibility and spatial proximity
7. minimum_covering_images(): given a list of object_ids, find a minimal set of viewpoint images that
   cover those objects, optionally requiring that the images have saved storage paths
"""


class SceneGraphProcessing:
    """
    Loader + retrieval utilities for scene graph JSON and snapshot state.h5.

    Expected input:
    - scene graph JSON published/saved by `streaming_mapper` (`{"objects":[...], "snapshot": {...}}`)
    - snapshot H5 pointed by `snapshot.state_path` or resolved via `latest.json`.
    """

    @_log_runtime
    def __init__(
        self,
        scene_graph_json_path: str | Path | None = None,
        *,
        snapshot_root: str | Path | None = None,
        siglip2_service_name: str = "/spot/mapping/siglip2_text_embed",
        siglip2_service_timeout_s: float = 0.35,
        siglip2_service_retries: int = 3,
        siglip2_service_backoff_s: float = 0.05,
        siglip2_client_id: int = 0,
        siglip2_prompt_topn_mean: int = 3,
        siglip2_logit_scale: float = 1.0,
        siglip2_logit_bias: float = 0.0,
    ) -> None:
        # ``scene_graph_json_path`` is None for the in-memory entry point
        # (:meth:`from_scene_state`), where data flows directly from a
        # :class:`SceneState` dict instead of from JSON+state.h5 on disk.
        self.scene_graph_json_path: Optional[Path] = (
            Path(scene_graph_json_path).expanduser() if scene_graph_json_path is not None else None
        )
        self.snapshot_root = Path(snapshot_root).expanduser() if snapshot_root else None
        # True when populated via from_scene_state(); load_latest() is a no-op
        # in that mode and the retriever's reload-on-mtime-change path skips it.
        self._loaded_from_scene_state: bool = False
        # Optional source path the in-memory state was loaded from (a .pt) — used by
        # the retriever to invalidate caches when the same processor is asked to
        # reload from a newer file.
        self._scene_state_path: Optional[Path] = None

        self.scene_graph_payload: Dict[str, Any] = {}
        self.objects: List[Dict[str, Any]] = []
        self.objects_by_id: Dict[str, Dict[str, Any]] = {}
        self.object_ids: List[str] = []
        self._object_key_to_index: Dict[str, int] = {}
        self.means_xyz = np.empty((0, 3), dtype=np.float32)
        # 6D packed covariance per object: [xx, xy, xz, yy, yz, zz] (matches utils.geometry.cov6_to_matrix).
        # Populated by from_scene_state(); the JSON+H5 path leaves it empty (callers that need
        # bboxes from cov6 should use the in-memory entry point).
        self.cov6 = np.empty((0, 6), dtype=np.float32)
        # Per-object sparse voxel point cloud. ``object_voxel_keys`` is a list
        # aligned with ``object_ids`` (post-keep_idx filtering); each entry is
        # an int64 ndarray of bit-packed voxel keys at level
        # ``object_voxel_levels[i]``. Decoded lazily by retrieval-time code.
        self.object_voxel_keys: List[np.ndarray] = []
        self.object_voxel_levels: List[int] = []

        self.snapshot_state_path: Optional[Path] = None
        self.snapshot_manifest_path: Optional[Path] = None
        self.snapshot_latest_path: Optional[Path] = None

        self._snapshot_object_id_to_row: Dict[str, int] = {}
        self._object_image_ids: List[List[int]] = []
        self._object_viewpoint_image_ids: List[List[int]] = []

        self.caption_embeddings = np.empty((0, 0), dtype=np.float32)
        self.siglip2_embeddings = np.empty((0, 0), dtype=np.float32)
        self.qwen3_vl_embeddings = np.empty((0, 0), dtype=np.float32)
        self.caption_embedding_history: List[np.ndarray] = []
        self.siglip2_embedding_history: List[np.ndarray] = []
        self.qwen3_vl_embedding_history: List[np.ndarray] = []
        self.caption_text_history: List[List[str]] = []

        self.image_ids = np.empty((0,), dtype=np.int64)
        self.image_poses = np.empty((0, 4, 4), dtype=np.float32)
        self.image_camera_ids: List[str] = []
        self.image_storage_paths: List[str] = []
        self._image_id_to_camera: Dict[int, str] = {}
        self._image_id_to_pose: Dict[int, np.ndarray] = {}
        self._image_id_to_position: Dict[int, np.ndarray] = {}
        self._image_id_to_storage_path: Dict[int, str] = {}
        self._image_id_to_object_indices: Dict[int, List[int]] = {}

        self.covisibility_matrix = np.zeros((0, 0), dtype=bool)

        self._siglip2_service_name = str(siglip2_service_name or "/spot/mapping/siglip2_text_embed")
        self._siglip2_service_timeout_s = max(0.05, float(siglip2_service_timeout_s))
        self._siglip2_service_retries = max(1, int(siglip2_service_retries))
        self._siglip2_service_backoff_s = max(0.0, float(siglip2_service_backoff_s))
        self._siglip2_client_id = int(siglip2_client_id)
        self._siglip2_prompt_topn_mean = max(1, int(siglip2_prompt_topn_mean))
        self._siglip2_logit_scale = float(siglip2_logit_scale)
        self._siglip2_logit_bias = float(siglip2_logit_bias)
        self._siglip2_client_lock = threading.Lock()
        self._siglip2_ros_context_owner = False
        self._siglip2_ros_node = None
        self._siglip2_ros_client = None

    @staticmethod
    @_log_runtime(level=logging.DEBUG)
    def _id_to_key(object_id: Any) -> str:
        return str(object_id)

    @staticmethod
    @_log_runtime(level=logging.DEBUG)
    def _safe_float(value: Any, default: float = float("nan")) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    @_log_runtime(level=logging.DEBUG)
    def _to_string_list(values: Sequence[Any]) -> List[str]:
        out: List[str] = []
        for value in values:
            if isinstance(value, bytes):
                with np.errstate(all="ignore"):
                    out.append(value.decode("utf-8", errors="replace"))
            else:
                out.append(str(value))
        return out

    @staticmethod
    @_log_runtime(level=logging.DEBUG)
    def _cosine_search(query: np.ndarray, vectors: np.ndarray, top_k: int) -> List[tuple[int, float]]:
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.shape[1] != query.shape[0]:
            return []

        query_norm = float(np.linalg.norm(query))
        if query_norm <= 0.0:
            return []
        query = query / query_norm

        vec_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vec_norms[vec_norms == 0] = 1.0
        vectors = vectors / vec_norms

        scores = vectors.dot(query)
        top_k = max(1, min(int(top_k), int(scores.shape[0])))
        idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in idx.tolist()]

    @staticmethod
    @_log_runtime(level=logging.DEBUG)
    def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
        mat = np.asarray(vectors, dtype=np.float32)
        if mat.ndim != 2 or mat.shape[0] == 0:
            return np.empty((0, 0), dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return mat / norms

    @staticmethod
    @_log_runtime(level=logging.DEBUG)
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(x, dtype=np.float32), -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    @staticmethod
    @_log_runtime(level=logging.DEBUG)
    def _build_siglip2_prompt_ensemble(task_description: str) -> List[str]:
        query = " ".join(str(task_description or "").strip().split())
        if not query:
            return []
        q_lower = query.lower()

        article_prompt = query
        if not q_lower.startswith(("a ", "an ", "the ")):
            article = "an" if q_lower[:1] in {"a", "e", "i", "o", "u"} else "a"
            article_prompt = f"{article} {query}"

        candidates = [
            query,
            f"a photo of {query}",
            f"a picture of {query}",
            f"{query}, product photo",
            f"{query} on a table",
            f"close-up of {query}",
            f"{query}, blurry low-resolution photo",
            article_prompt,
        ]
        out: List[str] = []
        seen: set[str] = set()
        for text in candidates:
            text_norm = str(text or "").strip()
            if not text_norm:
                continue
            key = text_norm.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text_norm)
        return out

    @_log_runtime(level=logging.DEBUG)
    def _resolve_existing_path(self, candidate: str | Path | None) -> Optional[Path]:
        if not candidate:
            return None
        raw = Path(candidate).expanduser()
        if raw.is_absolute() and raw.exists():
            return raw.resolve()

        bases: List[Path] = [Path.cwd(), self.scene_graph_json_path.parent, self.scene_graph_json_path.parent.parent]
        if self.snapshot_root is not None:
            bases.extend([self.snapshot_root, self.snapshot_root.parent])

        # Snapshot metadata is often produced on a different machine and can contain
        # absolute paths like ".../snapshots/v000027/state.h5". When those absolute
        # paths do not exist locally, recover by remapping the suffix under the local
        # snapshot roots.
        remapped_suffixes: List[Path] = []
        if raw.is_absolute():
            parts = list(raw.parts)
            if "snapshots" in parts:
                snap_idx = parts.index("snapshots")
                if snap_idx + 1 < len(parts):
                    remapped_suffixes.append(Path(*parts[snap_idx + 1 :]))
            remapped_suffixes.append(Path(raw.name))

        # Preserve insertion order while removing duplicates.
        dedup_bases: List[Path] = []
        seen_bases: set[str] = set()
        for base in bases:
            key = str(base)
            if key in seen_bases:
                continue
            seen_bases.add(key)
            dedup_bases.append(base)

        # Candidate-relative paths in scene graph snapshot metadata are often rooted at "log/...".
        # If we already have a base ending in ".../log", avoid producing ".../log/log/...".
        trimmed_raw: Optional[Path] = None
        if len(raw.parts) >= 2 and raw.parts[0] == "log":
            trimmed_raw = Path(*raw.parts[1:])

        trimmed_snapshots_raw: Optional[Path] = None
        if len(raw.parts) >= 3 and raw.parts[0] == "log" and raw.parts[1] == "snapshots":
            trimmed_snapshots_raw = Path(*raw.parts[2:])

        tests: List[Path] = []
        for base in dedup_bases:
            tests.append(base / raw)
            for suffix in remapped_suffixes:
                tests.append(base / suffix)
                if base.name != "snapshots":
                    tests.append(base / "snapshots" / suffix)
            if trimmed_raw is not None and base.name == "log":
                tests.append(base / trimmed_raw)
            if trimmed_snapshots_raw is not None and base.name == "snapshots":
                tests.append(base / trimmed_snapshots_raw)

        # Also try literal relative path against cwd as a fallback.
        tests.append(raw)

        seen_tests: set[str] = set()
        for test_path in tests:
            key = str(test_path)
            if key in seen_tests:
                continue
            seen_tests.add(key)
            if test_path.exists():
                return test_path.resolve()
        return None

    @_log_runtime
    def _resolve_snapshot_paths(self, payload: Dict[str, Any]) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
        snapshot = payload.get("snapshot") if isinstance(payload.get("snapshot"), dict) else {}

        state_path = self._resolve_existing_path(snapshot.get("state_path"))
        manifest_path = self._resolve_existing_path(snapshot.get("manifest_path"))
        latest_path = self._resolve_existing_path(snapshot.get("latest_path"))

        if state_path is None and manifest_path is not None and manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest_data = json.load(handle) or {}
            state_from_manifest = (
                ((manifest_data.get("paths") or {}).get("state_h5")) if isinstance(manifest_data, dict) else None
            )
            state_path = self._resolve_existing_path(state_from_manifest)

        if state_path is None:
            latest_candidates: List[Path] = []
            if self.snapshot_root is not None:
                latest_candidates.append(self.snapshot_root / "latest.json")
            latest_candidates.append(self.scene_graph_json_path.parent / "snapshots" / "latest.json")
            latest_candidates.append(Path("log/snapshots/latest.json").expanduser())

            latest_found = next((path.resolve() for path in latest_candidates if path.exists()), None)
            if latest_found is not None:
                latest_path = latest_found
                with latest_found.open("r", encoding="utf-8") as handle:
                    latest_data = json.load(handle) or {}
                state_path = self._resolve_existing_path(latest_data.get("state_path"))
                manifest_path = manifest_path or self._resolve_existing_path(latest_data.get("manifest_path"))

        return state_path, manifest_path, latest_path

    @_log_runtime
    def _build_object_tables(self, payload: Dict[str, Any]) -> None:
        objects_raw = payload.get("objects")
        if not isinstance(objects_raw, list):
            objects_raw = []

        self.objects = []
        self.objects_by_id = {}
        self.object_ids = []
        self._object_key_to_index = {}

        means_rows: List[List[float]] = []
        for obj in objects_raw:
            if not isinstance(obj, dict):
                continue
            object_id = obj.get("object_id", obj.get("id"))
            if object_id is None:
                continue
            object_key = self._id_to_key(object_id)
            if object_key in self.objects_by_id:
                continue
            self.objects.append(obj)
            self.objects_by_id[object_key] = obj
            self.object_ids.append(object_key)
            self._object_key_to_index[object_key] = len(self.object_ids) - 1

            mean = obj.get("mean")
            if isinstance(mean, (list, tuple)) and len(mean) >= 3:
                means_rows.append([
                    self._safe_float(mean[0]),
                    self._safe_float(mean[1]),
                    self._safe_float(mean[2]),
                ])
            else:
                means_rows.append([float("nan"), float("nan"), float("nan")])

        self.means_xyz = np.asarray(means_rows, dtype=np.float32) if means_rows else np.empty((0, 3), dtype=np.float32)

    @_log_runtime
    def _load_snapshot_h5(self, state_path: Path) -> None:
        if h5py is None:
            raise RuntimeError("h5py is required to load scene graph snapshots")
        if not state_path.exists():
            raise FileNotFoundError(f"Snapshot file not found: {state_path}")

        with h5py.File(state_path, "r") as handle:
            snapshot_object_ids = handle["object_id"][:] if "object_id" in handle else np.empty((0,), dtype=np.int64)
            snapshot_object_keys = [self._id_to_key(x) for x in np.asarray(snapshot_object_ids).tolist()]
            self._snapshot_object_id_to_row = {key: idx for idx, key in enumerate(snapshot_object_keys)}

            obj_img_row_ptr = (
                handle["object_image_row_ptr"][:] if "object_image_row_ptr" in handle else np.zeros((1,), np.int64)
            )
            obj_img_ids = (
                handle["object_image_ids"][:] if "object_image_ids" in handle else np.empty((0,), dtype=np.int64)
            )
            vp_row_ptr = handle["viewpoint_row_ptr"][:] if "viewpoint_row_ptr" in handle else np.zeros((1,), np.int64)
            vp_image_ids = (
                handle["viewpoint_image_ids"][:] if "viewpoint_image_ids" in handle else np.empty((0,), dtype=np.int64)
            )

            caption_all = (
                np.asarray(handle["caption_embedding"][:], dtype=np.float32)
                if "caption_embedding" in handle
                else np.empty((len(snapshot_object_keys), 0), dtype=np.float32)
            )
            siglip_all = (
                np.asarray(handle["siglip2_embedding"][:], dtype=np.float32)
                if "siglip2_embedding" in handle
                else np.empty((len(snapshot_object_keys), 0), dtype=np.float32)
            )
            qwen3_vl_all = (
                np.asarray(handle["qwen3_vl_embedding"][:], dtype=np.float32)
                if "qwen3_vl_embedding" in handle
                else np.empty((len(snapshot_object_keys), 0), dtype=np.float32)
            )
            caption_hist_row_ptr = (
                np.asarray(handle["caption_embedding_hist_row_ptr"][:], dtype=np.int64)
                if "caption_embedding_hist_row_ptr" in handle
                else np.zeros((len(snapshot_object_keys) + 1,), dtype=np.int64)
            )
            caption_hist_all = (
                np.asarray(handle["caption_embedding_hist"][:], dtype=np.float32)
                if "caption_embedding_hist" in handle
                else np.empty((0, 0), dtype=np.float32)
            )
            siglip2_hist_row_ptr = (
                np.asarray(handle["siglip2_embedding_hist_row_ptr"][:], dtype=np.int64)
                if "siglip2_embedding_hist_row_ptr" in handle
                else np.zeros((len(snapshot_object_keys) + 1,), dtype=np.int64)
            )
            siglip2_hist_all = (
                np.asarray(handle["siglip2_embedding_hist"][:], dtype=np.float32)
                if "siglip2_embedding_hist" in handle
                else np.empty((0, 0), dtype=np.float32)
            )
            qwen3_vl_hist_row_ptr = (
                np.asarray(handle["qwen3_vl_embedding_hist_row_ptr"][:], dtype=np.int64)
                if "qwen3_vl_embedding_hist_row_ptr" in handle
                else np.zeros((len(snapshot_object_keys) + 1,), dtype=np.int64)
            )
            qwen3_vl_hist_all = (
                np.asarray(handle["qwen3_vl_embedding_hist"][:], dtype=np.float32)
                if "qwen3_vl_embedding_hist" in handle
                else np.empty((0, 0), dtype=np.float32)
            )
            caption_text_hist_row_ptr = (
                np.asarray(handle["caption_text_hist_row_ptr"][:], dtype=np.int64)
                if "caption_text_hist_row_ptr" in handle
                else np.zeros((len(snapshot_object_keys) + 1,), dtype=np.int64)
            )
            caption_text_hist_all = (
                self._to_string_list(handle["caption_text_hist"][:]) if "caption_text_hist" in handle else []
            )

            self.image_ids = handle["image_ids"][:] if "image_ids" in handle else np.empty((0,), dtype=np.int64)
            self.image_poses = (
                np.asarray(handle["image_poses"][:], dtype=np.float32)
                if "image_poses" in handle
                else np.empty((0, 4, 4), dtype=np.float32)
            )
            self.image_camera_ids = (
                self._to_string_list(handle["image_camera_ids"][:]) if "image_camera_ids" in handle else []
            )
            self.image_storage_paths = (
                self._to_string_list(handle["image_storage_paths"][:]) if "image_storage_paths" in handle else []
            )

        self._image_id_to_camera = {}
        self._image_id_to_pose = {}
        self._image_id_to_position = {}
        self._image_id_to_storage_path = {}
        for idx, image_id in enumerate(np.asarray(self.image_ids).tolist()):
            try:
                image_id_int = int(image_id)
            except Exception:
                continue
            if idx < len(self.image_poses):
                pose = np.asarray(self.image_poses[idx], dtype=np.float32)
                if pose.shape == (4, 4):
                    self._image_id_to_pose[image_id_int] = pose
                    position = np.asarray(pose[:3, 3], dtype=np.float32)
                    if position.shape == (3,) and np.all(np.isfinite(position)):
                        self._image_id_to_position[image_id_int] = position
            if idx < len(self.image_camera_ids):
                self._image_id_to_camera[image_id_int] = self.image_camera_ids[idx]
            if idx < len(self.image_storage_paths):
                self._image_id_to_storage_path[image_id_int] = self.image_storage_paths[idx]

        self._image_id_to_object_indices = {}

        object_count = len(self.object_ids)
        cap_dim = int(caption_all.shape[1]) if caption_all.ndim == 2 else 0
        sig_dim = int(siglip_all.shape[1]) if siglip_all.ndim == 2 else 0
        qwen3_vl_dim = int(qwen3_vl_all.shape[1]) if qwen3_vl_all.ndim == 2 else 0

        self.caption_embeddings = np.zeros((object_count, cap_dim), dtype=np.float32)
        self.siglip2_embeddings = np.zeros((object_count, sig_dim), dtype=np.float32)
        self.qwen3_vl_embeddings = np.zeros((object_count, qwen3_vl_dim), dtype=np.float32)
        self.caption_embedding_history = [
            np.zeros((0, cap_dim), dtype=np.float32) if cap_dim > 0 else np.empty((0, 0), dtype=np.float32)
            for _ in range(object_count)
        ]
        self.siglip2_embedding_history = [
            np.zeros((0, sig_dim), dtype=np.float32) if sig_dim > 0 else np.empty((0, 0), dtype=np.float32)
            for _ in range(object_count)
        ]
        self.qwen3_vl_embedding_history = [
            np.zeros((0, qwen3_vl_dim), dtype=np.float32) if qwen3_vl_dim > 0 else np.empty((0, 0), dtype=np.float32)
            for _ in range(object_count)
        ]
        self.caption_text_history = [[] for _ in range(object_count)]
        self._object_image_ids = [[] for _ in range(object_count)]
        self._object_viewpoint_image_ids = [[] for _ in range(object_count)]

        for obj_idx, object_key in enumerate(self.object_ids):
            snapshot_row = self._snapshot_object_id_to_row.get(object_key)
            if snapshot_row is None:
                continue

            if cap_dim > 0 and 0 <= snapshot_row < caption_all.shape[0]:
                self.caption_embeddings[obj_idx] = caption_all[snapshot_row]
            if sig_dim > 0 and 0 <= snapshot_row < siglip_all.shape[0]:
                self.siglip2_embeddings[obj_idx] = siglip_all[snapshot_row]
            if qwen3_vl_dim > 0 and 0 <= snapshot_row < qwen3_vl_all.shape[0]:
                self.qwen3_vl_embeddings[obj_idx] = qwen3_vl_all[snapshot_row]

            if (
                caption_hist_all.ndim == 2
                and caption_hist_all.shape[1] > 0
                and 0 <= snapshot_row + 1 < len(caption_hist_row_ptr)
            ):
                start = int(caption_hist_row_ptr[snapshot_row])
                end = int(caption_hist_row_ptr[snapshot_row + 1])
                if 0 <= start <= end <= int(caption_hist_all.shape[0]):
                    self.caption_embedding_history[obj_idx] = np.asarray(caption_hist_all[start:end], dtype=np.float32)

            if (
                siglip2_hist_all.ndim == 2
                and siglip2_hist_all.shape[1] > 0
                and 0 <= snapshot_row + 1 < len(siglip2_hist_row_ptr)
            ):
                start = int(siglip2_hist_row_ptr[snapshot_row])
                end = int(siglip2_hist_row_ptr[snapshot_row + 1])
                if 0 <= start <= end <= int(siglip2_hist_all.shape[0]):
                    self.siglip2_embedding_history[obj_idx] = np.asarray(siglip2_hist_all[start:end], dtype=np.float32)

            if (
                qwen3_vl_hist_all.ndim == 2
                and qwen3_vl_hist_all.shape[1] > 0
                and 0 <= snapshot_row + 1 < len(qwen3_vl_hist_row_ptr)
            ):
                start = int(qwen3_vl_hist_row_ptr[snapshot_row])
                end = int(qwen3_vl_hist_row_ptr[snapshot_row + 1])
                if 0 <= start <= end <= int(qwen3_vl_hist_all.shape[0]):
                    self.qwen3_vl_embedding_history[obj_idx] = np.asarray(
                        qwen3_vl_hist_all[start:end], dtype=np.float32
                    )

            if 0 <= snapshot_row + 1 < len(caption_text_hist_row_ptr):
                start = int(caption_text_hist_row_ptr[snapshot_row])
                end = int(caption_text_hist_row_ptr[snapshot_row + 1])
                if 0 <= start <= end <= len(caption_text_hist_all):
                    self.caption_text_history[obj_idx] = [
                        str(x or "").strip() for x in caption_text_hist_all[start:end] if str(x or "").strip()
                    ]

            full_ids: List[int] = []
            if 0 <= snapshot_row + 1 < len(obj_img_row_ptr):
                start = int(obj_img_row_ptr[snapshot_row])
                end = int(obj_img_row_ptr[snapshot_row + 1])
                if 0 <= start <= end <= len(obj_img_ids):
                    full_ids = [int(x) for x in obj_img_ids[start:end].tolist()]
            self._object_image_ids[obj_idx] = full_ids

            viewpoint_ids: List[int] = []
            if 0 <= snapshot_row + 1 < len(vp_row_ptr):
                start = int(vp_row_ptr[snapshot_row])
                end = int(vp_row_ptr[snapshot_row + 1])
                if 0 <= start <= end <= len(vp_image_ids):
                    viewpoint_ids = [int(x) for x in vp_image_ids[start:end].tolist()]

            # Prefer full per-object image connectivity when available; fall back to
            # the reduced viewpoint subset for backwards compatibility with older snapshots.
            active_image_ids = full_ids if full_ids else viewpoint_ids
            self._object_viewpoint_image_ids[obj_idx] = active_image_ids
            for image_id_int in active_image_ids:
                self._image_id_to_object_indices.setdefault(int(image_id_int), []).append(int(obj_idx))

    @_log_runtime
    def load_latest(self) -> None:
        if self._loaded_from_scene_state:
            # Data was supplied directly via from_scene_state(); no JSON/H5 to refresh.
            return
        if self.scene_graph_json_path is None:
            raise RuntimeError(
                "SceneGraphProcessing.load_latest() requires scene_graph_json_path to be set "
                "(or use SceneGraphProcessing.from_scene_state(...) for the in-memory path)."
            )
        if not self.scene_graph_json_path.exists():
            raise FileNotFoundError(f"Scene graph JSON not found: {self.scene_graph_json_path}")
        with self.scene_graph_json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError("Scene graph JSON must be a JSON object")

        self.scene_graph_payload = payload
        self._build_object_tables(payload)

        state_path, manifest_path, latest_path = self._resolve_snapshot_paths(payload)
        self.snapshot_state_path = state_path
        self.snapshot_manifest_path = manifest_path
        self.snapshot_latest_path = latest_path

        if state_path is None:
            raise RuntimeError("Unable to resolve snapshot state.h5 from scene graph JSON or latest.json")
        self._load_snapshot_h5(state_path)
        self.covisibility_matrix = np.zeros((len(self.object_ids), len(self.object_ids)), dtype=bool)

    # ------------------------------------------------------------------
    # In-memory entry point: build a processor directly from a SceneState dict.
    # Bypasses the JSON-manifest / state.h5 round-trip the live ROS pipeline uses,
    # so the offline runner's saved scene_state.pt feeds the retriever directly.
    # ------------------------------------------------------------------
    @classmethod
    @_log_runtime
    def from_scene_state(
        cls,
        state: "Dict[str, Any] | str | Path",
        *,
        snapshot_root: "str | Path | None" = None,
        feature_dim: Optional[int] = None,
        scene_state_path: "str | Path | None" = None,
        min_obs_for_retrieval: int = 1,
    ) -> "SceneGraphProcessing":
        """Build a fully-populated :class:`SceneGraphProcessing` from a SceneState.

        Args:
            state: Either an in-memory SceneState dict (the shape produced by
                :func:`scene_graph.map_update.models.initialize_scene_graph_state`
                and round-tripped by :func:`scene_graph.scene_state_io.save_scene_state`)
                or a path to a saved ``.pt`` file.
            snapshot_root: Optional directory the retriever uses for image-path
                resolution; mostly useful when ``ImageRecord.storage_path`` values
                are relative.
            feature_dim: Required when ``state`` is a path. Read from the .pt
                metadata when not supplied.
            scene_state_path: Stored on the processor for cache invalidation;
                set automatically when ``state`` is a path.
        """
        if torch is None:
            raise RuntimeError("from_scene_state requires PyTorch; install torch to use this entry point.")
        # 1. Resolve the state dict (load the .pt if needed).
        if isinstance(state, (str, Path)):
            from scene_graph.scene_state_io import load_scene_state

            pt_path = Path(state).expanduser()
            if feature_dim is None:
                payload = torch.load(pt_path, map_location="cpu", weights_only=False)
                if isinstance(payload, dict):
                    feature_dim = payload.get("feature_dim")
                    if feature_dim is None:
                        inner = payload.get("state") or payload
                        feats = inner.get("features") if isinstance(inner, dict) else None
                        if isinstance(feats, torch.Tensor) and feats.ndim == 2:
                            feature_dim = int(feats.shape[1])
                if feature_dim is None:
                    raise ValueError(f"Could not infer feature_dim from {pt_path}")
            state_dict = load_scene_state(pt_path, feature_dim=int(feature_dim), device="cpu")
            scene_state_path = scene_state_path or pt_path
        elif isinstance(state, dict):
            state_dict = state
        else:
            raise TypeError(f"state must be a dict or path, got {type(state).__name__}")

        instance = cls(scene_graph_json_path=None, snapshot_root=snapshot_root)
        instance._loaded_from_scene_state = True
        if scene_state_path is not None:
            instance._scene_state_path = Path(scene_state_path).expanduser()
        instance._populate_from_scene_state(state_dict, min_obs_for_retrieval=min_obs_for_retrieval)
        return instance

    @staticmethod
    def _pose_matrix_to_xyz_quat(matrix: np.ndarray) -> Optional[List[float]]:
        """Decompose a 4×4 SE(3) matrix into ``[x, y, z, qx, qy, qz, qw]``.

        Returns None for non-finite or non-(4,4) input. Mirrors the convention
        in ``mapping.lib.geometric_transforms.xyz_quat_from_matrix`` so the
        viewpoints emitted here match what the live ROS path emits.
        """
        if matrix is None:
            return None
        m = np.asarray(matrix, dtype=np.float32)
        if m.shape != (4, 4) or not np.all(np.isfinite(m)):
            return None
        rot = m[:3, :3]
        trans = m[:3, 3]
        trace = float(np.trace(rot))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (rot[2, 1] - rot[1, 2]) / s
            qy = (rot[0, 2] - rot[2, 0]) / s
            qz = (rot[1, 0] - rot[0, 1]) / s
        elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            s = math.sqrt(max(0.0, 1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])) * 2.0
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s if s else 0.0
            qz = (rot[0, 2] + rot[2, 0]) / s if s else 0.0
            qw = (rot[2, 1] - rot[1, 2]) / s if s else 0.0
        elif rot[1, 1] > rot[2, 2]:
            s = math.sqrt(max(0.0, 1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])) * 2.0
            qx = (rot[0, 1] + rot[1, 0]) / s if s else 0.0
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s if s else 0.0
            qw = (rot[0, 2] - rot[2, 0]) / s if s else 0.0
        else:
            s = math.sqrt(max(0.0, 1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])) * 2.0
            qx = (rot[0, 2] + rot[2, 0]) / s if s else 0.0
            qy = (rot[1, 2] + rot[2, 1]) / s if s else 0.0
            qz = 0.25 * s
            qw = (rot[1, 0] - rot[0, 1]) / s if s else 0.0
        return [float(trans[0]), float(trans[1]), float(trans[2]),
                float(qx), float(qy), float(qz), float(qw)]

    @staticmethod
    def _vec_to_np(value: Any, *, dim_hint: int = 0) -> np.ndarray:
        """Coerce a per-object embedding entry to a 1-D float32 array."""
        if value is None:
            return np.zeros(dim_hint, dtype=np.float32) if dim_hint > 0 else np.empty(0, dtype=np.float32)
        if isinstance(value, np.ndarray):
            arr = value
        elif isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        elif isinstance(value, (list, tuple)):
            if not value:
                return np.zeros(dim_hint, dtype=np.float32) if dim_hint > 0 else np.empty(0, dtype=np.float32)
            arr = np.asarray(value, dtype=np.float32)
        else:
            return np.zeros(dim_hint, dtype=np.float32) if dim_hint > 0 else np.empty(0, dtype=np.float32)
        return arr.astype(np.float32, copy=False).reshape(-1)

    @_log_runtime
    def _populate_from_scene_state(self, state: Dict[str, Any], *,
                                    min_obs_for_retrieval: int = 1) -> None:
        """Mirror what ``_build_object_tables`` + ``_load_snapshot_h5`` do, but
        from an in-memory SceneState dict instead of JSON+H5 on disk.
        """
        # ---- raw pulls (handle torch tensors / numpy / lists uniformly) -----
        def _to_np_1d(t: Any, dtype: Any) -> np.ndarray:
            if isinstance(t, torch.Tensor):
                return t.detach().cpu().numpy().astype(dtype, copy=False).reshape(-1)
            if t is None:
                return np.empty((0,), dtype=dtype)
            return np.asarray(t, dtype=dtype).reshape(-1)

        def _to_np_2d(t: Any, dtype: Any) -> np.ndarray:
            if isinstance(t, torch.Tensor):
                return t.detach().cpu().numpy().astype(dtype, copy=False)
            if t is None:
                return np.empty((0, 0), dtype=dtype)
            return np.asarray(t, dtype=dtype)

        means_np = _to_np_2d(state.get("means"), np.float32)
        if means_np.ndim != 2 or means_np.shape[1] != 3:
            means_np = np.empty((0, 3), dtype=np.float32)
        cov6_np = _to_np_2d(state.get("cov6"), np.float32)
        if cov6_np.ndim != 2 or cov6_np.shape[1] != 6 or cov6_np.shape[0] != means_np.shape[0]:
            cov6_np = np.empty((0, 6), dtype=np.float32)
        active_np = _to_np_1d(state.get("active"), bool) if state.get("active") is not None else \
            np.ones((means_np.shape[0],), dtype=bool)
        object_id_np = _to_np_1d(state.get("object_id"), np.int64) if state.get("object_id") is not None else \
            np.arange(means_np.shape[0], dtype=np.int64)

        captions_seq: Sequence[str] = state.get("object_caption") or []
        cap_emb_seq: Sequence[Any] = state.get("object_caption_embedding") or []
        sig_emb_seq: Sequence[Any] = state.get("object_siglip2_embedding") or []
        qwn_emb_seq: Sequence[Any] = state.get("object_qwen3_vl_embedding") or []
        cap_text_hist_seq: Sequence[Sequence[str]] = state.get("object_caption_history") or []
        cap_emb_hist_seq: Sequence[Any] = state.get("object_caption_embedding_history") or []
        sig_emb_hist_seq: Sequence[Any] = state.get("object_siglip2_embedding_history") or []
        qwn_emb_hist_seq: Sequence[Any] = state.get("object_qwen3_vl_embedding_history") or []
        obj_image_ids_seq: Sequence[Sequence[int]] = state.get("object_image_ids") or []
        viewpoint_image_ids_seq: Sequence[Sequence[int]] = state.get("viewpoint_image_ids") or []
        images_seq: Sequence[Any] = state.get("images") or []

        # ---- 1. images table (so we can resolve viewpoint poses below) ------
        n_images = len(images_seq)
        image_ids_list: List[int] = []
        image_poses = np.full((n_images, 4, 4), np.nan, dtype=np.float32)
        image_camera_ids: List[str] = []
        image_storage_paths: List[str] = []
        for idx, record in enumerate(images_seq):
            image_ids_list.append(int(getattr(record, "image_id", idx)))
            pose = getattr(record, "pose", None)
            if pose is not None:
                try:
                    pose_np = (
                        pose.detach().to("cpu", copy=False).numpy()
                        if isinstance(pose, torch.Tensor)
                        else np.asarray(pose, dtype=np.float32)
                    )
                    if pose_np.shape == (4, 4):
                        image_poses[idx] = pose_np.astype(np.float32, copy=False)
                except Exception:
                    pass
            image_camera_ids.append(str(getattr(record, "camera_id", "") or ""))
            image_storage_paths.append(str(getattr(record, "storage_path", "") or ""))

        self.image_ids = np.asarray(image_ids_list, dtype=np.int64)
        self.image_poses = image_poses
        self.image_camera_ids = image_camera_ids
        self.image_storage_paths = image_storage_paths

        self._image_id_to_camera = {}
        self._image_id_to_pose = {}
        self._image_id_to_position = {}
        self._image_id_to_storage_path = {}
        for idx, image_id in enumerate(image_ids_list):
            try:
                image_id_int = int(image_id)
            except Exception:
                continue
            pose = image_poses[idx]
            if pose.shape == (4, 4) and np.all(np.isfinite(pose)):
                self._image_id_to_pose[image_id_int] = pose
                position = np.asarray(pose[:3, 3], dtype=np.float32)
                if position.shape == (3,) and np.all(np.isfinite(position)):
                    self._image_id_to_position[image_id_int] = position
            self._image_id_to_camera[image_id_int] = image_camera_ids[idx]
            self._image_id_to_storage_path[image_id_int] = image_storage_paths[idx]

        # ---- 2. determine which objects to keep --------------------------
        # Mirror take_scene_graph_snapshot: active AND finite mean AND non-empty caption.
        # P3 transient filter: require >= ``min_obs_for_retrieval`` observations
        # to participate in retrieval. Default 1 (no filtering).
        n_total = int(min(means_np.shape[0], active_np.shape[0], object_id_np.shape[0]))
        min_obs = max(1, int(min_obs_for_retrieval))
        keep_idx: List[int] = []
        for idx in range(n_total):
            if not bool(active_np[idx]):
                continue
            if not np.all(np.isfinite(means_np[idx])):
                continue
            cap = str(captions_seq[idx]).strip() if idx < len(captions_seq) else ""
            if not cap:
                continue
            if min_obs > 1:
                ids_seq = obj_image_ids_seq[idx] if idx < len(obj_image_ids_seq) else []
                if not isinstance(ids_seq, (list, tuple)):
                    n_obs = 0
                else:
                    n_obs = len(ids_seq)
                if n_obs < min_obs:
                    continue
            keep_idx.append(idx)

        # ---- 3. object dicts ---------------------------------------------
        self.objects = []
        self.objects_by_id = {}
        self.object_ids = []
        self._object_key_to_index = {}
        means_rows: List[List[float]] = []
        for new_idx, obj_idx in enumerate(keep_idx):
            object_id = int(object_id_np[obj_idx])
            object_key = self._id_to_key(object_id)
            mean_vec = means_np[obj_idx]
            mean_list = [float(mean_vec[0]), float(mean_vec[1]), float(mean_vec[2])]
            caption = str(captions_seq[obj_idx]).strip()

            viewpoints_payload: List[List[float]] = []
            viewpoint_camera_ids_payload: List[str] = []
            vp_ids = viewpoint_image_ids_seq[obj_idx] if obj_idx < len(viewpoint_image_ids_seq) else []
            for image_id in vp_ids:
                try:
                    image_id_int = int(image_id)
                except Exception:
                    continue
                if image_id_int < 0 or image_id_int >= len(images_seq):
                    continue
                pose = getattr(images_seq[image_id_int], "pose", None)
                if pose is None:
                    continue
                try:
                    pose_np = (
                        pose.detach().to("cpu", copy=False).numpy()
                        if isinstance(pose, torch.Tensor)
                        else np.asarray(pose, dtype=np.float32)
                    )
                except Exception:
                    continue
                payload = self._pose_matrix_to_xyz_quat(pose_np)
                if payload is None:
                    continue
                viewpoints_payload.append(payload)
                viewpoint_camera_ids_payload.append(
                    str(getattr(images_seq[image_id_int], "camera_id", "") or "")
                )

            obj_dict: Dict[str, Any] = {
                "object_id": object_id,
                "mean": mean_list,
                "object_caption": caption,
                "viewpoints": viewpoints_payload,
                "viewpoint_camera_ids": viewpoint_camera_ids_payload,
            }
            self.objects.append(obj_dict)
            self.objects_by_id[object_key] = obj_dict
            self.object_ids.append(object_key)
            self._object_key_to_index[object_key] = new_idx
            means_rows.append(mean_list)

        self.means_xyz = (
            np.asarray(means_rows, dtype=np.float32) if means_rows else np.empty((0, 3), dtype=np.float32)
        )
        if cov6_np.shape[0] > 0 and keep_idx:
            self.cov6 = np.asarray([cov6_np[i] for i in keep_idx], dtype=np.float32)
        else:
            self.cov6 = np.empty((len(keep_idx), 6), dtype=np.float32)
        self._snapshot_object_id_to_row = {key: i for i, key in enumerate(self.object_ids)}

        # ---- voxel cloud (sparse) sliced by keep_idx ----
        voxel_keys_flat_t = state.get("object_voxel_keys_flat")
        voxel_offsets_t = state.get("object_voxel_keys_offsets")
        voxel_levels_t = state.get("object_voxel_levels")
        if (
            isinstance(voxel_keys_flat_t, torch.Tensor)
            and isinstance(voxel_offsets_t, torch.Tensor)
            and isinstance(voxel_levels_t, torch.Tensor)
            and voxel_offsets_t.numel() > 1
        ):
            keys_np = voxel_keys_flat_t.detach().cpu().numpy().astype(np.int64, copy=False)
            offsets_np = voxel_offsets_t.detach().cpu().numpy().astype(np.int64, copy=False)
            levels_np = voxel_levels_t.detach().cpu().numpy().astype(np.int8, copy=False)
            n_persisted = max(0, offsets_np.shape[0] - 1)
            obj_keys: List[np.ndarray] = []
            obj_levels: List[int] = []
            for obj_idx in keep_idx:
                if 0 <= obj_idx < n_persisted:
                    s, e = int(offsets_np[obj_idx]), int(offsets_np[obj_idx + 1])
                    obj_keys.append(keys_np[s:e])
                    obj_levels.append(int(levels_np[obj_idx]) if obj_idx < levels_np.shape[0] else 0)
                else:
                    obj_keys.append(np.empty((0,), dtype=np.int64))
                    obj_levels.append(0)
            self.object_voxel_keys = obj_keys
            self.object_voxel_levels = obj_levels
        else:
            self.object_voxel_keys = [np.empty((0,), dtype=np.int64) for _ in keep_idx]
            self.object_voxel_levels = [0 for _ in keep_idx]

        # ---- 4. embedding matrices (with consistent dim per modality) -----
        def _detect_dim(rows: Sequence[Any], indices: Sequence[int]) -> int:
            for j in indices:
                if j >= len(rows):
                    continue
                vec = self._vec_to_np(rows[j])
                if vec.size > 0:
                    return int(vec.size)
            return 0

        cap_dim = _detect_dim(cap_emb_seq, keep_idx)
        sig_dim = _detect_dim(sig_emb_seq, keep_idx)
        qwn_dim = _detect_dim(qwn_emb_seq, keep_idx)
        n = len(keep_idx)
        self.caption_embeddings = np.zeros((n, cap_dim), dtype=np.float32)
        self.siglip2_embeddings = np.zeros((n, sig_dim), dtype=np.float32)
        self.qwen3_vl_embeddings = np.zeros((n, qwn_dim), dtype=np.float32)

        for new_idx, obj_idx in enumerate(keep_idx):
            if cap_dim > 0 and obj_idx < len(cap_emb_seq):
                v = self._vec_to_np(cap_emb_seq[obj_idx], dim_hint=cap_dim)
                if v.size == cap_dim:
                    self.caption_embeddings[new_idx] = v
            if sig_dim > 0 and obj_idx < len(sig_emb_seq):
                v = self._vec_to_np(sig_emb_seq[obj_idx], dim_hint=sig_dim)
                if v.size == sig_dim:
                    self.siglip2_embeddings[new_idx] = v
            if qwn_dim > 0 and obj_idx < len(qwn_emb_seq):
                v = self._vec_to_np(qwn_emb_seq[obj_idx], dim_hint=qwn_dim)
                if v.size == qwn_dim:
                    self.qwen3_vl_embeddings[new_idx] = v

        # ---- 5. embedding histories + caption text history ---------------
        def _stack_history(rows_for_obj: Any, dim: int) -> np.ndarray:
            if not isinstance(rows_for_obj, (list, tuple)) or not rows_for_obj or dim <= 0:
                return np.zeros((0, dim), dtype=np.float32) if dim > 0 else np.empty((0, 0), dtype=np.float32)
            stacked: List[np.ndarray] = []
            for entry in rows_for_obj:
                v = self._vec_to_np(entry, dim_hint=dim)
                if v.size == dim:
                    stacked.append(v)
            if not stacked:
                return np.zeros((0, dim), dtype=np.float32)
            return np.stack(stacked, axis=0).astype(np.float32, copy=False)

        self.caption_embedding_history = []
        self.siglip2_embedding_history = []
        self.qwen3_vl_embedding_history = []
        self.caption_text_history = []
        for obj_idx in keep_idx:
            self.caption_embedding_history.append(
                _stack_history(cap_emb_hist_seq[obj_idx] if obj_idx < len(cap_emb_hist_seq) else [], cap_dim)
            )
            self.siglip2_embedding_history.append(
                _stack_history(sig_emb_hist_seq[obj_idx] if obj_idx < len(sig_emb_hist_seq) else [], sig_dim)
            )
            self.qwen3_vl_embedding_history.append(
                _stack_history(qwn_emb_hist_seq[obj_idx] if obj_idx < len(qwn_emb_hist_seq) else [], qwn_dim)
            )
            text_hist_row = cap_text_hist_seq[obj_idx] if obj_idx < len(cap_text_hist_seq) else []
            text_hist: List[str] = []
            if isinstance(text_hist_row, (list, tuple)):
                for x in text_hist_row:
                    s = str(x or "").strip()
                    if s:
                        text_hist.append(s)
            self.caption_text_history.append(text_hist)

        # ---- 6. per-object image_id lists + reverse index ----------------
        self._object_image_ids = []
        self._object_viewpoint_image_ids = []
        self._image_id_to_object_indices = {}
        for new_idx, obj_idx in enumerate(keep_idx):
            full_ids: List[int] = []
            row = obj_image_ids_seq[obj_idx] if obj_idx < len(obj_image_ids_seq) else []
            if isinstance(row, (list, tuple)):
                for image_id in row:
                    try:
                        full_ids.append(int(image_id))
                    except Exception:
                        continue
            self._object_image_ids.append(full_ids)

            viewpoint_ids: List[int] = []
            row = viewpoint_image_ids_seq[obj_idx] if obj_idx < len(viewpoint_image_ids_seq) else []
            if isinstance(row, (list, tuple)):
                for image_id in row:
                    try:
                        viewpoint_ids.append(int(image_id))
                    except Exception:
                        continue
            # Mirror _load_snapshot_h5: prefer full per-object image set, fall back to viewpoints.
            active_image_ids = full_ids if full_ids else viewpoint_ids
            self._object_viewpoint_image_ids.append(active_image_ids)
            for image_id_int in active_image_ids:
                self._image_id_to_object_indices.setdefault(int(image_id_int), []).append(int(new_idx))

        # ---- 7. covisibility — initialized empty; retriever calls
        #         build_covisibility_graph() to compute from positions. -----
        self.covisibility_matrix = np.zeros((n, n), dtype=bool)

        # ---- 8. region data (optional — absent in pre-region saves) ------
        raw_region_ids = state.get("region_ids") or []
        raw_region_labels = state.get("region_labels") or []
        raw_region_object_lists = state.get("region_object_lists") or []
        raw_region_centroids = state.get("region_centroids") or []
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_idx)}
        self.region_ids: List[int] = []
        self.region_labels: List[str] = list(raw_region_labels)
        self.region_object_lists: List[List[int]] = []
        self.region_centroids: List[List[float]] = list(raw_region_centroids)
        for obj_idx in keep_idx:
            rid = raw_region_ids[obj_idx] if obj_idx < len(raw_region_ids) else -1
            self.region_ids.append(int(rid) if isinstance(rid, (int, float)) else -1)
        for old_list in raw_region_object_lists:
            if not isinstance(old_list, (list, tuple)):
                self.region_object_lists.append([])
                continue
            self.region_object_lists.append(sorted(
                old_to_new[oi] for oi in old_list if oi in old_to_new
            ))

        # No on-disk snapshot artifacts when loaded from .pt.
        self.snapshot_state_path = None
        self.snapshot_manifest_path = None
        self.snapshot_latest_path = None
        self.scene_graph_payload = {"objects": list(self.objects)}

    def get_region_label_for_object(self, obj_index: int) -> Optional[str]:
        """Return the region label for the given object index, or None."""
        if not self.region_ids or obj_index >= len(self.region_ids):
            return None
        rid = self.region_ids[obj_index]
        if rid < 0 or rid >= len(self.region_labels):
            return None
        return self.region_labels[rid]

    @_log_runtime(level=logging.DEBUG)
    def _distance(self, idx_a: int, idx_b: int) -> float:
        if idx_a < 0 or idx_b < 0 or idx_a >= len(self.means_xyz) or idx_b >= len(self.means_xyz):
            return float("inf")
        vec_a = self.means_xyz[idx_a]
        vec_b = self.means_xyz[idx_b]
        if not np.all(np.isfinite(vec_a)) or not np.all(np.isfinite(vec_b)):
            return float("inf")
        return float(np.linalg.norm(vec_a - vec_b))

    @_log_runtime(level=logging.DEBUG)
    def _save_json_debug(self, debug_log_path: str | Path, payload: Dict[str, Any]) -> None:
        path = Path(debug_log_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    @_log_runtime(level=logging.DEBUG)
    def _save_covisibility_topdown_map(
        self,
        graph: np.ndarray,
        *,
        debug_plot_path: str | Path,
        title: str = "Covisibility Graph (top-down x-y)",
    ) -> None:
        if plt is None:
            return
        coords = np.asarray(self.means_xyz, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] < 2 or coords.shape[0] == 0:
            return
        finite_xy = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
        if not bool(np.any(finite_xy)):
            return

        path = Path(debug_plot_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
        edge_pairs = np.transpose(np.nonzero(np.triu(graph, k=1)))
        max_edges = 80000
        if edge_pairs.shape[0] > max_edges:
            rng = np.random.default_rng(0)
            keep = rng.choice(edge_pairs.shape[0], size=max_edges, replace=False)
            edge_pairs = edge_pairs[keep]
        for i, j in edge_pairs.tolist():
            if not (finite_xy[int(i)] and finite_xy[int(j)]):
                continue
            xi, yi = float(coords[int(i), 0]), float(coords[int(i), 1])
            xj, yj = float(coords[int(j), 0]), float(coords[int(j), 1])
            ax.plot([xi, xj], [yi, yj], color="steelblue", alpha=0.12, linewidth=0.35, zorder=1)

        xy = coords[finite_xy, :2]
        ax.scatter(xy[:, 0], xy[:, 1], s=8, c="#005f73", alpha=0.9, label="objects", zorder=2)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.axis("equal")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    @_log_runtime(level=logging.DEBUG)
    def _save_cluster_topdown_map(
        self,
        *,
        selected_indices: Sequence[int],
        clusters_indices: Sequence[Sequence[int]],
        debug_plot_path: str | Path,
        retrieved_indices: Optional[Sequence[int]] = None,
        title: str = "Clustered retrieval objects (top-down x-y)",
    ) -> None:
        if plt is None:
            return
        coords = np.asarray(self.means_xyz, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] < 2 or coords.shape[0] == 0:
            return
        finite_xy = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
        if not bool(np.any(finite_xy)):
            return

        path = Path(debug_plot_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
        base_xy = coords[finite_xy, :2]
        ax.scatter(base_xy[:, 0], base_xy[:, 1], s=6, c="#d9d9d9", alpha=0.8, label="all objects", zorder=1)

        if selected_indices:
            valid_selected = [int(i) for i in selected_indices if 0 <= int(i) < coords.shape[0] and finite_xy[int(i)]]
            if valid_selected:
                sel_xy = coords[np.asarray(valid_selected, dtype=np.int64), :2]
                ax.scatter(sel_xy[:, 0], sel_xy[:, 1], s=24, c="#1b1f3b", alpha=0.9, label="retrieved", zorder=2)

        cmap = plt.cm.get_cmap("tab20", max(1, len(clusters_indices)))
        for cluster_idx, cluster in enumerate(clusters_indices):
            valid_cluster = [int(i) for i in cluster if 0 <= int(i) < coords.shape[0] and finite_xy[int(i)]]
            if not valid_cluster:
                continue
            cxy = coords[np.asarray(valid_cluster, dtype=np.int64), :2]
            ax.scatter(
                cxy[:, 0],
                cxy[:, 1],
                s=60,
                c=[cmap(cluster_idx)],
                alpha=0.85,
                edgecolors="black",
                linewidths=0.3,
                label=f"cluster {cluster_idx + 1}",
                zorder=3,
            )

        if retrieved_indices:
            valid_retrieved = [int(i) for i in retrieved_indices if 0 <= int(i) < coords.shape[0] and finite_xy[int(i)]]
            if valid_retrieved:
                rxy = coords[np.asarray(valid_retrieved, dtype=np.int64), :2]
                ax.scatter(
                    rxy[:, 0],
                    rxy[:, 1],
                    s=55,
                    marker="x",
                    c="#ee6c4d",
                    alpha=0.9,
                    linewidths=1.2,
                    label="retrieved (raw)",
                    zorder=4,
                )

        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.axis("equal")
        ax.grid(True, alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            uniq: Dict[str, Any] = {}
            for h, l in zip(handles, labels):
                if l not in uniq:
                    uniq[l] = h
            ax.legend(list(uniq.values()), list(uniq.keys()), loc="best")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    @_log_runtime
    def build_covisibility_graph(
        self,
        *,
        min_neighbors: int = 10,
        top_fraction: float = 0.5,
        two_hop_enabled: bool = True,
        two_hop_seed_cap: int = 50,
        two_hop_max_additional: Optional[int] = None,
        max_views_per_seed: int = 20,
        rng: Optional[np.random.Generator] = None,
        debug_plot_path: str | Path | None = None,
        debug_title: str | None = None,
    ) -> np.ndarray:
        object_count = len(self.object_ids)
        graph = np.zeros((object_count, object_count), dtype=bool)
        if object_count == 0:
            self.covisibility_matrix = graph
            return graph

        coords = np.asarray(self.means_xyz, dtype=np.float32)

        marks = np.zeros((object_count,), dtype=np.int32)
        stamp = 0

        two_hop_marks = np.zeros((object_count,), dtype=np.int32)
        two_hop_stamp = 0

        max_views_per_seed = max(0, int(max_views_per_seed))
        if rng is None:
            rng = np.random.default_rng()

        for idx in range(object_count):
            stamp += 1
            candidates_list: List[int] = []

            # 1-hop covisible candidates: share at least one image_id.
            for image_id in self._object_viewpoint_image_ids[idx]:
                for nbr in self._image_id_to_object_indices.get(int(image_id), []):
                    nbr_int = int(nbr)
                    if nbr_int == idx:
                        continue
                    if marks[nbr_int] == stamp:
                        continue
                    marks[nbr_int] = stamp
                    candidates_list.append(nbr_int)

            if not candidates_list:
                continue

            keep_count = max(int(min_neighbors), int(math.ceil(float(top_fraction) * float(len(candidates_list)))))
            keep_count = max(1, min(keep_count, len(candidates_list)))

            candidates_arr = np.asarray(candidates_list, dtype=np.int64)
            idx_pos = coords[idx] if 0 <= idx < coords.shape[0] else np.asarray([np.nan, np.nan, np.nan], np.float32)
            cand_pos = coords[candidates_arr]
            valid = bool(np.isfinite(idx_pos).all()) and (np.isfinite(cand_pos).all(axis=1))
            dist2 = np.full((len(candidates_list),), float("inf"), dtype=np.float32)
            if bool(np.any(valid)):
                delta = cand_pos[valid] - idx_pos[None, :]
                dist2[valid] = np.sum(delta * delta, axis=1).astype(np.float32, copy=False)

            if keep_count < len(candidates_list):
                chosen = np.argpartition(dist2, kth=keep_count - 1)[:keep_count]
                selected = candidates_arr[chosen].tolist()
            else:
                selected = candidates_list

            # 2-hop expansion: covisible-of-covisible, using only the already distance-pruned 1-hop set.
            two_hop_selected: List[int] = []
            if two_hop_enabled and selected:
                seed_cap = max(0, int(two_hop_seed_cap))
                seeds = selected if seed_cap <= 0 else selected[: min(seed_cap, len(selected))]

                two_hop_stamp += 1
                two_hop_candidates: List[int] = []
                for seed in seeds:
                    seed_views = self._object_viewpoint_image_ids[int(seed)]
                    if max_views_per_seed and len(seed_views) > max_views_per_seed:
                        pick = rng.choice(len(seed_views), size=max_views_per_seed, replace=False)
                        image_iter = (seed_views[int(i)] for i in pick.tolist())
                    else:
                        image_iter = seed_views

                    for image_id in image_iter:
                        for nbr in self._image_id_to_object_indices.get(int(image_id), []):
                            nbr_int = int(nbr)
                            if nbr_int == idx:
                                continue
                            if marks[nbr_int] == stamp:
                                # Already 1-hop candidate; not needed as an "additional" 2-hop neighbor.
                                continue
                            if two_hop_marks[nbr_int] == two_hop_stamp:
                                continue
                            two_hop_marks[nbr_int] = two_hop_stamp
                            two_hop_candidates.append(nbr_int)

                if two_hop_candidates:
                    if two_hop_max_additional is None:
                        max_additional = min(len(two_hop_candidates), max(1, keep_count))
                    else:
                        max_additional = min(len(two_hop_candidates), max(0, int(two_hop_max_additional)))
                    if max_additional > 0:
                        two_arr = np.asarray(two_hop_candidates, dtype=np.int64)
                        two_pos = coords[two_arr]
                        valid2 = bool(np.isfinite(idx_pos).all()) and (np.isfinite(two_pos).all(axis=1))
                        dist2_2 = np.full((len(two_hop_candidates),), float("inf"), dtype=np.float32)
                        if bool(np.any(valid2)):
                            delta2 = two_pos[valid2] - idx_pos[None, :]
                            dist2_2[valid2] = np.sum(delta2 * delta2, axis=1).astype(np.float32, copy=False)
                        if max_additional < len(two_hop_candidates):
                            chosen2 = np.argpartition(dist2_2, kth=max_additional - 1)[:max_additional]
                            two_hop_selected = two_arr[chosen2].tolist()
                        else:
                            two_hop_selected = two_hop_candidates

            covisible_seeds = selected + two_hop_selected

            for nbr in covisible_seeds:
                if nbr == idx:
                    continue
                graph[idx, int(nbr)] = True
                graph[int(nbr), idx] = True

        np.fill_diagonal(graph, False)
        self.covisibility_matrix = graph
        if debug_plot_path is not None:
            with contextlib.suppress(Exception):
                self._save_covisibility_topdown_map(
                    graph,
                    debug_plot_path=debug_plot_path,
                    title=debug_title or "Covisibility Graph (top-down x-y)",
                )
        return graph

    @_log_runtime
    def get_covisible_neighbors(self, object_id: Any) -> List[Dict[str, Any]]:
        object_key = self._id_to_key(object_id)
        if object_key not in self.objects_by_id:
            return []
        if self.covisibility_matrix.shape[0] != len(self.object_ids):
            self.build_covisibility_graph()

        index = self._object_key_to_index[object_key]
        neighbors = np.where(self.covisibility_matrix[index])[0].tolist()
        neighbors.sort(key=lambda nbr: self._distance(index, int(nbr)))
        return [self.objects[int(nbr)] for nbr in neighbors]

    @staticmethod
    @_log_runtime(level=logging.DEBUG)
    def _object_text(obj: Dict[str, Any]) -> str:
        caption = str(obj.get("object_caption", "") or "").strip()
        mean = obj.get("mean")
        if isinstance(mean, (list, tuple)) and len(mean) >= 3:
            try:
                return f"{caption} at ({float(mean[0]):.2f}, {float(mean[1]):.2f}, {float(mean[2]):.2f})"
            except Exception:
                return caption
        return caption

    @staticmethod
    @_log_runtime(level=logging.DEBUG)
    def _encode_query(encoder: Any, task_description: str) -> np.ndarray:
        query = str(task_description or "").strip()
        if not query:
            raise ValueError("task_description cannot be empty")
        if encoder is None or not hasattr(encoder, "encode"):
            raise ValueError("encoder must provide encode(text) -> np.ndarray")
        return np.asarray(encoder.encode(query), dtype=np.float32).reshape(-1)

    @staticmethod
    @_log_runtime(level=logging.DEBUG)
    def _is_transient_siglip2_error(error: str) -> bool:
        err = str(error or "").strip().lower()
        return err in {"busy", "timeout", "temporarily_unavailable"}

    @_log_runtime(level=logging.DEBUG)
    def _ensure_siglip2_service_client(self) -> bool:
        if self._siglip2_ros_client is not None and self._siglip2_ros_node is not None:
            return True
        if rclpy is None or Siglip2TextEmbed is None:
            return False
        if not rclpy.ok():
            rclpy.init(args=None)
            self._siglip2_ros_context_owner = True
        if self._siglip2_ros_node is None:
            node_name = f"scene_graph_processing_siglip2_client_{int(time.time() * 1e6) % 10000000}"
            self._siglip2_ros_node = rclpy.create_node(node_name)
        if self._siglip2_ros_client is None:
            self._siglip2_ros_client = self._siglip2_ros_node.create_client(
                Siglip2TextEmbed, self._siglip2_service_name
            )
        return True

    @_log_runtime
    def _call_siglip2_service_embeddings(self, texts: Sequence[str], *, normalize: bool = True) -> np.ndarray:
        text_list = [str(x or "").strip() for x in texts]
        if not text_list:
            return np.empty((0, 0), dtype=np.float32)
        if not all(text_list):
            raise ValueError("siglip2 service request contains empty text")

        with self._siglip2_client_lock:
            if not self._ensure_siglip2_service_client():
                raise RuntimeError("ROS2 SigLIP2 embedding service unavailable (missing rclpy or mapping_msgs.srv)")
            client = self._siglip2_ros_client
            node = self._siglip2_ros_node
            if client is None or node is None:
                raise RuntimeError("ROS2 SigLIP2 embedding client initialization failed")

            last_error = "unknown_error"
            for attempt in range(self._siglip2_service_retries):
                if not client.wait_for_service(timeout_sec=0.05):
                    last_error = "temporarily_unavailable"
                else:
                    request = Siglip2TextEmbed.Request()
                    request.texts = text_list
                    request.normalize = bool(normalize)
                    request.client_id = int(self._siglip2_client_id)
                    future = client.call_async(request)
                    rclpy.spin_until_future_complete(node, future, timeout_sec=self._siglip2_service_timeout_s)
                    if not future.done():
                        last_error = "timeout"
                    else:
                        try:
                            response = future.result()
                        except Exception as exc:
                            response = None
                            last_error = str(exc)
                        if response is not None:
                            if bool(response.ok):
                                n_texts = int(response.n_texts)
                                dim = int(response.dim)
                                if n_texts != len(text_list):
                                    raise RuntimeError(
                                        "siglip2 response n_texts mismatch:"
                                        f" response={n_texts} request={len(text_list)}"
                                    )
                                if dim <= 0:
                                    raise RuntimeError("siglip2 response dim <= 0")
                                flat = np.asarray(list(response.embeddings_flat), dtype=np.float32)
                                expected = int(n_texts * dim)
                                if flat.size != expected:
                                    raise RuntimeError(
                                        f"siglip2 response size mismatch: flat={int(flat.size)} expected={expected}"
                                    )
                                return flat.reshape(n_texts, dim)
                            last_error = str(response.error or "embed_failed")

                if attempt + 1 < self._siglip2_service_retries and self._is_transient_siglip2_error(last_error):
                    backoff = self._siglip2_service_backoff_s * (2**attempt)
                    jitter = random.uniform(0.0, 0.03)
                    time.sleep(max(0.0, backoff + jitter))
                    continue
                raise RuntimeError(f"siglip2_service_call_failed: {last_error}")

        raise RuntimeError("siglip2_service_call_failed: unknown_error")

    @_log_runtime(level=logging.DEBUG)
    def _encode_query_siglip2_service(self, task_description: str) -> np.ndarray:
        vectors = self._call_siglip2_service_embeddings([str(task_description or "").strip()], normalize=True)
        if vectors.ndim != 2 or vectors.shape[0] != 1:
            raise RuntimeError("siglip2 service returned invalid query embedding shape")
        return vectors[0]

    @_log_runtime(level=logging.DEBUG)
    def _encode_queries_siglip2(
        self,
        texts: Sequence[str],
        *,
        embedder: Any | None,
        use_ros_service: bool,
    ) -> np.ndarray:
        text_list = [str(x or "").strip() for x in texts]
        if not text_list:
            return np.empty((0, 0), dtype=np.float32)
        if not all(text_list):
            raise ValueError("siglip2 query text cannot be empty")

        if embedder is not None:
            if hasattr(embedder, "encode_batch"):
                vectors = np.asarray(embedder.encode_batch(text_list), dtype=np.float32)
            else:
                vectors = np.vstack([np.asarray(embedder.encode(t), dtype=np.float32).reshape(-1) for t in text_list])
        elif use_ros_service:
            vectors = self._call_siglip2_service_embeddings(text_list, normalize=True)
        else:
            raise ValueError("embedder is required when use_ros_service is False")

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2 or vectors.shape[0] != len(text_list):
            raise RuntimeError(
                f"siglip2 query embedding shape mismatch: got={tuple(vectors.shape)} expected_rows={len(text_list)}"
            )
        return vectors

    @_log_runtime(level=logging.DEBUG)
    def _object_embedding_rows(
        self,
        object_index: int,
        *,
        history_rows: Sequence[np.ndarray],
        fallback_matrix: np.ndarray,
    ) -> np.ndarray:
        rows = (
            np.asarray(history_rows[object_index], dtype=np.float32)
            if 0 <= object_index < len(history_rows)
            else np.empty((0, 0), dtype=np.float32)
        )
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.ndim != 2:
            rows = np.empty((0, 0), dtype=np.float32)
        if rows.shape[0] > 0:
            return rows

        if (
            isinstance(fallback_matrix, np.ndarray)
            and fallback_matrix.ndim == 2
            and 0 <= object_index < fallback_matrix.shape[0]
            and fallback_matrix.shape[1] > 0
        ):
            return np.asarray(fallback_matrix[object_index], dtype=np.float32).reshape(1, -1)
        return np.empty((0, 0), dtype=np.float32)

    @_log_runtime(level=logging.DEBUG)
    def _object_max_similarity_scores(
        self,
        query_vec: np.ndarray,
        *,
        history_rows: Sequence[np.ndarray],
        fallback_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        q_dim = int(query.shape[0])
        if q_dim <= 0:
            return np.full((len(self.object_ids),), -1.0, dtype=np.float32), np.zeros(
                (len(self.object_ids),), dtype=bool
            )
        q_norm = float(np.linalg.norm(query))
        if q_norm <= 0.0:
            return np.full((len(self.object_ids),), -1.0, dtype=np.float32), np.zeros(
                (len(self.object_ids),), dtype=bool
            )
        query = query / q_norm

        scores = np.full((len(self.object_ids),), -1.0, dtype=np.float32)
        valid_mask = np.zeros((len(self.object_ids),), dtype=bool)
        for obj_idx in range(len(self.object_ids)):
            rows = self._object_embedding_rows(
                obj_idx,
                history_rows=history_rows,
                fallback_matrix=fallback_matrix,
            )
            if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] != q_dim:
                continue
            norms = np.linalg.norm(rows, axis=1, keepdims=True)
            valid = norms.reshape(-1) > 0.0
            if not bool(np.any(valid)):
                continue
            rows_norm = rows[valid] / np.maximum(norms[valid], 1e-12)
            sims = rows_norm.dot(query)
            if sims.size <= 0:
                continue
            scores[obj_idx] = float(np.max(sims))
            valid_mask[obj_idx] = True
        return scores, valid_mask

    @_log_runtime
    def retrieve_by_caption(
        self,
        task_description: str,
        *,
        embedder: Any,
        reranker: Any | None = None,
        top_k: int = 10,
        retrieval_k: int | None = None,
        debug_log_path: str | Path | None = None,
        debug_top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if len(self.object_ids) == 0:
            return []
        query_vec = self._encode_query(embedder, task_description)
        object_scores, valid_mask = self._object_max_similarity_scores(
            query_vec,
            history_rows=self.caption_embedding_history,
            fallback_matrix=self.caption_embeddings,
        )
        if object_scores.size == 0 or not bool(np.any(valid_mask)):
            return []

        retrieval_k = int(retrieval_k) if retrieval_k is not None else max(int(top_k) * 4, int(top_k))
        retrieval_k = max(1, min(int(retrieval_k), int(np.sum(valid_mask))))
        valid_indices = np.nonzero(valid_mask)[0]
        order = valid_indices[np.argsort(-object_scores[valid_indices])]
        retrieval_hits = [(int(idx), float(object_scores[int(idx)])) for idx in order[: max(retrieval_k, int(top_k))]]
        if not retrieval_hits:
            return []

        if debug_log_path is not None:
            with contextlib.suppress(Exception):
                debug_path = Path(debug_log_path).expanduser()
                debug_retrieval_path = debug_path.parent / "caption_retrieval_topK.json"
                top_n_retrieval = max(1, int(retrieval_k))
                payload = {
                    "method": "caption_retrieval",
                    "query": str(task_description),
                    "retrieval_k": int(retrieval_k),
                    "results": [
                        {
                            "rank": rank + 1,
                            "object_id": self.objects[int(idx)].get("object_id", self.objects[int(idx)].get("id")),
                            "caption": str(self.objects[int(idx)].get("object_caption", "") or ""),
                            "retrieval_score": float(score),
                            "caption_embedding_norm": float(np.linalg.norm(self.caption_embeddings[int(idx)])),
                        }
                        for rank, (idx, score) in enumerate(retrieval_hits[:top_n_retrieval])
                        if 0 <= int(idx) < len(self.objects)
                    ],
                }
                self._save_json_debug(debug_retrieval_path, payload)

        candidates: List[Dict[str, Any]] = []
        for idx, score in retrieval_hits:
            obj = dict(self.objects[idx])
            obj["retrieval_score"] = float(score)
            obj["object_id"] = self.objects[idx].get("object_id", self.objects[idx].get("id"))
            candidates.append(obj)

        if reranker is None or not hasattr(reranker, "rerank"):
            out = candidates[: int(top_k)]
            if debug_log_path is not None:
                with contextlib.suppress(Exception):
                    top_n = max(1, int(debug_top_k))
                    payload = {
                        "method": "caption",
                        "query": str(task_description),
                        "results": [
                            {
                                "rank": rank + 1,
                                "object_id": item.get("object_id"),
                                "caption": str(item.get("object_caption", "") or ""),
                                "retrieval_score": float(item.get("retrieval_score", 0.0)),
                                "rerank_score": item.get("rerank_score"),
                            }
                            for rank, item in enumerate(out[:top_n])
                        ],
                    }
                    self._save_json_debug(debug_log_path, payload)
            return out

        # Reranker should score the caption text itself (ignore pose/viewpoint/location).
        docs = [str(obj.get("object_caption", "") or "").strip() for obj in candidates]
        rerank_hits = reranker.rerank(str(task_description), docs, top_k=min(int(top_k), len(docs)))

        reranked: List[Dict[str, Any]] = []
        for doc_idx, rerank_score in rerank_hits:
            if doc_idx < 0 or doc_idx >= len(candidates):
                continue
            obj = dict(candidates[int(doc_idx)])
            obj["rerank_score"] = float(rerank_score)
            reranked.append(obj)
        if debug_log_path is not None:
            with contextlib.suppress(Exception):
                top_n = max(1, int(debug_top_k))
                payload = {
                    "method": "caption",
                    "query": str(task_description),
                    "results": [
                        {
                            "rank": rank + 1,
                            "object_id": item.get("object_id"),
                            "caption": str(item.get("object_caption", "") or ""),
                            "retrieval_score": float(item.get("retrieval_score", 0.0)),
                            "rerank_score": float(item.get("rerank_score", 0.0)),
                        }
                        for rank, item in enumerate(reranked[:top_n])
                    ],
                }
                self._save_json_debug(debug_log_path, payload)
        return reranked

    @_log_runtime
    def retrieve_by_siglip2(
        self,
        task_description: str,
        *,
        embedder: Any | None = None,
        use_ros_service: bool = True,
        top_k: int = 10,
        debug_log_path: str | Path | None = None,
        debug_top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if len(self.object_ids) == 0:
            return []
        prompts = self._build_siglip2_prompt_ensemble(task_description)
        if not prompts:
            raise ValueError("task_description cannot be empty")
        query_matrix = self._encode_queries_siglip2(prompts, embedder=embedder, use_ros_service=use_ros_service)
        query_matrix = self._normalize_rows(query_matrix)
        if query_matrix.shape[0] == 0:
            return []
        query_dim = int(query_matrix.shape[1]) if query_matrix.ndim == 2 else 0
        if query_dim <= 0:
            return []

        agg_similarity = np.full((len(self.object_ids),), -1.0, dtype=np.float32)
        max_similarity = np.full((len(self.object_ids),), -1.0, dtype=np.float32)
        valid_mask = np.zeros((len(self.object_ids),), dtype=bool)
        prompt_topn = max(1, min(self._siglip2_prompt_topn_mean, int(query_matrix.shape[0])))
        for idx in range(len(self.object_ids)):
            rows = self._object_embedding_rows(
                idx,
                history_rows=self.siglip2_embedding_history,
                fallback_matrix=self.siglip2_embeddings,
            )
            if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] != query_dim:
                continue
            rows_norm = self._normalize_rows(rows)
            if rows_norm.shape[0] == 0:
                continue
            similarity = rows_norm.dot(query_matrix.T)  # (num_rows, num_prompts)
            if similarity.ndim != 2 or similarity.shape[0] == 0:
                continue
            # Max over history rows first, then aggregate over prompt ensemble.
            prompt_similarity = np.max(similarity, axis=0)
            if prompt_topn >= int(prompt_similarity.shape[0]):
                agg = float(np.mean(prompt_similarity))
            else:
                kth = int(prompt_similarity.shape[0] - prompt_topn)
                topn = np.partition(prompt_similarity, kth=kth)[-prompt_topn:]
                agg = float(np.mean(topn))
            agg_similarity[idx] = float(agg)
            max_similarity[idx] = float(np.max(prompt_similarity))
            valid_mask[idx] = True

        if not bool(np.any(valid_mask)):
            return []

        logits = (self._siglip2_logit_scale * agg_similarity) + self._siglip2_logit_bias
        sigmoid_scores = self._sigmoid(logits)

        top_k = max(1, min(int(top_k), int(np.sum(valid_mask))))
        valid_indices = np.nonzero(valid_mask)[0]
        indices = valid_indices[np.argsort(-agg_similarity[valid_indices])[:top_k]]
        out: List[Dict[str, Any]] = []
        for idx in indices.tolist():
            obj = dict(self.objects[idx])
            obj["retrieval_score"] = float(sigmoid_scores[idx])
            obj["siglip2_similarity"] = float(agg_similarity[idx])
            obj["siglip2_max_similarity"] = float(max_similarity[idx])
            obj["siglip2_logit"] = float(logits[idx])
            obj["object_id"] = self.objects[idx].get("object_id", self.objects[idx].get("id"))
            out.append(obj)
        if debug_log_path is not None:
            with contextlib.suppress(Exception):
                top_n = max(1, int(debug_top_k))
                payload = {
                    "method": "siglip2",
                    "query": str(task_description),
                    "prompts": prompts,
                    "prompt_topn_mean": int(prompt_topn),
                    "logit_scale": float(self._siglip2_logit_scale),
                    "logit_bias": float(self._siglip2_logit_bias),
                    "results": [
                        {
                            "rank": rank + 1,
                            "object_id": item.get("object_id"),
                            "caption": str(item.get("object_caption", "") or ""),
                            "retrieval_score": float(item.get("retrieval_score", 0.0)),
                            "siglip2_similarity": float(item.get("siglip2_similarity", 0.0)),
                            "siglip2_max_similarity": float(item.get("siglip2_max_similarity", 0.0)),
                            "siglip2_logit": float(item.get("siglip2_logit", 0.0)),
                        }
                        for rank, item in enumerate(out[:top_n])
                    ],
                }
                self._save_json_debug(debug_log_path, payload)
        return out

    @_log_runtime
    def retrieve_by_qwen3_vl(
        self,
        task_description: str,
        *,
        embedder: Any,
        top_k: int = 10,
        debug_log_path: str | Path | None = None,
        debug_top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if len(self.object_ids) == 0:
            return []
        query_vec = self._encode_query(embedder, task_description)
        scores, valid_mask = self._object_max_similarity_scores(
            query_vec,
            history_rows=self.qwen3_vl_embedding_history,
            fallback_matrix=self.qwen3_vl_embeddings,
        )
        if scores.size == 0 or not bool(np.any(valid_mask)):
            return []

        top_k = max(1, min(int(top_k), int(np.sum(valid_mask))))
        valid_indices = np.nonzero(valid_mask)[0]
        order = valid_indices[np.argsort(-scores[valid_indices])]
        hits = [(int(idx), float(scores[int(idx)])) for idx in order[:top_k]]
        if not hits:
            return []

        out: List[Dict[str, Any]] = []
        for idx, score in hits:
            obj = dict(self.objects[idx])
            obj["retrieval_score"] = float(score)
            obj["qwen3_vl_similarity"] = float(score)
            obj["object_id"] = self.objects[idx].get("object_id", self.objects[idx].get("id"))
            out.append(obj)

        if debug_log_path is not None:
            with contextlib.suppress(Exception):
                top_n = max(1, int(debug_top_k))
                payload = {
                    "method": "qwen3_vl",
                    "query": str(task_description),
                    "results": [
                        {
                            "rank": rank + 1,
                            "object_id": item.get("object_id"),
                            "caption": str(item.get("object_caption", "") or ""),
                            "retrieval_score": float(item.get("retrieval_score", 0.0)),
                            "qwen3_vl_similarity": float(item.get("qwen3_vl_similarity", 0.0)),
                        }
                        for rank, item in enumerate(out[:top_n])
                    ],
                }
                self._save_json_debug(debug_log_path, payload)
        return out

    @_log_runtime
    def cluster_objects(
        self,
        object_ids: Sequence[Any],
        *,
        distance_threshold_m: float = 5.0,
        debug_plot_path: str | Path | None = None,
        retrieved_object_ids: Optional[Sequence[Any]] = None,
        debug_title: str | None = None,
    ) -> List[List[Any]]:
        if self.covisibility_matrix.shape[0] != len(self.object_ids):
            self.build_covisibility_graph()
        if len(self.object_ids) == 0:
            return []

        keys = [self._id_to_key(x) for x in object_ids]
        indices = [self._object_key_to_index[key] for key in keys if key in self._object_key_to_index]
        if not indices:
            return []

        selected = sorted(set(indices))
        index_set = set(selected)
        adjacency: Dict[int, List[int]] = {idx: [] for idx in selected}
        threshold = max(0.0, float(distance_threshold_m))

        for idx in selected:
            for nbr in selected:
                if idx == nbr:
                    continue
                if not bool(self.covisibility_matrix[idx, nbr]):
                    continue
                if self._distance(idx, nbr) <= threshold:
                    adjacency[idx].append(nbr)

        clusters: List[List[Any]] = []
        cluster_indices: List[List[int]] = []
        visited: set[int] = set()
        for root in selected:
            if root in visited:
                continue
            stack = [root]
            component: List[int] = []
            while stack:
                node = stack.pop()
                if node in visited or node not in index_set:
                    continue
                visited.add(node)
                component.append(node)
                for nbr in adjacency.get(node, []):
                    if nbr not in visited:
                        stack.append(nbr)

            raw_ids = [
                self.objects[idx].get("object_id", self.objects[idx].get("id"))
                for idx in sorted(component, key=lambda i: self.object_ids[i])
            ]
            cluster_indices.append(sorted(component))
            clusters.append(raw_ids)
        if debug_plot_path is not None:
            with contextlib.suppress(Exception):
                retrieved_indices: List[int] = []
                if retrieved_object_ids:
                    retrieved_keys = [self._id_to_key(x) for x in retrieved_object_ids]
                    retrieved_indices = [
                        self._object_key_to_index[key] for key in retrieved_keys if key in self._object_key_to_index
                    ]
                self._save_cluster_topdown_map(
                    selected_indices=selected,
                    clusters_indices=cluster_indices,
                    debug_plot_path=debug_plot_path,
                    retrieved_indices=retrieved_indices,
                    title=debug_title or "Clustered retrieval objects (top-down x-y)",
                )
        return clusters

    @_log_runtime
    def minimum_covering_images(
        self,
        object_ids: Sequence[Any],
        *,
        require_saved_path: bool = True,
    ) -> Dict[str, Any]:
        target_keys = [self._id_to_key(x) for x in object_ids if self._id_to_key(x) in self.objects_by_id]
        uncovered = set(target_keys)
        if not uncovered:
            return {"selected_images": [], "uncovered_object_ids": []}

        image_to_objects: Dict[int, set[str]] = {}
        for key in target_keys:
            idx = self._object_key_to_index[key]
            for image_id in self._object_viewpoint_image_ids[idx]:
                path = self._image_id_to_storage_path.get(int(image_id), "")
                if require_saved_path and not str(path).strip():
                    continue
                image_to_objects.setdefault(int(image_id), set()).add(key)

        selected: List[Dict[str, Any]] = []
        while uncovered:
            best_image = None
            best_cover: set[str] = set()
            for image_id, covers in image_to_objects.items():
                gain = covers & uncovered
                if len(gain) > len(best_cover):
                    best_cover = gain
                    best_image = image_id
            if best_image is None or not best_cover:
                break

            selected.append({
                "image_id": int(best_image),
                "camera_id": self._image_id_to_camera.get(int(best_image), ""),
                "storage_path": self._image_id_to_storage_path.get(int(best_image), ""),
                "covered_object_ids": [
                    self.objects[self._object_key_to_index[key]].get(
                        "object_id", self.objects[self._object_key_to_index[key]].get("id")
                    )
                    for key in sorted(best_cover)
                ],
            })
            uncovered -= best_cover

        unresolved = [
            self.objects[self._object_key_to_index[key]].get(
                "object_id", self.objects[self._object_key_to_index[key]].get("id")
            )
            for key in sorted(uncovered)
        ]
        return {"selected_images": selected, "uncovered_object_ids": unresolved}


@_log_runtime
def load_scene_graph_processing(
    scene_graph_json_path: str | Path,
    *,
    snapshot_root: str | Path | None = None,
    siglip2_service_name: str = "/spot/mapping/siglip2_text_embed",
    siglip2_service_timeout_s: float = 0.35,
    siglip2_service_retries: int = 3,
    siglip2_service_backoff_s: float = 0.05,
    siglip2_client_id: int = 0,
    siglip2_prompt_topn_mean: int = 3,
    siglip2_logit_scale: float = 1.0,
    siglip2_logit_bias: float = 0.0,
) -> SceneGraphProcessing:
    processor = SceneGraphProcessing(
        scene_graph_json_path,
        snapshot_root=snapshot_root,
        siglip2_service_name=siglip2_service_name,
        siglip2_service_timeout_s=siglip2_service_timeout_s,
        siglip2_service_retries=siglip2_service_retries,
        siglip2_service_backoff_s=siglip2_service_backoff_s,
        siglip2_client_id=siglip2_client_id,
        siglip2_prompt_topn_mean=siglip2_prompt_topn_mean,
        siglip2_logit_scale=siglip2_logit_scale,
        siglip2_logit_bias=siglip2_logit_bias,
    )
    processor.load_latest()
    return processor
