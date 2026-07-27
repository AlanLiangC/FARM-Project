"""Predicate evaluators: fast-path (geometric/semantic) and VLM-based."""

from __future__ import annotations

from collections.abc import Mapping
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from scene_graph.llm_utils import EmbedInterface, LLMInterface

from .fuzzy_ops import gaussian_membership, normalize_vlm_score, sigmoid_membership
from .models import PredicateResult
from .prompts import PREDICATE_PROMPTS

logger = logging.getLogger(__name__)

# Predicates that are superlative (rank-based across ALL candidates)
SUPERLATIVE_PREDICATES = frozenset(["Closest", "Farthest"])


def _axis_from_name(value: Any) -> Optional[int]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"0", "x", "+x", "-x", "x-up", "x_up"}:
        return 0
    if text in {"1", "y", "+y", "-y", "y-up", "y_up", "habitat", "hm3d"}:
        return 1
    if text in {"2", "z", "+z", "-z", "z-up", "z_up", "scannet"}:
        return 2
    if "y-up" in text or "y_up" in text or "habitat" in text or "hm3d" in text:
        return 1
    if "z-up" in text or "z_up" in text or "scannet" in text:
        return 2
    return None


def _infer_vertical_axis(scene_state: Dict[str, Any]) -> int:
    """Infer the world-frame vertical axis for spatial predicates.

    ScanNet-style reconstructions are z-up, while HM3D/Habitat trajectories and
    scene states are y-up.  Older scene_state files do not carry explicit frame
    metadata, so we also inspect image source refs written by the offline HM3D
    frame source.
    """

    for key in ("vertical_axis", "up_axis", "coordinate_frame", "world_frame", "dataset", "bench"):
        axis = _axis_from_name(scene_state.get(key))
        if axis is not None:
            return axis

    metadata = scene_state.get("metadata")
    if isinstance(metadata, dict):
        for key in ("vertical_axis", "up_axis", "coordinate_frame", "world_frame", "dataset", "bench"):
            axis = _axis_from_name(metadata.get(key))
            if axis is not None:
                return axis

    images = scene_state.get("images") or []
    for record in images[: min(16, len(images))]:
        if isinstance(record, dict):
            text = " ".join(
                str(record.get(k, "") or "") for k in ("source_ref", "storage_path", "camera_id")
            ).lower()
        else:
            text = " ".join(
                str(getattr(record, k, "") or "") for k in ("source_ref", "storage_path", "camera_id")
            ).lower()
        if "hm3d" in text or "iref_vla" in text or "rendered_trajectory" in text:
            return 1

    return 2


def _horizontal_axes(vertical_axis: int) -> Tuple[int, int]:
    axes = [0, 1, 2]
    vertical_axis = int(vertical_axis)
    if vertical_axis not in axes:
        vertical_axis = 2
    axes.remove(vertical_axis)
    return int(axes[0]), int(axes[1])


def _euclidean_distance(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    return float(np.linalg.norm(pos_a - pos_b))


def _horizontal_distance(pos_a: np.ndarray, pos_b: np.ndarray, *, vertical_axis: int = 2) -> float:
    axes = _horizontal_axes(vertical_axis)
    return float(np.linalg.norm(pos_a[list(axes)] - pos_b[list(axes)]))


def _height_diff(pos_a: np.ndarray, pos_b: np.ndarray, *, vertical_axis: int = 2) -> float:
    return float(pos_a[int(vertical_axis)] - pos_b[int(vertical_axis)])


def _format_pos(pos: np.ndarray) -> str:
    return f"{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}"


def _parse_vlm_score(response: str) -> Optional[float]:
    """Extract a numeric score from VLM JSON response."""
    try:
        data = json.loads(response)
        if isinstance(data, dict) and "score" in data:
            return normalize_vlm_score(data["score"])
    except json.JSONDecodeError:
        pass
    match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', response)
    if match:
        return normalize_vlm_score(float(match.group(1)))
    match = re.search(r"\b(\d{1,3})\b", response)
    if match:
        val = int(match.group(1))
        if 0 <= val <= 100:
            return normalize_vlm_score(val)
    return None


def _observation_image_id(observation: Mapping[str, Any]) -> Optional[int]:
    try:
        return int(observation.get("image_id"))
    except Exception:
        return None


def _as_bbox_xyxy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < 4:
        return None
    bbox = arr[:4]
    if not np.all(np.isfinite(bbox)):
        return None
    if float(bbox[2]) <= float(bbox[0]) or float(bbox[3]) <= float(bbox[1]):
        return None
    return bbox


def _first_bbox_xyxy(*values: Any) -> Optional[np.ndarray]:
    for value in values:
        bbox = _as_bbox_xyxy(value)
        if bbox is not None:
            return bbox
    return None


def _as_image_shape(value: Any) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.int64).reshape(-1)
    except Exception:
        return None
    if arr.size < 2:
        return None
    h, w = int(arr[0]), int(arr[1])
    if h <= 0 or w <= 0:
        return None
    return h, w


def _resolve_mask_observation_path(path_raw: Any) -> Optional[Path]:
    if not path_raw:
        return None
    path = Path(str(path_raw)).expanduser()
    return path if path.exists() else None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


class PredicateEvaluator:
    """Evaluates predicates against scene state using fast-path and VLM."""

    def __init__(
        self,
        scene_state: Dict[str, Any],
        llm: LLMInterface,
        embedder: EmbedInterface,
        *,
        near_sigma: float = 0.5,
        near_prune_distance: float = 5.0,
        on_max_horiz: float = 1.0,
        on_height_band: Tuple[float, float] = (0.0, 0.5),
        verbose: bool = False,
    ):
        self.scene_state = scene_state
        self.llm = llm
        self.embedder = embedder
        self.near_sigma = near_sigma
        self.near_prune_distance = near_prune_distance
        self.on_max_horiz = on_max_horiz
        self.on_height_band = on_height_band
        self.verbose = verbose
        self.vertical_axis = _infer_vertical_axis(scene_state)
        self.horizontal_axes = _horizontal_axes(self.vertical_axis)
        # Optional proximity prior for the view-dependent directional
        # predicates (LeftOf/RightOf/InFrontOf/Behind): their image-plane
        # sigmoids saturate, so "A left of B" scores the same at 0.5 m and
        # 15 m. When FARM_DIRECTIONAL_DISTANCE_SIGMA is set (> 0, metres),
        # directional scores are multiplied by gaussian(||A-B||, sigma) so the
        # closest satisfying candidate wins. Off by default: enabling it
        # changes the paper's locked scoring and needs a benchmark re-run
        # before it can become the eval default.
        try:
            self.directional_distance_sigma = float(os.getenv("FARM_DIRECTIONAL_DISTANCE_SIGMA", "0") or 0.0)
        except ValueError:
            self.directional_distance_sigma = 0.0

        self._means = self._get_means()
        self._captions: List[str] = scene_state.get("object_caption", [])
        self._categories: List[str] = scene_state.get("object_category", [])
        self._key_attributes: List[List[str]] = scene_state.get("object_key_attributes", [])
        self._region_ids: List[int] = scene_state.get("region_ids", [])
        self._region_labels: List[str] = scene_state.get("region_labels", [])
        self._region_object_lists: List[List[int]] = scene_state.get("region_object_lists", [])
        self._region_confidence: List[float] = scene_state.get("region_label_confidence", [])
        self._mask_observations: List[Any] = scene_state.get("object_mask_observations", []) or []
        self._mask_obs_by_image_cache: Dict[int, Dict[int, Mapping[str, Any]]] = {}
        self._mask_geometry_cache: Dict[Tuple[int, int], Optional[Dict[str, Any]]] = {}
        self._image_pose_cache: Dict[int, Optional[np.ndarray]] = {}
        self._has_any_mask_observations = any(
            isinstance(row, (list, tuple)) and any(isinstance(obs, Mapping) for obs in row)
            for row in self._mask_observations
        )

        self._vlm_cache: Dict[str, float] = {}

    def _get_means(self) -> Optional[np.ndarray]:
        means = self.scene_state.get("means")
        if means is None:
            return None
        if hasattr(means, "cpu"):
            return means.cpu().numpy()
        return np.asarray(means)

    def _get_position(self, obj_idx: int) -> Optional[np.ndarray]:
        if self._means is None or obj_idx >= len(self._means):
            return None
        return self._means[obj_idx]

    def _get_caption(self, obj_idx: int) -> str:
        if obj_idx < len(self._captions):
            return self._captions[obj_idx]
        return ""

    def _get_category(self, obj_idx: int) -> str:
        if obj_idx < len(self._categories):
            return self._categories[obj_idx]
        return ""

    def _get_key_attributes(self, obj_idx: int) -> List[str]:
        if obj_idx < len(self._key_attributes):
            return self._key_attributes[obj_idx] or []
        return []

    def _get_region_label(self, obj_idx: int) -> Tuple[str, float]:
        """Return (label, confidence) for the region containing obj_idx."""
        if obj_idx >= len(self._region_ids):
            return "", 0.0
        rid = self._region_ids[obj_idx]
        if rid < 0 or rid >= len(self._region_labels):
            return "", 0.0
        label = self._region_labels[rid]
        conf = self._region_confidence[rid] if rid < len(self._region_confidence) else 0.5
        return label, conf

    def _load_crop(self, obj_idx: int) -> Optional[np.ndarray]:
        """Load a single crop image for an object."""
        from scene_graph.scene_state_io import load_scene_state_image

        observations = self.scene_state.get("rgb_observations", [])
        if obj_idx >= len(observations) or not observations[obj_idx]:
            return None
        obs = observations[obj_idx][0]
        img_ref = obs.get("image_caption") or obs.get("image")
        if img_ref is None:
            return None
        try:
            return load_scene_state_image(img_ref)
        except Exception:
            return None

    # =========================================================================
    # Fast-path evaluators
    # =========================================================================

    def fast_path(self, predicate_name: str, subject_idx: int, anchor_idx: Optional[int] = None, **kwargs: Any) -> Optional[float]:
        """Evaluate a predicate via fast path. Returns score or None if undecidable."""
        fn = self._fast_path_registry.get(predicate_name)
        if fn is None:
            return None
        return fn(self, subject_idx, anchor_idx, **kwargs)

    def _fast_near(self, subject_idx: int, anchor_idx: Optional[int], **kwargs: Any) -> Optional[float]:
        if anchor_idx is None:
            return None
        pos_a = self._get_position(subject_idx)
        pos_b = self._get_position(anchor_idx)
        if pos_a is None or pos_b is None:
            return None
        dist = _euclidean_distance(pos_a, pos_b)
        if dist > self.near_prune_distance:
            return 0.0
        sigma = kwargs.get("sigma", self.near_sigma)
        return gaussian_membership(dist, sigma)

    def _fast_on(self, subject_idx: int, anchor_idx: Optional[int], **kwargs: Any) -> Optional[float]:
        if anchor_idx is None:
            return None
        pos_a = self._get_position(subject_idx)
        pos_b = self._get_position(anchor_idx)
        if pos_a is None or pos_b is None:
            return None
        hdist = _horizontal_distance(pos_a, pos_b, vertical_axis=self.vertical_axis)
        hdiff = _height_diff(pos_a, pos_b, vertical_axis=self.vertical_axis)
        if hdist > self.on_max_horiz:
            return 0.0
        lo, hi = self.on_height_band
        if hdiff < lo - 0.2 or hdiff > hi + 0.3:
            return 0.0
        horiz_score = gaussian_membership(hdist, self.on_max_horiz * 0.7)
        height_score = sigmoid_membership(hdiff, lo, 0.1) * sigmoid_membership(hi - hdiff + lo, lo, 0.1)
        return horiz_score * min(1.0, height_score)

    def _fast_above(self, subject_idx: int, anchor_idx: Optional[int], **kwargs: Any) -> Optional[float]:
        if anchor_idx is None:
            return None
        pos_a = self._get_position(subject_idx)
        pos_b = self._get_position(anchor_idx)
        if pos_a is None or pos_b is None:
            return None
        hdiff = _height_diff(pos_a, pos_b, vertical_axis=self.vertical_axis)
        hdist = _horizontal_distance(pos_a, pos_b, vertical_axis=self.vertical_axis)
        if hdiff < -0.1:
            return 0.0
        if hdist > 2.0:
            return 0.0
        return sigmoid_membership(hdiff, 0.2, 0.15) * gaussian_membership(hdist, 1.0)

    def _fast_below(self, subject_idx: int, anchor_idx: Optional[int], **kwargs: Any) -> Optional[float]:
        if anchor_idx is None:
            return None
        pos_a = self._get_position(subject_idx)
        pos_b = self._get_position(anchor_idx)
        if pos_a is None or pos_b is None:
            return None
        hdiff = _height_diff(pos_b, pos_a, vertical_axis=self.vertical_axis)
        hdist = _horizontal_distance(pos_a, pos_b, vertical_axis=self.vertical_axis)
        if hdiff < -0.1:
            return 0.0
        if hdist > 2.0:
            return 0.0
        return sigmoid_membership(hdiff, 0.2, 0.15) * gaussian_membership(hdist, 1.0)

    def _fast_next_to(self, subject_idx: int, anchor_idx: Optional[int], **kwargs: Any) -> Optional[float]:
        if anchor_idx is None:
            return None
        pos_a = self._get_position(subject_idx)
        pos_b = self._get_position(anchor_idx)
        if pos_a is None or pos_b is None:
            return None
        hdist = _horizontal_distance(pos_a, pos_b, vertical_axis=self.vertical_axis)
        height_diff = abs(_height_diff(pos_a, pos_b, vertical_axis=self.vertical_axis))
        if hdist > 2.0 or height_diff > 1.0:
            return 0.0
        return gaussian_membership(hdist, 1.0)

    def _fast_between(self, subject_idx: int, anchor_idx: Optional[int], *, ref_b_idx: Optional[int] = None, **kwargs: Any) -> Optional[float]:
        ref_a_idx = anchor_idx
        if ref_a_idx is None or ref_b_idx is None:
            return None
        pos_s = self._get_position(subject_idx)
        pos_a = self._get_position(ref_a_idx)
        pos_b = self._get_position(ref_b_idx)
        if pos_s is None or pos_a is None or pos_b is None:
            return None
        axes = list(self.horizontal_axes)
        pos_s_h = pos_s[axes]
        pos_a_h = pos_a[axes]
        pos_b_h = pos_b[axes]
        ab = pos_b_h - pos_a_h
        ab_len = float(np.linalg.norm(ab))
        if ab_len < 0.01:
            return 0.0
        ab_unit = ab / ab_len
        as_vec = pos_s_h - pos_a_h
        t = float(np.dot(as_vec, ab_unit)) / ab_len
        if t < -0.1 or t > 1.1:
            return 0.0
        perp_dist = float(np.linalg.norm(as_vec - t * ab_len * ab_unit))
        t_score = 1.0 - 2.0 * abs(t - 0.5)
        perp_score = gaussian_membership(perp_dist, 1.0)
        return t_score * perp_score

    def _fast_inside(self, subject_idx: int, anchor_idx: Optional[int], **kwargs: Any) -> Optional[float]:
        if anchor_idx is None:
            return None
        pos_s = self._get_position(subject_idx)
        pos_a = self._get_position(anchor_idx)
        if pos_s is None or pos_a is None:
            return None

        means = self.scene_state.get("means")
        cov6 = self.scene_state.get("cov6")
        if means is None or cov6 is None:
            return gaussian_membership(_euclidean_distance(pos_s, pos_a), 0.75)
        try:
            means_np = means.cpu().numpy() if hasattr(means, "cpu") else np.asarray(means)
            cov6_np = cov6.cpu().numpy() if hasattr(cov6, "cpu") else np.asarray(cov6)
            if anchor_idx >= len(means_np) or anchor_idx >= len(cov6_np):
                return gaussian_membership(_euclidean_distance(pos_s, pos_a), 0.75)
            c6 = np.asarray(cov6_np[anchor_idx], dtype=np.float32)
            center = np.asarray(means_np[anchor_idx], dtype=np.float32)
            # cov6 layout is [xx, xy, xz, yy, yz, zz].
            diag = np.maximum(np.asarray([c6[0], c6[3], c6[5]], dtype=np.float32), 1e-6)
            half = 2.5 * np.sqrt(diag) + 0.10
            bmin = center - half
            bmax = center + half
            delta = np.maximum(np.maximum(bmin - pos_s, pos_s - bmax), 0.0)
            outside_dist = float(np.linalg.norm(delta))
            if outside_dist <= 1e-6:
                return 1.0
            return gaussian_membership(outside_dist, 0.35)
        except Exception:
            return gaussian_membership(_euclidean_distance(pos_s, pos_a), 0.75)

    def _fast_in_region(self, subject_idx: int, anchor_idx: Optional[int], *, region: str = "", **kwargs: Any) -> Optional[float]:
        if not region:
            return None
        label, confidence = self._get_region_label(subject_idx)
        if not label:
            return None
        if label.lower() == region.lower():
            return 1.0
        if confidence > 0.7:
            return 0.0
        return None

    def _fast_has_attribute(self, subject_idx: int, anchor_idx: Optional[int], *, attribute: str = "", **kwargs: Any) -> Optional[float]:
        if not attribute:
            return None
        attr_lower = attribute.lower()
        caption = self._get_caption(subject_idx).lower()
        if attr_lower in caption:
            return 1.0
        key_attrs = [a.lower() for a in self._get_key_attributes(subject_idx)]
        if attr_lower in key_attrs:
            return 1.0
        return None

    def _fast_is_category(self, subject_idx: int, anchor_idx: Optional[int], *, category: str = "", **kwargs: Any) -> Optional[float]:
        if not category:
            return None
        cat = self._get_category(subject_idx).lower()
        if cat and cat == category.lower():
            return 1.0
        caption = self._get_caption(subject_idx).lower()
        if category.lower() in caption:
            return 0.9
        return None

    # ---- Superlative predicates (need all candidates; fast_path returns None) ----

    def _fast_closest(self, subject_idx: int, anchor_idx: Optional[int], **kwargs: Any) -> Optional[float]:
        """Cannot evaluate per-candidate; requires all candidates. Returns None."""
        return None

    def _fast_farthest(self, subject_idx: int, anchor_idx: Optional[int], **kwargs: Any) -> Optional[float]:
        """Cannot evaluate per-candidate; requires all candidates. Returns None."""
        return None

    # ---- View-dependent predicates ----

    def _object_mask_rows(self, obj_idx: int) -> List[Mapping[str, Any]]:
        if obj_idx < 0 or obj_idx >= len(self._mask_observations):
            return []
        row = self._mask_observations[obj_idx]
        if not isinstance(row, (list, tuple)):
            return []
        return [obs for obs in row if isinstance(obs, Mapping)]

    def _has_object_mask_observations(self, obj_idx: Optional[int]) -> bool:
        if obj_idx is None:
            return False
        return bool(self._object_mask_rows(int(obj_idx)))

    def _observation_quality_key(self, observation: Mapping[str, Any]) -> Tuple[float, float]:
        pixels = _safe_float(observation.get("raw_pixels"), 0.0)
        if pixels <= 0.0:
            bbox = _first_bbox_xyxy(
                observation.get("raw_bbox_xyxy"),
                observation.get("bbox_xyxy"),
                observation.get("bbox"),
            )
            if bbox is not None:
                pixels = float(max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
        return pixels, _safe_float(observation.get("score"), 1.0)

    def _mask_observations_by_image(self, obj_idx: int) -> Dict[int, Mapping[str, Any]]:
        cached = self._mask_obs_by_image_cache.get(int(obj_idx))
        if cached is not None:
            return cached
        by_image: Dict[int, Mapping[str, Any]] = {}
        for observation in self._object_mask_rows(int(obj_idx)):
            image_id = _observation_image_id(observation)
            if image_id is None:
                continue
            current = by_image.get(image_id)
            if current is None or self._observation_quality_key(observation) > self._observation_quality_key(current):
                by_image[image_id] = observation
        self._mask_obs_by_image_cache[int(obj_idx)] = by_image
        return by_image

    def _mask_observation_geometry(self, observation: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        image_id = _observation_image_id(observation)
        cache_key = (int(id(observation)), int(image_id) if image_id is not None else -1)
        if cache_key in self._mask_geometry_cache:
            return self._mask_geometry_cache[cache_key]

        image_shape = _as_image_shape(observation.get("image_shape"))
        bbox = _first_bbox_xyxy(
            observation.get("raw_bbox_xyxy"),
            observation.get("bbox_xyxy"),
            observation.get("bbox"),
        )
        pixels = _safe_float(observation.get("raw_pixels"), 0.0)
        center: Optional[np.ndarray] = None

        path = _resolve_mask_observation_path(observation.get("path") or observation.get("mask_path"))
        if path is not None:
            try:
                with np.load(str(path), allow_pickle=False) as data:
                    if "image_shape" in data.files:
                        loaded_shape = _as_image_shape(data["image_shape"])
                        if loaded_shape is not None:
                            image_shape = loaded_shape
                    if "raw_bbox_xyxy" in data.files:
                        loaded_bbox = _as_bbox_xyxy(data["raw_bbox_xyxy"])
                        if loaded_bbox is not None:
                            bbox = loaded_bbox
                    if bbox is not None and "raw_bits" in data.files and "raw_shape" in data.files:
                        raw_shape = _as_image_shape(data["raw_shape"])
                        if raw_shape is not None:
                            crop_h, crop_w = raw_shape
                            flat = np.unpackbits(
                                np.asarray(data["raw_bits"], dtype=np.uint8),
                                bitorder="little",
                            )[: crop_h * crop_w]
                            if flat.size == crop_h * crop_w:
                                crop = flat.reshape(crop_h, crop_w).astype(bool, copy=False)
                                ys, xs = np.nonzero(crop)
                                if xs.size:
                                    center = np.asarray(
                                        [float(bbox[0]) + float(xs.mean()), float(bbox[1]) + float(ys.mean())],
                                        dtype=np.float32,
                                    )
                                    pixels = float(xs.size)
            except Exception:
                pass

        if bbox is None or image_shape is None:
            self._mask_geometry_cache[cache_key] = None
            return None
        if center is None:
            center = np.asarray(
                [0.5 * (float(bbox[0]) + float(bbox[2])), 0.5 * (float(bbox[1]) + float(bbox[3]))],
                dtype=np.float32,
            )
        if pixels <= 0.0:
            pixels = float(max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))

        geometry = {
            "center": center,
            "bbox": bbox.astype(np.float32, copy=False),
            "image_shape": image_shape,
            "pixels": float(pixels),
            "score": _safe_float(observation.get("score"), 1.0),
        }
        self._mask_geometry_cache[cache_key] = geometry
        return geometry

    @staticmethod
    def _bbox_edge_score(geometry: Dict[str, Any]) -> float:
        h, w = geometry["image_shape"]
        x0, y0, x1, y1 = [float(v) for v in geometry["bbox"]]
        margin = min(x0, y0, float(w) - x1, float(h) - y1)
        scale = max(16.0, 0.05 * float(min(h, w)))
        return max(0.0, min(1.0, margin / scale))

    def _image_pose(self, image_id: int) -> Optional[np.ndarray]:
        cached = self._image_pose_cache.get(int(image_id))
        if int(image_id) in self._image_pose_cache:
            return cached
        pose_raw = None
        images = self.scene_state.get("images") or []
        if 0 <= int(image_id) < len(images):
            record = images[int(image_id)]
            if isinstance(record, Mapping):
                pose_raw = record.get("pose")
                if pose_raw is None:
                    pose_raw = record.get("T_world_cam")
            else:
                pose_raw = getattr(record, "pose", None)
                if pose_raw is None:
                    pose_raw = getattr(record, "T_world_cam", None)
        pose: Optional[np.ndarray] = None
        if pose_raw is not None:
            try:
                arr = pose_raw.detach().cpu().numpy() if hasattr(pose_raw, "detach") else np.asarray(pose_raw)
                arr = np.asarray(arr, dtype=np.float64).reshape(4, 4)
                if np.all(np.isfinite(arr)):
                    pose = arr
            except Exception:
                pose = None
        self._image_pose_cache[int(image_id)] = pose
        return pose

    def _shared_mask_view_score(
        self,
        subject_idx: int,
        anchor_idx: Optional[int],
        score_fn: Callable[[int, Dict[str, Any], Dict[str, Any]], Optional[float]],
    ) -> Optional[float]:
        """Pick the best shared stored view and score a relation in that view."""
        if anchor_idx is None:
            return None
        subject_by_image = self._mask_observations_by_image(int(subject_idx))
        anchor_by_image = self._mask_observations_by_image(int(anchor_idx))
        if not subject_by_image or not anchor_by_image:
            return None

        best_quality = -1.0
        best_score: Optional[float] = None
        for image_id in sorted(set(subject_by_image).intersection(anchor_by_image)):
            subj_geom = self._mask_observation_geometry(subject_by_image[image_id])
            anchor_geom = self._mask_observation_geometry(anchor_by_image[image_id])
            if subj_geom is None or anchor_geom is None:
                continue
            score = score_fn(image_id, subj_geom, anchor_geom)
            if score is None:
                continue
            visibility = min(float(subj_geom["pixels"]), float(anchor_geom["pixels"]))
            confidence = max(0.01, min(float(subj_geom["score"]), float(anchor_geom["score"])))
            edge_score = min(self._bbox_edge_score(subj_geom), self._bbox_edge_score(anchor_geom))
            quality = visibility * confidence * (0.25 + 0.75 * edge_score)
            if quality > best_quality:
                best_quality = quality
                best_score = score

        return best_score

    def _image_plane_lateral_score(
        self,
        subject_idx: int,
        anchor_idx: Optional[int],
        component: str,
        **kwargs: Any,
    ) -> Optional[float]:
        """Score left/right in the saved RGB image plane using shared mask views."""
        if component not in {"left", "right"}:
            return None
        temperature = max(1e-4, _safe_float(kwargs.get("image_plane_temperature"), 0.03))

        def _score(_image_id: int, subj_geom: Dict[str, Any], anchor_geom: Dict[str, Any]) -> Optional[float]:
            _, subj_w = subj_geom["image_shape"]
            _, anchor_w = anchor_geom["image_shape"]
            image_width = float(max(1, subj_w, anchor_w))
            subj_x = float(subj_geom["center"][0])
            anchor_x = float(anchor_geom["center"][0])
            direction_px = anchor_x - subj_x if component == "left" else subj_x - anchor_x
            return sigmoid_membership(direction_px / image_width, 0.0, temperature)

        return self._shared_mask_view_score(subject_idx, anchor_idx, _score)

    def _image_plane_depth_score(
        self,
        subject_idx: int,
        anchor_idx: Optional[int],
        component: str,
        **kwargs: Any,
    ) -> Optional[float]:
        """Score front/behind by camera-z in the saved shared RGB view."""
        if component not in {"front", "behind"}:
            return None
        if anchor_idx is None:
            return None
        pos_target = self._get_position(subject_idx)
        pos_anchor = self._get_position(anchor_idx)
        if pos_target is None or pos_anchor is None:
            return None
        temperature = max(1e-4, _safe_float(kwargs.get("image_depth_temperature_m"), 0.30))

        def _score(image_id: int, _subj_geom: Dict[str, Any], _anchor_geom: Dict[str, Any]) -> Optional[float]:
            pose = self._image_pose(image_id)
            if pose is None:
                return None
            try:
                cam_from_world = np.linalg.inv(pose)
                pts = np.asarray([pos_target, pos_anchor], dtype=np.float64)
                pts_h = np.concatenate([pts, np.ones((2, 1), dtype=np.float64)], axis=1)
                pts_cam = (cam_from_world @ pts_h.T).T[:, :3]
            except Exception:
                return None
            if not np.all(np.isfinite(pts_cam)):
                return None
            subj_z = float(pts_cam[0, 2])
            anchor_z = float(pts_cam[1, 2])
            if subj_z <= 1e-6 or anchor_z <= 1e-6:
                return None
            direction_m = anchor_z - subj_z if component == "front" else subj_z - anchor_z
            return sigmoid_membership(direction_m, 0.0, temperature)

        return self._shared_mask_view_score(subject_idx, anchor_idx, _score)

    def _view_dependent_score(self, subject_idx: int, anchor_idx: Optional[int], component: str, **kwargs: Any) -> Optional[float]:
        """Shared logic for view-dependent predicates.

        Left/right predicates are evaluated in stored shared image views, using
        mask centroids or bbox centers on the image plane. Front/behind
        predicates are evaluated from the same shared stored views using the
        saved frame pose and camera-z depth. If a pair is never jointly visible
        in a stored view, the view-dependent relation is unsupported and scores
        0.0.
        """
        if component in {"left", "right"}:
            score = self._image_plane_lateral_score(subject_idx, anchor_idx, component, **kwargs)
            if score is not None:
                return self._apply_directional_distance_prior(score, subject_idx, anchor_idx)
            return 0.0
        if component in {"front", "behind"}:
            score = self._image_plane_depth_score(subject_idx, anchor_idx, component, **kwargs)
            if score is not None:
                return self._apply_directional_distance_prior(score, subject_idx, anchor_idx)
            return 0.0
        return None

    def _apply_directional_distance_prior(
        self, score: float, subject_idx: int, anchor_idx: Optional[int]
    ) -> float:
        """Soft proximity prior for directional predicates (see __init__)."""
        sigma = self.directional_distance_sigma
        if sigma <= 0.0 or score <= 0.0 or anchor_idx is None:
            return float(score)
        pos_a = self._get_position(subject_idx)
        pos_b = self._get_position(anchor_idx)
        if pos_a is None or pos_b is None:
            return float(score)
        return float(score) * gaussian_membership(_euclidean_distance(pos_a, pos_b), sigma)

    def _fast_left_of(self, subject_idx: int, anchor_idx: Optional[int], **kwargs: Any) -> Optional[float]:
        return self._view_dependent_score(subject_idx, anchor_idx, "left", **kwargs)

    def _fast_right_of(self, subject_idx: int, anchor_idx: Optional[int], **kwargs: Any) -> Optional[float]:
        return self._view_dependent_score(subject_idx, anchor_idx, "right", **kwargs)

    def _fast_in_front_of(self, subject_idx: int, anchor_idx: Optional[int], **kwargs: Any) -> Optional[float]:
        return self._view_dependent_score(subject_idx, anchor_idx, "front", **kwargs)

    def _fast_behind(self, subject_idx: int, anchor_idx: Optional[int], **kwargs: Any) -> Optional[float]:
        return self._view_dependent_score(subject_idx, anchor_idx, "behind", **kwargs)

    _fast_path_registry: Dict[str, Callable] = {
        "Near": _fast_near,
        "On": _fast_on,
        "Above": _fast_above,
        "Below": _fast_below,
        "NextTo": _fast_next_to,
        "Between": _fast_between,
        "Inside": _fast_inside,
        "InRegion": _fast_in_region,
        "HasAttribute": _fast_has_attribute,
        "IsCategory": _fast_is_category,
        "Closest": _fast_closest,
        "Farthest": _fast_farthest,
        "LeftOf": _fast_left_of,
        "RightOf": _fast_right_of,
        "InFrontOf": _fast_in_front_of,
        "Behind": _fast_behind,
    }

    # =========================================================================
    # VLM evaluators
    # =========================================================================

    def vlm_evaluate(
        self,
        predicate_name: str,
        subject_idx: int,
        anchor_idx: Optional[int] = None,
        *,
        ref_b_idx: Optional[int] = None,
        attribute: str = "",
        category: str = "",
        region: str = "",
    ) -> Optional[float]:
        """Evaluate a predicate via VLM. Returns score in [0,1] or None on failure."""
        cache_key = f"{predicate_name}:{subject_idx}:{anchor_idx}:{ref_b_idx}:{attribute}:{category}:{region}"
        if cache_key in self._vlm_cache:
            return self._vlm_cache[cache_key]

        prompt_template = PREDICATE_PROMPTS.get(predicate_name)
        if prompt_template is None:
            return None

        images: List[np.ndarray] = []
        format_kwargs: Dict[str, Any] = {}

        subject_crop = self._load_crop(subject_idx)
        if subject_crop is not None:
            images.append(subject_crop)
        format_kwargs["subject_description"] = self._get_caption(subject_idx) or f"object #{subject_idx}"
        pos_s = self._get_position(subject_idx)
        format_kwargs["subject_pos"] = _format_pos(pos_s) if pos_s is not None else "unknown"

        if predicate_name in ("Near", "On", "Above", "Below", "NextTo", "Inside"):
            if anchor_idx is None:
                return None
            anchor_crop = self._load_crop(anchor_idx)
            if anchor_crop is not None:
                images.append(anchor_crop)
            format_kwargs["anchor_description"] = self._get_caption(anchor_idx) or f"object #{anchor_idx}"
            pos_b = self._get_position(anchor_idx)
            format_kwargs["anchor_pos"] = _format_pos(pos_b) if pos_b is not None else "unknown"

            if pos_s is not None and pos_b is not None:
                format_kwargs["distance"] = _euclidean_distance(pos_s, pos_b)
                format_kwargs["horiz_dist"] = _horizontal_distance(
                    pos_s, pos_b, vertical_axis=self.vertical_axis
                )
                format_kwargs["height_diff"] = _height_diff(
                    pos_s, pos_b, vertical_axis=self.vertical_axis
                )
            else:
                format_kwargs["distance"] = 0.0
                format_kwargs["horiz_dist"] = 0.0
                format_kwargs["height_diff"] = 0.0

        elif predicate_name == "Between":
            if anchor_idx is None or ref_b_idx is None:
                return None
            crop_a = self._load_crop(anchor_idx)
            crop_b = self._load_crop(ref_b_idx)
            if crop_a is not None:
                images.append(crop_a)
            if crop_b is not None:
                images.append(crop_b)
            format_kwargs["ref_a_description"] = self._get_caption(anchor_idx) or f"object #{anchor_idx}"
            format_kwargs["ref_b_description"] = self._get_caption(ref_b_idx) or f"object #{ref_b_idx}"
            pos_a = self._get_position(anchor_idx)
            pos_b = self._get_position(ref_b_idx)
            format_kwargs["ref_a_pos"] = _format_pos(pos_a) if pos_a is not None else "unknown"
            format_kwargs["ref_b_pos"] = _format_pos(pos_b) if pos_b is not None else "unknown"
            if pos_s is not None and pos_a is not None and pos_b is not None:
                format_kwargs["dist_ab"] = _euclidean_distance(pos_s, pos_a)
                format_kwargs["dist_ac"] = _euclidean_distance(pos_s, pos_b)
                format_kwargs["dist_bc"] = _euclidean_distance(pos_a, pos_b)
            else:
                format_kwargs["dist_ab"] = 0.0
                format_kwargs["dist_ac"] = 0.0
                format_kwargs["dist_bc"] = 0.0

        elif predicate_name == "HasAttribute":
            format_kwargs["attribute"] = attribute

        elif predicate_name == "IsCategory":
            format_kwargs["category"] = category

        elif predicate_name == "InRegion":
            format_kwargs["region"] = region
            label, conf = self._get_region_label(subject_idx)
            format_kwargs["assigned_region"] = label or "unassigned"
            format_kwargs["region_confidence"] = conf
            region_objs = self._get_region_context_objects(subject_idx, max_objects=5)
            format_kwargs["region_objects"] = region_objs

        try:
            prompt = prompt_template.format(**format_kwargs)
        except KeyError as e:
            logger.warning("Failed to format prompt for %s: missing key %s", predicate_name, e)
            return None

        try:
            original_max_tokens = self.llm.config.max_tokens
            self.llm.config.max_tokens = min(256, original_max_tokens)
            try:
                response = self.llm.query(prompt, images=images if images else None)
            finally:
                self.llm.config.max_tokens = original_max_tokens
        except Exception as e:
            logger.warning("VLM call failed for %s: %s", predicate_name, e)
            return None

        score = _parse_vlm_score(response)
        if score is not None:
            self._vlm_cache[cache_key] = score
        return score

    def _get_region_context_objects(self, obj_idx: int, max_objects: int = 5) -> str:
        """Get captions of other objects in the same region for context."""
        if obj_idx >= len(self._region_ids):
            return "unknown"
        rid = self._region_ids[obj_idx]
        if rid < 0 or rid >= len(self._region_object_lists):
            return "unknown"
        obj_list = self._region_object_lists[rid]
        context = []
        for idx in obj_list:
            if idx == obj_idx:
                continue
            cap = self._get_caption(idx)
            if cap:
                context.append(cap)
            if len(context) >= max_objects:
                break
        return ", ".join(context) if context else "none"

    # =========================================================================
    # Unified evaluate
    # =========================================================================

    def _should_skip_vlm(self, predicate_name: str, fast_score: Optional[float]) -> bool:
        """Predicate-type-specific VLM gate. Returns True if VLM can be skipped."""
        if fast_score is None:
            return False

        # Above/Below: geometry is reliable for vertical relations
        if predicate_name in ("Above", "Below"):
            return fast_score > 0.70 or fast_score <= 0.05

        # Near/NextTo: invoke VLM only in uncertain band (0.30, 0.70)
        if predicate_name in ("Near", "NextTo"):
            return fast_score <= 0.30 or fast_score >= 0.70

        # InRegion: skip VLM when exact match found
        if predicate_name == "InRegion":
            return fast_score == 1.0 or fast_score <= 0.05

        # Superlative and view-dependent: geometry only, no VLM.
        if predicate_name in ("Closest", "Farthest", "LeftOf", "RightOf", "InFrontOf", "Behind"):
            return True

        # Default: skip VLM only at extremes
        return fast_score >= 0.95 or fast_score <= 0.05

    def evaluate(
        self,
        predicate_name: str,
        subject_idx: int,
        anchor_idx: Optional[int] = None,
        *,
        ref_b_idx: Optional[int] = None,
        attribute: str = "",
        category: str = "",
        region: str = "",
        use_vlm: bool = True,
    ) -> PredicateResult:
        """Evaluate a predicate: try fast path first, fall back to VLM if needed."""
        fast_kwargs: Dict[str, Any] = {}
        if attribute:
            fast_kwargs["attribute"] = attribute
        if category:
            fast_kwargs["category"] = category
        if region:
            fast_kwargs["region"] = region
        if ref_b_idx is not None:
            fast_kwargs["ref_b_idx"] = ref_b_idx

        fast_score = self.fast_path(predicate_name, subject_idx, anchor_idx, **fast_kwargs)

        if fast_score is not None:
            if self._should_skip_vlm(predicate_name, fast_score):
                return PredicateResult(name=predicate_name, score=fast_score, status="fast_path")
            if not use_vlm:
                return PredicateResult(name=predicate_name, score=fast_score, status="fast_path")

        if not use_vlm:
            if fast_score is not None:
                return PredicateResult(name=predicate_name, score=fast_score, status="fast_path")
            return PredicateResult(name=predicate_name, score=1.0, status="dropped", drop_reason="no_vlm_and_undecidable")

        vlm_score = self.vlm_evaluate(
            predicate_name, subject_idx, anchor_idx,
            ref_b_idx=ref_b_idx, attribute=attribute, category=category, region=region,
        )

        if vlm_score is not None:
            return PredicateResult(name=predicate_name, score=vlm_score, status="evaluated")

        if fast_score is not None:
            return PredicateResult(name=predicate_name, score=fast_score, status="fast_path")

        return PredicateResult(name=predicate_name, score=1.0, status="dropped", drop_reason="evaluation_failed")
