"""Unified scoring for the canonical predictions schema.

The canonical, apples-to-apples evaluation pipeline. One scoring code path
serves every method × benchmark:
the per-method differences live in the canonical preds JSON
(``pred_mask_source`` + ``chosen_view_image_id``); the scorer is identical.

Headline metric per locked defaults: ``acc@1@mask_iou=0.1`` (visible-mask
single-view IoU). The single canonical view per candidate is resolved with
the precedence in :func:`scene_graph.eval.view_selection.resolve_chosen_view_image_id`.

Public entry points:

- :func:`load_pred_mask`: pred-mask loader that dispatches on
  ``candidate["pred_mask_source"]``.
- :func:`score_one_candidate`: score a single (candidate, chosen-view, GT)
  triple and return the IoU + image_id.
- :func:`score_predictions_unified`: full sweep — loads canonical preds,
  scores every record, returns the metrics JSON dict.
- :func:`aggregate_overall`: pure aggregator over per-utterance scores.
"""

from __future__ import annotations

import gzip
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from scene_graph.eval.view_selection import resolve_chosen_view_image_id
from scene_graph.eval.visible_mask import (
    EvalFrame,
    GTMeshMaskProvider,
    SceneStateMaskIndex,
    mask_overlap,
    resize_bool_mask,
    visible_points_mask,
)


# ---------------------------------------------------------------------
# Frame source abstractions
# ---------------------------------------------------------------------


class _SceneStateFrameSource:
    """Load EvalFrame instances from ours' SceneStateMaskIndex frame_resolver."""

    def __init__(self, mask_index: SceneStateMaskIndex) -> None:
        self._mask_index = mask_index

    def load(self, image_id: int) -> Optional[EvalFrame]:
        return self._mask_index.frame_resolver.load(int(image_id))


class _BBQExtractedFramesSource:
    """Load EvalFrame from a BBQ ``bbq_frames/<scene>/`` dir layout.

    Expected layout::

        <dir>/pose/<image_id:06d>.txt          # 4x4 c2w pose
        <dir>/depth/<image_id:06d>.png         # uint16 depth in mm
        <dir>/intrinsics/intrinsic_color.txt   # 4x4 intrinsic, fx/fy/cx/cy
    """

    def __init__(self, frames_dir: Path) -> None:
        self._dir = Path(frames_dir)
        # BBQ stores intrinsics either flat (HM3D magnet layout:
        # ``<dir>/intrinsic_color.txt``) or under ``intrinsics/`` (older
        # ScanNet extracts). Try both.
        cand_paths = (
            self._dir / "intrinsic_color.txt",
            self._dir / "intrinsics" / "intrinsic_color.txt",
        )
        for cand in cand_paths:
            if cand.exists():
                self._K_color = self._load_intrinsic(cand)
                break
        else:  # nobreak
            raise FileNotFoundError(
                f"no intrinsic_color.txt found under {self._dir} (tried: {[str(p) for p in cand_paths]})"
            )
        self._frame_cache: "Dict[int, Optional[EvalFrame]]" = {}

    @staticmethod
    def _load_intrinsic(path: Path) -> np.ndarray:
        with path.open("r") as fp:
            rows = [
                [float(x) for x in ln.strip().split()]
                for ln in fp
                if ln.strip()
            ]
        arr = np.asarray(rows, dtype=np.float64)
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = arr[0, 0]
        K[1, 1] = arr[1, 1]
        K[0, 2] = arr[0, 2]
        K[1, 2] = arr[1, 2]
        return K

    @staticmethod
    def _load_pose(path: Path) -> np.ndarray:
        with path.open("r") as fp:
            rows = [
                [float(x) for x in ln.strip().split()]
                for ln in fp
                if ln.strip()
            ]
        return np.asarray(rows, dtype=np.float32).reshape(4, 4)

    @staticmethod
    def _load_depth(path: Path) -> Optional[np.ndarray]:
        try:
            import imageio.v2 as imageio
        except Exception:  # noqa: BLE001
            try:
                import imageio  # type: ignore
            except Exception:
                return None
        arr = np.asarray(imageio.imread(str(path)))
        if arr.dtype == np.uint16:
            return arr.astype(np.float32) / 1000.0
        if arr.dtype in (np.float32, np.float64):
            return arr.astype(np.float32)
        return arr.astype(np.float32) / 1000.0

    def load(self, image_id: int) -> Optional[EvalFrame]:
        iid = int(image_id)
        if iid in self._frame_cache:
            return self._frame_cache[iid]
        # BBQ extracts use zero-padded 6-digit stems (HM3D, e.g. `000000.png`)
        # OR unpadded stems (ScanNet, e.g. `0.jpg`). RGB extension is `.jpg`
        # on ScanNet and `.png`/`.jpg` on HM3D; depth is always `.png`. Try
        # all combinations for robustness.
        for stem in (f"{iid:06d}", f"{iid}"):
            pose_p = self._dir / "pose" / f"{stem}.txt"
            if not pose_p.exists():
                continue
            depth_p = self._dir / "depth" / f"{stem}.png"
            color_p: Optional[Path] = None
            for color_ext in ("jpg", "png"):
                cand = self._dir / "color" / f"{stem}.{color_ext}"
                if cand.exists():
                    color_p = cand
                    break
            try:
                pose = self._load_pose(pose_p)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("bbq pose load failed %s: %s", pose_p, exc)
                continue
            depth = self._load_depth(depth_p) if depth_p.exists() else None
            # Width/height come from the color image. Read shape lazily.
            rgb: Optional[np.ndarray] = None
            height: int = 0
            width: int = 0
            if color_p is not None:
                try:
                    import imageio.v2 as imageio
                    rgb = np.asarray(imageio.imread(str(color_p)))
                    height, width = int(rgb.shape[0]), int(rgb.shape[1])
                except Exception:
                    rgb = None
            if (height == 0 or width == 0) and depth is not None:
                height, width = int(depth.shape[0]), int(depth.shape[1])
            if height == 0 or width == 0:
                continue
            # Resize depth to color resolution if mismatch (ScanNet bbq_frames
            # have depth at 480x640 but color at 968x1296). The K matrix is
            # for color resolution; without resize, visible_points_mask
            # returns empty.
            if depth is not None and depth.shape[:2] != (height, width):
                try:
                    from PIL import Image as _PILImage
                    d_img = _PILImage.fromarray(depth)
                    depth = np.asarray(d_img.resize((width, height), resample=_PILImage.NEAREST),
                                       dtype=np.float32)
                except Exception:
                    depth = None
            frame = EvalFrame(
                image_id=iid,
                pose_world_cam=pose,
                K=self._K_color,
                width=width,
                height=height,
                depth=depth,
                rgb=rgb,
                source_ref=f"bbq_frames:{self._dir.name}:{stem}",
            )
            self._frame_cache[iid] = frame
            if len(self._frame_cache) > 256:
                self._frame_cache.pop(next(iter(self._frame_cache)))
            return frame
        self._frame_cache[iid] = None
        return None


class _ScanNetSensDirectFrameSource:
    """Load EvalFrame directly from a ScanNet ``.sens`` file at stride=1.

    Used for methods (e.g. RynnBrain ScanNet) whose ``frame_idx`` is the
    native unstrided sens frame index — distinct from ours' scene_state
    ``image_id`` which corresponds to a strided sub-sample of the same trajectory.
    """

    def __init__(self, scan_id: str, scans_dir: Path) -> None:
        self._scan_id = str(scan_id)
        self._sens_path = Path(scans_dir) / scan_id / f"{scan_id}.sens"
        self._cache: "Dict[int, Optional[EvalFrame]]" = {}

    def load(self, image_id: int) -> Optional[EvalFrame]:
        iid = int(image_id)
        if iid in self._cache:
            return self._cache[iid]
        try:
            from scene_graph.offline.frame_sources.sens import SensFrameSource
            src = SensFrameSource(str(self._sens_path), stride=1, start=iid, end=iid + 1)
            item = next(iter(src))
        except (StopIteration, Exception) as exc:  # noqa: BLE001
            LOGGER.debug("sens-direct frame load failed scan=%s idx=%d: %s", self._scan_id, iid, exc)
            self._cache[iid] = None
            return None
        rgb = np.asarray(item.get("rgb"))
        depth = np.asarray(item.get("depth_f32")) if item.get("depth_f32") is not None else None
        pose = np.asarray(item.get("T_world_cam"), dtype=np.float32).reshape(4, 4)
        ri = item.get("rgb_instrinsics") or item.get("rgb_intrinsics") or {}
        K = np.array(
            [[float(ri.get("fx", 0.0)), 0.0, float(ri.get("cx", 0.0))],
             [0.0, float(ri.get("fy", 0.0)), float(ri.get("cy", 0.0))],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        H, W = rgb.shape[:2]
        # ScanNet's .sens has color at 1296x968 but depth at 640x480.
        # Resize depth to color resolution (nearest neighbor) so projected GT
        # points pass the depth-tolerance test below.
        if depth is not None and depth.shape[:2] != (H, W):
            try:
                from PIL import Image as _PILImage
                d_img = _PILImage.fromarray(depth)
                depth = np.asarray(d_img.resize((W, H), resample=_PILImage.NEAREST), dtype=np.float32)
            except Exception:
                depth = None
        frame = EvalFrame(
            image_id=iid,
            pose_world_cam=pose,
            K=K,
            width=int(W),
            height=int(H),
            depth=depth,
            rgb=rgb,
            source_ref=f"{self._sens_path}#frame={iid}",
        )
        self._cache[iid] = frame
        if len(self._cache) > 64:
            self._cache.pop(next(iter(self._cache)))
        return frame


def make_frame_source(
    *,
    frame_source_kind: str,
    mask_index: Optional[SceneStateMaskIndex] = None,
    frames_dir: Optional[Path] = None,
    scan_id: Optional[str] = None,
    scans_dir: Optional[Path] = None,
) -> Any:
    """Resolve a frame source by kind.

    - ``ours_scene_state``: wrap ours' ``SceneStateMaskIndex.frame_resolver``.
    - ``bbq_extracted_frames``: load from a ``bbq_frames/<scene>/`` dir layout.
    - ``scannet_sens_direct``: read ScanNet .sens directly at stride=1 (for
      methods whose frame_idx is the unstride'd sens frame).
    """
    if frame_source_kind == "ours_scene_state":
        if mask_index is None:
            raise ValueError("ours_scene_state frame source requires mask_index")
        return _SceneStateFrameSource(mask_index)
    if frame_source_kind == "bbq_extracted_frames":
        if frames_dir is None:
            raise ValueError("bbq_extracted_frames frame source requires frames_dir")
        return _BBQExtractedFramesSource(Path(frames_dir))
    if frame_source_kind == "scannet_sens_direct":
        if scan_id is None or scans_dir is None:
            raise ValueError("scannet_sens_direct frame source requires scan_id and scans_dir")
        return _ScanNetSensDirectFrameSource(scan_id, Path(scans_dir))
    raise ValueError(f"unknown frame_source kind={frame_source_kind!r}")

LOGGER = logging.getLogger("scene_graph.eval.unified_scoring")


# ---------------------------------------------------------------------
# Canonical schema constants
# ---------------------------------------------------------------------

VALID_PRED_MASK_SOURCES = {
    "ours_state",
    "bbq_pickle",
    "sam2_npz",
    "voxel_projection",
}

# Default IoU thresholds reported in `overall` (locked). Recall is computed at
# "any positive overlap" as well as the IoU thresholds.
DEFAULT_IOU_THRESHOLDS: Tuple[float, ...] = (0.1, 0.25, 0.5)
DEFAULT_RECALL_KS: Tuple[int, ...] = (1, 3, 5, 10)
HEADLINE_IOU = 0.1


# ---------------------------------------------------------------------
# Pred-mask loader dispatch
# ---------------------------------------------------------------------

_BBQ_PICKLE_CACHE: "Dict[str, Dict[int, Mapping[str, Any]]]" = {}
_BBQ_PICKLE_LRU_LIMIT = 4


def _load_bbq_pickle_objects(pkl_path: Path) -> Dict[int, Mapping[str, Any]]:
    key = str(pkl_path)
    cached = _BBQ_PICKLE_CACHE.get(key)
    if cached is not None:
        return cached
    with gzip.open(key, "rb") as fh:
        data = pickle.load(fh)
    if isinstance(data, dict):
        objs = data.get("objects") or []
    else:
        objs = list(data)
    out = {idx: obj for idx, obj in enumerate(objs)}
    while len(_BBQ_PICKLE_CACHE) >= _BBQ_PICKLE_LRU_LIMIT:
        _BBQ_PICKLE_CACHE.pop(next(iter(_BBQ_PICKLE_CACHE)))
    _BBQ_PICKLE_CACHE[key] = out
    return out


def _evidence_object_id(candidate: Mapping[str, Any]) -> int:
    raw = candidate.get("evidence_object_id")
    if raw is None:
        raw = candidate.get("object_id", -1)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1


def load_pred_mask(
    candidate: Mapping[str, Any],
    chosen_image_id: int,
    *,
    mask_index: Optional[SceneStateMaskIndex],
    frame: EvalFrame,
    depth_tolerance_m: float = 0.15,  # locked 2026-05-16: matches the legacy visible-mask protocol
    point_radius_px: int = 3,
    max_points: int = 50000,
    require_depth: bool = True,
) -> Optional[np.ndarray]:
    """Return the predicted 2D mask in ``frame`` for ``candidate``.

    Dispatches on ``candidate['pred_mask_source']`` (one of
    :data:`VALID_PRED_MASK_SOURCES`). Returns ``None`` when the mask can't
    be resolved; the scorer should record an error and treat IoU as 0.
    """

    src = str(candidate.get("pred_mask_source") or "ours_state").strip().lower()
    if src not in VALID_PRED_MASK_SOURCES:
        raise ValueError(
            f"unknown pred_mask_source={src!r}; valid: {sorted(VALID_PRED_MASK_SOURCES)}"
        )

    H, W = int(frame.height), int(frame.width)

    if src == "ours_state":
        if mask_index is None:
            return None
        evidence_oid = _evidence_object_id(candidate)
        idx = mask_index.object_id_to_index.get(int(evidence_oid))
        if idx is None or idx >= len(mask_index.object_mask_observations):
            return None
        raw = mask_index.object_mask_observations[idx]
        if not isinstance(raw, (list, tuple)):
            return None
        target = int(chosen_image_id)
        for obs in raw:
            if isinstance(obs, dict) and int(obs.get("image_id", -1)) == target:
                return mask_index.load_predicted_observation_mask(
                    obs, kind="raw", frame=frame
                )
        return None

    if src == "bbq_pickle":
        pkl_path = candidate.get("pred_mask_path")
        if not pkl_path:
            return None
        objs = _load_bbq_pickle_objects(Path(pkl_path))
        obj_idx = int(candidate.get("object_id", -1))
        obj = objs.get(obj_idx)
        if obj is None:
            return None
        mask = obj.get("mask") if isinstance(obj, dict) else getattr(obj, "mask", None)
        if mask is None:
            return None
        m = np.asarray(mask).astype(bool)
        if m.shape != (H, W):
            m = resize_bool_mask(m, H, W)
        return m

    if src == "sam2_npz":
        npz_path = candidate.get("pred_mask_path")
        if not npz_path:
            return None
        with np.load(npz_path) as blob:
            keys = set(blob.files)
            if "mask" in keys:
                m = np.asarray(blob["mask"]).astype(bool)
            elif "raw_bits" in keys and "mask_shape" in keys:
                shape = tuple(int(x) for x in blob["mask_shape"])
                m = np.unpackbits(np.asarray(blob["raw_bits"], dtype=np.uint8))
                m = m[: int(np.prod(shape))].reshape(shape).astype(bool)
            else:
                return None
        if m.shape != (H, W):
            m = resize_bool_mask(m, H, W)
        return m

    if src == "voxel_projection":
        if mask_index is None:
            return None
        evidence_oid = _evidence_object_id(candidate)
        points = mask_index.object_points(int(evidence_oid))
        if points.size == 0:
            return None
        return visible_points_mask(
            points,
            frame,
            depth_tolerance_m=depth_tolerance_m,
            point_radius_px=point_radius_px,
            max_points=max_points,
            require_depth=require_depth,
        )

    # Unreachable due to the validation above, but keep the explicit return.
    return None


# ---------------------------------------------------------------------
# Per-candidate scoring
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class SingleViewScore:
    """One (candidate, chosen-view) IoU result."""

    image_id: Optional[int]
    iou: float
    precision: float
    recall: float
    pred_pixels: int
    gt_pixels: int
    error: Optional[str] = None


def _render_gt_mask(
    *,
    frame: EvalFrame,
    gt_mask_provider: Optional[GTMeshMaskProvider],
    gt_instance: Optional[Any],
    gt_points: Optional[np.ndarray],
    depth_tolerance_m: float,
    point_radius_px: int,
    max_points: int,
    require_depth: bool,
) -> Optional[np.ndarray]:
    if gt_mask_provider is not None and gt_instance is not None:
        return gt_mask_provider.render_mask(
            gt_instance,
            frame,
            depth_tolerance_m=depth_tolerance_m,
            require_depth=require_depth,
        )
    if gt_points is not None and gt_points.size > 0:
        return visible_points_mask(
            np.asarray(gt_points, dtype=np.float32).reshape(-1, 3),
            frame,
            depth_tolerance_m=depth_tolerance_m,
            point_radius_px=point_radius_px,
            max_points=max_points,
            require_depth=require_depth,
        )
    return None


def score_one_candidate(
    candidate: Mapping[str, Any],
    *,
    mask_index: Optional[SceneStateMaskIndex],
    frame: EvalFrame,
    gt_mask_provider: Optional[GTMeshMaskProvider] = None,
    gt_instance: Optional[Any] = None,
    gt_points: Optional[np.ndarray] = None,
    depth_tolerance_m: float = 0.15,  # locked 2026-05-16: matches the legacy visible-mask protocol
    point_radius_px: int = 3,
    min_gt_pixels: int = 20,
    max_points: int = 50000,
    require_depth: bool = True,
) -> SingleViewScore:
    """Score one (candidate, chosen-view) IoU.

    Does NOT pick the chosen view — caller must have already resolved it
    via :func:`resolve_chosen_view_image_id`.
    """
    gt_mask = _render_gt_mask(
        frame=frame,
        gt_mask_provider=gt_mask_provider,
        gt_instance=gt_instance,
        gt_points=gt_points,
        depth_tolerance_m=depth_tolerance_m,
        point_radius_px=point_radius_px,
        max_points=max_points,
        require_depth=require_depth,
    )
    if gt_mask is None:
        return SingleViewScore(int(frame.image_id), 0.0, 0.0, 0.0, 0, 0, error="no_gt_mask")
    if int(gt_mask.sum()) < int(min_gt_pixels):
        return SingleViewScore(int(frame.image_id), 0.0, 0.0, 0.0, 0, int(gt_mask.sum()), error="gt_mask_below_min_pixels")
    pred_mask = load_pred_mask(
        candidate,
        int(frame.image_id),
        mask_index=mask_index,
        frame=frame,
        depth_tolerance_m=depth_tolerance_m,
        point_radius_px=point_radius_px,
        max_points=max_points,
        require_depth=require_depth,
    )
    if pred_mask is None:
        return SingleViewScore(int(frame.image_id), 0.0, 0.0, 0.0, 0, int(gt_mask.sum()), error="no_pred_mask")
    if pred_mask.shape != gt_mask.shape:
        pred_mask = resize_bool_mask(pred_mask, gt_mask.shape[0], gt_mask.shape[1])
    overlap = mask_overlap(pred_mask, gt_mask)
    return SingleViewScore(
        image_id=int(frame.image_id),
        iou=float(overlap.iou),
        precision=float(overlap.precision),
        recall=float(overlap.recall),
        pred_pixels=int(overlap.pred_pixels),
        gt_pixels=int(overlap.gt_pixels),
    )


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------


def _first_hit_rank(ious: Sequence[float], threshold: float) -> int:
    """Return the index of the first IoU >= ``threshold``, or ``-1``."""
    for idx, iou in enumerate(ious):
        if iou >= threshold:
            return idx
    return -1


def aggregate_overall(per_utterance: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-utterance results into the canonical overall metrics block.

    Each record must contain ``top1_iou`` and ``per_candidate_ious`` (list of
    floats, ranked). Returns the standard schema used by every method/benchmark
    metrics JSON.

    Includes:
      - ``acc@1@mask_iou={0.1,0.25,0.5}`` and ``acc@1@any_overlap``
      - ``recall@{1,3,5,10}@mask_iou={0.1,0.25,0.5}`` and ``recall@K@any_overlap``
      - ``mrr@mask_iou={0.1,0.25,0.5}`` and ``mrr@any_overlap``
      - ``median_rank@mask_iou={0.1,0.25,0.5}`` (median 1-indexed rank of first hit; 0 = no hit in any of the scored candidates)
      - ``mean_top1_iou``, ``mean_pred_pixels``, ``mean_gt_pixels``
    """
    n = len(per_utterance)
    if n == 0:
        return {"n": 0, "n_total": 0}
    out: Dict[str, Any] = {"n": n, "n_total": int(per_utterance[0].get("n_total") or n)}
    sum_iou = 0.0
    sum_pred_px = 0
    sum_gt_px = 0
    mrr_at_thr: Dict[str, float] = {f"mrr@mask_iou={t}": 0.0 for t in DEFAULT_IOU_THRESHOLDS}
    mrr_at_thr["mrr@any_overlap"] = 0.0
    ranks_at_thr: Dict[str, List[int]] = {f"median_rank@mask_iou={t}": [] for t in DEFAULT_IOU_THRESHOLDS}
    for thr in DEFAULT_IOU_THRESHOLDS:
        out[f"acc@1@mask_iou={thr}"] = 0
        for k in DEFAULT_RECALL_KS:
            out[f"recall@{k}@mask_iou={thr}"] = 0
    for k in DEFAULT_RECALL_KS:
        out[f"recall@{k}@any_overlap"] = 0
    out["acc@1@any_overlap"] = 0
    for rec in per_utterance:
        top1 = float(rec.get("top1_iou", 0.0))
        sum_iou += top1
        sum_pred_px += int(rec.get("top1_pred_pixels", 0) or 0)
        sum_gt_px += int(rec.get("top1_gt_pixels", 0) or 0)
        ious = list(rec.get("per_candidate_ious") or [])
        for thr in DEFAULT_IOU_THRESHOLDS:
            if top1 >= thr:
                out[f"acc@1@mask_iou={thr}"] += 1
            first_hit = _first_hit_rank(ious, thr)
            ranks_at_thr[f"median_rank@mask_iou={thr}"].append(first_hit + 1 if first_hit >= 0 else 0)
            if first_hit >= 0:
                mrr_at_thr[f"mrr@mask_iou={thr}"] += 1.0 / float(first_hit + 1)
            for k in DEFAULT_RECALL_KS:
                if 0 <= first_hit < k:
                    out[f"recall@{k}@mask_iou={thr}"] += 1
        first_any = _first_hit_rank(ious, 1e-9)
        if first_any >= 0:
            mrr_at_thr["mrr@any_overlap"] += 1.0 / float(first_any + 1)
        if top1 > 0:
            out["acc@1@any_overlap"] += 1
        for k in DEFAULT_RECALL_KS:
            if 0 <= first_any < k:
                out[f"recall@{k}@any_overlap"] += 1
    for key in list(out.keys()):
        if key in ("n", "n_total"):
            continue
        out[key] = float(out[key]) / float(n)
    for key, accumulator in mrr_at_thr.items():
        out[key] = float(accumulator) / float(n)
    for key, ranks in ranks_at_thr.items():
        # Median 1-indexed rank of the first hit. 0 means no hit.
        ranks_sorted = sorted(ranks)
        if ranks_sorted:
            mid = ranks_sorted[len(ranks_sorted) // 2]
            out[key] = int(mid)
        else:
            out[key] = 0
    out["mean_top1_iou"] = sum_iou / float(n)
    out["mean_pred_pixels"] = sum_pred_px / float(n)
    out["mean_gt_pixels"] = sum_gt_px / float(n)
    out["primary_metric"] = f"acc@1@mask_iou={HEADLINE_IOU}"
    return out
