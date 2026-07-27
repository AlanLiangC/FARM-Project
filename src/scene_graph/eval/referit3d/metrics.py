"""ReferIt3D metrics over a predictions JSON.

The default scorer preserves the original protocol: ranked predicted objects
are matched to ScanNet GT instances by 3D IoU on axis-aligned bounding boxes.
When ``match_mode='visible_mask'``, candidates are instead matched by
occlusion-aware projected visible-mask agreement using the predicted object's
persisted voxel support and associated RGB-D views.

Headline numbers for bbox mode:

- ``acc_at_1@iou=0.25``, ``acc_at_1@iou=0.5`` — top-1 IoU vs. GT target ≥ thr
- ``recall_at_k@iou=0.25`` for K ∈ {1, 3, 5, 10} — any of top-K hits the target
- ``mrr@iou=0.25`` — reciprocal rank of first hit
- ``median_rank@iou=0.25`` — median rank of first hit (∞ → ranked-list-length+1)

For ``match_mode='visible_mask'`` the primary protocol is any positive mask
overlap: ``acc@1@any_overlap`` with Recall@K/MRR at the same threshold.
Stricter mask thresholds ``0.1``, ``0.25``, and ``0.5`` are still reported as
secondary diagnostics.

Plus per-class / per-reference-type / easy-vs-hard / NR3D language-flag
breakdowns. All breakdowns use the same metric set; the result is a nested
dict keyed by (split_kind, split_value, metric_name).
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .matching import iou_3d_aabb
from .scannet_gt import GTInstance, load_scene_gt, load_scene_gt_points
from scene_graph.eval.visible_mask import (
    SceneStateMaskIndex,
    VisibleMaskMatch,
    discover_scene_state_paths,
    first_hit_rank_from_scores,
)
from scene_graph.eval.view_selection import resolve_chosen_view_image_id as _resolve_chosen_view_image_id

LOGGER = logging.getLogger("scene_graph.eval.referit3d.metrics")

ANY_OVERLAP_THRESHOLD: float = 1e-9
DEFAULT_IOU_THRESHOLDS: Tuple[float, ...] = (0.25, 0.5)
DEFAULT_VISIBLE_MASK_IOU_THRESHOLDS: Tuple[float, ...] = (
    ANY_OVERLAP_THRESHOLD,
    0.1,
    0.25,
    0.5,
)
DEFAULT_RECALL_KS: Tuple[int, ...] = (1, 3, 5, 10)
PRIMARY_THRESHOLD: float = 0.25  # used for MRR / median rank / per-split aggregates
VISIBLE_MASK_PRIMARY_THRESHOLD: float = ANY_OVERLAP_THRESHOLD
DEFAULT_MATCH_MODE: str = "bbox"


# ---------------------------------------------------------------------
# Per-utterance scoring
# ---------------------------------------------------------------------


@dataclass
class UtteranceScore:
    """Per-utterance evaluation outcome (filled even on errors)."""

    uid: str
    scan_id: str
    dataset: str
    instance_type: str
    target_id: int
    n_distractors: int
    top1_iou: float                  # IoU(top-1 predicted bbox, GT target bbox); 0 if no preds
    first_hit_rank_at: Dict[float, Optional[int]]  # threshold → 1-based rank of first hit, or None
    n_predictions: int
    error: Optional[str] = None
    score_kind: str = "iou"

    # Side info preserved for breakdowns:
    reference_type: Optional[str] = None
    coarse_reference_type: Optional[str] = None
    mentions_target_class: Optional[bool] = None
    uses_object_lang: Optional[bool] = None
    uses_spatial_lang: Optional[bool] = None
    uses_color_lang: Optional[bool] = None
    uses_shape_lang: Optional[bool] = None
    top1_region_label: Optional[str] = None
    top1_precision: Optional[float] = None
    top1_recall: Optional[float] = None
    top1_n_valid_views: Optional[int] = None
    top1_best_image_id: Optional[int] = None
    top1_best_iou: Optional[float] = None
    top1_mean_topk_iou: Optional[float] = None
    top1_weighted_iou: Optional[float] = None

    @property
    def difficulty(self) -> str:
        return "hard" if self.n_distractors >= 3 else "easy"

    def hit_at_k(self, k: int, threshold: float) -> bool:
        rank = self.first_hit_rank_at.get(float(threshold))
        return rank is not None and 1 <= rank <= k

    def reciprocal_rank(self, threshold: float = PRIMARY_THRESHOLD) -> float:
        rank = self.first_hit_rank_at.get(float(threshold))
        if rank is None or rank < 1:
            return 0.0
        return 1.0 / float(rank)


def _ranked_to_aabb(ranked: Sequence[Dict[str, Any]]) -> List[Tuple[np.ndarray, np.ndarray]]:
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for entry in ranked:
        bb_min = np.asarray(entry.get("bbox_min", [0.0, 0.0, 0.0]), dtype=np.float32)
        bb_max = np.asarray(entry.get("bbox_max", [0.0, 0.0, 0.0]), dtype=np.float32)
        if bb_min.shape != (3,) or bb_max.shape != (3,):
            continue
        out.append((bb_min, bb_max))
    return out


def score_utterance(
    record: Dict[str, Any],
    gt: Dict[int, GTInstance],
    *,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
) -> UtteranceScore:
    """Score one utterance record (from the runner's predictions JSON)."""
    target_id = int(record.get("target_id", -1))
    distractors = list(record.get("distractor_ids") or [])

    target = gt.get(target_id)
    ranked_dicts = list(record.get("ranked") or [])
    ranked_aabb = _ranked_to_aabb(ranked_dicts)

    base = UtteranceScore(
        uid=str(record.get("uid", "")),
        scan_id=str(record.get("scan_id", "")),
        dataset=str(record.get("dataset", "")),
        instance_type=str(record.get("instance_type", "")),
        target_id=target_id,
        n_distractors=len(distractors),
        top1_iou=0.0,
        first_hit_rank_at={float(t): None for t in iou_thresholds},
        n_predictions=len(ranked_dicts),
        error=record.get("error"),
        reference_type=record.get("reference_type"),
        coarse_reference_type=record.get("coarse_reference_type"),
        mentions_target_class=record.get("mentions_target_class"),
        uses_object_lang=record.get("uses_object_lang"),
        uses_spatial_lang=record.get("uses_spatial_lang"),
        uses_color_lang=record.get("uses_color_lang"),
        uses_shape_lang=record.get("uses_shape_lang"),
    )

    if ranked_dicts:
        base.top1_region_label = ranked_dicts[0].get("region_label") if ranked_dicts[0] else None

    if target is None:
        base.error = base.error or f"no GT instance {target_id} in scene {base.scan_id}"
        return base
    if not ranked_aabb:
        return base

    # Compute top-1 IoU and first-hit rank per threshold.
    ious = [
        iou_3d_aabb(bmin, bmax, target.bbox_min, target.bbox_max) for bmin, bmax in ranked_aabb
    ]
    base.top1_iou = float(ious[0])
    for thr in iou_thresholds:
        thr_f = float(thr)
        base.first_hit_rank_at[thr_f] = next(
            (i + 1 for i, iou in enumerate(ious) if iou >= thr_f),
            None,
        )
    return base


def _empty_visible_mask_score(
    record: Dict[str, Any],
    *,
    iou_thresholds: Sequence[float],
    error: Optional[str] = None,
) -> UtteranceScore:
    ranked_dicts = list(record.get("ranked") or [])
    base = UtteranceScore(
        uid=str(record.get("uid", "")),
        scan_id=str(record.get("scan_id", "")),
        dataset=str(record.get("dataset", "")),
        instance_type=str(record.get("instance_type", "")),
        target_id=int(record.get("target_id", -1)),
        n_distractors=len(list(record.get("distractor_ids") or [])),
        top1_iou=0.0,
        first_hit_rank_at={float(t): None for t in iou_thresholds},
        n_predictions=len(ranked_dicts),
        error=error if error is not None else record.get("error"),
        score_kind="mask_iou",
        reference_type=record.get("reference_type"),
        coarse_reference_type=record.get("coarse_reference_type"),
        mentions_target_class=record.get("mentions_target_class"),
        uses_object_lang=record.get("uses_object_lang"),
        uses_spatial_lang=record.get("uses_spatial_lang"),
        uses_color_lang=record.get("uses_color_lang"),
        uses_shape_lang=record.get("uses_shape_lang"),
    )
    if ranked_dicts:
        base.top1_region_label = ranked_dicts[0].get("region_label") if ranked_dicts[0] else None
    return base


def score_utterance_visible_mask(
    record: Dict[str, Any],
    *,
    mask_index: SceneStateMaskIndex,
    gt_points: Dict[int, Any],
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
    depth_tolerance_m: float = 0.15,  # locked 2026-05-16: matches unified scorer / legacy protocol
    point_radius_px: int = 3,
    min_gt_pixels: int = 20,
    topk: int = 3,
    max_views: Optional[int] = 50,
    max_points: int = 50000,
    score_aggregation: str = "best_iou",
    require_depth: bool = True,
    view_picker_name: str = "v1_largest_mask",
) -> UtteranceScore:
    """Score one utterance with view-level projected visible-mask agreement."""

    target_id = int(record.get("target_id", -1))
    target_points = gt_points.get(target_id)
    ranked_dicts = list(record.get("ranked") or [])
    base = _empty_visible_mask_score(record, iou_thresholds=iou_thresholds)
    if target_points is None:
        base.error = base.error or f"no GT points for instance {target_id} in scene {base.scan_id}"
        return base
    if not ranked_dicts:
        return base

    target_pts = getattr(target_points, "points", target_points)
    matches: List[VisibleMaskMatch] = []
    scores: List[float] = []
    for candidate in ranked_dicts:
        chosen_view_image_id = _resolve_chosen_view_image_id(
            mask_index, candidate, picker_name=view_picker_name
        )
        match = mask_index.score_candidate(
            candidate,
            target_id,
            target_pts,
            depth_tolerance_m=depth_tolerance_m,
            point_radius_px=point_radius_px,
            min_gt_pixels=min_gt_pixels,
            topk=topk,
            max_views=max_views,
            max_points=max_points,
            require_depth=require_depth,
            chosen_view_image_id=chosen_view_image_id,
        )
        matches.append(match)
        scores.append(match.score(score_aggregation))

    if not matches:
        return base

    top1 = matches[0]
    base.top1_iou = float(scores[0]) if scores else 0.0
    base.top1_precision = float(top1.best_precision)
    base.top1_recall = float(top1.best_recall)
    base.top1_n_valid_views = int(top1.n_valid_views)
    base.top1_best_image_id = top1.best_image_id
    base.top1_best_iou = float(top1.best_iou)
    base.top1_mean_topk_iou = float(top1.mean_topk_iou)
    base.top1_weighted_iou = float(top1.weighted_iou)
    for thr in iou_thresholds:
        base.first_hit_rank_at[float(thr)] = first_hit_rank_from_scores(scores, float(thr))
    return base


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------


def _safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom) if denom > 0 else 0.0


def _score_kind_for_summary(scores: Sequence[UtteranceScore]) -> str:
    for score in scores:
        kind = str(getattr(score, "score_kind", "iou") or "iou")
        if kind:
            return kind
    return "iou"


def _mean_top1_key(score_kind: str) -> str:
    return "mean_top1_iou" if score_kind == "iou" else f"mean_top1_{score_kind}"


def _is_any_overlap_threshold(threshold: float) -> bool:
    return math.isclose(float(threshold), ANY_OVERLAP_THRESHOLD, rel_tol=0.0, abs_tol=1e-12)


def _threshold_display(threshold: float) -> str:
    return "any_overlap" if _is_any_overlap_threshold(threshold) else f"{float(threshold):g}"


def _threshold_metric_name(prefix: str, score_kind: str, threshold: float) -> str:
    if score_kind != "iou" and _is_any_overlap_threshold(threshold):
        return f"{prefix}@any_overlap"
    metric = "iou" if score_kind == "iou" else score_kind
    return f"{prefix}@{metric}={float(threshold):g}"


def _resolved_thresholds(
    mode: str,
    iou_thresholds: Optional[Sequence[float]],
) -> Tuple[Tuple[float, ...], float]:
    if iou_thresholds is None:
        if str(mode).strip().lower() == "visible_mask":
            return DEFAULT_VISIBLE_MASK_IOU_THRESHOLDS, VISIBLE_MASK_PRIMARY_THRESHOLD
        return DEFAULT_IOU_THRESHOLDS, PRIMARY_THRESHOLD
    thresholds = tuple(float(t) for t in iou_thresholds)
    primary = VISIBLE_MASK_PRIMARY_THRESHOLD if str(mode).strip().lower() == "visible_mask" else PRIMARY_THRESHOLD
    return thresholds, primary


def _summary_for_scores(
    scores: Sequence[UtteranceScore],
    *,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
    recall_ks: Sequence[int] = DEFAULT_RECALL_KS,
    primary_threshold: float = PRIMARY_THRESHOLD,
) -> Dict[str, Any]:
    """Compute the metric set over a homogeneous slice of scores."""
    scored = [s for s in scores if s.error is None]
    n = len(scored)
    n_total = len(scores)
    score_kind = _score_kind_for_summary(scored or scores)

    out: Dict[str, Any] = {
        "n": n,
        "n_total": n_total,
        "n_dropped": n_total - n,
        "score_kind": score_kind,
        _mean_top1_key(score_kind): _safe_div(sum(s.top1_iou for s in scored), n),
    }
    if score_kind != "iou":
        out["mean_top1_precision"] = _safe_div(
            sum(float(s.top1_precision or 0.0) for s in scored), n
        )
        out["mean_top1_recall"] = _safe_div(
            sum(float(s.top1_recall or 0.0) for s in scored), n
        )
        out["mean_top1_valid_views"] = _safe_div(
            sum(float(s.top1_n_valid_views or 0) for s in scored), n
        )
        out["mean_top1_best_view_iou"] = _safe_div(
            sum(float(s.top1_best_iou or 0.0) for s in scored), n
        )
        out["mean_top1_mean_topk_iou"] = _safe_div(
            sum(float(s.top1_mean_topk_iou or 0.0) for s in scored), n
        )
        out["mean_top1_weighted_iou"] = _safe_div(
            sum(float(s.top1_weighted_iou or 0.0) for s in scored), n
        )

    for thr in iou_thresholds:
        thr_f = float(thr)
        n_acc = sum(1 for s in scored if s.first_hit_rank_at.get(thr_f) == 1)
        out[_threshold_metric_name("acc@1", score_kind, thr_f)] = _safe_div(n_acc, n)

    primary = float(primary_threshold)
    for k in recall_ks:
        n_hit = sum(1 for s in scored if s.hit_at_k(int(k), primary))
        out[_threshold_metric_name(f"recall@{int(k)}", score_kind, primary)] = _safe_div(n_hit, n)

    out[_threshold_metric_name("mrr", score_kind, primary)] = _safe_div(
        sum(s.reciprocal_rank(primary) for s in scored), n
    )
    ranks = [s.first_hit_rank_at.get(primary) for s in scored]
    finite_ranks = [r for r in ranks if r is not None]
    out[_threshold_metric_name("median_rank", score_kind, primary)] = (
        float(np.median(finite_ranks)) if finite_ranks else math.inf
    )
    out[_threshold_metric_name("hit_rate@any_rank", score_kind, primary)] = _safe_div(
        len(finite_ranks), n
    )

    return out


# ---------------------------------------------------------------------
# Breakdown helpers — closures returning (key, sortable_label) per score
# ---------------------------------------------------------------------


def _bool_flag(value: Optional[bool]) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"


def _per_class_key(s: UtteranceScore) -> str:
    return s.instance_type or "<unknown>"


def _per_dataset_key(s: UtteranceScore) -> str:
    return s.dataset or "<unknown>"


def _per_difficulty_key(s: UtteranceScore) -> str:
    return s.difficulty


def _per_sr3d_ref_key(s: UtteranceScore) -> Optional[str]:
    if s.dataset != "sr3d":
        return None
    return s.reference_type or "<none>"


def _per_sr3d_coarse_key(s: UtteranceScore) -> Optional[str]:
    if s.dataset != "sr3d":
        return None
    return s.coarse_reference_type or "<none>"


def _per_nr3d_object_lang_key(s: UtteranceScore) -> Optional[str]:
    return f"uses_object_lang={_bool_flag(s.uses_object_lang)}" if s.dataset == "nr3d" else None


def _per_nr3d_spatial_lang_key(s: UtteranceScore) -> Optional[str]:
    return f"uses_spatial_lang={_bool_flag(s.uses_spatial_lang)}" if s.dataset == "nr3d" else None


def _per_nr3d_color_lang_key(s: UtteranceScore) -> Optional[str]:
    return f"uses_color_lang={_bool_flag(s.uses_color_lang)}" if s.dataset == "nr3d" else None


def _per_nr3d_shape_lang_key(s: UtteranceScore) -> Optional[str]:
    return f"uses_shape_lang={_bool_flag(s.uses_shape_lang)}" if s.dataset == "nr3d" else None


def _per_region_key(s: UtteranceScore) -> Optional[str]:
    return s.top1_region_label if s.top1_region_label else None


_DEFAULT_BREAKDOWNS: List[Tuple[str, Callable[[UtteranceScore], Optional[str]]]] = [
    ("dataset", _per_dataset_key),
    ("difficulty", _per_difficulty_key),
    ("sr3d_reference_type", _per_sr3d_ref_key),
    ("sr3d_coarse_reference_type", _per_sr3d_coarse_key),
    ("nr3d_uses_object_lang", _per_nr3d_object_lang_key),
    ("nr3d_uses_spatial_lang", _per_nr3d_spatial_lang_key),
    ("nr3d_uses_color_lang", _per_nr3d_color_lang_key),
    ("nr3d_uses_shape_lang", _per_nr3d_shape_lang_key),
    ("region", _per_region_key),
    # per-class is large; keep it but expect a long list
    ("instance_type", _per_class_key),
]


def aggregate(
    scores: Sequence[UtteranceScore],
    *,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
    recall_ks: Sequence[int] = DEFAULT_RECALL_KS,
    primary_threshold: float = PRIMARY_THRESHOLD,
    breakdowns: Optional[Sequence[Tuple[str, Callable[[UtteranceScore], Optional[str]]]]] = None,
    min_per_bucket: int = 5,
) -> Dict[str, Any]:
    """Compute the overall metric set + per-split breakdowns."""
    if breakdowns is None:
        breakdowns = _DEFAULT_BREAKDOWNS

    overall = _summary_for_scores(
        scores,
        iou_thresholds=iou_thresholds,
        recall_ks=recall_ks,
        primary_threshold=primary_threshold,
    )

    breakdowns_out: Dict[str, Dict[str, Any]] = {}
    for split_name, key_fn in breakdowns:
        buckets: Dict[str, List[UtteranceScore]] = defaultdict(list)
        for s in scores:
            try:
                k = key_fn(s)
            except Exception:  # noqa: BLE001
                k = None
            if k is None:
                continue
            buckets[k].append(s)
        per_split: Dict[str, Any] = {}
        for k, bucket in sorted(buckets.items()):
            if len(bucket) < min_per_bucket:
                continue
            per_split[k] = _summary_for_scores(
                bucket,
                iou_thresholds=iou_thresholds,
                recall_ks=recall_ks,
                primary_threshold=primary_threshold,
            )
        breakdowns_out[split_name] = per_split

    return {
        "overall": overall,
        "breakdowns": breakdowns_out,
    }


# ---------------------------------------------------------------------
# End-to-end: predictions JSON + scans dir → metrics dict
# ---------------------------------------------------------------------


def score_predictions(
    predictions_path: Path,
    *,
    scans_dir: Optional[Path] = None,
    match_mode: str = DEFAULT_MATCH_MODE,
    scene_state_dir: Optional[Path] = None,
    mask_depth_tolerance_m: float = 0.15,  # locked 2026-05-16: matches unified scorer / legacy protocol
    mask_point_radius_px: int = 3,
    mask_min_gt_pixels: int = 20,
    mask_topk: int = 3,
    mask_max_views: Optional[int] = 50,
    mask_max_points: int = 50000,
    mask_score_aggregation: str = "best_iou",
    mask_require_depth: bool = True,
    iou_thresholds: Optional[Sequence[float]] = None,
    recall_ks: Sequence[int] = DEFAULT_RECALL_KS,
    view_picker_name: str = "v1_largest_mask",
) -> Tuple[List[UtteranceScore], Dict[str, Any]]:
    """Load predictions, fetch GT for each scene, compute per-utterance scores
    and aggregate metrics. Returns ``(scores, aggregate_dict)``.
    """
    predictions = json.loads(Path(predictions_path).read_text(encoding="utf-8")) or []
    by_scene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in predictions:
        scan = str(r.get("scan_id", ""))
        if not scan:
            continue
        by_scene[scan].append(r)

    mode = str(match_mode or DEFAULT_MATCH_MODE).strip().lower()
    if mode not in {"bbox", "visible_mask"}:
        raise ValueError(f"unknown match_mode: {match_mode!r}")
    resolved_iou_thresholds, primary_threshold = _resolved_thresholds(mode, iou_thresholds)

    scene_state_paths: Dict[str, Path] = {}
    if mode == "visible_mask":
        if scene_state_dir is None:
            raise ValueError("scene_state_dir is required for match_mode='visible_mask'")
        scene_state_paths = discover_scene_state_paths(Path(scene_state_dir))

    LOGGER.info(
        "scoring %d predictions across %d scenes (predictions=%s, match_mode=%s)",
        len(predictions), len(by_scene), predictions_path, mode,
    )

    scores: List[UtteranceScore] = []
    for scan, records in sorted(by_scene.items()):
        try:
            gt = load_scene_gt(scan, scans_dir=scans_dir)
            gt_points = load_scene_gt_points(scan, scans_dir=scans_dir) if mode == "visible_mask" else {}
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("could not load GT for %s: %s", scan, exc)
            for r in records:
                if mode == "visible_mask":
                    err_score = _empty_visible_mask_score(
                        r,
                        iou_thresholds=resolved_iou_thresholds,
                        error=f"GT load failed: {exc}",
                    )
                else:
                    err_score = score_utterance(
                        {**r, "error": f"GT load failed: {exc}"},
                        gt={},
                        iou_thresholds=resolved_iou_thresholds,
                    )
                scores.append(err_score)
            continue

        if mode == "bbox":
            for r in records:
                scores.append(score_utterance(r, gt=gt, iou_thresholds=resolved_iou_thresholds))
            continue

        scene_state_path = scene_state_paths.get(scan)
        if scene_state_path is None:
            msg = f"scene_state not found for {scan} under {scene_state_dir}"
            LOGGER.error(msg)
            for r in records:
                scores.append(_empty_visible_mask_score(r, iou_thresholds=resolved_iou_thresholds, error=msg))
            continue

        mask_index: Optional[SceneStateMaskIndex] = None
        try:
            mask_index = SceneStateMaskIndex.from_path(scene_state_path)
            for r in records:
                # Preserve the existing GT-id error behavior before doing mask work.
                if int(r.get("target_id", -1)) not in gt:
                    scores.append(
                        _empty_visible_mask_score(
                            r,
                            iou_thresholds=resolved_iou_thresholds,
                            error=f"no GT instance {int(r.get('target_id', -1))} in scene {scan}",
                        )
                    )
                    continue
                scores.append(
                    score_utterance_visible_mask(
                        r,
                        mask_index=mask_index,
                        gt_points=gt_points,
                        iou_thresholds=resolved_iou_thresholds,
                        depth_tolerance_m=mask_depth_tolerance_m,
                        point_radius_px=mask_point_radius_px,
                        min_gt_pixels=mask_min_gt_pixels,
                        topk=mask_topk,
                        max_views=mask_max_views,
                        max_points=mask_max_points,
                        score_aggregation=mask_score_aggregation,
                        require_depth=mask_require_depth,
                        view_picker_name=view_picker_name,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("visible-mask scoring failed for %s: %s", scan, exc)
            for r in records:
                scores.append(
                    _empty_visible_mask_score(
                        r,
                        iou_thresholds=resolved_iou_thresholds,
                        error=f"visible-mask scoring failed: {exc}",
                    )
                )
        finally:
            if mask_index is not None:
                mask_index.close()

    aggregate_dict = aggregate(
        scores,
        iou_thresholds=resolved_iou_thresholds,
        recall_ks=recall_ks,
        primary_threshold=primary_threshold,
    )
    aggregate_dict["match_mode"] = mode
    aggregate_dict["primary_threshold"] = float(primary_threshold)
    aggregate_dict["primary_metric"] = _threshold_display(primary_threshold)
    aggregate_dict["iou_thresholds"] = [float(t) for t in resolved_iou_thresholds]
    if mode == "visible_mask":
        aggregate_dict["visible_mask_params"] = {
            "scene_state_dir": str(scene_state_dir),
            "depth_tolerance_m": float(mask_depth_tolerance_m),
            "point_radius_px": int(mask_point_radius_px),
            "min_gt_pixels": int(mask_min_gt_pixels),
            "topk": int(mask_topk),
            "max_views": None if mask_max_views is None else int(mask_max_views),
            "max_points": int(mask_max_points),
            "score_aggregation": str(mask_score_aggregation),
            "require_depth": bool(mask_require_depth),
            "primary_metric": "acc@1@any_overlap",
            "view_picker": str(view_picker_name),
        }
    return scores, aggregate_dict


def write_metrics_json(
    aggregate_dict: Dict[str, Any],
    *,
    output_path: Path,
) -> None:
    """Persist an aggregate dict to ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregate_dict, indent=2, ensure_ascii=False), encoding="utf-8")


def format_overall_table(aggregate_dict: Dict[str, Any]) -> str:
    """Pretty-print the overall metric block + dataset breakdown."""
    lines: List[str] = []
    overall = aggregate_dict.get("overall") or {}
    lines.append(f"# Overall (n={overall.get('n', 0)} / total={overall.get('n_total', 0)})")
    for k, v in overall.items():
        if isinstance(v, float) and not math.isfinite(v):
            v_disp = "inf"
        elif isinstance(v, float):
            v_disp = f"{v:.4f}"
        else:
            v_disp = str(v)
        lines.append(f"  {k:<32s} {v_disp}")

    by_dataset = (aggregate_dict.get("breakdowns") or {}).get("dataset") or {}
    if by_dataset:
        score_kind = str(overall.get("score_kind") or "iou")
        mean_key = _mean_top1_key(score_kind)
        primary_threshold = float(aggregate_dict.get("primary_threshold", PRIMARY_THRESHOLD))
        primary_label = _threshold_display(primary_threshold)
        acc_primary_key = _threshold_metric_name("acc@1", score_kind, primary_threshold)
        acc50_key = _threshold_metric_name("acc@1", score_kind, 0.5)
        r5_primary_key = _threshold_metric_name("recall@5", score_kind, primary_threshold)
        mrr_primary_key = _threshold_metric_name("mrr", score_kind, primary_threshold)
        med_primary_key = _threshold_metric_name("median_rank", score_kind, primary_threshold)
        lines.append("")
        lines.append("# By dataset")
        for ds, m in by_dataset.items():
            lines.append(
                f"  {ds:<6s} n={m.get('n', 0):>5d} "
                f"mean_top1={m.get(mean_key, 0.0):.3f} "
                f"acc@1@{primary_label}={m.get(acc_primary_key, 0.0):.3f} "
                f"acc@1@0.5={m.get(acc50_key, 0.0):.3f} "
                f"recall@5@{primary_label}={m.get(r5_primary_key, 0.0):.3f} "
                f"mrr@{primary_label}={m.get(mrr_primary_key, 0.0):.3f} "
                f"median_rank@{primary_label}={m.get(med_primary_key, float('inf')):.1f}"
            )
    return "\n".join(lines)
