"""IRef-VLA metrics over a predictions JSON.

Evaluates ranked predicted objects against IRef-VLA GT instances by 3D IoU on
axis-aligned bounding boxes. The predicted bboxes come from the runner's
``ranked`` list (already AABBs); GT bboxes come from
:func:`scene_graph.eval.iref_vla.iref_vla_gt.load_scene_objects`.

Headline numbers for bbox mode (matching what the IRef-VLA paper reports):

- ``acc_at_1@iou=0.25``, ``acc_at_1@iou=0.5`` — top-1 IoU vs. GT target ≥ thr
- ``recall_at_k@iou=0.25`` for K ∈ {1, 3, 5, 10}
- ``mrr@iou=0.25`` — reciprocal rank of first hit
- ``median_rank@iou=0.25``

For ``match_mode='visible_mask'`` the primary protocol is any positive mask
overlap: ``acc@1@any_overlap`` with Recall@K/MRR at the same threshold.
Stricter mask thresholds ``0.1``, ``0.25``, and ``0.5`` are still reported as
secondary diagnostics.

Plus per-relation / per-region-class / per-difficulty / multi-room
breakdowns. All breakdowns share the metric set; the result is a nested dict
keyed by ``(split_kind, split_value, metric_name)``.
"""

from __future__ import annotations

import json
import logging
import math
import contextlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from scene_graph.eval.referit3d.matching import iou_3d_aabb
from scene_graph.eval.visible_mask import (
    GTMeshMaskProvider,
    SceneStateMaskIndex,
    VisibleMaskMatch,
    discover_scene_state_paths,
    first_hit_rank_from_scores,
    save_mask_debug_image,
    sample_aabb_surface_points,
)
from scene_graph.eval.view_selection import resolve_chosen_view_image_id as _resolve_chosen_view_image_id

from .iref_vla_gt import GTInstance, RegionInfo, load_scene_objects, load_scene_regions

LOGGER = logging.getLogger("scene_graph.eval.iref_vla.metrics")

ANY_OVERLAP_THRESHOLD: float = 1e-9
DEFAULT_IOU_THRESHOLDS: Tuple[float, ...] = (0.25, 0.5)
DEFAULT_VISIBLE_MASK_IOU_THRESHOLDS: Tuple[float, ...] = (
    ANY_OVERLAP_THRESHOLD,
    0.1,
    0.25,
    0.5,
)
DEFAULT_RECALL_KS: Tuple[int, ...] = (1, 3, 5, 10)
PRIMARY_THRESHOLD: float = 0.25
VISIBLE_MASK_PRIMARY_THRESHOLD: float = ANY_OVERLAP_THRESHOLD
DEFAULT_MATCH_MODE: str = "bbox"


# ---------------------------------------------------------------------
# Per-statement scoring
# ---------------------------------------------------------------------


@dataclass
class StatementScore:
    """Per-statement evaluation outcome (filled even on errors)."""

    uid: str
    scene_id: str
    region_id: int
    region_label: str
    target_id: int
    target_class: str
    relation: str
    relation_type: str
    n_distractors: int
    top1_iou: float
    first_hit_rank_at: Dict[float, Optional[int]]
    n_predictions: int
    error: Optional[str] = None
    is_false_statement: bool = False
    score_kind: str = "iou"
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


def score_statement(
    record: Dict[str, Any],
    gt: Dict[int, GTInstance],
    region_lookup: Dict[int, RegionInfo],
    *,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
) -> StatementScore:
    """Score one statement record (from the runner's predictions JSON)."""
    target_id = int(record.get("target_id", -1))
    region_id = int(record.get("region_id", -1))
    distractors = list(record.get("distractor_ids") or [])

    target = gt.get(target_id)
    region_label = ""
    region_info = region_lookup.get(region_id)
    if region_info is not None:
        region_label = region_info.label

    ranked_dicts = list(record.get("ranked") or [])
    ranked_aabb = _ranked_to_aabb(ranked_dicts)

    base = StatementScore(
        uid=str(record.get("uid", "")),
        scene_id=str(record.get("scene_id", "")),
        region_id=region_id,
        region_label=region_label,
        target_id=target_id,
        target_class=str(record.get("target_class", "")),
        relation=str(record.get("relation", "")),
        relation_type=str(record.get("relation_type", "")),
        n_distractors=len(distractors),
        top1_iou=0.0,
        first_hit_rank_at={float(t): None for t in iou_thresholds},
        n_predictions=len(ranked_dicts),
        error=record.get("error"),
        is_false_statement=bool(record.get("is_false_statement", False)),
    )

    if target is None:
        base.error = base.error or f"no GT instance {target_id} in scene {base.scene_id}"
        return base
    if not ranked_aabb:
        return base

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
    region_lookup: Dict[int, RegionInfo],
    *,
    iou_thresholds: Sequence[float],
    error: Optional[str] = None,
) -> StatementScore:
    target_id = int(record.get("target_id", -1))
    region_id = int(record.get("region_id", -1))
    region_info = region_lookup.get(region_id)
    return StatementScore(
        uid=str(record.get("uid", "")),
        scene_id=str(record.get("scene_id", "")),
        region_id=region_id,
        region_label=region_info.label if region_info is not None else "",
        target_id=target_id,
        target_class=str(record.get("target_class", "")),
        relation=str(record.get("relation", "")),
        relation_type=str(record.get("relation_type", "")),
        n_distractors=len(list(record.get("distractor_ids") or [])),
        top1_iou=0.0,
        first_hit_rank_at={float(t): None for t in iou_thresholds},
        n_predictions=len(list(record.get("ranked") or [])),
        error=error if error is not None else record.get("error"),
        is_false_statement=bool(record.get("is_false_statement", False)),
        score_kind="mask_iou",
    )


def _record_statement_text(record: Dict[str, Any]) -> str:
    for key in ("statement", "utterance", "query", "text"):
        value = record.get(key)
        if value:
            return str(value)
    return ""


def _safe_debug_slug(value: object, *, limit: int = 120) -> str:
    text = str(value or "")
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return safe[:limit] or "debug"


def _candidate_debug_payload(candidate: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidate:
        return {}
    out: Dict[str, Any] = {}
    for key in ("rank", "object_id", "score", "label", "caption", "region_label"):
        if key in candidate:
            out[key] = candidate.get(key)
    for key in ("bbox_min", "bbox_max"):
        if key in candidate:
            out[key] = candidate.get(key)
    return out


def _mask_debug_metadata_lines(
    record: Dict[str, Any],
    *,
    target: GTInstance,
    candidate: Optional[Dict[str, Any]],
    match: VisibleMaskMatch,
    top1_score: float,
    image_id: Optional[int],
    image_reason: str,
) -> List[str]:
    statement = _record_statement_text(record)
    pred_id = candidate.get("object_id") if candidate else None
    pred_score = candidate.get("score") if candidate else None
    pred_label = candidate.get("label") if candidate else ""
    pred_caption = candidate.get("caption") if candidate else ""
    return [
        f"utterance: {statement}",
        (
            f"target: id={target.instance_id} class={target.label} "
            f"region={target.region_id} relation={record.get('relation', '')}"
        ),
        f"predicted: id={pred_id} score={pred_score} label={pred_label}",
        f"predicted caption: {pred_caption}",
        (
            f"mask: score={top1_score:.4f} best_iou={float(match.best_iou):.4f} "
            f"precision={float(match.best_precision):.4f} recall={float(match.best_recall):.4f} "
            f"valid_views={int(match.n_valid_views)} image_id={image_id} reason={image_reason}"
        ),
    ]


def _write_mask_debug_record(debug_dir: Path, sidecar_path: Path, payload: Dict[str, Any]) -> None:
    def _json_default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    try:
        debug_dir = Path(debug_dir)
        sidecar_path = Path(sidecar_path)
        debug_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        with (debug_dir / "manifest.jsonl").open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("could not write mask debug sidecar %s: %s", sidecar_path, exc)


def score_statement_visible_mask(
    record: Dict[str, Any],
    *,
    mask_index: SceneStateMaskIndex,
    gt: Dict[int, GTInstance],
    region_lookup: Dict[int, RegionInfo],
    gt_mask_provider: Optional[GTMeshMaskProvider] = None,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
    depth_tolerance_m: float = 0.15,  # locked 2026-05-16: matches unified scorer / legacy protocol
    point_radius_px: int = 3,
    min_gt_pixels: int = 20,
    topk: int = 3,
    max_views: Optional[int] = 50,
    max_points: int = 50000,
    score_aggregation: str = "best_iou",
    require_depth: bool = True,
    gt_point_spacing_m: float = 0.03,
    pred_mask_kind: str = "raw",
    allow_pred_point_projection: bool = False,
    debug_dir: Optional[Path] = None,
    view_picker_name: str = "v1_largest_mask",
) -> StatementScore:
    """Score one IRef-VLA statement with image-space visible-mask agreement."""

    target_id = int(record.get("target_id", -1))
    target = gt.get(target_id)
    ranked_dicts = list(record.get("ranked") or [])
    base = _empty_visible_mask_score(record, region_lookup, iou_thresholds=iou_thresholds)
    if target is None:
        base.error = base.error or f"no GT instance {target_id} in scene {base.scene_id}"
        return base
    if not ranked_dicts:
        return base

    target_points = None
    if gt_mask_provider is None:
        target_points = sample_aabb_surface_points(
            target.bbox_min,
            target.bbox_max,
            spacing=gt_point_spacing_m,
            max_points=max_points,
        )
    matches: List[VisibleMaskMatch] = []
    scores: List[float] = []
    for candidate in ranked_dicts:
        chosen_view_image_id = _resolve_chosen_view_image_id(
            mask_index, candidate, picker_name=view_picker_name
        )
        match = mask_index.score_candidate(
            candidate,
            target_id,
            target_points,
            gt_instance=target,
            gt_mask_provider=gt_mask_provider,
            pred_mask_kind=pred_mask_kind,
            allow_pred_point_projection=allow_pred_point_projection,
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
    if debug_dir is not None and ranked_dicts:
        debug_candidate = ranked_dicts[0]
        debug_image_id = top1.best_image_id
        debug_image_reason = "best_valid_mask_iou_view"
        if debug_image_id is None:
            debug_image_reason = "first_predicted_observation_no_valid_gt_view"
            with_context = mask_index.object_mask_observation_records(
                mask_index.candidate_evidence_object_id(debug_candidate),
                max_views=1,
            )
            if with_context:
                with contextlib.suppress(Exception):
                    debug_image_id = int(with_context[0].get("image_id"))
        uid = str(record.get("uid") or record.get("statement_id") or f"target_{target_id}")
        safe_uid = _safe_debug_slug(uid)
        debug_root = Path(debug_dir)
        debug_record: Dict[str, Any] = {
            "uid": uid,
            "scene_id": base.scene_id,
            "statement": _record_statement_text(record),
            "target": {
                "object_id": int(target.instance_id),
                "label": str(target.label),
                "region_id": int(target.region_id),
                "target_class": str(record.get("target_class", "")),
            },
            "relation": str(record.get("relation", "")),
            "relation_type": str(record.get("relation_type", "")),
            "top1": _candidate_debug_payload(debug_candidate),
            "mask": {
                "score": float(scores[0]) if scores else 0.0,
                "score_aggregation": str(score_aggregation),
                "best_iou": float(top1.best_iou),
                "precision": float(top1.best_precision),
                "recall": float(top1.best_recall),
                "n_valid_views": int(top1.n_valid_views),
                "best_image_id": top1.best_image_id,
                "debug_image_id": debug_image_id,
                "debug_image_reason": debug_image_reason,
            },
            "debug_image_path": None,
        }
        sidecar_path = debug_root / f"{base.scene_id}_{safe_uid}_debug.json"
        payload = mask_index.masks_for_candidate_view(
            debug_candidate,
            target_id,
            int(debug_image_id) if debug_image_id is not None else -1,
            gt_instance=target,
            gt_mask_provider=gt_mask_provider,
            gt_points=target_points,
            pred_mask_kind=pred_mask_kind,
            depth_tolerance_m=depth_tolerance_m,
            point_radius_px=point_radius_px,
            max_points=max_points,
            require_depth=require_depth,
        ) if debug_image_id is not None else None
        if payload is not None:
            frame, pred_mask, gt_mask = payload
            frame = mask_index.frame_resolver.frame_with_rgb(frame)
            image_path = debug_root / f"{base.scene_id}_{safe_uid}_img{int(debug_image_id):06d}.png"
            debug_record["debug_image_path"] = str(image_path)
            sidecar_path = image_path.with_suffix(".json")
            save_mask_debug_image(
                image_path,
                frame,
                pred_mask,
                gt_mask,
                title=f"target={target_id} top1={top1.candidate_object_id} iou={float(top1.best_iou):.3f}",
                metadata_lines=_mask_debug_metadata_lines(
                    record,
                    target=target,
                    candidate=debug_candidate,
                    match=top1,
                    top1_score=float(scores[0]) if scores else 0.0,
                    image_id=int(debug_image_id),
                    image_reason=debug_image_reason,
                ),
            )
        _write_mask_debug_record(debug_root, sidecar_path, debug_record)
    for thr in iou_thresholds:
        base.first_hit_rank_at[float(thr)] = first_hit_rank_from_scores(scores, float(thr))
    return base


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------


def _safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom) if denom > 0 else 0.0


def _score_kind_for_summary(scores: Sequence[StatementScore]) -> str:
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
    scores: Sequence[StatementScore],
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
        out["mean_top1_precision"] = _safe_div(sum(float(s.top1_precision or 0.0) for s in scored), n)
        out["mean_top1_recall"] = _safe_div(sum(float(s.top1_recall or 0.0) for s in scored), n)
        out["mean_top1_valid_views"] = _safe_div(sum(float(s.top1_n_valid_views or 0) for s in scored), n)
        out["mean_top1_best_view_iou"] = _safe_div(sum(float(s.top1_best_iou or 0.0) for s in scored), n)
        out["mean_top1_mean_topk_iou"] = _safe_div(sum(float(s.top1_mean_topk_iou or 0.0) for s in scored), n)
        out["mean_top1_weighted_iou"] = _safe_div(sum(float(s.top1_weighted_iou or 0.0) for s in scored), n)

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
    out[_threshold_metric_name("hit_rate@any_rank", score_kind, primary)] = _safe_div(len(finite_ranks), n)

    return out


# ---------------------------------------------------------------------
# Breakdown helpers
# ---------------------------------------------------------------------


def _per_relation_key(s: StatementScore) -> str:
    return s.relation or "<none>"


def _per_relation_type_key(s: StatementScore) -> str:
    return s.relation_type or "<none>"


def _per_region_class_key(s: StatementScore) -> str:
    return s.region_label or "<unknown>"


def _per_difficulty_key(s: StatementScore) -> str:
    return s.difficulty


def _per_target_class_key(s: StatementScore) -> str:
    return s.target_class or "<unknown>"


def _per_scene_key(s: StatementScore) -> str:
    return s.scene_id or "<unknown>"


_DEFAULT_BREAKDOWNS: List[Tuple[str, Callable[[StatementScore], Optional[str]]]] = [
    ("relation", _per_relation_key),
    ("relation_type", _per_relation_type_key),
    ("region_class", _per_region_class_key),
    ("difficulty", _per_difficulty_key),
    ("target_class", _per_target_class_key),
    ("scene", _per_scene_key),
]


def aggregate(
    scores: Sequence[StatementScore],
    *,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
    recall_ks: Sequence[int] = DEFAULT_RECALL_KS,
    primary_threshold: float = PRIMARY_THRESHOLD,
    breakdowns: Optional[Sequence[Tuple[str, Callable[[StatementScore], Optional[str]]]]] = None,
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
        buckets: Dict[str, List[StatementScore]] = defaultdict(list)
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
# End-to-end
# ---------------------------------------------------------------------


def score_predictions(
    predictions_path: Path,
    *,
    dataset_root: Optional[Path] = None,
    match_mode: str = DEFAULT_MATCH_MODE,
    scene_state_dir: Optional[Path] = None,
    hm3d_root: Optional[Path] = None,
    mask_depth_tolerance_m: float = 0.15,  # locked 2026-05-16: matches unified scorer / legacy protocol
    mask_point_radius_px: int = 3,
    mask_min_gt_pixels: int = 20,
    mask_topk: int = 3,
    mask_max_views: Optional[int] = 50,
    mask_max_points: int = 50000,
    mask_score_aggregation: str = "best_iou",
    mask_require_depth: bool = True,
    mask_gt_point_spacing_m: float = 0.03,
    mask_gt_object_margin_m: float = 0.02,
    mask_pred_kind: str = "raw",
    mask_allow_pred_point_projection: bool = False,
    mask_debug_dir: Optional[Path] = None,
    iou_thresholds: Optional[Sequence[float]] = None,
    recall_ks: Sequence[int] = DEFAULT_RECALL_KS,
    view_picker_name: str = "v1_largest_mask",
) -> Tuple[List[StatementScore], Dict[str, Any]]:
    """Load predictions, fetch GT for each scene, compute per-statement scores
    and aggregate metrics. Returns ``(scores, aggregate_dict)``.
    """
    predictions = json.loads(Path(predictions_path).read_text(encoding="utf-8")) or []
    by_scene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in predictions:
        scan = str(r.get("scene_id", ""))
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
        if hm3d_root is None:
            raise ValueError("hm3d_root is required for match_mode='visible_mask' mesh GT projection")
        scene_state_paths = discover_scene_state_paths(Path(scene_state_dir))

    LOGGER.info(
        "scoring %d predictions across %d scenes (predictions=%s, match_mode=%s)",
        len(predictions), len(by_scene), predictions_path, mode,
    )

    scores: List[StatementScore] = []
    for scan, records in sorted(by_scene.items()):
        try:
            gt = load_scene_objects(scan, dataset_root=dataset_root)
            regions = load_scene_regions(scan, dataset_root=dataset_root)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("could not load GT for %s: %s", scan, exc)
            for r in records:
                if mode == "visible_mask":
                    err_score = _empty_visible_mask_score(
                        r,
                        {},
                        iou_thresholds=resolved_iou_thresholds,
                        error=f"GT load failed: {exc}",
                    )
                else:
                    err_score = score_statement(
                        {**r, "error": f"GT load failed: {exc}"},
                        gt={},
                        region_lookup={},
                        iou_thresholds=resolved_iou_thresholds,
                    )
                scores.append(err_score)
            continue
        if mode == "bbox":
            for r in records:
                scores.append(score_statement(r, gt=gt, region_lookup=regions, iou_thresholds=resolved_iou_thresholds))
            continue

        scene_state_path = scene_state_paths.get(scan)
        if scene_state_path is None:
            msg = f"scene_state not found for {scan} under {scene_state_dir}"
            LOGGER.error(msg)
            for r in records:
                scores.append(_empty_visible_mask_score(r, regions, iou_thresholds=resolved_iou_thresholds, error=msg))
            continue

        mask_index: Optional[SceneStateMaskIndex] = None
        gt_mask_provider: Optional[GTMeshMaskProvider] = None
        try:
            mask_index = SceneStateMaskIndex.from_path(scene_state_path)
            gt_mask_provider = GTMeshMaskProvider.from_hm3d_root(
                scan,
                Path(hm3d_root),
                object_margin_m=mask_gt_object_margin_m,
            )
            scene_debug_dir = Path(mask_debug_dir) / scan if mask_debug_dir is not None else None
            if scene_debug_dir is not None:
                scene_debug_dir.mkdir(parents=True, exist_ok=True)
                (scene_debug_dir / "manifest.jsonl").write_text("", encoding="utf-8")
            for r in records:
                scores.append(
                    score_statement_visible_mask(
                        r,
                        mask_index=mask_index,
                        gt=gt,
                        region_lookup=regions,
                        gt_mask_provider=gt_mask_provider,
                        iou_thresholds=resolved_iou_thresholds,
                        depth_tolerance_m=mask_depth_tolerance_m,
                        point_radius_px=mask_point_radius_px,
                        min_gt_pixels=mask_min_gt_pixels,
                        topk=mask_topk,
                        max_views=mask_max_views,
                        max_points=mask_max_points,
                        score_aggregation=mask_score_aggregation,
                        require_depth=mask_require_depth,
                        gt_point_spacing_m=mask_gt_point_spacing_m,
                        pred_mask_kind=mask_pred_kind,
                        allow_pred_point_projection=mask_allow_pred_point_projection,
                        debug_dir=scene_debug_dir,
                        view_picker_name=view_picker_name,
                    )
                )
                mask_index.clear_visible_mask_cache()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("visible-mask scoring failed for %s: %s", scan, exc)
            for r in records:
                scores.append(
                    _empty_visible_mask_score(
                        r,
                        regions,
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
            "gt_point_spacing_m": float(mask_gt_point_spacing_m),
            "hm3d_root": str(hm3d_root),
            "gt_object_margin_m": float(mask_gt_object_margin_m),
            "pred_mask_kind": str(mask_pred_kind),
            "allow_pred_point_projection": bool(mask_allow_pred_point_projection),
            "debug_dir": None if mask_debug_dir is None else str(mask_debug_dir),
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
    """Pretty-print the overall metric block + key breakdowns."""
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

    breakdowns = aggregate_dict.get("breakdowns") or {}
    score_kind = str(overall.get("score_kind") or "iou")
    primary_threshold = float(aggregate_dict.get("primary_threshold", PRIMARY_THRESHOLD))
    primary_label = _threshold_display(primary_threshold)
    acc_primary_key = _threshold_metric_name("acc@1", score_kind, primary_threshold)
    r5_primary_key = _threshold_metric_name("recall@5", score_kind, primary_threshold)
    mrr_primary_key = _threshold_metric_name("mrr", score_kind, primary_threshold)
    acc50_key = _threshold_metric_name("acc@1", score_kind, 0.5)
    for kind in ("relation", "region_class", "difficulty"):
        section = breakdowns.get(kind) or {}
        if not section:
            continue
        lines.append("")
        lines.append(f"# By {kind}")
        for label, m in section.items():
            lines.append(
                f"  {label:<22s} n={m.get('n', 0):>5d} "
                f"acc@1@{primary_label}={m.get(acc_primary_key, 0.0):.3f} "
                f"acc@1@0.5={m.get(acc50_key, 0.0):.3f} "
                f"recall@5@{primary_label}={m.get(r5_primary_key, 0.0):.3f} "
                f"mrr@{primary_label}={m.get(mrr_primary_key, 0.0):.3f}"
            )
    return "\n".join(lines)
