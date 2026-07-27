"""Convenience wrapper around :mod:`metrics` for CLI scoring.

Reads a runner-produced ``predictions.json`` and writes a sibling
``<predictions_stem>-metrics.json`` with the aggregate metric block + all
breakdowns. Returns the aggregate dict so the caller can also print a
human-readable table via :func:`metrics.format_overall_table`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from .metrics import (
    DEFAULT_RECALL_KS,
    UtteranceScore,
    format_overall_table,
    score_predictions,
    write_metrics_json,
)

LOGGER = logging.getLogger("scene_graph.eval.referit3d.scoring")


def metrics_path_for(predictions_path: Path) -> Path:
    return predictions_path.with_name(predictions_path.stem + "-metrics.json")


def score_and_persist(
    predictions_path: Path,
    *,
    scans_dir: Optional[Path] = None,
    match_mode: str = "bbox",
    scene_state_dir: Optional[Path] = None,
    mask_depth_tolerance_m: float = 0.15,  # locked 2026-05-16: matches unified scorer / legacy protocol
    mask_point_radius_px: int = 3,
    mask_min_gt_pixels: int = 20,
    mask_topk: int = 3,
    mask_max_views: Optional[int] = 50,
    mask_max_points: int = 50000,
    mask_score_aggregation: str = "best_iou",
    mask_require_depth: bool = True,
    metrics_path: Optional[Path] = None,
    iou_thresholds: Optional[Sequence[float]] = None,
    recall_ks: Sequence[int] = DEFAULT_RECALL_KS,
    view_picker_name: str = "v1_largest_mask",
) -> Tuple[Dict[str, Any], Path]:
    """Run scoring + write metrics JSON. Returns ``(aggregate, metrics_path)``."""
    scores, aggregate = score_predictions(
        predictions_path,
        scans_dir=scans_dir,
        match_mode=match_mode,
        scene_state_dir=scene_state_dir,
        mask_depth_tolerance_m=mask_depth_tolerance_m,
        mask_point_radius_px=mask_point_radius_px,
        mask_min_gt_pixels=mask_min_gt_pixels,
        mask_topk=mask_topk,
        mask_max_views=mask_max_views,
        mask_max_points=mask_max_points,
        mask_score_aggregation=mask_score_aggregation,
        mask_require_depth=mask_require_depth,
        iou_thresholds=iou_thresholds,
        recall_ks=recall_ks,
        view_picker_name=view_picker_name,
    )
    out_path = metrics_path or metrics_path_for(predictions_path)
    write_metrics_json(aggregate, output_path=out_path)
    LOGGER.info("wrote metrics: %s", out_path)
    LOGGER.info("\n%s", format_overall_table(aggregate))
    return aggregate, out_path
