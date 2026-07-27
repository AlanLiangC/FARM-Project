#!/usr/bin/env python3
"""Predict + score for the IRef-VLA HM3D grounding benchmark.

Two phases:

* ``--phase predict`` — load ``scene_state.pt`` files, run retrieval over each
  IRef-VLA statement, write predictions JSON.
* ``--phase score`` — score predictions against IRef-VLA GT (object_result.csv
  -> AABB IoU by default, or visible-mask IoU with ``--match-mode visible_mask``);
  write metrics JSON.
* ``--phase all`` — both, in sequence.

Resume is on by default — re-running picks up where it left off.

Example (inside the container; see EVALUATION.md for the full protocol)::

    python scripts/eval_iref_vla.py --phase predict \\
        --scenes-dir /data/out/iref_vla \\
        --predictions-path /data/out/iref_vla/predictions.json \\
        --iref-vla-root /data/iref_vla/HM3D \\
        --uid-filter benchmarks/curated_utterances/iref_vla_hm3d_30.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

LOGGER = logging.getLogger("eval_iref_vla")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--phase",
        choices=("predict", "score", "all"),
        default="all",
        help="Run prediction, scoring, or both.",
    )
    p.add_argument(
        "--scenes-dir",
        type=Path,
        default=Path("/data/out/iref_vla"),
        help="Directory of <scene_id>.pt files.",
    )
    p.add_argument(
        "--predictions-path",
        type=Path,
        default=Path("/data/out/iref_vla/predictions.json"),
        help="Where to read/write the predictions JSON.",
    )
    p.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help="Override the metrics JSON path (default: <predictions>-metrics.json).",
    )
    p.add_argument(
        "--iref-vla-root",
        type=Path,
        default=None,
        help="IRef-VLA HM3D dataset root (defaults to default_iref_vla_root()).",
    )
    p.add_argument(
        "--match-mode",
        choices=("bbox", "visible_mask"),
        default="visible_mask",  # 2D visible-mask IoU is the headline metric (locked 2026-05-15)
        help="Evaluation match criterion: legacy 3D bbox IoU or occlusion-aware projected visible-mask IoU.",
    )
    p.add_argument("--mask-scene-state-dir", type=Path, default=None,
                   help="Scene-state dir for visible-mask scoring. Defaults to --scenes-dir.")
    p.add_argument("--hm3d-root", type=Path, default=None,
                   help="HM3D mesh root for visible-mask GT mesh projection.")
    p.add_argument("--mask-depth-tolerance-m", type=float, default=0.15,
                   help="Depth-tolerance for visible-mask GT projection (locked 2026-05-16). "
                        "Matches the eval_predictions.py visible-mask protocol.")
    p.add_argument("--mask-point-radius-px", type=int, default=3)
    p.add_argument("--mask-min-gt-pixels", type=int, default=20)
    p.add_argument("--mask-topk", type=int, default=3)
    p.add_argument("--mask-max-views", type=int, default=50,
                   help="Max associated views per object for visible-mask scoring. Use <=0 for all views.")
    p.add_argument("--mask-max-points", type=int, default=50000)
    p.add_argument("--mask-score-aggregation",
                   choices=("best_iou", "mean_topk_iou", "weighted_iou"),
                   default="best_iou")
    p.add_argument("--mask-gt-point-spacing-m", type=float, default=0.03,
                   help="Legacy fallback spacing if mesh GT is unavailable.")
    p.add_argument("--mask-gt-object-margin-m", type=float, default=0.02,
                   help="Expanded OBB margin used only to extract the GT object surface from the HM3D mesh.")
    p.add_argument("--mask-pred-kind", choices=("raw", "inlier"), default="raw",
                   help="Persisted predicted mask kind used for scoring.")
    p.add_argument("--mask-allow-pred-point-projection", action="store_true",
                   help="Debug fallback only: allow predicted voxel projection if saved masks are absent.")
    p.add_argument("--mask-debug-dir", type=Path, default=None,
                   help="Optional directory for GT/pred mask overlay debug PNGs.")
    p.add_argument("--mask-allow-raw-projection", action="store_true",
                   help="Allow scoring frames with no depth by raw projection. Default requires depth.")
    p.add_argument("--view-picker",
                   default="v1_largest_mask",
                   help="Canonical-view picker for single-view visible-mask IoU. v1_largest_mask is "
                        "the locked default (2026-05-16 ablation); "
                        "pass v0_multiview to fall back to the legacy best-of-N behavior. See "
                        "scene_graph.eval.view_selection for all picker variants.")
    p.add_argument(
        "--multi-room-only",
        action="store_true",
        help="Restrict to scenes with >=2 regions in region_result.csv.",
    )
    p.add_argument("--scene", action="append", default=None, help="Restrict to specific scene ids.")
    p.add_argument("--relation", action="append", default=None,
                   help="Restrict to specific relation labels (e.g. above, below, closest).")
    p.add_argument("--include-false-statements", action="store_true",
                   help="Also score the adversarial 'false_statements' variants.")
    p.add_argument("--max-statements", type=int, default=None)
    p.add_argument(
        "--max-per-scene",
        type=int,
        default=None,
        help="Cap statements per scene after filters. 0/None means uncapped.",
    )
    p.add_argument(
        "--uid-filter",
        type=Path,
        default=None,
        help="Optional path to a JSON file whose top-level shape is either a "
             "flat list of uids or a curated-utterances JSON as shipped in "
             "benchmarks/curated_utterances/. Only statements "
             "whose uid is in the union are kept.",
    )
    p.add_argument("--no-resume", action="store_true",
                   help="Do not resume from existing predictions.")
    p.add_argument("--k-sigma", type=float, default=2.5)
    p.add_argument("--max-predictions", type=int, default=20)
    p.add_argument(
        "--grounding-mode",
        choices=("spatial", "semantic"),
        default="spatial",
        help="Default spatial uses target/anchor/relation metadata; semantic is the legacy full-utterance retriever.",
    )
    p.add_argument("--spatial-method", default="unified_soft_w50",
                   help="Spatial disambiguation method. unified_soft_w50 is the "
                        "locked default (2026-05-17 "
                        "ablation; class_mismatch_floor=0.3). Replaces the prior "
                        "per-benchmark dispatch class_soft_w50.")
    p.add_argument("--parse-cache-path", type=Path, default=None,
                   help="Previous predictions JSON whose per-record 'predicates' + "
                        "'target_description' are reused as the QueryGraph for matching "
                        "uids — identical parses across method reruns, no new LLM calls "
                        "(mirrors eval_referit3d_spatial.py --parse-cache-path).")
    p.add_argument("--statement-query-graph", dest="text_only_query_graph",
                   action="store_false", default=True,
                   help="Diagnostic only: build the QueryGraph from the IRef-VLA "
                        "GT-annotated structured fields (target_class, anchor_classes, "
                        "relation) via statement_to_query_graph(stmt). Default is "
                        "text-only via parse_query(stmt.statement, llm) "
                        "(locked 2026-05-17).")
    p.add_argument("--pre-filter-k", type=int, default=-1,
                   help="Max candidates after semantic pre-filter. -1 = no filter "
                        "(score every region-scoped candidate; locked default, "
                        "2026-05-16 ablation).")
    p.add_argument("--retrieval-mode", choices=("caption", "multi"), default="multi")
    p.add_argument("--candidate-pool-mode", choices=("active", "all", "active_plus_redirect"), default="active")
    p.add_argument("--use-vlm", action="store_true", help="Allow VLM predicate checks after fast-path geometry.")
    p.add_argument("--no-voxel-aabb", action="store_true")
    p.add_argument(
        "--geometry-mode",
        choices=("default", "canonical_source", "alias_expand", "alias_text", "alias_text_expand"),
        default="alias_expand",
    )
    p.add_argument("--max-aliases-per-candidate", type=int, default=2)
    p.add_argument("--alias-order", choices=("source_first", "text_first", "count_first"), default="source_first")
    p.add_argument("--logger-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARN", "ERROR"))
    return p.parse_args(argv)


def _load_parse_cache(path: Optional[Path]) -> dict:
    """Load cached QueryGraph objects from a previous predictions JSON.

    Copied from ``eval_referit3d_spatial.py`` (same record schema). ``None``
    is a meaningful cached value: the prior parser returned no spatial query
    graph, so the runner takes the embedding fallback without a new LLM call.
    """
    import json as _json

    if path is None or not path.exists():
        return {}
    from scene_graph.retrieval.spatial_reasoning.models import Predicate, QueryGraph

    try:
        payload = _json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    cache: dict = {}
    for rec in payload or []:
        uid = str(rec.get("uid", ""))
        if not uid:
            continue
        if rec.get("method") == "fallback":
            cache[uid] = None
            continue
        pred_payload = rec.get("predicates")
        target = rec.get("target_description")
        if pred_payload is None and target is None:
            continue
        preds = []
        for item in pred_payload or []:
            if not isinstance(item, dict):
                continue
            preds.append(Predicate(
                name=str(item.get("name", "")),
                args=[str(x) for x in item.get("args", [])],
                kwargs=dict(item.get("kwargs", {}) or {}),
            ))
        cache[uid] = QueryGraph(
            target_description=str(target or ""),
            predicates=preds,
            reasoning=str(rec.get("reasoning", "")),
        )
    return cache


def _do_predict(args: argparse.Namespace) -> None:
    from scene_graph.eval.iref_vla.dataset import (
        filter_statements,
        load_all_statements,
        multi_room_scene_ids,
    )
    from scene_graph.eval.iref_vla.runner import RunnerConfig, run_predictions
    from scene_graph.eval.iref_vla.spatial_runner import SpatialRunnerConfig, run_spatial_predictions

    if args.multi_room_only:
        scenes = multi_room_scene_ids(dataset_root=args.iref_vla_root, min_regions=2)
    else:
        scenes = None

    if args.scene:
        if scenes is None:
            scenes = list(args.scene)
        else:
            scenes = [s for s in scenes if s in set(args.scene)]

    statements = load_all_statements(
        dataset_root=args.iref_vla_root,
        scene_filter=scenes,
        include_false_statements=bool(args.include_false_statements),
    )
    statements = filter_statements(
        statements,
        relations=args.relation,
        drop_false_statements=not args.include_false_statements,
    )
    if args.uid_filter is not None:
        import json as _json
        uid_doc = _json.loads(args.uid_filter.read_text())
        if isinstance(uid_doc, list):
            uid_allow = {str(x) for x in uid_doc}
        elif isinstance(uid_doc, dict) and "scenes" in uid_doc:
            uid_allow = set()
            for _, sc in uid_doc["scenes"].items():
                for u in sc.get("utterances", []):
                    if "uid" in u:
                        uid_allow.add(str(u["uid"]))
        else:
            uid_allow = set()
        before = len(statements)
        statements = [s for s in statements if getattr(s, "uid", None) in uid_allow]
        LOGGER.info(
            "--uid-filter kept %d / %d statements (allow=%d)",
            len(statements), before, len(uid_allow),
        )
    if args.max_per_scene is not None and int(args.max_per_scene) > 0:
        cap = int(args.max_per_scene)
        counts = {}
        limited = []
        for s in statements:
            n = counts.get(s.scene_id, 0)
            if n >= cap:
                continue
            counts[s.scene_id] = n + 1
            limited.append(s)
        statements = limited
    LOGGER.info("queued %d statements (after filters)", len(statements))

    if args.grounding_mode == "spatial":
        cfg = SpatialRunnerConfig(
            k_sigma=float(args.k_sigma),
            max_predictions=int(args.max_predictions),
            pre_filter_k=int(args.pre_filter_k),
            retrieval_mode=str(args.retrieval_mode),
            candidate_pool_mode=str(args.candidate_pool_mode),
            spatial_method=str(args.spatial_method),
            use_vlm=bool(args.use_vlm),
            prefer_voxel_aabb=not bool(args.no_voxel_aabb),
            geometry_mode=str(args.geometry_mode),
            max_aliases_per_candidate=int(args.max_aliases_per_candidate),
            alias_order=str(args.alias_order),
            text_only_query_graph=bool(getattr(args, "text_only_query_graph", True)),
        )
        parse_cache = _load_parse_cache(args.parse_cache_path)
        if parse_cache:
            LOGGER.info("Loaded %d cached parses from %s", len(parse_cache), args.parse_cache_path)
        run_spatial_predictions(
            scenes_dir=args.scenes_dir,
            output_path=args.predictions_path,
            cfg=cfg,
            statements=statements,
            scene_filter=args.scene,
            max_statements=args.max_statements,
            resume=not args.no_resume,
            dataset_root=args.iref_vla_root,
            parse_cache=parse_cache or None,
        )
    else:
        cfg = RunnerConfig(
            k_sigma=float(args.k_sigma),
            max_predictions=int(args.max_predictions),
            prefer_voxel_aabb=not bool(args.no_voxel_aabb),
        )
        run_predictions(
            scenes_dir=args.scenes_dir,
            output_path=args.predictions_path,
            cfg=cfg,
            statements=statements,
            scene_filter=args.scene,
            max_statements=args.max_statements,
            resume=not args.no_resume,
            dataset_root=args.iref_vla_root,
        )


def _do_score(args: argparse.Namespace) -> None:
    from scene_graph.eval.iref_vla.metrics import format_overall_table
    from scene_graph.eval.iref_vla.scoring import score_and_persist

    aggregate, metrics_path = score_and_persist(
        args.predictions_path,
        dataset_root=args.iref_vla_root,
        metrics_path=args.metrics_path,
        match_mode=args.match_mode,
        scene_state_dir=args.mask_scene_state_dir or args.scenes_dir,
        hm3d_root=args.hm3d_root,
        mask_depth_tolerance_m=args.mask_depth_tolerance_m,
        mask_point_radius_px=args.mask_point_radius_px,
        mask_min_gt_pixels=args.mask_min_gt_pixels,
        mask_topk=args.mask_topk,
        mask_max_views=None if args.mask_max_views <= 0 else args.mask_max_views,
        mask_max_points=args.mask_max_points,
        mask_score_aggregation=args.mask_score_aggregation,
        mask_require_depth=not args.mask_allow_raw_projection,
        mask_gt_point_spacing_m=args.mask_gt_point_spacing_m,
        mask_gt_object_margin_m=args.mask_gt_object_margin_m,
        mask_pred_kind=args.mask_pred_kind,
        mask_allow_pred_point_projection=args.mask_allow_pred_point_projection,
        mask_debug_dir=args.mask_debug_dir,
        view_picker_name=args.view_picker,
    )
    LOGGER.info("metrics → %s", metrics_path)
    print(format_overall_table(aggregate))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.logger_level), format="[%(levelname)s] %(name)s: %(message)s")
    if args.phase in ("predict", "all"):
        _do_predict(args)
    if args.phase in ("score", "all"):
        _do_score(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
