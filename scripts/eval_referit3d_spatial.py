"""Evaluate the spatial reasoning module on ReferIt3D (SR3D+ / NR3D).

Routes each utterance through the relational retrieval pipeline
(parse_query -> execute_spatial_query). Score phase can use either 3D bbox
IoU or the paper's projected visible-mask metric via
``--match-mode visible_mask``.

Flow per utterance:
  1. parse_query(utterance) -> QueryGraph (if spatial predicates found)
  2. execute_spatial_query(query_graph, scene_state, ...) -> ranked ScoredCandidate
  3. Convert to PredictedObject (bbox via gaussian_aabb) for IoU scoring
  4. Fallback: if parse returns None (simple lookup), use baseline retriever

Usage (inside the container; see EVALUATION.md for the full protocol)::

    # Paper protocol predict pass (needs the vLLM servers, ./run.sh vllm):
    python scripts/eval_referit3d_spatial.py --phase predict \
        --scenes-dir /data/out/scannet \
        --predictions-path /data/out/referit3d/preds.json \
        --uid-filter benchmarks/curated_utterances/scannet_30.json

    # Quick smoke (50 SR3D+ utterances):
    python scripts/eval_referit3d_spatial.py --phase predict \
        --scenes-dir /data/out/scannet \
        --predictions-path /tmp/preds_smoke.json \
        --max-utterances 50 --dataset sr3d

    # Score only (if predictions already exist):
    python scripts/eval_referit3d_spatial.py --phase score \
        --scenes-dir /data/out/scannet \
        --predictions-path /data/out/referit3d/preds.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scene_graph.eval.referit3d import (  # noqa: E402
    RunnerConfig,
    Utterance,
    discover_scene_states,
    format_overall_table,
    utterance_to_record,
    utterances_by_scene,
    val_local_subset,
)
from scene_graph.eval.referit3d.alias_geometry import AliasBox, AliasGeometryResolver  # noqa: E402
from scene_graph.eval.referit3d.dataset import partial_scene_ids  # noqa: E402
from scene_graph.eval.referit3d.matching import PredictedObject, gaussian_aabb, voxel_cloud_aabb  # noqa: E402
from scene_graph.eval.referit3d.retrieval_adapter import ranked_to_dicts  # noqa: E402
from scene_graph.eval.referit3d.scoring import score_and_persist  # noqa: E402
from scene_graph.retrieval.spatial_reasoning.methods import spatial_method_choices  # noqa: E402

LOGGER = logging.getLogger("eval_referit3d_spatial")


def _encode_query(embedder: Any, text: str) -> "np.ndarray":
    """Use Qwen3 retrieval query formatting when the embedder supports it."""
    import numpy as np

    fn = getattr(embedder, "encode_query", None)
    if callable(fn):
        return np.asarray(fn(text), dtype=np.float32)
    return np.asarray(embedder.encode(text), dtype=np.float32)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ReferIt3D eval with spatial reasoning module")
    p.add_argument("--phase", choices=("predict", "score", "both"), default="both")
    p.add_argument("--scenes-dir", type=Path, required=True,
                   help="Directory of reconstructed <scene_id>.pt scene states.")
    p.add_argument("--predictions-path", type=Path, required=True)
    p.add_argument("--metrics-path", type=Path, default=None)
    p.add_argument("--parse-cache-path", type=Path, default=None,
                   help="Optional predictions JSON from a prior spatial run. Reuses parsed query graphs by uid.")
    p.add_argument("--scans-dir", type=Path, default=None)
    p.add_argument(
        "--match-mode",
        choices=("bbox", "visible_mask"),
        default="visible_mask",  # 2D visible-mask IoU is the headline metric (locked 2026-05-15)
        help="Evaluation match criterion for score/both: legacy 3D bbox IoU or visible-mask IoU.",
    )
    p.add_argument("--mask-scene-state-dir", type=Path, default=None,
                   help="Scene-state dir for visible-mask scoring. Defaults to --scenes-dir.")
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
    p.add_argument("--mask-allow-raw-projection", action="store_true",
                   help="Allow scoring frames with no depth by raw projection. Default requires depth.")
    p.add_argument("--view-picker",
                   default="v1_largest_mask",
                   help="Canonical-view picker for single-view visible-mask IoU. v1_largest_mask is "
                        "the locked default (2026-05-16 ablation); "
                        "pass v0_multiview to fall back to the legacy best-of-N behavior. See "
                        "scene_graph.eval.view_selection for all picker variants.")
    p.add_argument("--scene", action="append", default=None)
    p.add_argument("--dataset", choices=("all", "nr3d", "sr3d"), default="all",
                   help="Filter to NR3D-only or SR3D-only utterances.")
    p.add_argument(
        "--uid-filter",
        type=Path,
        default=None,
        help="Optional path to a JSON file whose top-level shape is either a "
             "flat list of UIDs (e.g. nr3d_<assignmentid> / sr3d_<idx>) or the "
             "curated-utterances JSON as shipped in "
             "benchmarks/curated_utterances/. Only utterances "
             "whose uid is in the union are kept.",
    )
    p.add_argument("--max-utterances", type=int, default=None)
    p.add_argument(
        "--max-per-scene",
        type=int,
        default=None,
        help="Cap utterances per scan after scene/dataset filters. 0/None means uncapped.",
    )
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--no-vlm", action="store_true", help="Fast-path only (no VLM calls).")
    p.add_argument("--k-sigma", type=float, default=2.5)
    p.add_argument("--no-voxel-aabb", action="store_true",
                   help="Force gaussian_aabb even when scene state has voxel keys.")
    p.add_argument(
        "--geometry-mode",
        choices=("default", "canonical_source", "alias_expand", "alias_text", "alias_text_expand"),
        default="alias_expand",
        help=(
            "How to emit geometry for redirected objects. default preserves legacy raw object boxes; "
            "canonical_source canonicalizes ids but keeps the source box; alias_* exposes redirect-group aliases. "
            "alias_expand is the recommended V2 setting."
        ),
    )
    p.add_argument("--max-aliases-per-candidate", type=int, default=2)
    p.add_argument(
        "--alias-order",
        choices=("source_first", "text_first", "count_first"),
        default="source_first",
        help="Alias ordering prior used by alias geometry modes.",
    )
    p.add_argument("--max-predictions", type=int, default=20)
    p.add_argument("--pre-filter-k", type=int, default=-1,
                   help="Max candidates after semantic pre-filter. -1 = no filter "
                        "(score every region-scoped candidate; locked default, "
                        "2026-05-16 ablation).")
    p.add_argument("--max-candidates-for-vlm", type=int, default=10)
    p.add_argument("--vlm-rerank-candidates", action="store_true",
                   help="Use one VLM contact sheet to rerank target candidates before spatial predicates.")
    p.add_argument("--vlm-rerank-top-k", type=int, default=20)
    p.add_argument("--vlm-rerank-blend", type=float, default=0.65)
    p.add_argument("--retrieval-mode", choices=("caption", "multi"), default="multi",
                   help="Semantic candidate pre-filter: legacy caption-only or multi-embedding RRF.")
    p.add_argument("--candidate-pool-mode", choices=("active", "all", "active_plus_redirect"), default="active",
                   help="Object pool for semantic candidate retrieval before spatial scoring.")
    p.add_argument("--spatial-method", choices=spatial_method_choices(), default="unified_soft_w50",
                   help="Spatial disambiguation method/ablation profile. unified_soft_w50 is "
                        "the locked default (2026-05-17 "
                        "ablation; class_mismatch_floor=0.3). Replaces the prior "
                        "per-benchmark dispatch soft_predicates_w50.")
    p.add_argument("--predicate-calibration-path", type=Path, default=None,
                   help="Calibration JSON used by --spatial-method calibrated_logprob.")
    p.add_argument("--hard-threshold", type=float, default=0.5,
                   help="Predicate pass threshold for --spatial-method hard_predicates.")
    p.add_argument("--partial", action="store_true",
                   help="Restrict to the frozen 75-scene partial subset.")
    p.add_argument("--logger-level", default="INFO")
    return p.parse_args()


def _load_existing(output_path: Path) -> Dict[str, Dict[str, Any]]:
    if not output_path.exists():
        return {}
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {str(e.get("uid", "")): e for e in (payload or []) if e.get("uid")}


def _load_parse_cache(path: Optional[Path]) -> Dict[str, Any]:
    """Load cached QueryGraph objects from a previous predictions JSON.

    ``None`` is a meaningful cached value: it records that the prior parser
    returned no spatial query graph, so the evaluator should take the embedding
    fallback without spending another LLM call.
    """

    if path is None or not path.exists():
        return {}
    from scene_graph.retrieval.spatial_reasoning.models import Predicate, QueryGraph

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    cache: Dict[str, Any] = {}
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


def _persist(output_path: Path, ordered_uids: List[str], predictions: Dict[str, Dict[str, Any]]) -> None:
    out = [predictions[u] for u in ordered_uids if u in predictions]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(output_path)


def _scoring_detail(scored: List[Any], scene_state: Dict[str, Any], max_detail: int = 5) -> List[Dict[str, Any]]:
    """Build per-candidate scoring detail for top candidates."""
    captions = scene_state.get("object_caption", [])
    categories = scene_state.get("object_category", [])
    region_ids = scene_state.get("region_ids", [])
    region_labels = scene_state.get("region_labels", [])

    details = []
    for sc in scored[:max_detail]:
        caption = captions[sc.object_index] if sc.object_index < len(captions) else ""
        category = categories[sc.object_index] if sc.object_index < len(categories) else ""
        region = ""
        if sc.object_index < len(region_ids):
            rid = region_ids[sc.object_index]
            if 0 <= rid < len(region_labels):
                region = region_labels[rid]

        pred_details = []
        pred_scores = []
        for pr in sc.predicate_results:
            pd = {"name": pr.name, "score": round(pr.score, 4), "status": pr.status}
            if pr.drop_reason:
                pd["drop_reason"] = pr.drop_reason
            pred_details.append(pd)
            try:
                pred_scores.append(float(pr.score))
            except Exception:
                pass

        predicate_mean = float(sum(pred_scores) / len(pred_scores)) if pred_scores else None
        predicate_min = float(min(pred_scores)) if pred_scores else None
        predicate_max = float(max(pred_scores)) if pred_scores else None

        details.append({
            "object_index": sc.object_index,
            "object_id": sc.object_id,
            "composite_score": round(sc.composite_score, 4),
            "target_similarity": round(float(sc.target_similarity), 4) if sc.target_similarity is not None else None,
            "vlm_rerank_score": round(float(sc.vlm_rerank_score), 4) if sc.vlm_rerank_score is not None else None,
            "predicate_geo_mean": round(float(sc.predicate_geo_mean), 4) if sc.predicate_geo_mean is not None else None,
            "predicate_weight": round(float(sc.predicate_weight), 4) if sc.predicate_weight is not None else None,
            "predicate_score_mean": round(predicate_mean, 4) if predicate_mean is not None else None,
            "predicate_score_min": round(predicate_min, 4) if predicate_min is not None else None,
            "predicate_score_max": round(predicate_max, 4) if predicate_max is not None else None,
            "caption": caption,
            "category": category,
            "region": region,
            "predicates_evaluated": sc.predicates_evaluated,
            "predicates_dropped": sc.predicates_dropped,
            "predicate_results": pred_details,
            "matched_anchors": sc.matched_anchors,
        })
    return details


def run_spatial_predictions(
    *,
    scenes_dir: Path,
    output_path: Path,
    utterances: List[Utterance],
    use_vlm: bool = True,
    k_sigma: float = 2.5,
    max_predictions: int = 20,
    pre_filter_k: int = 40,
    max_candidates_for_vlm: int = 10,
    vlm_rerank_candidates: bool = False,
    vlm_rerank_top_k: int = 20,
    vlm_rerank_blend: float = 0.65,
    retrieval_mode: str = "multi",
    candidate_pool_mode: str = "active",
    spatial_method: str = "current",
    predicate_calibration_path: Optional[Path] = None,
    hard_threshold: float = 0.5,
    parse_cache_path: Optional[Path] = None,
    resume: bool = True,
    prefer_voxel_aabb: bool = True,
    geometry_mode: str = "alias_expand",
    max_aliases_per_candidate: int = 2,
    alias_order: str = "source_first",
) -> List[Dict[str, Any]]:
    """Run spatial reasoning predictions across all utterances."""
    import torch
    import numpy as np

    from scene_graph.llm_utils import EmbedInterface, LLMInterface
    from scene_graph.retrieval.spatial_reasoning import execute_spatial_query, parse_query
    from scene_graph.retrieval.spatial_reasoning.models import ScoredCandidate
    from scene_graph.eval.referit3d.matching import gaussian_aabb

    scene_states_paths = discover_scene_states(scenes_dir)
    LOGGER.info("Found %d scene_state.pt files under %s", len(scene_states_paths), scenes_dir)

    grouped = utterances_by_scene(utterances)
    existing = _load_existing(output_path) if resume else {}
    if existing:
        LOGGER.info("Resuming from %d existing predictions", len(existing))
    parse_cache = _load_parse_cache(parse_cache_path)
    if parse_cache:
        LOGGER.info("Loaded %d cached parses from %s", len(parse_cache), parse_cache_path)

    predictions: Dict[str, Dict[str, Any]] = dict(existing)
    ordered_uids = [u.uid for u in utterances]

    llm = LLMInterface(verbose=False, log_dir="/tmp/llm_logs_spatial")
    llm.config.max_tokens = 512
    embedder = EmbedInterface(verbose=False)

    n_scenes = 0
    n_total = 0
    n_spatial = 0
    n_fallback = 0
    n_fail = 0
    t_start = time.time()

    for scan_id in sorted(grouped.keys()):
        utts = [u for u in grouped[scan_id] if u.uid not in predictions]
        if not utts:
            continue
        pt_path = scene_states_paths.get(scan_id)
        if pt_path is None:
            LOGGER.warning("Skipping %s — no .pt found", scan_id)
            continue

        LOGGER.info("Loading %s (%d utterances)", scan_id, len(utts))
        try:
            payload = torch.load(pt_path, map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and "state" in payload:
                scene_state = payload["state"]
            else:
                scene_state = payload
        except Exception as e:
            LOGGER.error("Failed to load %s: %s", pt_path, e)
            continue

        n_scenes += 1
        means = scene_state.get("means")
        cov6 = scene_state.get("cov6")
        object_ids = scene_state.get("object_id")

        if means is not None and hasattr(means, "cpu"):
            means_np = means.cpu().numpy()
        else:
            means_np = np.asarray(means) if means is not None else np.empty((0, 3))

        if cov6 is not None and hasattr(cov6, "cpu"):
            cov6_np = cov6.cpu().numpy()
        else:
            cov6_np = np.asarray(cov6) if cov6 is not None else np.empty((0, 6))

        # Voxel-cloud AABB inputs (sparse-CSR per-object). Preferred over the Gaussian
        # box when present — gives a much tighter, evidence-grounded bbox that matters
        # at IoU=0.25/0.5 scoring. Set --no-voxel-aabb to disable.
        get_aabb = _make_aabb_resolver(scene_state, k_sigma, prefer_voxel=prefer_voxel_aabb)
        alias_resolver = AliasGeometryResolver(
            scene_state,
            k_sigma=k_sigma,
            prefer_voxel=prefer_voxel_aabb,
        )

        scene_t0 = time.time()

        for i, utt in enumerate(utts, start=1):
            t0 = time.time()
            record = utterance_to_record(utt)

            try:
                # Step 1: Parse query into predicates
                if utt.uid in parse_cache:
                    query_graph = parse_cache[utt.uid]
                else:
                    query_graph = parse_query(utt.utterance, llm)

                if query_graph is not None:
                    # Step 2: Execute spatial reasoning
                    scored = execute_spatial_query(
                        query_graph, scene_state, llm, embedder,
                        use_vlm=use_vlm,
                        pre_filter_k=pre_filter_k,
                        max_candidates_for_vlm=max_candidates_for_vlm,
                        vlm_rerank_enabled=vlm_rerank_candidates,
                        vlm_rerank_top_k=vlm_rerank_top_k,
                        vlm_rerank_blend=vlm_rerank_blend,
                        raw_query=utt.utterance,
                        retrieval_mode=retrieval_mode,
                        candidate_pool_mode=candidate_pool_mode,
                        spatial_method=spatial_method,
                        predicate_calibration_path=predicate_calibration_path,
                        hard_threshold=hard_threshold,
                        verbose=False,
                    )
                    # Step 3: Convert to PredictedObject for IoU scoring
                    ranked = _scored_to_predicted(
                        scored,
                        means_np,
                        cov6_np,
                        object_ids,
                        k_sigma,
                        max_predictions,
                        get_aabb=get_aabb,
                        alias_resolver=alias_resolver,
                        geometry_mode=geometry_mode,
                        query_text=f"{query_graph.target_description} {utt.utterance}",
                        max_aliases_per_candidate=max_aliases_per_candidate,
                        alias_order=alias_order,
                    )
                    record["method"] = "spatial"
                    record["spatial_method"] = spatial_method
                    record["vlm_rerank_candidates"] = bool(vlm_rerank_candidates)
                    record["predicates"] = [
                        {"name": p.name, "args": p.args, "kwargs": p.kwargs} for p in query_graph.predicates
                    ]
                    record["target_description"] = query_graph.target_description
                    record["target_class"] = getattr(query_graph, "target_class", None)
                    record["reasoning"] = query_graph.reasoning
                    record["scoring_detail"] = _scoring_detail(scored, scene_state, max_detail=max_predictions)
                    n_spatial += 1
                else:
                    # Fallback: simple embedding retrieval (no spatial predicates)
                    ranked = _embedding_fallback(
                        utt.utterance, scene_state, embedder, means_np, cov6_np, object_ids,
                        k_sigma, max_predictions, get_aabb=get_aabb,
                        alias_resolver=alias_resolver,
                        geometry_mode=geometry_mode,
                        max_aliases_per_candidate=max_aliases_per_candidate,
                        alias_order=alias_order,
                    )
                    record["method"] = "fallback"
                    record["spatial_method"] = spatial_method
                    record["predicates"] = []
                    n_fallback += 1

                record["ranked"] = ranked_to_dicts(ranked)
                record["error"] = None

            except Exception as e:
                LOGGER.error("Failed on %s: %s", utt.uid, e)
                record["ranked"] = []
                record["error"] = str(e)
                record["method"] = "error"
                record["spatial_method"] = spatial_method
                record["predicates"] = []
                n_fail += 1

            record["geometry_mode"] = geometry_mode
            record["max_aliases_per_candidate"] = int(max_aliases_per_candidate)
            record["alias_order"] = alias_order
            record["elapsed_s"] = round(time.time() - t0, 4)
            predictions[utt.uid] = record
            n_total += 1

            if i % 50 == 0 or i == len(utts):
                _persist(output_path, ordered_uids, predictions)
                elapsed_scene = time.time() - scene_t0
                LOGGER.info(
                    "  %s %d/%d (%.1f utt/s) [spatial=%d, fallback=%d, fail=%d]",
                    scan_id, i, len(utts), i / max(0.001, elapsed_scene),
                    n_spatial, n_fallback, n_fail,
                )

        _persist(output_path, ordered_uids, predictions)

    _persist(output_path, ordered_uids, predictions)
    elapsed = time.time() - t_start
    LOGGER.info(
        "Done. %d scenes, %d utterances (spatial=%d, fallback=%d, fail=%d) in %.1fs (%.2f utt/s)",
        n_scenes, n_total, n_spatial, n_fallback, n_fail, elapsed,
        n_total / max(0.001, elapsed),
    )
    return [predictions[u] for u in ordered_uids if u in predictions]


def _make_aabb_resolver(scene_state: Dict[str, Any], k_sigma: float, *, prefer_voxel: bool):
    """Return a callable ``idx -> (bbox_min, bbox_max)`` that prefers the per-object
    sparse voxel cloud AABB when present, falling back to ``gaussian_aabb``.

    Voxel data layout in the scene state (sparse-CSR):
      - ``object_voxel_keys_flat``    (M_total,) int64 packed voxel keys
      - ``object_voxel_keys_offsets`` (N+1,)    int64 row offsets per object
      - ``object_voxel_levels``       (N,)      int8/int64 per-object level

    The Gaussian box (`gaussian_aabb`) is a 2.5-sigma ellipsoid AABB which often
    overestimates extent; the voxel-derived box is the tight bbox over the actual
    accumulated 3D evidence and is what we want by default for IoU scoring.
    """
    import numpy as np

    means = scene_state.get("means")
    cov6 = scene_state.get("cov6")
    means_np_local = means.cpu().numpy() if hasattr(means, "cpu") else np.asarray(means)
    cov6_np_local = cov6.cpu().numpy() if hasattr(cov6, "cpu") else np.asarray(cov6)

    voxel_flat = scene_state.get("object_voxel_keys_flat") if prefer_voxel else None
    voxel_offsets = scene_state.get("object_voxel_keys_offsets") if prefer_voxel else None
    voxel_levels = scene_state.get("object_voxel_levels") if prefer_voxel else None
    has_voxels = (
        prefer_voxel
        and voxel_flat is not None and voxel_offsets is not None and voxel_levels is not None
    )
    if has_voxels:
        flat_np = voxel_flat.cpu().numpy().astype(np.int64) if hasattr(voxel_flat, "cpu") else np.asarray(voxel_flat, dtype=np.int64)
        off_np = voxel_offsets.cpu().numpy().astype(np.int64) if hasattr(voxel_offsets, "cpu") else np.asarray(voxel_offsets, dtype=np.int64)
        lvl_np = voxel_levels.cpu().numpy().astype(np.int64) if hasattr(voxel_levels, "cpu") else np.asarray(voxel_levels, dtype=np.int64)
    else:
        flat_np = off_np = lvl_np = None

    def _get(idx: int):
        # Prefer voxel-cloud AABB when this object has any voxels.
        if has_voxels and idx + 1 < len(off_np):
            s, e = int(off_np[idx]), int(off_np[idx + 1])
            if e > s:
                level = int(lvl_np[idx]) if idx < len(lvl_np) else 0
                box = voxel_cloud_aabb(flat_np[s:e], level)
                if box is not None:
                    return box
        # Fallback: Gaussian k-sigma AABB.
        mean = means_np_local[idx] if idx < len(means_np_local) else np.zeros(3)
        c6 = cov6_np_local[idx] if idx < len(cov6_np_local) else np.zeros(6)
        return gaussian_aabb(mean, c6, k_sigma=k_sigma)

    return _get


def _scored_to_predicted(
    scored: List[Any],
    means_np: "np.ndarray",
    cov6_np: "np.ndarray",
    object_ids: Any,
    k_sigma: float,
    max_predictions: int,
    *,
    get_aabb=None,
    alias_resolver: Optional[AliasGeometryResolver] = None,
    geometry_mode: str = "default",
    query_text: str = "",
    max_aliases_per_candidate: int = 2,
    alias_order: str = "source_first",
) -> List[PredictedObject]:
    """Convert ScoredCandidate list to PredictedObject list with bboxes."""
    import numpy as np

    ranked: List[PredictedObject] = []
    seen: set = set()
    mode = str(geometry_mode or "default").strip().lower()

    if mode in {"alias_expand", "alias_text_expand"} and alias_resolver is not None:
        order = "text_first" if mode == "alias_text_expand" else alias_order
        for sc in scored:
            for box in alias_resolver.alias_boxes(
                sc.object_index,
                query_text=query_text,
                max_aliases=max_aliases_per_candidate,
                order=order,
            ):
                key = (int(box.canonical_object_id), int(box.alias_index))
                if key in seen:
                    continue
                seen.add(key)
                ranked.append(_predicted_from_alias_box(box, float(sc.composite_score)))
                if len(ranked) >= max_predictions:
                    return ranked
        return ranked

    for sc in scored[:max_predictions]:
        idx = sc.object_index
        if mode in {"canonical_source", "alias_text"} and alias_resolver is not None:
            if mode == "alias_text":
                aliases = alias_resolver.alias_boxes(
                    idx,
                    query_text=query_text,
                    max_aliases=1,
                    order="text_first",
                )
                if not aliases:
                    continue
                box = aliases[0]
                oid = int(box.canonical_object_id)
                bbox_min, bbox_max = box.bbox_min, box.bbox_max
            else:
                box = alias_resolver.box_for_index(idx)
                if box is None:
                    continue
                oid = int(box.canonical_object_id)
                bbox_min, bbox_max = box.bbox_min, box.bbox_max
        else:
            if object_ids is not None and idx < len(object_ids):
                oid = int(object_ids[idx])
            else:
                oid = idx

            if get_aabb is not None:
                bbox_min, bbox_max = get_aabb(idx)
            else:
                mean = means_np[idx] if idx < len(means_np) else np.zeros(3)
                c6 = cov6_np[idx] if idx < len(cov6_np) else np.zeros(6)
                bbox_min, bbox_max = gaussian_aabb(mean, c6, k_sigma=k_sigma)

        if oid in seen:
            continue
        seen.add(oid)

        # Build caption/label from scene state
        captions = []
        if hasattr(sc, 'predicate_results'):
            pass  # ScoredCandidate from spatial reasoning

        ranked.append(PredictedObject(
            object_id=oid,
            score=sc.composite_score,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            label=None,
            caption=None,
            region_label=None,
        ))

    return ranked


def _embedding_fallback(
    query: str,
    scene_state: Dict[str, Any],
    embedder: Any,
    means_np: "np.ndarray",
    cov6_np: "np.ndarray",
    object_ids: Any,
    k_sigma: float,
    max_predictions: int,
    *,
    get_aabb=None,
    alias_resolver: Optional[AliasGeometryResolver] = None,
    geometry_mode: str = "default",
    max_aliases_per_candidate: int = 2,
    alias_order: str = "source_first",
) -> List[PredictedObject]:
    """Simple embedding similarity ranking as fallback for non-spatial queries."""
    import numpy as np

    caption_embeddings = scene_state.get("object_caption_embedding", [])
    active = scene_state.get("active")

    if not caption_embeddings:
        return []

    try:
        query_emb = _encode_query(embedder, query)
    except Exception:
        return []

    scores = []
    for idx, emb in enumerate(caption_embeddings):
        if active is not None:
            active_val = active[idx] if idx < len(active) else False
            if hasattr(active_val, "item"):
                active_val = active_val.item()
            if not active_val:
                continue
        if not emb:
            continue
        emb_np = np.asarray(emb, dtype=np.float32)
        if emb_np.size == 0:
            continue
        sim = float(np.dot(query_emb, emb_np) / (np.linalg.norm(query_emb) * np.linalg.norm(emb_np) + 1e-8))
        scores.append((idx, sim))

    scores.sort(key=lambda x: x[1], reverse=True)

    ranked: List[PredictedObject] = []
    seen: set = set()
    mode = str(geometry_mode or "default").strip().lower()
    if mode in {"alias_expand", "alias_text_expand"} and alias_resolver is not None:
        order = "text_first" if mode == "alias_text_expand" else alias_order
        for idx, sim in scores:
            for box in alias_resolver.alias_boxes(
                idx,
                query_text=query,
                max_aliases=max_aliases_per_candidate,
                order=order,
            ):
                key = (int(box.canonical_object_id), int(box.alias_index))
                if key in seen:
                    continue
                seen.add(key)
                ranked.append(_predicted_from_alias_box(box, float(sim)))
                if len(ranked) >= max_predictions:
                    return ranked
        return ranked

    for idx, sim in scores[:max_predictions]:
        if mode in {"canonical_source", "alias_text"} and alias_resolver is not None:
            if mode == "alias_text":
                aliases = alias_resolver.alias_boxes(idx, query_text=query, max_aliases=1, order="text_first")
                if not aliases:
                    continue
                box = aliases[0]
                oid = int(box.canonical_object_id)
                bbox_min, bbox_max = box.bbox_min, box.bbox_max
            else:
                box = alias_resolver.box_for_index(idx)
                if box is None:
                    continue
                oid = int(box.canonical_object_id)
                bbox_min, bbox_max = box.bbox_min, box.bbox_max
        else:
            if object_ids is not None and idx < len(object_ids):
                oid = int(object_ids[idx])
            else:
                oid = idx
            if get_aabb is not None:
                bbox_min, bbox_max = get_aabb(idx)
            else:
                mean = means_np[idx] if idx < len(means_np) else np.zeros(3)
                c6 = cov6_np[idx] if idx < len(cov6_np) else np.zeros(6)
                bbox_min, bbox_max = gaussian_aabb(mean, c6, k_sigma=k_sigma)
        if oid in seen:
            continue
        seen.add(oid)

        ranked.append(PredictedObject(
            object_id=oid,
            score=sim,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            label=None,
            caption=None,
            region_label=None,
        ))

    return ranked


def _predicted_from_alias_box(box: AliasBox, score: float) -> PredictedObject:
    label = f"alias:{box.alias_object_id}:{box.source}"
    return PredictedObject(
        object_id=int(box.canonical_object_id),
        score=float(score),
        bbox_min=box.bbox_min,
        bbox_max=box.bbox_max,
        label=label,
        caption=None,
        region_label=None,
    )


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.logger_level.upper()),
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    if args.phase in ("predict", "both"):
        if args.uid_filter is not None:
            # When a uid filter is supplied, skip the val/local restriction so
            # train-split scenes (e.g. top-30 by # GT objects for the final
            # benchmark) can be evaluated.
            from scene_graph.eval.referit3d.dataset import load_all as _load_all
            utterances = _load_all()
        else:
            utterances = val_local_subset()

        if args.partial:
            allow_partial = set(partial_scene_ids())
            utterances = [u for u in utterances if u.scan_id in allow_partial]

        if args.dataset == "sr3d":
            utterances = [u for u in utterances if u.dataset == "sr3d"]
        elif args.dataset == "nr3d":
            utterances = [u for u in utterances if u.dataset == "nr3d"]

        if args.scene:
            allow = set(args.scene)
            utterances = [u for u in utterances if u.scan_id in allow]
        if args.uid_filter is not None:
            uid_doc = json.loads(args.uid_filter.read_text())
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
            before = len(utterances)
            utterances = [u for u in utterances if u.uid in uid_allow]
            LOGGER.info(
                "--uid-filter kept %d / %d utterances (allow=%d)",
                len(utterances), before, len(uid_allow),
            )
        if args.max_per_scene is not None and int(args.max_per_scene) > 0:
            cap = int(args.max_per_scene)
            counts = {}
            limited = []
            for u in utterances:
                n = counts.get(u.scan_id, 0)
                if n >= cap:
                    continue
                counts[u.scan_id] = n + 1
                limited.append(u)
            utterances = limited
        if args.max_utterances is not None:
            utterances = utterances[:int(args.max_utterances)]

        LOGGER.info("Running %d utterances (dataset=%s, partial=%s)", len(utterances), args.dataset, args.partial)

        run_spatial_predictions(
            scenes_dir=args.scenes_dir,
            output_path=args.predictions_path,
            utterances=utterances,
            use_vlm=not args.no_vlm,
            k_sigma=args.k_sigma,
            max_predictions=args.max_predictions,
            pre_filter_k=args.pre_filter_k,
            max_candidates_for_vlm=args.max_candidates_for_vlm,
            vlm_rerank_candidates=args.vlm_rerank_candidates,
            vlm_rerank_top_k=args.vlm_rerank_top_k,
            vlm_rerank_blend=args.vlm_rerank_blend,
            retrieval_mode=args.retrieval_mode,
            candidate_pool_mode=args.candidate_pool_mode,
            spatial_method=args.spatial_method,
            predicate_calibration_path=args.predicate_calibration_path,
            hard_threshold=args.hard_threshold,
            parse_cache_path=args.parse_cache_path,
            resume=not args.no_resume,
            prefer_voxel_aabb=not args.no_voxel_aabb,
            geometry_mode=args.geometry_mode,
            max_aliases_per_candidate=args.max_aliases_per_candidate,
            alias_order=args.alias_order,
        )

    if args.phase in ("score", "both"):
        if not args.predictions_path.exists():
            print(f"ERROR: predictions not found at {args.predictions_path}", file=sys.stderr)
            return 2
        agg, metrics_path = score_and_persist(
            args.predictions_path,
            scans_dir=args.scans_dir,
            metrics_path=args.metrics_path,
            match_mode=args.match_mode,
            scene_state_dir=args.mask_scene_state_dir or args.scenes_dir,
            mask_depth_tolerance_m=args.mask_depth_tolerance_m,
            mask_point_radius_px=args.mask_point_radius_px,
            mask_min_gt_pixels=args.mask_min_gt_pixels,
            mask_topk=args.mask_topk,
            mask_max_views=None if args.mask_max_views <= 0 else args.mask_max_views,
            mask_max_points=args.mask_max_points,
            mask_score_aggregation=args.mask_score_aggregation,
            mask_require_depth=not args.mask_allow_raw_projection,
            view_picker_name=args.view_picker,
        )
        print(format_overall_table(agg))
        print(f"\nmetrics -> {metrics_path}")

        # Also print method breakdown
        predictions = json.loads(args.predictions_path.read_text())
        n_spatial = sum(1 for p in predictions if p.get("method") == "spatial")
        n_fallback = sum(1 for p in predictions if p.get("method") == "fallback")
        n_error = sum(1 for p in predictions if p.get("method") == "error")
        print(f"\nMethod breakdown: spatial={n_spatial}, fallback={n_fallback}, error={n_error}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
