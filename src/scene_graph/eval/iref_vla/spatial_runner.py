"""Spatial-grounding runner for IRef-VLA statements.

IRef-VLA already provides target class, anchor class(es), and relation labels,
so this runner builds QueryGraph objects directly from benchmark metadata and
routes them through the shared spatial reasoning executor.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from scene_graph.eval.referit3d.alias_geometry import AliasBox, AliasGeometryResolver
from scene_graph.eval.referit3d.matching import PredictedObject, gaussian_aabb, voxel_cloud_aabb
from scene_graph.eval.referit3d.retrieval_adapter import ranked_to_dicts
from scene_graph.retrieval.spatial_reasoning.models import Predicate, QueryGraph

from .dataset import Statement, load_all_statements, statements_by_scene
from .runner import _load_existing, _persist, discover_scene_states, statement_to_record

LOGGER = logging.getLogger("scene_graph.eval.iref_vla.spatial_runner")


@dataclass
class SpatialRunnerConfig:
    k_sigma: float = 2.5
    max_predictions: int = 20
    pre_filter_k: int = -1  # locked 2026-05-16: no pre-filter, score all region-scoped candidates.
    retrieval_mode: str = "multi"
    candidate_pool_mode: str = "active"
    spatial_method: str = "unified_soft_w50"  # locked 2026-05-17.
    use_vlm: bool = False
    prefer_voxel_aabb: bool = True
    geometry_mode: str = "alias_expand"
    max_aliases_per_candidate: int = 2
    alias_order: str = "source_first"
    log_every: int = 50
    # Locked 2026-05-17. When True (locked
    # default), build the QueryGraph from RAW STATEMENT TEXT via
    # ``parse_query(stmt.statement, llm)`` — same code path ScanNet uses.
    # When False, consume the GT-annotated structured fields
    # (``stmt.target_class``, ``stmt.anchor_classes``, ``stmt.relation``,
    # ``stmt.anchor_ids``) via ``statement_to_query_graph(stmt)`` — the
    # legacy behavior, retained as a diagnostic to measure parser-vs-GT
    # quality.
    text_only_query_graph: bool = True


_RELATION_SPLIT_PATTERNS = (
    r"\bsecond\s+closest\s+to\b",
    r"\bthird\s+closest\s+to\b",
    r"\bclosest\s+to\b",
    r"\bsecond\s+farthest\s+from\b",
    r"\bthird\s+farthest\s+from\b",
    r"\bfarthest\s+from\b",
    r"\bbetween\b",
    r"\babove\b",
    r"\bover\b",
    r"\bbelow\b",
    r"\bunder\b",
    r"\bnear\b",
    r"\bnext\s+to\b",
    r"\bon\b",
    r"\bin\b",
)


def _clean_phrase(text: str) -> str:
    out = " ".join(str(text or "").strip().split())
    out = re.sub(r"^(the|a|an)\s+", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+(that|which)\s+(is|are)?\s*$", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+(that|which)\s*$", "", out, flags=re.IGNORECASE)
    return out.strip(" ,.;:")


def _target_description(statement: Statement) -> str:
    text = str(statement.statement or "").strip()
    prefix = text
    for pattern in _RELATION_SPLIT_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            prefix = text[: match.start()]
            break
    desc = _clean_phrase(prefix)
    target_class = _clean_phrase(statement.target_class)
    if not desc:
        return target_class
    if target_class and target_class.lower() not in desc.lower():
        return target_class
    return desc or target_class


def _anchor_descriptions(statement: Statement) -> List[str]:
    anchors: List[str] = []
    for idx, cls in enumerate(statement.anchor_classes):
        text = _clean_phrase(cls)
        if not text and idx < len(statement.anchor_ids):
            text = f"object {int(statement.anchor_ids[idx])}"
        if text:
            anchors.append(text)
    return anchors


def _relation_predicate(statement: Statement, anchors: Sequence[str]) -> Optional[Predicate]:
    relation = str(statement.relation or "").strip().lower().replace("-", "_")
    if not relation:
        return None

    def anchor_at(index: int = 0) -> Optional[str]:
        if 0 <= index < len(anchors):
            return anchors[index]
        return None

    if relation == "between":
        if len(anchors) >= 2:
            return Predicate("Between", ["$target", anchors[0], anchors[1]])
        return None
    if relation in {"near", "close", "close_to"}:
        anchor = anchor_at()
        return Predicate("Near", ["$target", anchor]) if anchor else None
    if relation in {"above", "over"}:
        anchor = anchor_at()
        return Predicate("Above", ["$target", anchor]) if anchor else None
    if relation in {"below", "under"}:
        anchor = anchor_at()
        return Predicate("Below", ["$target", anchor]) if anchor else None
    if relation in {"on", "on_top_of"}:
        anchor = anchor_at()
        return Predicate("On", ["$target", anchor]) if anchor else None
    if relation in {"in", "inside", "within"}:
        anchor = anchor_at()
        return Predicate("Inside", ["$target", anchor]) if anchor else None

    ordinal_rank = {
        "closest": ("Closest", 0),
        "second_closest": ("Closest", 1),
        "third_closest": ("Closest", 2),
        "farthest": ("Farthest", 0),
        "second_farthest": ("Farthest", 1),
        "third_farthest": ("Farthest", 2),
    }.get(relation)
    if ordinal_rank is not None:
        name, rank = ordinal_rank
        anchor = anchor_at()
        return Predicate(name, ["$target", anchor], {"rank": rank}) if anchor else None

    return None


def statement_to_query_graph(statement: Statement) -> QueryGraph:
    """Build a spatial QueryGraph from IRef-VLA's structured annotation."""

    target = _target_description(statement)
    anchors = _anchor_descriptions(statement)
    predicates: List[Predicate] = []
    pred = _relation_predicate(statement, anchors)
    if pred is not None:
        predicates.append(pred)
    return QueryGraph(
        target_description=target,
        predicates=predicates,
        reasoning="Built from IRef-VLA target, anchor, and relation metadata.",
    )


def _to_numpy(value: Any, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32 if dtype is None else dtype)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _object_id_at(object_ids: Any, idx: int) -> int:
    try:
        value = object_ids[idx]
        return int(value.item()) if hasattr(value, "item") else int(value)
    except Exception:
        return int(idx)


def _make_aabb_resolver(scene_state: Dict[str, Any], k_sigma: float, *, prefer_voxel: bool):
    means_np = _to_numpy(scene_state.get("means"), dtype=np.float32)
    cov6_np = _to_numpy(scene_state.get("cov6"), dtype=np.float32)
    flat = scene_state.get("object_voxel_keys_flat") if prefer_voxel else None
    offsets = scene_state.get("object_voxel_keys_offsets") if prefer_voxel else None
    levels = scene_state.get("object_voxel_levels") if prefer_voxel else None
    has_voxels = flat is not None and offsets is not None and levels is not None
    flat_np = _to_numpy(flat, dtype=np.int64) if has_voxels else None
    off_np = _to_numpy(offsets, dtype=np.int64) if has_voxels else None
    lvl_np = _to_numpy(levels, dtype=np.int64) if has_voxels else None

    def _get(idx: int):
        if has_voxels and flat_np is not None and off_np is not None and lvl_np is not None and idx + 1 < len(off_np):
            s, e = int(off_np[idx]), int(off_np[idx + 1])
            if 0 <= s < e <= len(flat_np):
                level = int(lvl_np[idx]) if idx < len(lvl_np) else 0
                box = voxel_cloud_aabb(flat_np[s:e], level)
                if box is not None:
                    return box
        mean = means_np[idx] if idx < len(means_np) else np.zeros(3, dtype=np.float32)
        c6 = cov6_np[idx] if idx < len(cov6_np) else np.zeros(6, dtype=np.float32)
        return gaussian_aabb(mean, c6, k_sigma=k_sigma)

    return _get


def _caption_at(scene_state: Dict[str, Any], idx: int) -> Optional[str]:
    captions = scene_state.get("object_caption", []) or []
    if 0 <= int(idx) < len(captions):
        text = str(captions[int(idx)] or "").strip()
        return text or None
    return None


def _predicted_from_alias_box(scene_state: Dict[str, Any], box: AliasBox, score: float) -> PredictedObject:
    return PredictedObject(
        object_id=int(box.canonical_object_id),
        score=float(score),
        bbox_min=box.bbox_min,
        bbox_max=box.bbox_max,
        label=f"alias:{box.alias_object_id}:{box.source}",
        caption=_caption_at(scene_state, int(box.alias_index)),
        region_label=None,
    )


def _scored_to_predicted(
    scored: List[Any],
    scene_state: Dict[str, Any],
    *,
    k_sigma: float,
    max_predictions: int,
    get_aabb: Any,
    alias_resolver: AliasGeometryResolver,
    geometry_mode: str,
    query_text: str,
    max_aliases_per_candidate: int,
    alias_order: str,
) -> List[PredictedObject]:
    object_ids = scene_state.get("object_id")
    mode = str(geometry_mode or "default").strip().lower()
    ranked: List[PredictedObject] = []
    seen: set[Any] = set()

    if mode in {"alias_expand", "alias_text_expand"}:
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
                ranked.append(_predicted_from_alias_box(scene_state, box, float(sc.composite_score)))
                if len(ranked) >= max_predictions:
                    return ranked
        return ranked

    means_np = _to_numpy(scene_state.get("means"), dtype=np.float32)
    cov6_np = _to_numpy(scene_state.get("cov6"), dtype=np.float32)
    for sc in scored[:max_predictions]:
        idx = int(sc.object_index)
        if mode in {"canonical_source", "alias_text"}:
            order = "text_first" if mode == "alias_text" else alias_order
            aliases = alias_resolver.alias_boxes(idx, query_text=query_text, max_aliases=1, order=order)
            if not aliases:
                continue
            box = aliases[0]
            oid = int(box.canonical_object_id)
            bbox_min, bbox_max = box.bbox_min, box.bbox_max
            caption = _caption_at(scene_state, int(box.alias_index))
            label = f"alias:{box.alias_object_id}:{box.source}"
        else:
            oid = _object_id_at(object_ids, idx)
            if get_aabb is not None:
                bbox_min, bbox_max = get_aabb(idx)
            else:
                mean = means_np[idx] if idx < len(means_np) else np.zeros(3, dtype=np.float32)
                c6 = cov6_np[idx] if idx < len(cov6_np) else np.zeros(6, dtype=np.float32)
                bbox_min, bbox_max = gaussian_aabb(mean, c6, k_sigma=k_sigma)
            caption = _caption_at(scene_state, idx)
            label = None
        if oid in seen:
            continue
        seen.add(oid)
        ranked.append(
            PredictedObject(
                object_id=int(oid),
                score=float(sc.composite_score),
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                label=label,
                caption=caption,
                region_label=None,
            )
        )
    return ranked


def _scoring_detail(scored: List[Any], scene_state: Dict[str, Any], max_detail: int = 5) -> List[Dict[str, Any]]:
    categories = scene_state.get("object_category", []) or []
    out: List[Dict[str, Any]] = []
    for sc in scored[:max_detail]:
        idx = int(sc.object_index)
        preds = [
            {"name": pr.name, "score": round(float(pr.score), 4), "status": pr.status}
            for pr in getattr(sc, "predicate_results", [])
        ]
        out.append({
            "object_index": idx,
            "object_id": int(sc.object_id),
            "composite_score": round(float(sc.composite_score), 4),
            "target_similarity": round(float(sc.target_similarity), 4)
            if sc.target_similarity is not None
            else None,
            "predicate_geo_mean": round(float(sc.predicate_geo_mean), 4)
            if sc.predicate_geo_mean is not None
            else None,
            "predicate_weight": round(float(sc.predicate_weight), 4)
            if sc.predicate_weight is not None
            else None,
            "caption": _caption_at(scene_state, idx),
            "category": str(categories[idx]) if idx < len(categories) else "",
            "predicate_results": preds,
            "matched_anchors": dict(getattr(sc, "matched_anchors", {}) or {}),
        })
    return out


def run_spatial_predictions(
    *,
    scenes_dir: Path,
    output_path: Path,
    cfg: Optional[SpatialRunnerConfig] = None,
    statements: Optional[Iterable[Statement]] = None,
    scene_filter: Optional[Sequence[str]] = None,
    max_statements: Optional[int] = None,
    resume: bool = True,
    dataset_root: Optional[Path] = None,
    parse_cache: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run spatial IRef-VLA grounding and persist a predictions JSON.

    ``parse_cache`` maps statement uid -> QueryGraph (or None for a recorded
    parse miss) from a previous predictions JSON, so reruns reuse identical
    parses without new LLM calls (see ``--parse-cache-path``).
    """

    cfg = cfg or SpatialRunnerConfig()
    import torch
    from scene_graph.llm_utils import EmbedInterface, LLMInterface
    from scene_graph.retrieval.spatial_reasoning import execute_spatial_query, parse_query

    scene_states = discover_scene_states(scenes_dir)
    if statements is None:
        statements = load_all_statements(dataset_root=dataset_root, scene_filter=sorted(scene_states.keys()))
    statements = list(statements)
    if scene_filter is not None:
        allow = set(scene_filter)
        statements = [s for s in statements if s.scene_id in allow]
    if max_statements is not None:
        statements = statements[: int(max_statements)]

    LOGGER.info(
        "found %d scene_state.pt under %s; %d spatial statements queued",
        len(scene_states),
        scenes_dir,
        len(statements),
    )

    grouped = statements_by_scene(statements)
    existing = _load_existing(output_path) if resume else {}
    predictions: Dict[str, Dict[str, Any]] = dict(existing)
    ordered_uids = [s.uid for s in statements]

    embedder = EmbedInterface(verbose=False)
    llm = LLMInterface(verbose=False, log_dir="/tmp/llm_logs_iref_vla_spatial")
    llm.config.max_tokens = 512

    n_scenes_done = 0
    n_stmts_done = 0
    n_spatial = 0
    n_fallback = 0
    n_fail = 0
    t_start = time.time()

    for scene_id in sorted(grouped.keys()):
        stmts_for_scene = [s for s in grouped[scene_id] if s.uid not in predictions]
        if not stmts_for_scene:
            continue
        pt = scene_states.get(scene_id)
        if pt is None:
            LOGGER.warning("skipping %s — no scene_state.pt found", scene_id)
            continue

        LOGGER.info("loading scene %s (%d new statements) from %s", scene_id, len(stmts_for_scene), pt.name)
        try:
            payload = torch.load(pt, map_location="cpu", weights_only=False)
            scene_state = payload["state"] if isinstance(payload, dict) and "state" in payload else payload
            get_aabb = _make_aabb_resolver(scene_state, cfg.k_sigma, prefer_voxel=cfg.prefer_voxel_aabb)
            alias_resolver = AliasGeometryResolver(
                scene_state,
                k_sigma=cfg.k_sigma,
                prefer_voxel=cfg.prefer_voxel_aabb,
            )
        except Exception as exc:
            LOGGER.error("failed to load scene %s: %s", scene_id, exc)
            continue

        n_scenes_done += 1
        scene_started = time.time()
        for i, stmt in enumerate(stmts_for_scene, start=1):
            t0 = time.time()
            record = statement_to_record(stmt)
            try:
                # Locked 2026-05-17: build the QueryGraph from raw
                # statement text via ``parse_query`` (same code path ScanNet
                # uses) rather than consuming IRef-VLA's GT-annotated
                # ``target_class``/``anchor_classes``/``relation`` fields —
                # apples-to-apples with ScanNet, no GT-structure leakage at
                # retrieval time. Set ``cfg.text_only_query_graph = False`` to
                # fall back to ``statement_to_query_graph`` for diagnostic
                # parser-vs-GT comparisons.
                if cfg.text_only_query_graph:
                    if parse_cache is not None and stmt.uid in parse_cache:
                        parsed = parse_cache[stmt.uid]
                    else:
                        parsed = parse_query(stmt.statement, llm)
                    if parsed is None:
                        # parse_query returned no predicates (or LLM failed).
                        # Build a degenerate QueryGraph: use the raw statement
                        # text as ``target_description`` so the
                        # semantic-only path can still run.
                        query_graph = QueryGraph(
                            target_description=stmt.statement,
                            predicates=[],
                            reasoning="parse_query returned None (text-only fallback to raw utterance)",
                            target_class=None,
                        )
                    else:
                        query_graph = parsed
                else:
                    query_graph = statement_to_query_graph(stmt)
                if query_graph.predicates:
                    scored = execute_spatial_query(
                        query_graph,
                        scene_state,
                        llm,
                        embedder,
                        use_vlm=cfg.use_vlm,
                        pre_filter_k=cfg.pre_filter_k,
                        max_output_candidates=cfg.max_predictions,
                        raw_query=stmt.statement,
                        retrieval_mode=cfg.retrieval_mode,
                        candidate_pool_mode=cfg.candidate_pool_mode,
                        spatial_method=cfg.spatial_method,
                        verbose=False,
                    )
                    ranked = _scored_to_predicted(
                        scored,
                        scene_state,
                        k_sigma=cfg.k_sigma,
                        max_predictions=cfg.max_predictions,
                        get_aabb=get_aabb,
                        alias_resolver=alias_resolver,
                        geometry_mode=cfg.geometry_mode,
                        query_text=f"{query_graph.target_description} {stmt.statement}",
                        max_aliases_per_candidate=cfg.max_aliases_per_candidate,
                        alias_order=cfg.alias_order,
                    )
                    record["method"] = "spatial"
                    n_spatial += 1
                    record["scoring_detail"] = _scoring_detail(scored, scene_state, max_detail=cfg.max_predictions)
                else:
                    scored = execute_spatial_query(
                        QueryGraph(
                            target_description=query_graph.target_description,
                            predicates=[Predicate("IsCategory", ["$target", query_graph.target_description])],
                            reasoning=query_graph.reasoning,
                        ),
                        scene_state,
                        llm,
                        embedder,
                        use_vlm=False,
                        pre_filter_k=cfg.pre_filter_k,
                        max_output_candidates=cfg.max_predictions,
                        raw_query=stmt.statement,
                        retrieval_mode=cfg.retrieval_mode,
                        candidate_pool_mode=cfg.candidate_pool_mode,
                        spatial_method="semantic_only",
                        verbose=False,
                    )
                    ranked = _scored_to_predicted(
                        scored,
                        scene_state,
                        k_sigma=cfg.k_sigma,
                        max_predictions=cfg.max_predictions,
                        get_aabb=get_aabb,
                        alias_resolver=alias_resolver,
                        geometry_mode=cfg.geometry_mode,
                        query_text=f"{query_graph.target_description} {stmt.statement}",
                        max_aliases_per_candidate=cfg.max_aliases_per_candidate,
                        alias_order=cfg.alias_order,
                    )
                    record["method"] = "fallback"
                    n_fallback += 1
                    record["scoring_detail"] = _scoring_detail(scored, scene_state, max_detail=cfg.max_predictions)

                record["spatial_method"] = cfg.spatial_method
                record["target_description"] = query_graph.target_description
                record["target_class"] = getattr(query_graph, "target_class", None)
                record["reasoning"] = getattr(query_graph, "reasoning", None)
                record["predicates"] = [
                    {"name": p.name, "args": p.args, "kwargs": p.kwargs} for p in query_graph.predicates
                ]
                record["ranked"] = ranked_to_dicts(ranked)
                record["error"] = None
            except Exception as exc:
                LOGGER.error("spatial retrieve failed on %s: %s", stmt.uid, exc)
                record["ranked"] = []
                record["error"] = str(exc)
                record["method"] = "error"
                record["spatial_method"] = cfg.spatial_method
                record["predicates"] = []
                n_fail += 1
            record["geometry_mode"] = cfg.geometry_mode
            record["max_aliases_per_candidate"] = int(cfg.max_aliases_per_candidate)
            record["alias_order"] = cfg.alias_order
            record["elapsed_s"] = round(time.time() - t0, 4)
            predictions[stmt.uid] = record
            n_stmts_done += 1

            if i % cfg.log_every == 0 or i == len(stmts_for_scene):
                _persist(output_path, ordered_uids, predictions)
                elapsed_scene = time.time() - scene_started
                LOGGER.info(
                    "  %s %d/%d (%.2f stmt/s) [spatial=%d, fallback=%d, fail=%d]",
                    scene_id,
                    i,
                    len(stmts_for_scene),
                    i / max(1e-3, elapsed_scene),
                    n_spatial,
                    n_fallback,
                    n_fail,
                )
        _persist(output_path, ordered_uids, predictions)

    _persist(output_path, ordered_uids, predictions)
    LOGGER.info(
        "done. %d scenes, %d statements in %.1fs [spatial=%d, fallback=%d, fail=%d]",
        n_scenes_done,
        n_stmts_done,
        time.time() - t_start,
        n_spatial,
        n_fallback,
        n_fail,
    )
    return [predictions[u] for u in ordered_uids if u in predictions]
