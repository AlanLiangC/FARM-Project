"""Execute a QueryGraph against scene state to produce ranked candidates."""

from __future__ import annotations

import logging
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from scene_graph.llm_utils import EmbedInterface, LLMInterface

from .calibration import SpatialCalibrator, clamp_probability, logprob_score
from .methods import get_spatial_method
from .models import Predicate, PredicateResult, QueryGraph, ScoredCandidate
from .predicates import PredicateEvaluator, SUPERLATIVE_PREDICATES
from .semantic_retrieval import SemanticRetrievalResult, retrieve_semantic_candidates


def _predicate_geo_mean(predicate_scores: List[float]) -> Optional[float]:
    """Return the geometric mean used by the soft-predicate scorer."""
    if not predicate_scores:
        return None
    product = 1.0
    for s in predicate_scores:
        product *= max(float(s), 1e-6)
    return float(product ** (1.0 / len(predicate_scores)))


def _current_composite_score(
    predicate_scores: List[float],
    target_sim: float,
    *,
    predicate_weight: float = 0.25,
) -> float:
    """Combine predicate scores with target similarity.

    Uses the geometric mean of predicate scores (robust to a single near-zero)
    as a light penalty on the target-description similarity. Predicate scores
    are memberships in ``[0, 1]``; flooring the result at ``target_sim`` would
    make the spatial terms unable to affect ranking, while a full product is
    too brittle when anchors/parses are noisy.
    """
    geo_mean = _predicate_geo_mean(predicate_scores)
    if geo_mean is None:
        return target_sim
    predicate_weight = max(0.0, min(1.0, float(predicate_weight)))
    spatial_factor = (1.0 - predicate_weight) + predicate_weight * geo_mean
    return target_sim * spatial_factor


def _composite_score(
    predicate_results: List[PredicateResult],
    target_sim: float,
    *,
    composition: str = "current",
    hard_threshold: float = 0.5,
    calibrator: Optional[SpatialCalibrator] = None,
    predicate_weight: Optional[float] = None,
) -> float:
    """Compose per-predicate results into the candidate score."""

    scores = [r.score for r in predicate_results]
    mode = str(composition or "current").strip().lower()
    if mode == "semantic":
        return float(target_sim)
    if mode == "hard":
        evaluated = [r for r in predicate_results if r.status != "dropped"]
        if not evaluated:
            return float(target_sim)
        ok = all(float(r.score) >= float(hard_threshold) for r in evaluated)
        return float(target_sim) if ok else 0.0
    if mode == "logprob":
        return logprob_score(
            ((r.name, r.score) for r in predicate_results if r.status != "dropped"),
            target_score=clamp_probability(target_sim),
            calibrator=calibrator,
        )
    return _current_composite_score(
        scores,
        float(target_sim),
        predicate_weight=0.25 if predicate_weight is None else float(predicate_weight),
    )

logger = logging.getLogger(__name__)

TARGET_VAR = "$target"

_LOCAL_COVISIBILITY_PREDICATES = {
    "Near",
    "NextTo",
    "On",
    "Above",
    "Below",
    "Inside",
    "LeftOf",
    "RightOf",
    "InFrontOf",
    "Behind",
}


def _to_numpy(value: Any) -> np.ndarray:
    """Best-effort conversion for torch tensors / numpy arrays / lists."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    return np.asarray(value)


def _scene_object_count(scene_state: Dict[str, Any]) -> int:
    means = scene_state.get("means")
    if means is not None:
        with np.errstate(all="ignore"):
            try:
                return int(len(means))
            except Exception:
                pass
    object_ids = scene_state.get("object_id")
    if object_ids is not None:
        with np.errstate(all="ignore"):
            try:
                return int(len(object_ids))
            except Exception:
                pass
    return 0


def _add_undirected_edge(adjacency: List[Set[int]], i: int, j: int) -> None:
    if i == j:
        return
    n = len(adjacency)
    if 0 <= i < n and 0 <= j < n:
        adjacency[i].add(j)
        adjacency[j].add(i)


def _adjacency_from_covisibility_bitset(scene_state: Dict[str, Any], object_count: int) -> Optional[List[Set[int]]]:
    raw = scene_state.get("covisibility_adj_u64")
    if raw is None or object_count <= 0:
        return None
    try:
        arr = _to_numpy(raw)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return None
    n_rows = min(int(object_count), int(arr.shape[0]))
    n_blocks = min(int(arr.shape[1]), int(math.ceil(float(object_count) / 64.0)))
    adjacency: List[Set[int]] = [set() for _ in range(int(object_count))]
    edge_count = 0
    for i in range(n_rows):
        for block in range(n_blocks):
            word = int(arr[i, block]) & ((1 << 64) - 1)
            while word:
                lsb = word & -word
                bit = int(lsb.bit_length() - 1)
                j = block * 64 + bit
                if j < object_count and j != i:
                    before = len(adjacency[i])
                    _add_undirected_edge(adjacency, i, j)
                    if len(adjacency[i]) != before:
                        edge_count += 1
                word &= word - 1
    if edge_count == 0:
        return None
    return adjacency


def _image_rows_from_scene_state(scene_state: Dict[str, Any]) -> Sequence[Sequence[int]]:
    object_rows = scene_state.get("object_image_ids") or []
    if object_rows:
        any_object = any(bool(row) for row in object_rows if isinstance(row, (list, tuple)))
        if any_object:
            return object_rows
    viewpoint_rows = scene_state.get("viewpoint_image_ids") or []
    if viewpoint_rows:
        any_viewpoint = any(bool(row) for row in viewpoint_rows if isinstance(row, (list, tuple)))
        if any_viewpoint:
            return viewpoint_rows
    return []


def _shared_image_lookup(
    scene_state: Dict[str, Any],
    object_count: int,
) -> Optional[Tuple[List[Set[int]], Dict[int, Set[int]]]]:
    rows = _image_rows_from_scene_state(scene_state)
    if not rows or object_count <= 0:
        return None
    object_images: List[Set[int]] = [set() for _ in range(object_count)]
    image_to_objects: Dict[int, Set[int]] = {}
    for idx in range(min(object_count, len(rows))):
        row = rows[idx]
        if not isinstance(row, (list, tuple)):
            continue
        for image_id in row:
            try:
                image_int = int(image_id)
            except Exception:
                continue
            object_images[idx].add(image_int)
            image_to_objects.setdefault(image_int, set()).add(int(idx))
    if not image_to_objects:
        return None
    return object_images, image_to_objects


class _CovisibilityConstraint:
    """k-hop reachability over the saved covisibility graph."""

    def __init__(self, adjacency: List[Set[int]], hops: int) -> None:
        self.adjacency = adjacency
        self.hops = max(0, int(hops))
        self._reachable_cache: Dict[int, Set[int]] = {}

    @property
    def enabled(self) -> bool:
        return self.hops > 0 and bool(self.adjacency) and any(self.adjacency)

    def reachable(self, object_idx: int) -> Set[int]:
        idx = int(object_idx)
        cached = self._reachable_cache.get(idx)
        if cached is not None:
            return cached
        if not self.enabled or idx < 0 or idx >= len(self.adjacency):
            out: Set[int] = set()
            self._reachable_cache[idx] = out
            return out
        visited: Set[int] = {idx}
        frontier: Set[int] = {idx}
        for _depth in range(self.hops):
            next_frontier: Set[int] = set()
            for node in frontier:
                if 0 <= node < len(self.adjacency):
                    next_frontier.update(self.adjacency[node])
            next_frontier.difference_update(visited)
            if not next_frontier:
                break
            visited.update(next_frontier)
            frontier = next_frontier
        visited.discard(idx)
        self._reachable_cache[idx] = visited
        return visited

    def filter_anchors(self, target_idx: int, anchor_candidates: Sequence[int]) -> List[int]:
        if not self.enabled:
            return [int(idx) for idx in anchor_candidates]
        target = int(target_idx)
        # Anchor lists are small and reused across all target candidates for a
        # query. Expanding from anchors lets the reachability cache pay off
        # after only a few calls, instead of expanding from every target.
        return [int(idx) for idx in anchor_candidates if target in self.reachable(int(idx))]


class _SharedImageCovisibilityConstraint:
    """k-hop reachability where one hop means sharing at least one frame id."""

    def __init__(
        self,
        object_images: List[Set[int]],
        image_to_objects: Dict[int, Set[int]],
        hops: int,
    ) -> None:
        self.object_images = object_images
        self.image_to_objects = image_to_objects
        self.hops = max(0, int(hops))
        self._reachable_cache: Dict[int, Set[int]] = {}

    @property
    def enabled(self) -> bool:
        return self.hops > 0 and bool(self.object_images) and bool(self.image_to_objects)

    def reachable(self, object_idx: int) -> Set[int]:
        idx = int(object_idx)
        cached = self._reachable_cache.get(idx)
        if cached is not None:
            return cached
        if not self.enabled or idx < 0 or idx >= len(self.object_images):
            out: Set[int] = set()
            self._reachable_cache[idx] = out
            return out
        visited: Set[int] = {idx}
        frontier: Set[int] = {idx}
        for _depth in range(self.hops):
            frontier_images: Set[int] = set()
            for node in frontier:
                if 0 <= node < len(self.object_images):
                    frontier_images.update(self.object_images[node])
            next_frontier: Set[int] = set()
            for image_id in frontier_images:
                next_frontier.update(self.image_to_objects.get(image_id, set()))
            next_frontier.difference_update(visited)
            if not next_frontier:
                break
            visited.update(next_frontier)
            frontier = next_frontier
        visited.discard(idx)
        self._reachable_cache[idx] = visited
        return visited

    def filter_anchors(self, target_idx: int, anchor_candidates: Sequence[int]) -> List[int]:
        if not self.enabled:
            return [int(idx) for idx in anchor_candidates]
        target = int(target_idx)
        # Anchor lists are small and reused across all target candidates for a
        # query. Expanding from anchors avoids a full shared-frame BFS for every
        # target object in dense scenes like HM3D 00475.
        return [int(idx) for idx in anchor_candidates if target in self.reachable(int(idx))]


def _scene_covisibility_cache(scene_state: Dict[str, Any]) -> Dict[Tuple[str, int], Any]:
    cache = scene_state.get("_spatial_covisibility_cache")
    if not isinstance(cache, dict):
        cache = {}
        scene_state["_spatial_covisibility_cache"] = cache
    return cache


def _build_covisibility_constraint(
    scene_state: Dict[str, Any],
    *,
    hops: int,
    source: str = "auto",
    verbose: bool,
) -> Optional[_CovisibilityConstraint]:
    hops = int(hops)
    if hops <= 0:
        return None
    object_count = _scene_object_count(scene_state)
    if object_count <= 0:
        return None
    source_norm = str(source or "auto").strip().lower()
    if source_norm not in {"auto", "bitset", "covisibility_adj_u64", "shared_images", "image_ids"}:
        source_norm = "auto"

    cache = _scene_covisibility_cache(scene_state)
    adjacency: Optional[List[Set[int]]] = None
    shared_lookup: Optional[Tuple[List[Set[int]], Dict[int, Set[int]]]] = None
    source_used = ""
    if source_norm in {"shared_images", "image_ids"}:
        key = ("shared_images", object_count)
        shared_lookup = cache.get(key)
        if shared_lookup is None:
            shared_lookup = _shared_image_lookup(scene_state, object_count)
            cache[key] = shared_lookup
        source_used = "shared_image_ids"
    elif source_norm in {"bitset", "covisibility_adj_u64"}:
        key = ("covisibility_adj_u64", object_count)
        adjacency = cache.get(key)
        if adjacency is None:
            adjacency = _adjacency_from_covisibility_bitset(scene_state, object_count)
            cache[key] = adjacency
        source_used = "covisibility_adj_u64"
    else:
        key = ("shared_images", object_count)
        shared_lookup = cache.get(key)
        if shared_lookup is None:
            shared_lookup = _shared_image_lookup(scene_state, object_count)
            cache[key] = shared_lookup
        source_used = "shared_image_ids"
        if shared_lookup is None:
            key = ("covisibility_adj_u64", object_count)
            adjacency = cache.get(key)
            if adjacency is None:
                adjacency = _adjacency_from_covisibility_bitset(scene_state, object_count)
                cache[key] = adjacency
            source_used = "covisibility_adj_u64"
    if shared_lookup is not None:
        object_images, image_to_objects = shared_lookup
        constraint = _SharedImageCovisibilityConstraint(object_images, image_to_objects, hops)
        if not constraint.enabled:
            if verbose:
                logger.info("Covisibility %d-hop constraint disabled: shared-image graph has no edges", hops)
            return None
        if verbose:
            degrees = np.asarray([len(row) for row in object_images], dtype=np.float32)
            logger.info(
                "Covisibility %d-hop constraint enabled from %s: objects=%d images=%d mean_object_frames=%.2f",
                hops,
                source_used,
                object_count,
                len(image_to_objects),
                float(degrees.mean()) if degrees.size else 0.0,
            )
        return constraint
    if adjacency is None:
        if verbose:
            logger.info("Covisibility %d-hop constraint disabled: no graph available", hops)
        return None
    constraint = _CovisibilityConstraint(adjacency, hops)
    if not constraint.enabled:
        if verbose:
            logger.info("Covisibility %d-hop constraint disabled: graph has no edges", hops)
        return None
    if verbose:
        degrees = np.asarray([len(row) for row in adjacency], dtype=np.float32)
        logger.info(
            "Covisibility %d-hop constraint enabled from %s: objects=%d mean_degree=%.2f",
            hops,
            source_used,
            object_count,
            float(degrees.mean()) if degrees.size else 0.0,
        )
    return constraint

_CLASS_FILTER_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "back",
    "black",
    "blue",
    "brown",
    "choose",
    "dark",
    "front",
    "glass",
    "gray",
    "green",
    "is",
    "large",
    "left",
    "light",
    "near",
    "nearest",
    "next",
    "object",
    "of",
    "on",
    "one",
    "pink",
    "right",
    "silver",
    "small",
    "target",
    "the",
    "to",
    "white",
    "with",
    "wood",
    "wooden",
    "yellow",
}

_CLASS_FILTER_GENERIC_CATEGORIES = {
    "none",
    "object",
    "other",
    "target",
    "thing",
    "unknown",
}

_CLASS_FILTER_SYNONYMS = {
    "armchair": {"armchair", "arm chair", "chair"},
    "backpack": {"backpack", "bag"},
    "briefcase": {"briefcase", "case", "bag"},
    "cabinet": {"cabinet", "cabinets", "drawer", "cupboard"},
    "cabinets": {"cabinet", "cabinets", "drawer", "cupboard"},
    "chair": {"chair", "armchair", "arm chair", "seat"},
    "conference table": {"conference table", "table"},
    "couch": {"couch", "sofa"},
    "desk": {"desk", "table"},
    "dining table": {"dining table", "table"},
    "door": {"door", "doors", "double door", "glass door", "front door"},
    "doors": {"door", "doors", "double door", "glass door", "front door"},
    "kitchen cabinets": {"kitchen cabinet", "kitchen cabinets", "cabinet", "cabinets"},
    "laptop": {"laptop", "computer"},
    "ottoman": {"ottoman", "footstool"},
    "painting": {"painting", "picture", "picture frame", "photo", "artwork", "wall art"},
    "picture": {"picture", "painting", "picture frame", "photo", "artwork", "wall art"},
    "picture frame": {"picture", "painting", "picture frame", "photo", "artwork", "wall art"},
    "refrigerator": {"refrigerator", "fridge"},
    "shelf": {"shelf", "shelves", "bookcase"},
    "table": {"table", "desk", "countertop", "counter"},
    "trash can": {"trash can", "trashcan", "garbage can", "bin"},
    "trashcan": {"trash can", "trashcan", "garbage can", "bin"},
    "window": {"window", "windows"},
}


def _encode_query(embedder: EmbedInterface, text: str) -> np.ndarray:
    """Encode query text using the retrieval query side when available."""
    fn = getattr(embedder, "encode_query", None)
    if callable(fn):
        return np.asarray(fn(text), dtype=np.float32)
    return np.asarray(embedder.encode(text), dtype=np.float32)


def _encode_document(embedder: EmbedInterface, text: str) -> np.ndarray:
    """Encode document-side text when the embedder exposes retrieval wrappers."""
    fn = getattr(embedder, "encode_document", None)
    if callable(fn):
        return np.asarray(fn(text), dtype=np.float32)
    return np.asarray(embedder.encode(text), dtype=np.float32)


def _cosine_or_none(a: Any, b: Any) -> Optional[float]:
    """Cosine similarity for embedding vectors; ``None`` on invalid/mismatched dims."""
    va = np.asarray(a, dtype=np.float32).reshape(-1)
    vb = np.asarray(b, dtype=np.float32).reshape(-1)
    if va.size == 0 or vb.size == 0 or va.shape[0] != vb.shape[0]:
        return None
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 0.0:
        return None
    return float(np.dot(va, vb) / (denom + 1e-8))


def _normalize_class_text(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _singularize_class_token(token: str) -> str:
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _target_class_terms(target_description: str) -> List[str]:
    norm = _normalize_class_text(target_description)
    if not norm or norm in {"target", "object", "one"}:
        return []
    tokens = [
        _singularize_class_token(tok)
        for tok in norm.split()
        if tok and tok not in _CLASS_FILTER_STOPWORDS
    ]
    if not tokens:
        return []
    terms: List[str] = [norm]
    if tokens:
        terms.extend(tokens)
        terms.append(" ".join(tokens))
    for key, values in _CLASS_FILTER_SYNONYMS.items():
        if norm == key or norm in values or any(tok in values for tok in tokens):
            terms.extend(values)

    out: List[str] = []
    seen: set[str] = set()
    for term in terms:
        clean = _normalize_class_text(term)
        if not clean or clean in _CLASS_FILTER_STOPWORDS:
            continue
        if clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


def _text_matches_any_class_term(text: Any, terms: Iterable[str]) -> bool:
    norm = _normalize_class_text(text)
    if not norm:
        return False
    padded = f" {norm} "
    for term in terms:
        t = _normalize_class_text(term)
        if not t:
            continue
        if f" {t} " in padded:
            return True
        parts = [p for p in t.split() if p and p not in _CLASS_FILTER_STOPWORDS]
        if parts and all(f" {p} " in padded for p in parts):
            return True
    return False


def _category_is_informative(text: Any) -> bool:
    norm = _normalize_class_text(text)
    return bool(norm) and norm not in _CLASS_FILTER_GENERIC_CATEGORIES


def _target_class_filter_candidates(
    target_description: str,
    candidates: List[int],
    scene_state: Dict[str, Any],
) -> Tuple[List[int], List[str]]:
    """Keep candidates whose category/caption matches the parsed target class."""

    terms = _target_class_terms(target_description)
    if not terms:
        return [], []
    captions = scene_state.get("object_caption", []) or []
    categories = scene_state.get("object_category", []) or []
    filtered: List[int] = []
    for idx in candidates:
        category: Any = None
        if 0 <= int(idx) < len(categories):
            category = categories[int(idx)]
        if _category_is_informative(category):
            if _text_matches_any_class_term(category, terms):
                filtered.append(int(idx))
            continue

        texts: List[Any] = []
        if 0 <= int(idx) < len(captions):
            texts.append(captions[int(idx)])
        if any(_text_matches_any_class_term(text, terms) for text in texts):
            filtered.append(int(idx))
    return filtered, terms


def _target_class_match_per_candidate(
    target_description: str,
    candidates: List[int],
    scene_state: Dict[str, Any],
    *,
    class_match_source: str = "tiered",
    class_match_tiered_conf_threshold: Optional[float] = None,
) -> Tuple[Dict[int, bool], List[str]]:
    """Per-candidate boolean: does its category/caption match the target class?

    Variant of :func:`_target_class_filter_candidates` that returns a per-id
    match flag instead of filtering, used by the soft-class composition path
    in :func:`execute_spatial_query` for `unified_soft_w50_*` profiles.

    Returns ``({obj_id: matches}, terms)``. When ``terms`` is empty (no
    target class could be parsed from ``target_description``), all candidates
    are reported as matching — there's no information to use class factor.

    ``class_match_source`` (Track B, 2026-05-17) selects which candidate signal
    to match against:
      - ``"tiered"`` (default): YOLOE ``object_category`` if informative,
        else fall back to ``object_caption``. Current behavior.
      - ``"yoloe"``: only the YOLOE ``object_category``; no caption fallback.
      - ``"caption"``: only the VLM ``object_caption``; skip the category.
      - ``"both"``: match if EITHER the category OR caption contains a term.
    """

    terms = _target_class_terms(target_description)
    if not terms:
        return {int(idx): True for idx in candidates}, []
    captions = scene_state.get("object_caption", []) or []
    categories = scene_state.get("object_category", []) or []
    det_conf_list = scene_state.get("object_detection_category_conf", []) or []
    out: Dict[int, bool] = {}
    src = str(class_match_source or "tiered").strip().lower()
    if src not in {"tiered", "yoloe", "caption", "both"}:
        raise ValueError(
            f"unknown class_match_source={class_match_source!r}; valid: "
            "tiered, yoloe, caption, both"
        )
    conf_thr: Optional[float] = (
        float(class_match_tiered_conf_threshold)
        if class_match_tiered_conf_threshold is not None
        else None
    )
    for idx in candidates:
        oid = int(idx)
        category: Any = None
        if 0 <= oid < len(categories):
            category = categories[oid]
        caption: Any = None
        if 0 <= oid < len(captions):
            caption = captions[oid]
        cat_hit = (
            _category_is_informative(category)
            and _text_matches_any_class_term(category, terms)
        )
        cap_hit = bool(caption) and _text_matches_any_class_term(caption, terms)
        if src == "yoloe":
            out[oid] = bool(cat_hit)
        elif src == "caption":
            out[oid] = bool(cap_hit)
        elif src == "both":
            out[oid] = bool(cat_hit or cap_hit)
        else:  # tiered
            yoloe_is_trustworthy = _category_is_informative(category)
            if yoloe_is_trustworthy and conf_thr is not None:
                # Lookup conf for the chosen category.
                conf_dict: Dict[str, float] = (
                    det_conf_list[oid] if 0 <= oid < len(det_conf_list)
                    and isinstance(det_conf_list[oid], dict) else {}
                )
                # Try exact match first, then case-normalized.
                conf = conf_dict.get(str(category)) if category is not None else None
                if conf is None and category is not None:
                    norm_cat = _normalize_class_text(category)
                    for k, v in conf_dict.items():
                        if _normalize_class_text(k) == norm_cat:
                            conf = float(v)
                            break
                if conf is None or float(conf) < conf_thr:
                    yoloe_is_trustworthy = False
            if yoloe_is_trustworthy:
                out[oid] = bool(cat_hit)
            else:
                out[oid] = bool(cap_hit)
    return out, terms


def _to_hwc_rgb_uint8(image_payload: Any, encoding: str = "") -> Optional[np.ndarray]:
    try:
        arr = image_payload.detach().to("cpu", copy=False).numpy() if hasattr(image_payload, "detach") else np.asarray(image_payload)
    except Exception:
        return None
    if not isinstance(arr, np.ndarray):
        return None
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        return None
    if arr.shape[-1] in {3, 4}:
        hwc = arr
    elif arr.shape[0] in {3, 4}:
        hwc = np.transpose(arr, (1, 2, 0))
    else:
        return None
    if hwc.shape[-1] == 4:
        hwc = hwc[..., :3]
    if hwc.dtype != np.uint8:
        with np.errstate(all="ignore"):
            vmax = float(np.nanmax(hwc)) if hwc.size else 0.0
        if vmax <= 1.0:
            hwc = hwc * 255.0
        hwc = np.clip(hwc, 0, 255).astype(np.uint8)
    else:
        hwc = np.ascontiguousarray(hwc)
    if str(encoding or "").strip().lower() in {"bgr8", "bgr", "bgra8", "bgra"}:
        hwc = hwc[..., [2, 1, 0]]
    return np.ascontiguousarray(hwc)


def _bbox_area(raw_bbox: Any, size: Any) -> float:
    if not isinstance(raw_bbox, (list, tuple, np.ndarray)) or len(raw_bbox) < 4:
        return 0.0
    try:
        x0, y0, x1, y1 = [float(raw_bbox[i]) for i in range(4)]
        w_img, h_img = float(size[0]), float(size[1])
    except Exception:
        return 0.0
    if not np.isfinite([x0, y0, x1, y1, w_img, h_img]).all() or w_img <= 0.0 or h_img <= 0.0:
        return 0.0
    w = max(0.0, min(w_img, x1) - max(0.0, x0))
    h = max(0.0, min(h_img, y1) - max(0.0, y0))
    return float(w * h)


def _object_caption_crop_rgb(scene_state: Dict[str, Any], object_index: int) -> Optional[np.ndarray]:
    obs_rows = scene_state.get("rgb_observations") or []
    if int(object_index) < 0 or int(object_index) >= len(obs_rows):
        return None
    obs_list = obs_rows[int(object_index)]
    obs_candidates = list(obs_list) if isinstance(obs_list, (list, tuple)) else [obs_list]
    best_img: Optional[np.ndarray] = None
    best_area = -1.0
    for obs in obs_candidates:
        image_payload = obs
        encoding = ""
        bbox = None
        size = None
        if isinstance(obs, dict):
            image_payload = obs.get("image_caption")
            if image_payload is None:
                image_payload = obs.get("image")
            encoding = str(obs.get("encoding", "") or obs.get("color_encoding", "") or "")
            bbox = obs.get("bbox_caption", obs.get("bbox"))
            size = obs.get("size_caption", obs.get("size"))
        rgb = _to_hwc_rgb_uint8(image_payload, encoding=encoding)
        if rgb is None:
            continue
        area = _bbox_area(bbox, size or (rgb.shape[1], rgb.shape[0]))
        if area <= 0.0:
            area = float(rgb.shape[0] * rgb.shape[1])
        if area > best_area:
            best_area = area
            best_img = rgb
    return best_img


def _build_vlm_contact_sheet(crops: List[Tuple[int, np.ndarray]], *, tile_size: int = 160) -> Optional[np.ndarray]:
    if not crops:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception:
        return None

    n = len(crops)
    cols = min(5, max(1, int(math.ceil(math.sqrt(n)))))
    rows = int(math.ceil(n / cols))
    tile = max(96, int(tile_size))
    label_h = 28
    gap = 6
    canvas_w = cols * tile + (cols + 1) * gap
    canvas_h = rows * (tile + label_h) + (rows + 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for slot, (candidate_id, arr) in enumerate(crops):
        row = slot // cols
        col = slot % cols
        x = gap + col * (tile + gap)
        y = gap + row * (tile + label_h + gap)
        img = Image.fromarray(arr, mode="RGB")
        img.thumbnail((tile, tile), Image.Resampling.BILINEAR)
        ox = x + (tile - img.width) // 2
        oy = y + label_h + (tile - img.height) // 2
        draw.rectangle([x, y, x + tile - 1, y + label_h - 1], fill=(20, 20, 20))
        draw.text((x + 8, y + 4), f"{candidate_id}", fill=(255, 255, 255), font=font)
        canvas.paste(img, (ox, oy))
        draw.rectangle([x, y, x + tile - 1, y + tile + label_h - 1], outline=(20, 20, 20), width=2)
    return np.asarray(canvas, dtype=np.uint8)


def _parse_vlm_rerank_scores(text: str) -> Dict[int, float]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    payload: Any
    try:
        payload = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
        except Exception:
            return {}
    rows = payload.get("scores") if isinstance(payload, dict) else None
    if rows is None and isinstance(payload, dict):
        rows = payload.get("candidates")
    if not isinstance(rows, list):
        return {}
    out: Dict[int, float] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        try:
            cid = int(item.get("id", item.get("candidate_id")))
            score = float(item.get("score", item.get("relevance", 0.0)))
        except Exception:
            continue
        if np.isfinite(score):
            out[cid] = float(max(0.0, min(1.0, score)))
    return out


def _vlm_rerank_target_candidates(
    *,
    target_description: str,
    raw_query: Optional[str],
    target_candidates: List[int],
    scene_state: Dict[str, Any],
    llm: LLMInterface,
    target_similarities: Dict[int, float],
    top_k: int,
    blend: float,
    verbose: bool,
) -> Tuple[List[int], Dict[int, float], Dict[int, float], Dict[str, Any]]:
    """Use one numbered VLM contact sheet to adjust target-class relevance.

    This happens before spatial predicate scoring. The prompt asks the VLM to
    judge only target identity and visible attributes, not spatial relations.
    """

    n = max(0, min(int(top_k), len(target_candidates)))
    if n <= 0:
        return target_candidates, target_similarities, {}, {"applied": False, "reason": "top_k_zero"}

    numbered_crops: List[Tuple[int, np.ndarray]] = []
    id_to_idx: Dict[int, int] = {}
    for display_id, idx in enumerate(target_candidates[:n], start=1):
        crop = _object_caption_crop_rgb(scene_state, int(idx))
        if crop is None:
            continue
        numbered_crops.append((display_id, crop))
        id_to_idx[display_id] = int(idx)

    if not numbered_crops:
        return target_candidates, target_similarities, {}, {"applied": False, "reason": "no_candidate_crops"}

    sheet = _build_vlm_contact_sheet(numbered_crops)
    if sheet is None:
        return target_candidates, target_similarities, {}, {"applied": False, "reason": "contact_sheet_failed"}

    target_text = " ".join(str(target_description or "").strip().split())
    full_query = " ".join(str(raw_query or "").strip().split())
    prompt = (
        "You are reranking object candidates before spatial reasoning.\n"
        "The image is a contact sheet of numbered candidate object crops.\n"
        "Score whether each numbered crop is the target object class and visible attributes.\n"
        "Do not use left/right/near/far/between spatial relations for this score.\n"
        "Use the full query only to clarify non-spatial attributes.\n\n"
        f"TARGET DESCRIPTION: {target_text}\n"
        f"FULL QUERY: {full_query}\n\n"
        "Return strict JSON only:\n"
        '{"scores":[{"id":1,"score":0.0},{"id":2,"score":1.0}]}\n'
        "Score is 1 for a clear match, 0 for a clear non-match, and intermediate for uncertainty."
    )

    old_tokens = getattr(llm.config, "max_tokens", None)
    old_temp = getattr(llm.config, "temperature", None)
    with np.errstate(all="ignore"):
        try:
            llm.config.max_tokens = int(os.getenv("VLM_RETRIEVAL_RERANK_MAX_TOKENS", "256"))
            llm.config.temperature = float(os.getenv("VLM_RETRIEVAL_RERANK_TEMPERATURE", "0.0"))
            response = llm.query(prompt, images=sheet, json_mode=True, model=os.getenv("VLM_RETRIEVAL_RERANK_MODEL"))
        except Exception as exc:
            if verbose:
                logger.warning("VLM target rerank failed: %s", exc)
            return target_candidates, target_similarities, {}, {"applied": False, "reason": repr(exc)}
        finally:
            if old_tokens is not None:
                llm.config.max_tokens = old_tokens
            if old_temp is not None:
                llm.config.temperature = old_temp

    display_scores = _parse_vlm_rerank_scores(response)
    rerank_scores: Dict[int, float] = {}
    for display_id, idx in id_to_idx.items():
        if display_id in display_scores:
            rerank_scores[idx] = float(display_scores[display_id])

    if not rerank_scores:
        return target_candidates, target_similarities, {}, {"applied": False, "reason": "no_scores_parsed"}

    alpha = max(0.0, min(1.0, float(blend)))
    updated = dict(target_similarities)
    for idx, vlm_score in rerank_scores.items():
        old = float(target_similarities.get(idx, 0.0))
        updated[idx] = float((1.0 - alpha) * old + alpha * float(vlm_score))

    reranked = sorted(
        target_candidates,
        key=lambda idx: (
            float(updated.get(idx, 0.0)),
            float(target_similarities.get(idx, 0.0)),
        ),
        reverse=True,
    )
    meta = {
        "applied": True,
        "top_k": int(n),
        "n_with_crops": int(len(numbered_crops)),
        "n_scores": int(len(rerank_scores)),
        "blend": float(alpha),
    }
    return reranked, updated, rerank_scores, meta


_VIEW_DEPENDENT_PROXIMITY_PREDICATES = frozenset({"LeftOf", "RightOf", "InFrontOf", "Behind"})


def _anchor_pool_sim_ratio() -> float:
    """FARM_ANCHOR_POOL_SIM_RATIO=<0..1> switches anchor resolution from a
    fixed top-k to a similarity threshold: every candidate scoring at least
    ratio * top_anchor_score joins the pool (deep-retrieved, capped at 30),
    and the per-candidate proximity ordering picks which pool members are
    actually tried. Fixes count-truncation with many same-class instances
    (e.g. 18 ladders, top-5 pool, right ladder ranked #7 semantically).
    Unset/0 = off (the paper's locked behavior)."""
    raw = (os.getenv("FARM_ANCHOR_POOL_SIM_RATIO") or "").strip()
    if not raw:
        return 0.0
    try:
        val = float(raw)
    except ValueError:
        return 0.0
    return val if 0.0 < val < 1.0 else 0.0


_ANCHOR_POOL_DEEP_K = 30


def _anchor_proximity_pool_k() -> int:
    """FARM_ANCHOR_PROXIMITY_SELECTION=<pool_k> enables target-correlated
    anchor selection: anchors are resolved into a *pool_k*-deep semantic pool
    and, per target candidate, the anchors nearest to that candidate are the
    ones tried (instead of the global semantic top few). 0/unset = off (the
    paper's locked behavior). Superlatives (Closest/Farthest) are scored via
    a separate path with the global semantic anchor and are unaffected."""
    raw = (os.getenv("FARM_ANCHOR_PROXIMITY_SELECTION") or "").strip().lower()
    if not raw or raw in {"0", "off", "false", "no"}:
        return 0
    try:
        return max(0, int(float(raw)))
    except ValueError:
        return 0


def _order_anchors_by_proximity(target_idx: int, anchor_candidates: List[int], means) -> List[int]:
    """Order semantically-valid anchor candidates by distance to *target_idx*."""
    if means is None or target_idx >= len(means):
        return anchor_candidates
    target_pos = np.asarray(means[target_idx], dtype=np.float64)

    def _dist(idx: int) -> float:
        if idx >= len(means):
            return float("inf")
        d = float(np.linalg.norm(np.asarray(means[idx], dtype=np.float64) - target_pos))
        return d if math.isfinite(d) else float("inf")

    return sorted(anchor_candidates, key=_dist)


def _directional_proximity_tau() -> Optional[object]:
    raw = (os.getenv("FARM_DIRECTIONAL_PROXIMITY_TAU") or "").strip().lower()
    if not raw or raw in {"0", "off", "false", "no"}:
        return None
    if raw == "adaptive":
        return "adaptive"
    try:
        val = float(raw)
    except ValueError:
        return None
    return val if val > 0 else None


def _apply_directional_proximity(scored, evaluator, query_graph) -> None:
    """Pool-relative proximity prior for view-dependent directional predicates.

    "the A left of B" pragmatically means the *closest* A that is left of B,
    so among candidates whose directional predicate held, the nearest
    candidate→matched-anchor pair keeps its score and farther pairs decay by
    exp(-(d - d_min)/tau). Absolute distance is never penalised on its own —
    a lone satisfying pair keeps full score however far apart it is, and the
    prior is scale-invariant across room- and warehouse-sized scenes.

    Enabled via FARM_DIRECTIONAL_PROXIMITY_TAU=<metres|adaptive>; unset
    (default) leaves the paper's locked scoring untouched. "adaptive" uses
    tau = max(1 m, 0.35 * d_min).
    """
    tau_cfg = _directional_proximity_tau()
    if tau_cfg is None or not scored:
        return
    means = getattr(evaluator, "_means", None)
    if means is None:
        return
    predicates = getattr(query_graph, "predicates", None) or []
    for pred in predicates:
        if pred.name not in _VIEW_DEPENDENT_PROXIMITY_PREDICATES:
            continue
        anchor_desc = pred.args[1] if len(pred.args) > 1 else ""
        entries = []
        for cand in scored:
            result = next(
                (
                    r
                    for r in (cand.predicate_results or [])
                    if r.name == pred.name and r.status != "dropped" and r.score > 0.05
                ),
                None,
            )
            if result is None:
                continue
            anchor_idx = (cand.matched_anchors or {}).get(anchor_desc)
            if anchor_idx is None:
                continue
            if cand.object_index >= len(means) or int(anchor_idx) >= len(means):
                continue
            dist = float(
                np.linalg.norm(
                    np.asarray(means[cand.object_index], dtype=np.float64)
                    - np.asarray(means[int(anchor_idx)], dtype=np.float64)
                )
            )
            if math.isfinite(dist):
                entries.append((cand, dist))
        if len(entries) < 2:
            continue  # a single satisfying pair has nothing to be preferred over
        d_min = min(d for _, d in entries)
        tau = max(1.0, 0.35 * d_min) if tau_cfg == "adaptive" else float(tau_cfg)
        for cand, dist in entries:
            cand.composite_score *= math.exp(-max(0.0, dist - d_min) / max(1e-6, tau))


def execute_spatial_query(
    query_graph: QueryGraph,
    scene_state: Dict[str, Any],
    llm: LLMInterface,
    embedder: EmbedInterface,
    *,
    region_sim_threshold: float = 0.7,
    pre_filter_k: int = 40,
    anchor_k: int = 5,
    max_candidates_for_vlm: int = 10,
    max_anchors_per_description: int = 3,
    max_output_candidates: int = 20,
    use_vlm: bool = True,
    raw_query: Optional[str] = None,
    retrieval_mode: str = "multi",
    candidate_pool_mode: str = "active",
    spatial_method: str = "current",
    predicate_calibration_path: Optional[Path | str] = None,
    predicate_calibrator: Optional[SpatialCalibrator] = None,
    hard_threshold: float = 0.5,
    vlm_rerank_enabled: bool = False,
    vlm_rerank_top_k: int = 20,
    vlm_rerank_blend: float = 0.65,
    verbose: bool = False,
) -> List[ScoredCandidate]:
    """Execute spatial query: region scope -> pre-filter -> score -> rank.

    Args:
        query_graph: Parsed query with target and predicates.
        scene_state: Scene state dict with means, captions, regions, etc.
        llm: LLM/VLM interface for predicate evaluation.
        embedder: Text embedding interface for retrieval.
        region_sim_threshold: Cosine similarity threshold for fuzzy region matching.
        pre_filter_k: Max candidates after semantic pre-filter.
        anchor_k: Max anchor candidates per description.
        max_candidates_for_vlm: Max candidates to send through VLM evaluation.
        max_anchors_per_description: Max anchors to try per description.
        max_output_candidates: Max candidates in final output (default 20).
        use_vlm: Whether to use VLM for predicate evaluation.
        raw_query: Original utterance, used as an auxiliary caption query in
            multi-embedding retrieval.
        retrieval_mode: "caption" for the legacy caption-only pre-filter, or
            "multi" for caption + SigLIP2 + Qwen3-VL RRF.
        candidate_pool_mode: "active", "all", or "active_plus_redirect".
        spatial_method: Named method profile: current, semantic_only,
        hard_predicates, soft_predicates, or calibrated_logprob.
        predicate_calibration_path: JSON calibration file for calibrated_logprob.
        predicate_calibrator: Pre-loaded calibration object; overrides path.
        hard_threshold: Predicate threshold for hard_predicates.
        vlm_rerank_enabled: Rerank top target candidates with one VLM contact sheet
            before spatial predicate scoring.
        vlm_rerank_top_k: Max target candidates shown to the reranker.
        vlm_rerank_blend: Blend between semantic target score and VLM target score.
        verbose: Enable verbose logging.

    Returns:
        Ranked list of ScoredCandidate.
    """
    if str(spatial_method or "").strip().lower() == "joint_v1":
        # Truncation-free vectorized joint retrieval (see joint_executor).
        from .joint_executor import execute_joint_v1

        return execute_joint_v1(
            query_graph,
            scene_state,
            llm,
            embedder,
            max_output_candidates=max_output_candidates,
            raw_query=raw_query or "",
            retrieval_mode=retrieval_mode,
            candidate_pool_mode=candidate_pool_mode,
            verbose=verbose,
        )

    method = get_spatial_method(spatial_method)
    use_vlm_effective = bool(use_vlm) and not bool(method.force_no_vlm)
    calibrator = predicate_calibrator
    if calibrator is None and predicate_calibration_path is not None:
        calibrator = SpatialCalibrator.load(predicate_calibration_path)
    if calibrator is None:
        calibrator = SpatialCalibrator.identity()

    evaluator = PredicateEvaluator(scene_state, llm, embedder, verbose=verbose)

    # Classify predicates by role
    target_predicates, anchor_predicates, anchor_descriptions = _classify_predicates(query_graph.predicates)

    # Separate superlative predicates from regular ones
    superlative_predicates = [p for p in target_predicates if p.name in SUPERLATIVE_PREDICATES]
    regular_predicates = [p for p in target_predicates if p.name not in SUPERLATIVE_PREDICATES]

    # Step 1: Region scoping
    candidate_pool, region_predicate_status = _region_scope(
        query_graph.predicates, scene_state, embedder, region_sim_threshold, verbose
    )

    # Step 2: Pre-filter by semantic rank fusion within scoped pool.
    # pre_filter_k <= 0 is a sentinel for "score every region-scoped candidate"
    # — the dataset-agnostic policy locked 2026-05-16.
    # We resolve the effective k against the scoped pool size so the
    # pre-filter becomes a no-op while preserving the rank-fusion ordering that
    # later spatial scoring consumes. When no region scope filtered the pool we
    # derive the pool size from the persisted scene_state by inspecting ``means``
    # (avoids ``scene_state["count"]``'s ambiguity as a torch tensor).
    effective_pre_filter_k = int(pre_filter_k)
    if effective_pre_filter_k <= 0:
        if candidate_pool is not None:
            effective_pre_filter_k = max(1, len(candidate_pool))
        else:
            means_obj = scene_state.get("means")
            try:
                n_active = int(len(means_obj)) if means_obj is not None else 0
            except TypeError:
                n_active = 0
            effective_pre_filter_k = max(1, n_active)
    target_retrieval = _pre_filter_semantic(
        query_graph.target_description,
        candidate_pool,
        scene_state,
        embedder,
        effective_pre_filter_k,
        verbose,
        raw_query=raw_query,
        retrieval_mode=retrieval_mode,
        candidate_pool_mode=candidate_pool_mode,
    )
    target_candidates = target_retrieval.ranked_indices

    if not target_candidates:
        if verbose:
            logger.info("No target candidates after pre-filter")
        return []

    # Compute target-description similarity for all candidates
    if str(retrieval_mode or "multi").strip().lower() == "multi":
        target_similarities = {
            idx: max(0.0, float(target_retrieval.fused_scores.get(idx, 0.0)))
            for idx in target_candidates
        }
    else:
        target_similarities = _compute_target_similarities(
            query_graph.target_description, target_candidates, scene_state, embedder,
        )

    # Locked 2026-05-17. Soft-class composition path: when
    # ``method.class_mismatch_floor`` is set, multiply each candidate's
    # target similarity by a per-candidate ``class_factor`` ∈
    # {class_mismatch_floor, 1.0} based on whether its category/caption
    # matches the parsed target class. No hard prune (mutually exclusive
    # with ``require_target_class_filter``). The composition becomes
    # ``score(c) = target_sim(c) × class_factor(c) × spatial_factor(c)``.
    #
    # Track B (2026-05-17): when ``query_graph.target_class`` is populated
    # by the LLM parser, prefer it over regex tokenization of
    # ``target_description``. Falls back to regex when the LLM returns
    # ``target_class = null`` (paraphrased queries the LLM is unsure about).
    if method.class_mismatch_floor is not None and target_candidates:
        floor = float(method.class_mismatch_floor)
        class_phrase = (
            getattr(query_graph, "target_class", None) or query_graph.target_description
        )
        match_map, class_terms = _target_class_match_per_candidate(
            class_phrase,
            target_candidates,
            scene_state,
            class_match_source=getattr(method, "class_match_source", "tiered"),
            class_match_tiered_conf_threshold=getattr(
                method, "class_match_tiered_conf_threshold", None
            ),
        )
        if class_terms:
            # Only apply the soft factor when a class term was actually parsed;
            # if not, the factor degenerates to 1.0 for every candidate (a no-op).
            n_match = sum(1 for v in match_map.values() if v)
            target_similarities = {
                idx: float(target_similarities.get(idx, 0.0))
                * (1.0 if match_map.get(int(idx), True) else floor)
                for idx in target_candidates
            }
            if verbose:
                logger.info(
                    "Soft-class factor applied (floor=%.2f, matched=%d/%d) on %r via terms=%s",
                    floor, n_match, len(target_candidates),
                    query_graph.target_description, class_terms,
                )

    if method.require_target_class_filter:
        original_candidate_count = len(target_candidates)
        filtered_candidates, filter_terms = _target_class_filter_candidates(
            query_graph.target_description,
            target_candidates,
            scene_state,
        )
        min_candidates = int(method.target_class_filter_min_candidates)
        if len(filtered_candidates) >= min_candidates:
            target_candidates = filtered_candidates
            target_similarities = {
                idx: float(target_similarities.get(idx, 0.0))
                for idx in target_candidates
            }
            if verbose:
                logger.info(
                    "Target-class filter kept %d/%d candidates for %r via terms=%s",
                    len(target_candidates),
                    original_candidate_count,
                    query_graph.target_description,
                    filter_terms,
                )
        else:
            if verbose:
                logger.info(
                    "Target-class filter failed for %r (kept %d/%d via terms=%s); semantic fallback",
                    query_graph.target_description,
                    len(filtered_candidates),
                    original_candidate_count,
                    filter_terms,
                )
            return _semantic_only_score_all(
                target_candidates,
                evaluator,
                target_similarities=target_similarities,
                vlm_rerank_scores={},
                max_output_candidates=max_output_candidates,
            )

    vlm_rerank_scores: Dict[int, float] = {}
    vlm_rerank_meta: Dict[str, Any] = {"applied": False, "reason": "disabled"}
    if bool(vlm_rerank_enabled):
        target_candidates, target_similarities, vlm_rerank_scores, vlm_rerank_meta = _vlm_rerank_target_candidates(
            target_description=query_graph.target_description,
            raw_query=raw_query,
            target_candidates=target_candidates,
            scene_state=scene_state,
            llm=llm,
            target_similarities=target_similarities,
            top_k=vlm_rerank_top_k,
            blend=vlm_rerank_blend,
            verbose=verbose,
        )
        if verbose:
            logger.info("VLM target rerank meta: %s", vlm_rerank_meta)

    if not method.evaluate_predicates:
        return _semantic_only_score_all(
            target_candidates,
            evaluator,
            target_similarities=target_similarities,
            vlm_rerank_scores=vlm_rerank_scores,
            max_output_candidates=max_output_candidates,
        )

    # Step 3: Resolve anchors within scoped pool. With target-correlated
    # anchor selection enabled, resolve a deeper semantic pool — the
    # per-candidate proximity ordering picks which ones actually get tried.
    resolved_anchors = _resolve_anchors(
        anchor_descriptions,
        anchor_predicates,
        candidate_pool,
        scene_state,
        evaluator,
        embedder,
        max(
            anchor_k,
            _anchor_proximity_pool_k(),
            _ANCHOR_POOL_DEEP_K if _anchor_pool_sim_ratio() > 0.0 else 0,
        ),
        verbose,
        retrieval_mode=retrieval_mode,
        candidate_pool_mode=candidate_pool_mode,
    )
    covisibility_constraint = _build_covisibility_constraint(
        scene_state,
        hops=int(getattr(method, "covisibility_hops", 0) or 0),
        source=str(getattr(method, "covisibility_source", "auto") or "auto"),
        verbose=verbose,
    )

    # Step 3.5: Evaluate superlative predicates across ALL candidates at once
    superlative_scores: Dict[int, Dict[str, float]] = {}
    if superlative_predicates:
        superlative_scores = _evaluate_superlative_predicates(
            target_candidates, superlative_predicates, resolved_anchors, evaluator, verbose
        )

    # Step 4: Fast-path scoring and pruning (regular predicates only)
    scored = _fast_path_score_all(
        target_candidates, regular_predicates, resolved_anchors, evaluator,
        region_predicate_status, max_anchors_per_description, verbose,
        target_similarities=target_similarities,
        superlative_scores=superlative_scores,
        composition=method.composition,
        predicate_weight=method.predicate_weight,
        hard_threshold=hard_threshold,
        calibrator=calibrator,
        vlm_rerank_scores=vlm_rerank_scores,
        covisibility_constraint=covisibility_constraint,
    )

    # Optional pool-relative proximity prior for directional predicates
    # (env-gated, off by default — see _apply_directional_proximity).
    _apply_directional_proximity(scored, evaluator, query_graph)

    # Sort by fast-path composite, keep top N for VLM
    scored.sort(key=lambda x: x.composite_score, reverse=True)
    vlm_candidates = scored[:max_candidates_for_vlm]

    if not use_vlm_effective:
        # When VLM is off, output is capped by max_output_candidates, NOT max_candidates_for_vlm
        return scored[:max_output_candidates]

    # Step 5: VLM evaluation on top candidates
    final = _vlm_evaluate_candidates(
        vlm_candidates, regular_predicates, resolved_anchors, evaluator,
        region_predicate_status, max_anchors_per_description, verbose,
        target_similarities=target_similarities,
        superlative_scores=superlative_scores,
        composition=method.composition,
        predicate_weight=method.predicate_weight,
        hard_threshold=hard_threshold,
        calibrator=calibrator,
        vlm_rerank_scores=vlm_rerank_scores,
        covisibility_constraint=covisibility_constraint,
    )

    final.sort(key=lambda x: x.composite_score, reverse=True)
    return final[:max_output_candidates]


def _classify_predicates(
    predicates: List[Predicate],
) -> Tuple[List[Predicate], Dict[str, List[Predicate]], Set[str]]:
    """Separate predicates into target-level and anchor-level.

    Returns:
        target_predicates: Predicates where subject is $target.
        anchor_predicates: Dict[anchor_description -> list of predicates about that anchor].
        anchor_descriptions: Set of unique non-$target entity descriptions.
    """
    target_predicates: List[Predicate] = []
    anchor_predicates: Dict[str, List[Predicate]] = {}
    anchor_descriptions: Set[str] = set()

    for pred in predicates:
        subject = pred.args[0] if pred.args else ""
        if subject == TARGET_VAR:
            target_predicates.append(pred)
            # Collect anchor descriptions from args[1:]
            for arg in pred.args[1:]:
                if arg != TARGET_VAR:
                    anchor_descriptions.add(arg)
        else:
            # This predicate constrains an anchor (e.g., Near("white table", "window"))
            anchor_predicates.setdefault(subject, []).append(pred)
            anchor_descriptions.add(subject)
            for arg in pred.args[1:]:
                if arg != TARGET_VAR:
                    anchor_descriptions.add(arg)

    return target_predicates, anchor_predicates, anchor_descriptions


def _region_scope(
    predicates: List[Predicate],
    scene_state: Dict[str, Any],
    embedder: EmbedInterface,
    sim_threshold: float,
    verbose: bool,
) -> Tuple[Optional[List[int]], Dict[str, PredicateResult]]:
    """If an InRegion predicate exists, scope the candidate pool.

    Returns:
        candidate_pool: List of object indices in the matched region, or None for all.
        region_predicate_status: Map of predicate key -> PredicateResult for resolved InRegion.
    """
    region_status: Dict[str, PredicateResult] = {}

    in_region_preds = [p for p in predicates if p.name == "InRegion"]
    if not in_region_preds:
        return None, region_status

    region_labels: List[str] = scene_state.get("region_labels", [])
    region_object_lists: List[List[int]] = scene_state.get("region_object_lists", [])

    if not region_labels or not region_object_lists:
        for pred in in_region_preds:
            region_name = pred.args[1] if len(pred.args) > 1 else ""
            key = f"InRegion:{region_name}"
            region_status[key] = PredicateResult(
                name="InRegion", score=1.0, status="dropped", drop_reason="no_regions_in_scene"
            )
        return None, region_status

    # Try to match each InRegion predicate
    matched_region_idx: Optional[int] = None
    for pred in in_region_preds:
        region_name = pred.args[1] if len(pred.args) > 1 else ""
        key = f"InRegion:{region_name}"

        # Exact match first
        exact_idx = None
        for i, label in enumerate(region_labels):
            if label.lower() == region_name.lower():
                exact_idx = i
                break

        if exact_idx is not None:
            matched_region_idx = exact_idx
            region_status[key] = PredicateResult(name="InRegion", score=1.0, status="fast_path")
            continue

        # Fuzzy match via embedding similarity
        best_sim = 0.0
        best_idx = -1
        try:
            query_emb = _encode_query(embedder, region_name)
            for i, label in enumerate(region_labels):
                label_emb = _encode_document(embedder, label)
                sim = _cosine_or_none(query_emb, label_emb)
                if sim is None:
                    continue
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
        except Exception as e:
            logger.warning("Embedding failed for region matching: %s", e)

        if best_sim >= sim_threshold and best_idx >= 0:
            matched_region_idx = best_idx
            region_status[key] = PredicateResult(name="InRegion", score=best_sim, status="fast_path")
        else:
            region_status[key] = PredicateResult(
                name="InRegion", score=1.0, status="dropped", drop_reason="region_not_found"
            )

    if matched_region_idx is not None and matched_region_idx < len(region_object_lists):
        pool = region_object_lists[matched_region_idx]
        if verbose:
            logger.info(
                "Region scoped to '%s' (%d objects)",
                region_labels[matched_region_idx],
                len(pool),
            )
        return pool, region_status

    return None, region_status


def _pre_filter(
    target_description: str,
    candidate_pool: Optional[List[int]],
    scene_state: Dict[str, Any],
    embedder: EmbedInterface,
    k: int,
    verbose: bool,
) -> List[int]:
    """Retrieve top-K candidates by embedding similarity within the pool.

    Fallback logic: if embedding-based filtering returns empty, try keyword
    matching in captions. As a last resort, return top-5 by score even if zero.
    """
    captions: List[str] = scene_state.get("object_caption", [])
    caption_embeddings = scene_state.get("object_caption_embedding", [])
    active = scene_state.get("active")

    if not captions or not caption_embeddings:
        if candidate_pool is not None:
            return candidate_pool[:k]
        n = len(captions) if captions else 0
        return list(range(min(n, k)))

    # Determine which indices to consider
    if candidate_pool is not None:
        indices = candidate_pool
    else:
        indices = list(range(len(captions)))

    # Filter to active objects
    if active is not None:
        if hasattr(active, "cpu"):
            active_np = active.cpu().numpy()
        else:
            active_np = np.asarray(active)
        indices = [i for i in indices if i < len(active_np) and active_np[i]]

    if not indices:
        # Fallback: keyword matching in captions
        all_indices = candidate_pool if candidate_pool is not None else list(range(len(captions)))
        keyword_matches = _keyword_fallback(target_description, all_indices, captions)
        if keyword_matches:
            return keyword_matches[:k]
        # Last resort: return top-5 from full pool
        return all_indices[:5]

    # Embed query
    try:
        query_emb = _encode_query(embedder, target_description)
    except Exception:
        return indices[:k]

    # Score each candidate
    scored: List[Tuple[int, float]] = []
    for idx in indices:
        if idx >= len(caption_embeddings) or not caption_embeddings[idx]:
            scored.append((idx, 0.0))
            continue
        emb = np.asarray(caption_embeddings[idx], dtype=np.float32)
        sim = _cosine_or_none(query_emb, emb)
        scored.append((idx, 0.0 if sim is None else sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    result = [idx for idx, _ in scored[:k]]

    if not result:
        # Fallback: keyword matching
        keyword_matches = _keyword_fallback(target_description, indices, captions)
        if keyword_matches:
            return keyword_matches[:k]
        # Last resort: return top-5 by score even if zero
        return [idx for idx, _ in scored[:5]]

    if verbose:
        logger.info("Pre-filter: %d candidates -> top %d", len(indices), len(result))

    return result


def _pre_filter_semantic(
    target_description: str,
    candidate_pool: Optional[List[int]],
    scene_state: Dict[str, Any],
    embedder: EmbedInterface,
    k: int,
    verbose: bool,
    *,
    raw_query: Optional[str] = None,
    retrieval_mode: str = "multi",
    candidate_pool_mode: str = "active",
) -> SemanticRetrievalResult:
    """Retrieve top candidates with explicit pool and embedding-channel mode."""

    mode = str(retrieval_mode or "multi").strip().lower()
    if mode in {"caption", "legacy", "caption_only"}:
        if str(candidate_pool_mode or "active").strip().lower() == "active":
            ranked = _pre_filter(target_description, candidate_pool, scene_state, embedder, k, verbose)
            scores = _compute_target_similarities(target_description, ranked, scene_state, embedder)
            return SemanticRetrievalResult(
                ranked_indices=ranked,
                fused_scores={idx: float(scores.get(idx, 0.0)) for idx in ranked},
                candidate_indices=[],
                channels={},
                mode="caption",
                candidate_pool_mode="active",
            )
        return retrieve_semantic_candidates(
            target_description,
            scene_state,
            embedder,
            raw_query=None,
            candidate_pool=candidate_pool,
            k=k,
            channel_k=max(60, int(k)),
            mode="caption",
            candidate_pool_mode=candidate_pool_mode,
            verbose=verbose,
        )

    result = retrieve_semantic_candidates(
        target_description,
        scene_state,
        embedder,
        raw_query=raw_query,
        candidate_pool=candidate_pool,
        k=k,
        channel_k=max(60, int(k)),
        mode="multi",
        candidate_pool_mode=candidate_pool_mode,
        verbose=verbose,
    )
    if verbose:
        enabled = [name for name, ch in result.channels.items() if ch.enabled]
        disabled = {name: ch.error for name, ch in result.channels.items() if not ch.enabled}
        logger.info(
            "Semantic pre-filter: pool=%s mode=%s candidates=%d top=%d enabled=%s disabled=%s",
            result.candidate_pool_mode,
            result.mode,
            len(result.candidate_indices),
            len(result.ranked_indices),
            enabled,
            disabled,
        )
    return result


def _keyword_fallback(
    target_description: str,
    indices: List[int],
    captions: List[str],
) -> List[int]:
    """Fall back to keyword matching in captions when embedding retrieval fails."""
    keywords = target_description.lower().split()
    matches: List[int] = []
    for idx in indices:
        if idx >= len(captions):
            continue
        caption_lower = captions[idx].lower()
        if any(kw in caption_lower for kw in keywords):
            matches.append(idx)
    return matches


def _resolve_anchors(
    anchor_descriptions: Set[str],
    anchor_predicates: Dict[str, List[Predicate]],
    candidate_pool: Optional[List[int]],
    scene_state: Dict[str, Any],
    evaluator: PredicateEvaluator,
    embedder: EmbedInterface,
    k: int,
    verbose: bool,
    *,
    retrieval_mode: str = "multi",
    candidate_pool_mode: str = "active",
) -> Dict[str, List[int]]:
    """Resolve each anchor description to candidate object indices."""
    resolved: Dict[str, List[int]] = {}

    sim_ratio = _anchor_pool_sim_ratio()
    for desc in anchor_descriptions:
        # Retrieve candidates for this anchor
        anchor_retrieval = _pre_filter_semantic(
            desc,
            candidate_pool,
            scene_state,
            embedder,
            k,
            verbose=False,
            retrieval_mode=retrieval_mode,
            candidate_pool_mode=candidate_pool_mode,
        )
        anchor_candidates = anchor_retrieval.ranked_indices
        # Similarity-threshold pooling (env-gated): keep everything scoring at
        # least ratio * top instead of trusting the fixed top-k order, so a
        # same-class instance with a modest caption still enters the pool.
        if sim_ratio > 0.0 and anchor_candidates:
            fused = getattr(anchor_retrieval, "fused_scores", None) or {}
            top_score = max((float(fused.get(i, 0.0)) for i in anchor_candidates), default=0.0)
            if top_score > 0.0:
                thresholded = [
                    i for i in anchor_candidates if float(fused.get(i, 0.0)) >= sim_ratio * top_score
                ]
                if thresholded:
                    anchor_candidates = thresholded

        # If anchor has its own predicates, evaluate fast-path to prune
        if desc in anchor_predicates:
            surviving = []
            for idx in anchor_candidates:
                ok = True
                for pred in anchor_predicates[desc]:
                    fast_kwargs: Dict[str, Any] = {}
                    if pred.name == "HasAttribute" and len(pred.args) > 1:
                        fast_kwargs["attribute"] = pred.args[1]
                    elif pred.name == "IsCategory" and len(pred.args) > 1:
                        fast_kwargs["category"] = pred.args[1]
                    score = evaluator.fast_path(pred.name, idx, **fast_kwargs)
                    if score is not None and score < 0.05:
                        ok = False
                        break
                if ok:
                    surviving.append(idx)
            anchor_candidates = surviving if surviving else anchor_candidates[:1]

        resolved[desc] = anchor_candidates
        if verbose:
            logger.info("Anchor '%s': resolved to %d candidates", desc, len(anchor_candidates))

    return resolved


def _compute_target_similarities(
    target_description: str,
    target_candidates: List[int],
    scene_state: Dict[str, Any],
    embedder: EmbedInterface,
) -> Dict[int, float]:
    """Compute embedding similarity between target description and each candidate."""
    caption_embeddings = scene_state.get("object_caption_embedding", [])
    if not caption_embeddings:
        return {}

    try:
        query_emb = _encode_query(embedder, target_description)
    except Exception:
        return {}

    similarities: Dict[int, float] = {}
    for idx in target_candidates:
        if idx >= len(caption_embeddings) or not caption_embeddings[idx]:
            similarities[idx] = 0.0
            continue
        emb = np.asarray(caption_embeddings[idx], dtype=np.float32)
        sim = _cosine_or_none(query_emb, emb)
        similarities[idx] = 0.0 if sim is None else max(0.0, sim)

    return similarities


def _evaluate_superlative_predicates(
    target_candidates: List[int],
    superlative_predicates: List[Predicate],
    resolved_anchors: Dict[str, List[int]],
    evaluator: PredicateEvaluator,
    verbose: bool,
) -> Dict[int, Dict[str, float]]:
    """Evaluate superlative (rank-based) predicates across ALL candidates simultaneously.

    Ranks all candidates by distance to anchor and assigns 1/(rank+1) scores.
    For Closest: rank 0 = nearest -> score 1.0, rank 1 -> 0.5, rank 2 -> 0.33, ...
    For Farthest: rank 0 = farthest -> score 1.0, rank 1 -> 0.5, ...
    If ``predicate.kwargs["rank"]`` is provided, scores peak at that zero-based
    ordinal rank. This covers IRef-VLA labels like second_closest/third_farthest.

    Returns:
        Dict[candidate_idx -> Dict[predicate_key -> score]]
    """
    scores: Dict[int, Dict[str, float]] = {idx: {} for idx in target_candidates}

    for pred in superlative_predicates:
        anchor_desc = pred.args[1] if len(pred.args) > 1 else ""
        anchor_candidates = resolved_anchors.get(anchor_desc, [])
        try:
            desired_rank = max(0, int(pred.kwargs.get("rank", 0)))
        except Exception:
            desired_rank = 0
        pred_key = f"{pred.name}:{anchor_desc}:rank={desired_rank}"

        if not anchor_candidates:
            # Cannot evaluate; give neutral score
            for idx in target_candidates:
                scores[idx][pred_key] = 1.0
            continue

        # Use the best (first) anchor
        anchor_idx = anchor_candidates[0]
        anchor_pos = evaluator._get_position(anchor_idx)
        if anchor_pos is None:
            for idx in target_candidates:
                scores[idx][pred_key] = 1.0
            continue

        # Compute distances from each candidate to anchor
        distances: List[Tuple[int, float]] = []
        for idx in target_candidates:
            if idx == anchor_idx:
                continue
            pos = evaluator._get_position(idx)
            if pos is None:
                distances.append((idx, float("inf")))
            else:
                dist = float(np.linalg.norm(pos - anchor_pos))
                distances.append((idx, dist))

        if not distances:
            continue

        # Sort by distance
        if pred.name == "Closest":
            distances.sort(key=lambda x: x[1])  # ascending: nearest first
        else:  # Farthest
            distances.sort(key=lambda x: x[1], reverse=True)  # descending: farthest first

        # Assign rank-based scores. Standard closest/farthest peaks at rank 0;
        # ordinal variants peak at their requested zero-based rank.
        for rank, (idx, _dist) in enumerate(distances):
            scores[idx][pred_key] = 1.0 / (abs(rank - desired_rank) + 1)

        # The anchor itself gets score 0 for Closest (it IS the anchor)
        if anchor_idx in scores:
            scores[anchor_idx][pred_key] = 0.0

        if verbose:
            top3 = distances[:3]
            logger.info(
                "Superlative %s (anchor=%s): top-3 = %s",
                pred.name, anchor_desc,
                [(idx, f"{d:.2f}m") for idx, d in top3],
            )

    return scores


def _semantic_only_score_all(
    target_candidates: List[int],
    evaluator: PredicateEvaluator,
    *,
    target_similarities: Optional[Dict[int, float]] = None,
    vlm_rerank_scores: Optional[Dict[int, float]] = None,
    max_output_candidates: int = 20,
) -> List[ScoredCandidate]:
    """Return semantic retrieval ranking as ScoredCandidate objects."""

    object_ids = evaluator.scene_state.get("object_id")
    results: List[ScoredCandidate] = []
    for target_idx in target_candidates:
        target_sim = target_similarities.get(target_idx, 0.0) if target_similarities else 0.0
        obj_id = int(object_ids[target_idx]) if object_ids is not None and target_idx < len(object_ids) else target_idx
        results.append(
            ScoredCandidate(
                object_index=target_idx,
                object_id=obj_id,
                predicate_results=[],
                composite_score=float(target_sim),
                matched_anchors={},
                target_similarity=float(target_sim),
                vlm_rerank_score=(
                    float(vlm_rerank_scores[target_idx])
                    if vlm_rerank_scores and target_idx in vlm_rerank_scores
                    else None
                ),
                predicate_geo_mean=None,
                predicate_weight=0.0,
            )
        )
    results.sort(key=lambda x: x.composite_score, reverse=True)
    return results[:max_output_candidates]


def _fast_path_score_all(
    target_candidates: List[int],
    target_predicates: List[Predicate],
    resolved_anchors: Dict[str, List[int]],
    evaluator: PredicateEvaluator,
    region_predicate_status: Dict[str, PredicateResult],
    max_anchors: int,
    verbose: bool,
    target_similarities: Optional[Dict[int, float]] = None,
    superlative_scores: Optional[Dict[int, Dict[str, float]]] = None,
    composition: str = "current",
    predicate_weight: Optional[float] = None,
    hard_threshold: float = 0.5,
    calibrator: Optional[SpatialCalibrator] = None,
    vlm_rerank_scores: Optional[Dict[int, float]] = None,
    covisibility_constraint: Optional[_CovisibilityConstraint] = None,
) -> List[ScoredCandidate]:
    """Score all target candidates using fast-path only."""
    results: List[ScoredCandidate] = []
    object_ids = evaluator.scene_state.get("object_id")

    for target_idx in target_candidates:
        pred_results, matched_anchors = _evaluate_predicates_for_candidate(
            target_idx, target_predicates, resolved_anchors, evaluator,
            region_predicate_status, max_anchors, use_vlm=False,
            covisibility_constraint=covisibility_constraint,
        )

        # Inject superlative predicate scores
        if superlative_scores and target_idx in superlative_scores:
            for pred_key, sup_score in superlative_scores[target_idx].items():
                pred_name = pred_key.split(":")[0]
                pred_results.append(PredicateResult(
                    name=pred_name, score=sup_score, status="fast_path",
                ))

        target_sim = target_similarities.get(target_idx, 0.5) if target_similarities else 0.5
        composite = _composite_score(
            pred_results,
            target_sim,
            composition=composition,
            predicate_weight=predicate_weight,
            hard_threshold=hard_threshold,
            calibrator=calibrator,
        )
        geo_mean = _predicate_geo_mean([r.score for r in pred_results])

        obj_id = int(object_ids[target_idx]) if object_ids is not None and target_idx < len(object_ids) else target_idx
        results.append(ScoredCandidate(
            object_index=target_idx,
            object_id=obj_id,
            predicate_results=pred_results,
            composite_score=composite,
            matched_anchors=matched_anchors,
            target_similarity=float(target_sim),
            vlm_rerank_score=(
                float(vlm_rerank_scores[target_idx])
                if vlm_rerank_scores and target_idx in vlm_rerank_scores
                else None
            ),
            predicate_geo_mean=geo_mean,
            predicate_weight=0.25 if predicate_weight is None else float(predicate_weight),
        ))

    return results


def _vlm_evaluate_candidates(
    candidates: List[ScoredCandidate],
    target_predicates: List[Predicate],
    resolved_anchors: Dict[str, List[int]],
    evaluator: PredicateEvaluator,
    region_predicate_status: Dict[str, PredicateResult],
    max_anchors: int,
    verbose: bool,
    target_similarities: Optional[Dict[int, float]] = None,
    superlative_scores: Optional[Dict[int, Dict[str, float]]] = None,
    composition: str = "current",
    predicate_weight: Optional[float] = None,
    hard_threshold: float = 0.5,
    calibrator: Optional[SpatialCalibrator] = None,
    vlm_rerank_scores: Optional[Dict[int, float]] = None,
    covisibility_constraint: Optional[_CovisibilityConstraint] = None,
) -> List[ScoredCandidate]:
    """Re-evaluate top candidates with VLM for predicates that need it."""
    results: List[ScoredCandidate] = []

    for candidate in candidates:
        pred_results, matched_anchors = _evaluate_predicates_for_candidate(
            candidate.object_index, target_predicates, resolved_anchors, evaluator,
            region_predicate_status, max_anchors, use_vlm=True,
            covisibility_constraint=covisibility_constraint,
        )

        # Inject superlative predicate scores
        if superlative_scores and candidate.object_index in superlative_scores:
            for pred_key, sup_score in superlative_scores[candidate.object_index].items():
                pred_name = pred_key.split(":")[0]
                pred_results.append(PredicateResult(
                    name=pred_name, score=sup_score, status="fast_path",
                ))

        target_sim = target_similarities.get(candidate.object_index, 0.5) if target_similarities else 0.5
        composite = _composite_score(
            pred_results,
            target_sim,
            composition=composition,
            predicate_weight=predicate_weight,
            hard_threshold=hard_threshold,
            calibrator=calibrator,
        )
        geo_mean = _predicate_geo_mean([r.score for r in pred_results])

        results.append(ScoredCandidate(
            object_index=candidate.object_index,
            object_id=candidate.object_id,
            predicate_results=pred_results,
            composite_score=composite,
            matched_anchors=matched_anchors,
            target_similarity=float(target_sim),
            vlm_rerank_score=(
                float(vlm_rerank_scores[candidate.object_index])
                if vlm_rerank_scores and candidate.object_index in vlm_rerank_scores
                else None
            ),
            predicate_geo_mean=geo_mean,
            predicate_weight=0.25 if predicate_weight is None else float(predicate_weight),
        ))

    return results


def _evaluate_predicates_for_candidate(
    target_idx: int,
    target_predicates: List[Predicate],
    resolved_anchors: Dict[str, List[int]],
    evaluator: PredicateEvaluator,
    region_predicate_status: Dict[str, PredicateResult],
    max_anchors: int,
    use_vlm: bool,
    covisibility_constraint: Optional[_CovisibilityConstraint] = None,
) -> Tuple[List[PredicateResult], Dict[str, int]]:
    """Evaluate all target predicates for a single candidate.

    Tries different anchor assignments and keeps the best scoring combination.
    """
    pred_results: List[PredicateResult] = []
    matched_anchors: Dict[str, int] = {}

    for pred in target_predicates:
        if pred.name == "InRegion":
            region_name = pred.args[1] if len(pred.args) > 1 else ""
            key = f"InRegion:{region_name}"
            if key in region_predicate_status:
                pred_results.append(region_predicate_status[key])
                continue
            # Evaluate directly
            result = evaluator.evaluate(
                "InRegion", target_idx, region=region_name, use_vlm=use_vlm,
            )
            pred_results.append(result)
            continue

        if pred.name == "HasAttribute":
            attribute = pred.args[1] if len(pred.args) > 1 else ""
            result = evaluator.evaluate(
                "HasAttribute", target_idx, attribute=attribute, use_vlm=use_vlm,
            )
            pred_results.append(result)
            continue

        if pred.name == "IsCategory":
            category = pred.args[1] if len(pred.args) > 1 else ""
            result = evaluator.evaluate(
                "IsCategory", target_idx, category=category, use_vlm=use_vlm,
            )
            pred_results.append(result)
            continue

        # Spatial predicates requiring anchors
        anchor_desc = pred.args[1] if len(pred.args) > 1 else ""
        anchor_candidates = resolved_anchors.get(anchor_desc, [])

        if not anchor_candidates:
            pred_results.append(PredicateResult(
                name=pred.name, score=1.0, status="dropped", drop_reason="anchor_not_found",
            ))
            continue

        if pred.name == "Between":
            ref_b_desc = pred.args[2] if len(pred.args) > 2 else ""
            ref_b_candidates = resolved_anchors.get(ref_b_desc, [])
            if not ref_b_candidates:
                pred_results.append(PredicateResult(
                    name=pred.name, score=1.0, status="dropped", drop_reason="anchor_not_found",
                ))
                continue
            if covisibility_constraint is not None and covisibility_constraint.enabled:
                anchor_candidates = covisibility_constraint.filter_anchors(target_idx, anchor_candidates)
                ref_b_candidates = covisibility_constraint.filter_anchors(target_idx, ref_b_candidates)
                if not anchor_candidates or not ref_b_candidates:
                    pred_results.append(PredicateResult(
                        name=pred.name,
                        score=0.0,
                        status="fast_path",
                        drop_reason=f"covisibility_no_anchor_within_{covisibility_constraint.hops}_hops",
                    ))
                    continue
            if _anchor_proximity_pool_k() > 0 or _anchor_pool_sim_ratio() > 0.0:
                means_ref = getattr(evaluator, "_means", None)
                anchor_candidates = _order_anchors_by_proximity(target_idx, anchor_candidates, means_ref)
                ref_b_candidates = _order_anchors_by_proximity(target_idx, ref_b_candidates, means_ref)
            best_result = _best_between_score(
                target_idx, anchor_candidates[:max_anchors], ref_b_candidates[:max_anchors],
                evaluator, use_vlm,
            )
            pred_results.append(best_result.result)
            if best_result.anchor_a is not None:
                matched_anchors[anchor_desc] = best_result.anchor_a
            if best_result.anchor_b is not None:
                matched_anchors[ref_b_desc] = best_result.anchor_b
            continue

        # Standard pairwise: Near, On, Above, Below, NextTo
        if (
            covisibility_constraint is not None
            and covisibility_constraint.enabled
            and pred.name in _LOCAL_COVISIBILITY_PREDICATES
        ):
            anchor_candidates = covisibility_constraint.filter_anchors(target_idx, anchor_candidates)
            if not anchor_candidates:
                pred_results.append(PredicateResult(
                    name=pred.name,
                    score=0.0,
                    status="fast_path",
                    drop_reason=f"covisibility_no_anchor_within_{covisibility_constraint.hops}_hops",
                ))
                continue

        best_score = -1.0
        best_result: Optional[PredicateResult] = None
        best_anchor_idx: Optional[int] = None

        if _anchor_proximity_pool_k() > 0 or _anchor_pool_sim_ratio() > 0.0:
            anchor_candidates = _order_anchors_by_proximity(
                target_idx, anchor_candidates, getattr(evaluator, "_means", None)
            )
        for anchor_idx in anchor_candidates[:max_anchors]:
            if anchor_idx == target_idx:
                continue
            result = evaluator.evaluate(
                pred.name, target_idx, anchor_idx, use_vlm=use_vlm,
            )
            if result.score > best_score:
                best_score = result.score
                best_result = result
                best_anchor_idx = anchor_idx

        if best_result is not None:
            pred_results.append(best_result)
            if best_anchor_idx is not None:
                matched_anchors[anchor_desc] = best_anchor_idx
        else:
            pred_results.append(PredicateResult(
                name=pred.name, score=1.0, status="dropped", drop_reason="no_valid_anchor",
            ))

    return pred_results, matched_anchors


class _BetweenResult:
    __slots__ = ("result", "anchor_a", "anchor_b")

    def __init__(self, result: PredicateResult, anchor_a: Optional[int], anchor_b: Optional[int]):
        self.result = result
        self.anchor_a = anchor_a
        self.anchor_b = anchor_b


def _best_between_score(
    target_idx: int,
    ref_a_candidates: List[int],
    ref_b_candidates: List[int],
    evaluator: PredicateEvaluator,
    use_vlm: bool,
) -> _BetweenResult:
    """Find the best (ref_a, ref_b) pair for a Between predicate."""
    best_score = -1.0
    best = _BetweenResult(
        PredicateResult(name="Between", score=1.0, status="dropped", drop_reason="no_valid_anchor"),
        None, None,
    )

    for a_idx in ref_a_candidates:
        if a_idx == target_idx:
            continue
        for b_idx in ref_b_candidates:
            if b_idx == target_idx or b_idx == a_idx:
                continue
            result = evaluator.evaluate(
                "Between", target_idx, a_idx, ref_b_idx=b_idx, use_vlm=use_vlm,
            )
            if result.score > best_score:
                best_score = result.score
                best = _BetweenResult(result, a_idx, b_idx)

    return best
