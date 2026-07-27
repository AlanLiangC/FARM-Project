"""joint_v1 — truncation-free, vectorized, jointly-aggregated relational retrieval.

Motivated by a concrete failure of the pipelined path ("cardboard box near a
metal ladder", FARM-Scenes warehouse): the fixed top-k anchor pool missed the
ladders that actually have boxes beside them (they ranked 6/33/42
semantically), every candidate scored Near=0, and the ranking silently
degenerated to semantics. Design principles:

1. **No anchor truncation.** Each anchor phrase grounds to a continuous
   affinity over *all* objects (caption-embedding cosine, normalised and
   sharpened) — never a top-k pool.
2. **Vectorized predicates.** Metric predicates (Near/On/Above/Below/NextTo)
   are evaluated as (candidates x anchors) matrices with the same membership
   functions and constants as the locked scalar fast paths. View-dependent
   predicates reuse the shared-view evaluator on a wide (not truncated)
   shortlist.
3. **Joint aggregation.** score(r) = sem(r) * prod_j soft(max_a A_j(a) *
   P_j(r, a)): the anchor assignment is optimised per candidate over the whole
   scene, so "right class, wrong instance" cannot happen by construction.
4. **Speaker pragmatics.** For view-dependent relations, satisfying pairs
   decay pool-relative to the nearest satisfying pair — "the A left of B"
   prefers the closest such A, while a lone far pair keeps full score.
5. **No silent degeneration.** A predicate with no signal anywhere is dropped
   with an explicit liveness flag (visible in PredicateResult.drop_reason and
   the log) instead of flooring every candidate equally.

Selected with ``spatial_method="joint_v1"`` in :func:`execute_spatial_query`;
the locked ``unified_soft_w50`` path is untouched.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from scene_graph.llm_utils import EmbedInterface, LLMInterface

from .models import PredicateResult, QueryGraph, ScoredCandidate
from .predicates import SUPERLATIVE_PREDICATES, PredicateEvaluator
from .semantic_retrieval import _encode_query, _rank_from_query_vector

logger = logging.getLogger(__name__)

_METRIC_PREDICATES = frozenset({"Near", "On", "Above", "Below", "NextTo", "Inside"})
_VIEW_PREDICATES = frozenset({"LeftOf", "RightOf", "InFrontOf", "Behind"})
_SEMANTIC_PREDICATES = frozenset({"IsCategory", "HasAttribute", "InRegion"})

_PREDICATE_FLOOR = 0.05          # soft floor per factor (locked path uses 0.5)
_LIVENESS_EPS = 0.02             # below this max, a predicate carries no signal
_AFFINITY_SHARPEN = 2.0          # suppress off-class anchors after max-normalising
_ANCHOR_AFFINITY_MIN = 0.30      # relational predicates ignore anchors below this
                                 # (post-normalisation): a gated-down off-class
                                 # object must never become a matched anchor just
                                 # because the real anchors lack predicate signal
_VIEW_CAND_LIMIT = 200           # view-dependent pairs: top candidates ...
_VIEW_ANCHOR_LIMIT = 16          # ... x top anchors by affinity


def _gaussian(d: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(d ** 2) / (2.0 * sigma ** 2))


def _sigmoid(v: np.ndarray, center: float, temperature: float) -> np.ndarray:
    e = np.clip(-(v - center) / temperature, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(e))


_OFFCLASS_PENALTY = 0.15  # affinity multiplier for objects failing the head-noun gate


def _head_noun_gate(scene_state: Dict[str, Any], phrase: str, n_objects: int) -> Optional[np.ndarray]:
    """Lexical gate for the phrase's head noun (word-boundary match against
    caption + category). Embedding cosine separates classes weakly ("metal
    ladder" vs "metal cabinet" differ by ~0.2), so without this gate any
    adjacent same-material object can be picked as the anchor. Returns None
    when no object matches (open-vocabulary phrase — pure embedding then)."""
    words = [w for w in re.findall(r"[a-z]+", (phrase or "").lower()) if len(w) >= 3]
    if not words:
        return None
    # Head noun, plus long content words ("tripod", "surveying" — the head can
    # be a hypernym absent from captions, e.g. "surveying tool" vs a caption
    # saying "tripod with an instrument"), plus the space-joined phrase
    # (compound spellings: "white board" vs "whiteboard").
    keys = {words[-1]} | {w for w in words if len(w) >= 6}
    joined = "".join(words)
    if len(joined) >= 6:
        keys.add(joined)
    pattern = re.compile("|".join(rf"\b{re.escape(k)}" for k in sorted(keys)))
    captions = scene_state.get("object_caption") or []
    categories = scene_state.get("object_category") or []
    gate = np.zeros(n_objects, dtype=bool)
    for i in range(n_objects):
        cap = captions[i] if i < len(captions) else ""
        cat = categories[i] if i < len(categories) else ""
        text = f"{cap} {cat}".lower()
        if pattern.search(text):
            gate[i] = True
    return gate if gate.any() else None


def _anchor_affinity(
    scene_state: Dict[str, Any],
    embedder: EmbedInterface,
    phrase: str,
    indices: Sequence[int],
    n_objects: int,
) -> Optional[np.ndarray]:
    """Continuous grounding affinity of *phrase* over all objects, in [0, 1]:
    caption-embedding cosine (modifiers, open vocab) gated by the head noun
    (class identity)."""
    try:
        q = _encode_query(embedder, phrase)
        ranked = _rank_from_query_vector(
            q, scene_state, indices, field="object_caption_embedding", top_k=len(indices)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("joint_v1: anchor affinity failed for %r: %s", phrase, exc)
        return None
    if not ranked:
        return None
    aff = np.zeros(n_objects, dtype=np.float64)
    for idx, score in ranked:
        if 0 <= int(idx) < n_objects:
            aff[int(idx)] = max(0.0, float(score))
    gate = _head_noun_gate(scene_state, phrase, n_objects)
    if gate is not None:
        aff = np.where(gate, aff, aff * _OFFCLASS_PENALTY)
    top = float(aff.max())
    if top <= 0.0:
        return None
    aff /= top
    return aff ** _AFFINITY_SHARPEN


def _metric_matrix(
    name: str,
    cand_pos: np.ndarray,   # (C, 3)
    anchor_pos: np.ndarray,  # (A, 3)
    vertical_axis: int,
    anchor_cov_diag: Optional[np.ndarray] = None,  # (A, 3) cov6 diagonal, Inside only
) -> np.ndarray:
    """(C, A) predicate scores, matching the scalar fast-path semantics."""
    haxes = [i for i in range(3) if i != vertical_axis]
    diff = cand_pos[:, None, :] - anchor_pos[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    hdist = np.linalg.norm(diff[:, :, haxes], axis=-1)
    hdiff = diff[:, :, vertical_axis]  # candidate height minus anchor height

    if name == "Near":
        # Furniture-scale "near": gentle absolute decay (0.22 at 3 m) with the
        # 5 m prune; the pool-relative pass afterwards makes it comparative —
        # "the door near the case" prefers the nearest satisfying door.
        out = np.exp(-dist / 2.0)
        out[dist > 5.0] = 0.0
        return out
    if name == "NextTo":
        out = _gaussian(hdist, 1.0)
        out[(hdist > 2.0) | (np.abs(hdiff) > 1.0)] = 0.0
        return out
    if name == "Above":
        out = _sigmoid(hdiff, 0.2, 0.15) * _gaussian(hdist, 1.0)
        out[(hdiff < -0.1) | (hdist > 2.0)] = 0.0
        return out
    if name == "Below":
        out = _sigmoid(-hdiff, 0.2, 0.15) * _gaussian(hdist, 1.0)
        out[(-hdiff < -0.1) | (hdist > 2.0)] = 0.0
        return out
    if name == "On":
        lo, hi = 0.0, 0.5
        horiz = _gaussian(hdist, 1.0 * 0.7)
        height = _sigmoid(hdiff, lo, 0.1) * _sigmoid(hi - hdiff + lo, lo, 0.1)
        out = horiz * np.minimum(1.0, height)
        out[(hdist > 1.0) | (hdiff < lo - 0.2) | (hdiff > hi + 0.3)] = 0.0
        return out
    if name == "Inside":
        # Mirrors the locked ``_fast_inside``: full membership within the
        # anchor's 2.5-sigma AABB (+10 cm margin), gaussian falloff outside.
        if anchor_cov_diag is None:
            return _gaussian(dist, 0.75)  # locked fallback when cov6 is absent
        half = 2.5 * np.sqrt(np.maximum(anchor_cov_diag, 1e-6)) + 0.10  # (A, 3)
        bmin = anchor_pos - half
        bmax = anchor_pos + half
        delta = np.maximum(
            np.maximum(bmin[None, :, :] - cand_pos[:, None, :], cand_pos[:, None, :] - bmax[None, :, :]),
            0.0,
        )
        outside = np.linalg.norm(delta, axis=-1)  # (C, A)
        return np.where(outside <= 1e-6, 1.0, _gaussian(outside, 0.35))
    raise ValueError(f"not a metric predicate: {name}")


def _anchor_cov_diag(scene_state: Dict[str, Any], rows: np.ndarray) -> Optional[np.ndarray]:
    """cov6 diagonal [xx, yy, zz] for the given object rows, or None."""
    cov6 = scene_state.get("cov6")
    if cov6 is None:
        return None
    try:
        cov_np = cov6.cpu().numpy() if hasattr(cov6, "cpu") else np.asarray(cov6)
        return np.asarray(cov_np[rows][:, [0, 3, 5]], dtype=np.float64)
    except Exception:
        return None


def _affinity_rows(aff: np.ndarray, cap: int = 24) -> np.ndarray:
    """Anchor row selection shared by Between: threshold, else top-5; cap by affinity."""
    rows = np.nonzero(aff >= _ANCHOR_AFFINITY_MIN)[0]
    if rows.size == 0:
        rows = np.argsort(-aff)[:5]
    elif rows.size > cap:
        rows = rows[np.argsort(-aff[rows])[:cap]]
    return rows


def _between_matrix(
    cand_pos: np.ndarray,   # (C, 3)
    pos_a: np.ndarray,      # (A, 3)
    pos_b: np.ndarray,      # (B, 3)
    horizontal_axes: Sequence[int],
) -> np.ndarray:
    """(C, A, B) corridor scores mirroring the locked ``_fast_between``.

    Horizontal-plane projection fraction t along a->b (hard zero outside
    [-0.1, 1.1]), t_score = 1 - 2|t - 0.5|, lateral gaussian sigma = 1 m.
    """
    axes = list(horizontal_axes)
    s = cand_pos[:, axes]                      # (C, 2)
    a = pos_a[:, axes]                         # (A, 2)
    b = pos_b[:, axes]                         # (B, 2)
    ab = b[None, :, :] - a[:, None, :]         # (A, B, 2): ab[i, j] = b[j] - a[i]
    ab_len = np.linalg.norm(ab, axis=-1)       # (A, B)
    safe_len = np.maximum(ab_len, 1e-9)
    as_vec = s[:, None, None, :] - a[None, :, None, :]                  # (C, A, 1, 2)
    t = np.sum(as_vec * ab[None, :, :, :], axis=-1) / (safe_len[None, :, :] ** 2)  # (C, A, B)
    proj = t[..., None] * ab[None, :, :, :]                             # (C, A, B, 2)
    perp = np.linalg.norm(as_vec - proj, axis=-1)                       # (C, A, B)
    t_score = np.maximum(0.0, 1.0 - 2.0 * np.abs(t - 0.5))
    out = t_score * _gaussian(perp, 1.0)
    out = np.where((t < -0.1) | (t > 1.1), 0.0, out)
    out = np.where(ab_len[None, :, :] < 0.01, 0.0, out)
    return out


def _camera_inverses(scene_state: Dict[str, Any]) -> Optional[np.ndarray]:
    """(P, 4, 4) world->camera transforms from the saved image records."""
    mats: List[np.ndarray] = []
    for rec in scene_state.get("images") or []:
        pose = rec.get("pose") if isinstance(rec, dict) else getattr(rec, "pose", None)
        if pose is None:
            continue
        try:
            p = np.asarray(pose.cpu().numpy() if hasattr(pose, "cpu") else pose, dtype=np.float64)
        except Exception:  # noqa: BLE001
            continue
        if p.shape == (4, 4) and np.isfinite(p).all():
            try:
                mats.append(np.linalg.inv(p))
            except np.linalg.LinAlgError:
                continue
    return np.stack(mats) if mats else None


def _projection_view_score(name: str, s: np.ndarray, a: np.ndarray) -> float:
    """Left/right/front/behind from camera-frame coordinates (P, 3) of subject
    and anchor — the pair need not share a stored *mask* view. Mirrors the
    mask path's convention: the relation is judged in the best stored
    viewpoint (both objects near and centred), not averaged over every frame
    — cameras on the far side of a pair see the opposite ordering, so an
    unweighted mean dilutes even an obvious relation toward 0.5."""
    zs, za = s[:, 2], a[:, 2]
    valid = (zs > 0.3) & (za > 0.3) & (zs < 15.0) & (za < 15.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        us, vs = s[:, 0] / zs, s[:, 1] / zs
        ua, va = a[:, 0] / za, a[:, 1] / za
    valid &= (np.abs(us) < 0.85) & (np.abs(ua) < 0.85) & (np.abs(vs) < 0.7) & (np.abs(va) < 0.7)
    if not valid.any():
        return 0.0
    if name in {"LeftOf", "RightOf"}:
        direction = (ua - us) if name == "LeftOf" else (us - ua)
        score = 1.0 / (1.0 + np.exp(np.clip(-direction / 0.08, -60, 60)))
    else:  # InFrontOf / Behind
        direction = (za - zs) if name == "InFrontOf" else (zs - za)
        score = 1.0 / (1.0 + np.exp(np.clip(-direction / 0.30, -60, 60)))
    # View quality: both objects centred and close. Blend the top few views.
    quality = np.where(
        valid,
        np.exp(-(us ** 2 + ua ** 2 + vs ** 2 + va ** 2)) / (1.0 + np.minimum(zs, za) / 4.0),
        0.0,
    )
    top = np.argsort(-quality)[:5]
    top = top[quality[top] > 0.0]
    if top.size == 0:
        return 0.0
    w = quality[top]
    return float(np.sum(w * score[top]) / np.sum(w))


def _view_matrix(
    name: str,
    cand_rows: np.ndarray,     # (C,) object indices
    anchor_rows: np.ndarray,   # (A,) object indices
    evaluator: PredicateEvaluator,
    cam_inv: Optional[np.ndarray] = None,
) -> np.ndarray:
    """(C, A) scores for view-dependent predicates: the shared-mask-view fast
    path where available, else the pose-projection fallback."""
    means = evaluator._means
    out = np.zeros((cand_rows.shape[0], anchor_rows.shape[0]), dtype=np.float64)
    proj_cache: Dict[int, np.ndarray] = {}

    def _proj(idx: int) -> np.ndarray:
        if idx not in proj_cache:
            p = np.asarray(means[idx], dtype=np.float64)
            proj_cache[idx] = cam_inv[:, :3, :3] @ p + cam_inv[:, :3, 3]
        return proj_cache[idx]

    for ci, c in enumerate(cand_rows):
        for ai, a in enumerate(anchor_rows):
            if int(c) == int(a):
                continue
            try:
                score = evaluator.fast_path(name, int(c), int(a))
            except Exception:  # noqa: BLE001
                score = None
            if (score is None or score <= 0.0) and cam_inv is not None and means is not None:
                score = _projection_view_score(name, _proj(int(c)), _proj(int(a)))
            if score is not None:
                out[ci, ai] = float(score)
    return out


def _pool_relative_decay(
    scores: np.ndarray,
    dists: np.ndarray,
    satisfied: np.ndarray,
    *,
    tau_floor: float = 1.0,
    tau_scale: float = 0.35,
) -> np.ndarray:
    """Speaker-pragmatics factor: satisfying pairs decay relative to the
    nearest satisfying pair; the nearest keeps full score. For Near the decay
    is tight (proximity IS the relation); for view-dependent relations use a
    wider tau — there, distance is a tiebreaker, not the relation itself."""
    if not satisfied.any():
        return scores
    d_min = float(dists[satisfied].min())
    tau = max(tau_floor, tau_scale * d_min)
    decay = np.exp(-np.maximum(0.0, dists - d_min) / tau)
    return np.where(satisfied, scores * decay, scores)


def execute_joint_v1(
    query_graph: QueryGraph,
    scene_state: Dict[str, Any],
    llm: LLMInterface,
    embedder: EmbedInterface,
    *,
    max_output_candidates: int = 20,
    raw_query: str = "",
    retrieval_mode: str = "multi",
    candidate_pool_mode: str = "active",
    verbose: bool = False,
) -> List[ScoredCandidate]:
    # Local import: executor imports this module lazily, avoid a cycle.
    from .executor import _pre_filter_semantic

    evaluator = PredicateEvaluator(scene_state, llm, embedder, verbose=verbose)
    means = evaluator._means
    if means is None or len(means) == 0:
        return []
    n_objects = len(means)
    object_ids = scene_state.get("object_id")
    ids_np = (
        np.asarray(object_ids.cpu().numpy() if hasattr(object_ids, "cpu") else object_ids).astype(int)
        if object_ids is not None
        else np.arange(n_objects)
    )

    # ---- target semantics: same multi-channel retrieval as the locked path
    target_retrieval = _pre_filter_semantic(
        query_graph.target_description,
        None,
        scene_state,
        embedder,
        n_objects,
        verbose,
        raw_query=raw_query,
        retrieval_mode=retrieval_mode,
        candidate_pool_mode=candidate_pool_mode,
    )
    cand_rows = np.asarray([int(i) for i in target_retrieval.ranked_indices], dtype=int)
    if cand_rows.size == 0:
        return []
    fused = target_retrieval.fused_scores or {}
    sem = np.asarray([max(0.0, float(fused.get(int(i), 0.0))) for i in cand_rows], dtype=np.float64)
    # Head-noun gate on the target side too: without it, off-class objects
    # with residual embedding similarity (a ladder for "cardboard box") can
    # ride a huge predicate score into the top ranks.
    target_gate = _head_noun_gate(scene_state, query_graph.target_description or "", n_objects)
    if target_gate is not None:
        sem = np.where(target_gate[cand_rows], sem, sem * _OFFCLASS_PENALTY)
    if sem.max() > 0:
        sem = sem / sem.max()
    means_all = np.asarray(means, dtype=np.float64)
    cand_pos = means_all[cand_rows]

    composite = sem.copy()
    pred_results_per_cand: List[List[PredicateResult]] = [[] for _ in range(cand_rows.size)]
    matched_per_cand: List[Dict[str, int]] = [{} for _ in range(cand_rows.size)]
    target_desc_l = (query_graph.target_description or "").strip().lower()
    cam_inverses: Optional[np.ndarray] = None  # lazy; only for view predicates

    for pred in query_graph.predicates or []:
        name = pred.name
        # -- anchor-free semantic predicates fold into the target affinity
        if name in _SEMANTIC_PREDICATES:
            phrase = pred.args[1] if len(pred.args) > 1 else ""
            if not phrase or phrase.strip().lower() == target_desc_l:
                continue
            aff = _anchor_affinity(scene_state, embedder, phrase, list(range(n_objects)), n_objects)
            if aff is None:
                continue
            factor = np.maximum(0.35, aff[cand_rows])
            composite *= factor
            for k in range(cand_rows.size):
                pred_results_per_cand[k].append(
                    PredicateResult(name=name, score=float(factor[k]), status="fast_path")
                )
            continue

        # -- two-anchor corridor: Between(target, A, B)
        if name == "Between" and len(pred.args) > 2:
            desc_a = str(pred.args[1] or "").strip()
            desc_b = str(pred.args[2] or "").strip()
            aff_a = _anchor_affinity(scene_state, embedder, desc_a, list(range(n_objects)), n_objects)
            aff_b = (
                aff_a
                if desc_b.lower() == desc_a.lower()
                else _anchor_affinity(scene_state, embedder, desc_b, list(range(n_objects)), n_objects)
            )
            if aff_a is None or aff_b is None or not desc_a or not desc_b:
                for k in range(cand_rows.size):
                    pred_results_per_cand[k].append(
                        PredicateResult(name=name, score=1.0, status="dropped", drop_reason="anchor_not_found")
                    )
                continue
            rows_a = _affinity_rows(aff_a)
            rows_b = _affinity_rows(aff_b)
            corridor = _between_matrix(
                cand_pos, means_all[rows_a], means_all[rows_b], evaluator.horizontal_axes
            )  # (C, A, B)
            joint3 = corridor * aff_a[rows_a][None, :, None] * aff_b[rows_b][None, None, :]
            # the two references must be distinct objects, and never the candidate
            joint3 = np.where((rows_a[:, None] == rows_b[None, :])[None, :, :], 0.0, joint3)
            joint3 = np.where((cand_rows[:, None] == rows_a[None, :])[:, :, None], 0.0, joint3)
            joint3 = np.where((cand_rows[:, None] == rows_b[None, :])[:, None, :], 0.0, joint3)
            flat = joint3.reshape(cand_rows.size, -1)
            best = np.argmax(flat, axis=1)
            m = flat[np.arange(cand_rows.size), best]
            if float(m.max(initial=0.0)) < _LIVENESS_EPS:
                logger.warning(
                    "joint_v1: predicate Between(%s, %s) has no signal anywhere — dropped (liveness)",
                    desc_a, desc_b,
                )
                for k in range(cand_rows.size):
                    pred_results_per_cand[k].append(
                        PredicateResult(name=name, score=1.0, status="dropped", drop_reason="no_predicate_signal")
                    )
                continue
            composite *= _PREDICATE_FLOOR + (1.0 - _PREDICATE_FLOOR) * m
            b_cols = max(1, int(rows_b.size))
            key_b = desc_b if desc_b.lower() != desc_a.lower() else f"{desc_b} (2)"
            for k in range(cand_rows.size):
                pred_results_per_cand[k].append(PredicateResult(name=name, score=float(m[k]), status="fast_path"))
                if m[k] >= _LIVENESS_EPS:
                    matched_per_cand[k][desc_a] = int(rows_a[int(best[k]) // b_cols])
                    matched_per_cand[k][key_b] = int(rows_b[int(best[k]) % b_cols])
            continue

        anchor_desc = pred.args[1] if len(pred.args) > 1 else ""
        if not anchor_desc:
            continue
        aff = _anchor_affinity(scene_state, embedder, anchor_desc, list(range(n_objects)), n_objects)
        if aff is None:
            for k in range(cand_rows.size):
                pred_results_per_cand[k].append(
                    PredicateResult(name=name, score=1.0, status="dropped", drop_reason="anchor_not_found")
                )
            continue

        anchor_rows = np.nonzero(aff >= _ANCHOR_AFFINITY_MIN)[0]
        if anchor_rows.size == 0:  # open-vocab phrase with a flat affinity — keep the best few
            anchor_rows = np.argsort(-aff)[:5]
        anchor_pos = np.asarray(means, dtype=np.float64)[anchor_rows]
        aff_row = aff[anchor_rows]

        if name in SUPERLATIVE_PREDICATES:
            # MAP anchor; rank candidates by distance to it.
            a_star = int(anchor_rows[int(np.argmax(aff_row))])
            d = np.linalg.norm(cand_pos - np.asarray(means[a_star], dtype=np.float64)[None, :], axis=-1)
            if name == "Closest":
                d_min = float(d.min())
                m = np.exp(-np.maximum(0.0, d - d_min) / max(1.0, 0.35 * max(d_min, 1e-6)))
            else:  # Farthest
                d_max = float(d.max())
                m = np.exp(-np.maximum(0.0, d_max - d) / max(1.0, 0.35 * d_max))
            m = np.where(cand_rows == a_star, 0.0, m)
            composite *= _PREDICATE_FLOOR + (1.0 - _PREDICATE_FLOOR) * m
            for k in range(cand_rows.size):
                pred_results_per_cand[k].append(PredicateResult(name=name, score=float(m[k]), status="fast_path"))
                matched_per_cand[k][anchor_desc] = a_star
            continue

        if name in _METRIC_PREDICATES:
            cov_diag = _anchor_cov_diag(scene_state, anchor_rows) if name == "Inside" else None
            pmat = _metric_matrix(
                name, cand_pos, anchor_pos, evaluator.vertical_axis, anchor_cov_diag=cov_diag
            )
        elif name in _VIEW_PREDICATES:
            if cam_inverses is None:
                cam_inverses = _camera_inverses(scene_state)
            top_c = min(_VIEW_CAND_LIMIT, cand_rows.size)
            top_a = min(_VIEW_ANCHOR_LIMIT, anchor_rows.size)
            c_sel = np.argsort(-sem)[:top_c]
            a_sel = np.argsort(-aff_row)[:top_a]
            pmat = np.zeros((cand_rows.size, anchor_rows.size), dtype=np.float64)
            sub = _view_matrix(name, cand_rows[c_sel], anchor_rows[a_sel], evaluator, cam_inverses)
            pmat[np.ix_(c_sel, a_sel)] = sub
        else:
            for k in range(cand_rows.size):
                pred_results_per_cand[k].append(
                    PredicateResult(name=name, score=1.0, status="dropped", drop_reason="unsupported_joint_v1")
                )
            continue

        # self-pairs never count
        same = cand_rows[:, None] == anchor_rows[None, :]
        pmat = np.where(same, 0.0, pmat)

        joint = pmat * aff_row[None, :]
        best_a = np.argmax(joint, axis=1)
        m = joint[np.arange(cand_rows.size), best_a]

        if name in _VIEW_PREDICATES:
            d_best = np.linalg.norm(cand_pos - anchor_pos[best_a], axis=-1)
            m = _pool_relative_decay(m, d_best, m > 0.05, tau_floor=3.0, tau_scale=0.5)
        elif name == "Near":
            d_best = np.linalg.norm(cand_pos - anchor_pos[best_a], axis=-1)
            m = _pool_relative_decay(m, d_best, m > 0.01)

        if float(m.max(initial=0.0)) < _LIVENESS_EPS:
            logger.warning(
                "joint_v1: predicate %s(%s) has no signal anywhere — dropped (liveness)",
                name, anchor_desc,
            )
            for k in range(cand_rows.size):
                pred_results_per_cand[k].append(
                    PredicateResult(name=name, score=1.0, status="dropped", drop_reason="no_predicate_signal")
                )
            continue

        composite *= _PREDICATE_FLOOR + (1.0 - _PREDICATE_FLOOR) * m
        for k in range(cand_rows.size):
            pred_results_per_cand[k].append(PredicateResult(name=name, score=float(m[k]), status="fast_path"))
            if m[k] >= _LIVENESS_EPS:  # no anchor edge for effectively-zero pairs
                matched_per_cand[k][anchor_desc] = int(anchor_rows[int(best_a[k])])

    order = np.argsort(-composite)[: max(1, int(max_output_candidates))]
    results: List[ScoredCandidate] = []
    for k in order:
        preds = pred_results_per_cand[int(k)]
        geo = float(np.prod([r.score for r in preds]) ** (1.0 / len(preds))) if preds else 1.0
        results.append(
            ScoredCandidate(
                object_index=int(cand_rows[int(k)]),
                object_id=int(ids_np[int(cand_rows[int(k)])]),
                predicate_results=preds,
                composite_score=float(composite[int(k)]),
                matched_anchors=matched_per_cand[int(k)],
                target_similarity=float(sem[int(k)]),
                vlm_rerank_score=None,
                predicate_geo_mean=geo,
                predicate_weight=1.0,
            )
        )
    return results
