"""Lightweight multi-embedding retrieval for spatial query execution."""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalChannel:
    """Per-channel ranking diagnostics."""

    name: str
    weight: float
    enabled: bool = True
    error: Optional[str] = None
    ranks: Dict[int, int] = field(default_factory=dict)
    scores: Dict[int, float] = field(default_factory=dict)


@dataclass
class SemanticRetrievalResult:
    """Fused semantic retrieval output."""

    ranked_indices: List[int]
    fused_scores: Dict[int, float]
    candidate_indices: List[int]
    channels: Dict[str, RetrievalChannel]
    mode: str
    candidate_pool_mode: str


_SIGLIP2_EMBEDDER: Optional[Any] = None
_SIGLIP2_FAILED = False
_QWEN3_VL_EMBEDDER: Optional[Any] = None
_QWEN3_VL_FAILED = False


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _to_numpy(x: Any) -> np.ndarray:
    if x is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)


def _object_count(scene_state: Dict[str, Any]) -> int:
    for key in ("object_caption", "object_id", "means", "active"):
        value = scene_state.get(key)
        if value is None:
            continue
        with contextlib.suppress(Exception):
            return int(len(value))
    return 0


def active_indices(scene_state: Dict[str, Any], candidate_pool: Optional[Sequence[int]] = None) -> List[int]:
    """Return active object indices, respecting an optional pre-scoped pool."""

    n = _object_count(scene_state)
    base = list(candidate_pool) if candidate_pool is not None else list(range(n))
    active = scene_state.get("active")
    if active is None:
        return [int(i) for i in base if 0 <= int(i) < n]
    active_np = _to_numpy(active).astype(bool).reshape(-1)
    return [int(i) for i in base if 0 <= int(i) < min(n, len(active_np)) and bool(active_np[int(i)])]


def all_indices(scene_state: Dict[str, Any], candidate_pool: Optional[Sequence[int]] = None) -> List[int]:
    """Return all object indices, respecting an optional pre-scoped pool."""

    n = _object_count(scene_state)
    base = list(candidate_pool) if candidate_pool is not None else list(range(n))
    return [int(i) for i in base if 0 <= int(i) < n]


def _object_id(scene_state: Dict[str, Any], idx: int) -> int:
    object_ids = scene_state.get("object_id")
    if object_ids is not None and 0 <= idx < len(object_ids):
        with contextlib.suppress(Exception):
            return int(object_ids[idx])
    return int(idx)


def _resolve_redirect(scene_state: Dict[str, Any], object_id: int) -> int:
    redirects = scene_state.get("id_redirect") or {}
    if not isinstance(redirects, dict) or not redirects:
        return int(object_id)
    cur = int(object_id)
    seen: set[int] = set()
    while cur not in seen:
        seen.add(cur)
        nxt = redirects.get(cur)
        if nxt is None:
            nxt = redirects.get(str(cur))
        if nxt is None:
            break
        with contextlib.suppress(Exception):
            cur = int(nxt)
            continue
        break
    return int(cur)


def candidate_indices(
    scene_state: Dict[str, Any],
    candidate_pool: Optional[Sequence[int]] = None,
    *,
    mode: str = "active",
) -> List[int]:
    """Resolve the object pool for retrieval ablations.

    Modes:
      - active: current production behavior.
      - all: include inactive objects.
      - active_plus_redirect: active objects plus inactive loser ids whose
        ``id_redirect`` chain resolves to an active object id.
    """

    mode_norm = str(mode or "active").strip().lower()
    base = all_indices(scene_state, candidate_pool)
    if mode_norm == "all":
        return base

    active = active_indices(scene_state, candidate_pool)
    if mode_norm not in {"active_plus_redirect", "redirect", "backfill", "active+redirect"}:
        return active

    active_set = set(active)
    active_resolved_ids = {
        _resolve_redirect(scene_state, _object_id(scene_state, idx))
        for idx in active
    }
    loser_ids = scene_state.get("loser_object_ids") or []
    for idx in active:
        if idx >= len(loser_ids):
            continue
        entry = loser_ids[idx]
        if isinstance(entry, (set, list, tuple)):
            for oid in entry:
                with contextlib.suppress(Exception):
                    active_resolved_ids.add(_resolve_redirect(scene_state, int(oid)))

    out = list(active)
    for idx in base:
        if idx in active_set:
            continue
        oid = _object_id(scene_state, idx)
        if _resolve_redirect(scene_state, oid) in active_resolved_ids:
            out.append(idx)
    return out


def _as_vector(value: Any) -> np.ndarray:
    arr = _to_numpy(value).astype(np.float32, copy=False).reshape(-1)
    if arr.size == 0 or float(np.linalg.norm(arr)) <= 0.0:
        return np.empty((0,), dtype=np.float32)
    return arr


def _embedding_rows(scene_state: Dict[str, Any], field: str, idx: int) -> np.ndarray:
    history = scene_state.get(f"{field}_history")
    if history is None:
        history = []
    if 0 <= idx < len(history):
        arr = _to_numpy(history[idx]).astype(np.float32, copy=False)
        if arr.ndim == 1 and arr.size > 0:
            arr = arr.reshape(1, -1)
        if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
            norms = np.linalg.norm(arr, axis=1)
            valid = norms > 0.0
            if bool(np.any(valid)):
                return arr[valid]

    primary = scene_state.get(field)
    if primary is None:
        primary = []
    if 0 <= idx < len(primary):
        row = _as_vector(primary[idx])
        if row.size > 0:
            return row.reshape(1, -1)
    return np.empty((0, 0), dtype=np.float32)


def has_embedding(scene_state: Dict[str, Any], field: str, idx: int) -> bool:
    """Whether object ``idx`` has a non-zero current or historical embedding."""

    rows = _embedding_rows(scene_state, field, idx)
    return bool(rows.ndim == 2 and rows.shape[0] > 0 and rows.shape[1] > 0)


def _encode_query(embedder: Any, text: str) -> np.ndarray:
    fn = getattr(embedder, "encode_query", None)
    if callable(fn):
        return np.asarray(fn(text), dtype=np.float32)
    return np.asarray(embedder.encode(text), dtype=np.float32)


def _rank_from_query_vector(
    query_vec: np.ndarray,
    scene_state: Dict[str, Any],
    indices: Sequence[int],
    *,
    field: str,
    top_k: int,
) -> List[Tuple[int, float]]:
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    q_norm = float(np.linalg.norm(q))
    if q.size == 0 or q_norm <= 0.0:
        return []
    q = q / max(q_norm, 1e-12)

    scored: List[Tuple[int, float]] = []
    for idx in indices:
        rows = _embedding_rows(scene_state, field, int(idx))
        if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] != q.shape[0]:
            continue
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        valid = norms.reshape(-1) > 0.0
        if not bool(np.any(valid)):
            continue
        rows_norm = rows[valid] / np.maximum(norms[valid], 1e-12)
        score = float(np.max(rows_norm.dot(q)))
        scored.append((int(idx), score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: max(1, int(top_k))]


def _build_siglip2_prompt_ensemble(text: str) -> List[str]:
    query = " ".join(str(text or "").strip().split())
    if not query:
        return []
    q_lower = query.lower()
    article_prompt = query
    if not q_lower.startswith(("a ", "an ", "the ")):
        article = "an" if q_lower[:1] in {"a", "e", "i", "o", "u"} else "a"
        article_prompt = f"{article} {query}"
    candidates = [
        query,
        f"a photo of {query}",
        f"a picture of {query}",
        f"{query}, product photo",
        f"{query} on a table",
        f"close-up of {query}",
        f"{query}, blurry low-resolution photo",
        article_prompt,
    ]
    out: List[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _get_siglip2_embedder(verbose: bool) -> Optional[Any]:
    global _SIGLIP2_EMBEDDER, _SIGLIP2_FAILED
    if not _env_enabled("LAM_SIGLIP2_LOCAL_ENABLED", True) or _SIGLIP2_FAILED:
        return None
    if _SIGLIP2_EMBEDDER is not None:
        return _SIGLIP2_EMBEDDER
    try:
        from scene_graph.retrieval.scene_graph_retriever import _Siglip2TextEmbedWrapper

        _SIGLIP2_EMBEDDER = _Siglip2TextEmbedWrapper(debug=verbose)
    except Exception as exc:  # noqa: BLE001
        _SIGLIP2_FAILED = True
        if verbose:
            logger.warning("SigLIP2 query embedder unavailable: %s", exc)
        return None
    return _SIGLIP2_EMBEDDER


def _get_qwen3_vl_embedder(verbose: bool) -> Optional[Any]:
    global _QWEN3_VL_EMBEDDER, _QWEN3_VL_FAILED
    if not _env_enabled("QWEN3_VL_EMBED_ENABLED", True) or _QWEN3_VL_FAILED:
        return None
    if _QWEN3_VL_EMBEDDER is not None:
        return _QWEN3_VL_EMBEDDER
    try:
        from scene_graph.retrieval.scene_graph_retriever import _Qwen3VLEmbedQueryWrapper

        _QWEN3_VL_EMBEDDER = _Qwen3VLEmbedQueryWrapper(debug=verbose)
    except Exception as exc:  # noqa: BLE001
        _QWEN3_VL_FAILED = True
        if verbose:
            logger.warning("Qwen3-VL query embedder unavailable: %s", exc)
        return None
    return _QWEN3_VL_EMBEDDER


def _add_rank_channel(
    *,
    result: SemanticRetrievalResult,
    name: str,
    weight: float,
    ranked: Sequence[Tuple[int, float]],
    rrf_offset: int,
) -> None:
    channel = RetrievalChannel(name=name, weight=float(weight))
    result.channels[name] = channel
    for rank, (idx, score) in enumerate(ranked, start=1):
        idx = int(idx)
        channel.ranks.setdefault(idx, int(rank))
        channel.scores[idx] = float(score)
        result.fused_scores[idx] = result.fused_scores.get(idx, 0.0) + float(weight) / float(rrf_offset + rank)


def _add_error_channel(result: SemanticRetrievalResult, name: str, weight: float, error: str) -> None:
    result.channels[name] = RetrievalChannel(
        name=name,
        weight=float(weight),
        enabled=False,
        error=str(error),
    )


def retrieve_semantic_candidates(
    target_description: str,
    scene_state: Dict[str, Any],
    embedder: Any,
    *,
    raw_query: Optional[str] = None,
    candidate_pool: Optional[Sequence[int]] = None,
    k: int = 40,
    channel_k: int = 60,
    mode: str = "multi",
    candidate_pool_mode: str = "active",
    rrf_offset: int = 5,
    verbose: bool = False,
) -> SemanticRetrievalResult:
    """Retrieve candidates with caption/SigLIP2/Qwen3-VL rank fusion.

    ``mode="caption"`` preserves the previous caption-only behavior, except
    that the active/all/redirect pool can now be selected explicitly.
    ``mode="multi"`` uses RRF over caption target text, caption raw utterance,
    SigLIP2 text-to-crop embeddings, and Qwen3-VL text-to-crop embeddings.
    """

    mode_norm = str(mode or "multi").strip().lower()
    pool_mode_norm = str(candidate_pool_mode or "active").strip().lower()
    indices = candidate_indices(scene_state, candidate_pool, mode=pool_mode_norm)
    result = SemanticRetrievalResult(
        ranked_indices=[],
        fused_scores={},
        candidate_indices=list(indices),
        channels={},
        mode=mode_norm,
        candidate_pool_mode=pool_mode_norm,
    )
    if not indices:
        return result

    target_text = str(target_description or raw_query or "").strip()
    raw_text = str(raw_query or "").strip()
    if not target_text:
        result.ranked_indices = indices[: max(1, int(k))]
        result.fused_scores = {idx: 0.0 for idx in result.ranked_indices}
        return result

    channel_limit = max(int(channel_k), int(k), 1)
    try:
        q = _encode_query(embedder, target_text)
        ranked = _rank_from_query_vector(
            q,
            scene_state,
            indices,
            field="object_caption_embedding",
            top_k=channel_limit,
        )
        _add_rank_channel(
            result=result,
            name="caption_target",
            weight=1.0,
            ranked=ranked,
            rrf_offset=rrf_offset,
        )
        if mode_norm == "caption":
            result.fused_scores = {idx: max(0.0, float(score)) for idx, score in ranked}
    except Exception as exc:  # noqa: BLE001
        _add_error_channel(result, "caption_target", 1.0, repr(exc))

    if mode_norm == "multi":
        if raw_text and raw_text.lower() != target_text.lower():
            try:
                q = _encode_query(embedder, raw_text)
                ranked = _rank_from_query_vector(
                    q,
                    scene_state,
                    indices,
                    field="object_caption_embedding",
                    top_k=channel_limit,
                )
                _add_rank_channel(
                    result=result,
                    name="caption_raw",
                    weight=0.5,
                    ranked=ranked,
                    rrf_offset=rrf_offset,
                )
            except Exception as exc:  # noqa: BLE001
                _add_error_channel(result, "caption_raw", 0.5, repr(exc))

        siglip = _get_siglip2_embedder(verbose)
        if siglip is None:
            _add_error_channel(result, "siglip2", 0.75, "disabled_or_unavailable")
        else:
            try:
                prompt_vecs = []
                for prompt in _build_siglip2_prompt_ensemble(target_text):
                    prompt_vecs.append(np.asarray(siglip.encode(prompt), dtype=np.float32).reshape(-1))
                ranked_by_idx: Dict[int, float] = {}
                for q in prompt_vecs:
                    for idx, score in _rank_from_query_vector(
                        q,
                        scene_state,
                        indices,
                        field="object_siglip2_embedding",
                        top_k=channel_limit,
                    ):
                        if idx not in ranked_by_idx or score > ranked_by_idx[idx]:
                            ranked_by_idx[idx] = float(score)
                ranked = sorted(ranked_by_idx.items(), key=lambda item: item[1], reverse=True)[:channel_limit]
                _add_rank_channel(
                    result=result,
                    name="siglip2",
                    weight=0.75,
                    ranked=ranked,
                    rrf_offset=rrf_offset,
                )
            except Exception as exc:  # noqa: BLE001
                _add_error_channel(result, "siglip2", 0.75, repr(exc))

        qwen3_vl = _get_qwen3_vl_embedder(verbose)
        if qwen3_vl is None:
            _add_error_channel(result, "qwen3_vl", 1.0, "disabled_or_unavailable")
        else:
            try:
                q = np.asarray(qwen3_vl.encode(target_text), dtype=np.float32).reshape(-1)
                ranked = _rank_from_query_vector(
                    q,
                    scene_state,
                    indices,
                    field="object_qwen3_vl_embedding",
                    top_k=channel_limit,
                )
                _add_rank_channel(
                    result=result,
                    name="qwen3_vl",
                    weight=1.0,
                    ranked=ranked,
                    rrf_offset=rrf_offset,
                )
            except Exception as exc:  # noqa: BLE001
                _add_error_channel(result, "qwen3_vl", 1.0, repr(exc))

    if result.fused_scores:
        ranked_indices = sorted(result.fused_scores, key=lambda idx: result.fused_scores[idx], reverse=True)
        result.ranked_indices = [int(i) for i in ranked_indices[: max(1, int(k))]]
        return result

    captions = scene_state.get("object_caption", []) or []
    keywords = target_text.lower().split()
    fallback = [
        int(idx)
        for idx in indices
        if idx < len(captions) and any(kw in str(captions[idx]).lower() for kw in keywords)
    ]
    if not fallback:
        fallback = list(indices)
    result.ranked_indices = fallback[: max(1, int(k))]
    result.fused_scores = {idx: 0.0 for idx in result.ranked_indices}
    return result


def channel_rank_summary(result: SemanticRetrievalResult, idx: int) -> Dict[str, Any]:
    """Compact per-object retrieval diagnostics for JSON output."""

    out: Dict[str, Any] = {"fused_score": float(result.fused_scores.get(int(idx), 0.0))}
    for name, channel in sorted(result.channels.items()):
        out[name] = {
            "enabled": bool(channel.enabled),
            "rank": channel.ranks.get(int(idx)),
            "score": channel.scores.get(int(idx)),
            "error": channel.error,
        }
    return out
