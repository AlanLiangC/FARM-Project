"""Canonical redirect-group geometry helpers for ReferIt3D-style scoring.

The scene graph keeps inactive redirected loser rows after object-object
merges. Their voxel buffers are normally cleared, but their Gaussian/caption
state can still be useful as an alias geometry hypothesis for the canonical
active winner. This module exposes those aliases without changing map
construction.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .matching import gaussian_aabb, voxel_cloud_aabb


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "near",
    "next",
    "of",
    "on",
    "one",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


@dataclass(frozen=True)
class AliasBox:
    """One geometry hypothesis for a canonical object."""

    canonical_index: int
    canonical_object_id: int
    alias_index: int
    alias_object_id: int
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    source: str
    alias_score: float = 0.0


def _to_numpy(x: Any, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if x is None:
        return np.empty((0,), dtype=np.float32 if dtype is None else dtype)
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _object_count(scene_state: Dict[str, Any]) -> int:
    for key in ("object_caption", "object_id", "means", "active"):
        value = scene_state.get(key)
        if value is not None:
            return int(len(value))
    return 0


def _tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(tok) > 1 and tok not in _STOPWORDS
    }


def _token_score(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    doc_tokens = _tokens(text)
    if not doc_tokens:
        return 0.0
    overlap = len(query_tokens & doc_tokens)
    if overlap == 0:
        return 0.0
    return float(overlap / math.sqrt(max(1, len(query_tokens)) * max(1, len(doc_tokens))))


class AliasGeometryResolver:
    """Resolve object indices to canonical object ids plus alias bboxes."""

    def __init__(
        self,
        scene_state: Dict[str, Any],
        *,
        k_sigma: float = 2.5,
        prefer_voxel: bool = True,
    ) -> None:
        self.scene_state = scene_state
        self.k_sigma = float(k_sigma)
        self.prefer_voxel = bool(prefer_voxel)
        self.n = _object_count(scene_state)

        means = scene_state.get("means")
        cov6 = scene_state.get("cov6")
        self.means = _to_numpy(means, dtype=np.float32)
        self.cov6 = _to_numpy(cov6, dtype=np.float32)
        self.object_ids = _to_numpy(scene_state.get("object_id"), dtype=np.int64)
        self.active = _to_numpy(scene_state.get("active"), dtype=bool)
        if self.active.shape[0] < self.n:
            padded = np.zeros((self.n,), dtype=bool)
            padded[: self.active.shape[0]] = self.active
            self.active = padded

        self.captions = list(scene_state.get("object_caption", []) or [])
        self.categories = list(scene_state.get("object_category", []) or [])
        self.counts = _to_numpy(scene_state.get("count"), dtype=np.float32)

        self._id_to_idx = {self.object_id(i): int(i) for i in range(self.n)}
        self._groups: Dict[int, List[int]] = {}
        for idx in range(self.n):
            canonical = self.canonical_index(idx)
            if canonical is not None:
                self._groups.setdefault(int(canonical), []).append(int(idx))
        for idx in self.active_indices:
            self._groups.setdefault(int(idx), [int(idx)])
        for key, values in list(self._groups.items()):
            self._groups[key] = sorted(set(values))

        self._box_cache: Dict[int, AliasBox] = {}
        self._voxel_flat: Optional[np.ndarray] = None
        self._voxel_offsets: Optional[np.ndarray] = None
        self._voxel_levels: Optional[np.ndarray] = None
        if self.prefer_voxel:
            flat = scene_state.get("object_voxel_keys_flat")
            offsets = scene_state.get("object_voxel_keys_offsets")
            levels = scene_state.get("object_voxel_levels")
            if flat is not None and offsets is not None and levels is not None:
                self._voxel_flat = _to_numpy(flat, dtype=np.int64)
                self._voxel_offsets = _to_numpy(offsets, dtype=np.int64)
                self._voxel_levels = _to_numpy(levels, dtype=np.int64)

    @property
    def active_indices(self) -> List[int]:
        return [int(i) for i in np.nonzero(self.active[: self.n])[0]]

    def object_id(self, idx: int) -> int:
        if 0 <= int(idx) < self.object_ids.shape[0]:
            return int(self.object_ids[int(idx)])
        return int(idx)

    def resolve_object_id(self, object_id: int) -> int:
        redirects = self.scene_state.get("id_redirect") or {}
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
            try:
                nxt_int = int(nxt)
            except Exception:
                break
            if nxt_int == cur:
                break
            cur = nxt_int
        return int(cur)

    def canonical_index(self, idx: int) -> Optional[int]:
        if idx < 0 or idx >= self.n:
            return None
        return self._id_to_idx.get(self.resolve_object_id(self.object_id(idx)))

    def canonical_object_id(self, idx: int) -> int:
        canonical = self.canonical_index(idx)
        if canonical is None:
            return self.resolve_object_id(self.object_id(idx))
        return self.object_id(canonical)

    def alias_group(self, idx: int) -> List[int]:
        canonical = self.canonical_index(idx)
        if canonical is None:
            return [int(idx)] if 0 <= int(idx) < self.n else []
        return list(self._groups.get(int(canonical), [int(canonical)]))

    def box_for_index(self, idx: int) -> Optional[AliasBox]:
        idx = int(idx)
        if idx in self._box_cache:
            return self._box_cache[idx]
        if idx < 0 or idx >= self.n:
            return None
        canonical = self.canonical_index(idx)
        if canonical is None:
            canonical = idx
        canonical_oid = self.object_id(canonical)
        alias_oid = self.object_id(idx)

        if (
            self._voxel_flat is not None
            and self._voxel_offsets is not None
            and self._voxel_levels is not None
            and idx + 1 < self._voxel_offsets.shape[0]
        ):
            s = int(self._voxel_offsets[idx])
            e = int(self._voxel_offsets[idx + 1])
            if 0 <= s < e <= self._voxel_flat.shape[0]:
                level = int(self._voxel_levels[idx]) if idx < self._voxel_levels.shape[0] else 0
                box = voxel_cloud_aabb(self._voxel_flat[s:e], level)
                if box is not None:
                    out = AliasBox(canonical, canonical_oid, idx, alias_oid, box[0], box[1], "voxel")
                    self._box_cache[idx] = out
                    return out

        if idx < self.means.shape[0] and idx < self.cov6.shape[0]:
            bb_min, bb_max = gaussian_aabb(self.means[idx], self.cov6[idx], k_sigma=self.k_sigma)
            source = "gaussian_fallback" if self.prefer_voxel else "gaussian"
            out = AliasBox(canonical, canonical_oid, idx, alias_oid, bb_min, bb_max, source)
            self._box_cache[idx] = out
            return out
        return None

    def get_aabb(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        box = self.box_for_index(idx)
        if box is not None:
            return box.bbox_min, box.bbox_max
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    def _alias_order_score(self, idx: int, source_idx: int, query_tokens: set[str], order: str) -> float:
        idx = int(idx)
        source_idx = int(source_idx)
        text_parts: List[str] = []
        if idx < len(self.captions):
            text_parts.append(str(self.captions[idx] or ""))
        if idx < len(self.categories):
            text_parts.append(str(self.categories[idx] or ""))
        text_score = _token_score(query_tokens, " ".join(text_parts))
        count_score = 0.0
        if idx < self.counts.shape[0] and self.counts.shape[0] > 0:
            denom = float(np.nanmax(self.counts)) if self.counts.size else 1.0
            if denom > 0.0:
                count_score = min(1.0, max(0.0, float(self.counts[idx]) / denom))
        source_bonus = 1.0 if idx == source_idx else 0.0
        active_bonus = 0.15 if idx < self.active.shape[0] and bool(self.active[idx]) else 0.0
        order_norm = str(order or "source_first").strip().lower()
        if order_norm == "text_first":
            return 2.0 * text_score + 0.20 * source_bonus + 0.05 * count_score + active_bonus
        if order_norm == "count_first":
            return count_score + 0.25 * text_score + 0.20 * source_bonus + active_bonus
        return source_bonus + 0.75 * text_score + 0.05 * count_score + active_bonus

    def alias_boxes(
        self,
        idx: int,
        *,
        query_text: str = "",
        max_aliases: int = 2,
        order: str = "source_first",
    ) -> List[AliasBox]:
        """Return alias boxes for ``idx``'s canonical group in a deterministic order."""

        idx = int(idx)
        query_tokens = _tokens(query_text)
        scored: List[tuple[float, int, AliasBox]] = []
        for alias_idx in self.alias_group(idx):
            box = self.box_for_index(alias_idx)
            if box is None:
                continue
            score = self._alias_order_score(alias_idx, idx, query_tokens, order)
            scored.append(
                (
                    score,
                    int(alias_idx),
                    AliasBox(
                        box.canonical_index,
                        box.canonical_object_id,
                        box.alias_index,
                        box.alias_object_id,
                        box.bbox_min,
                        box.bbox_max,
                        box.source,
                        alias_score=float(score),
                    ),
                )
            )
        scored.sort(key=lambda item: (item[0], item[1] == idx, -item[1]), reverse=True)
        limit = max(1, int(max_aliases))
        return [box for _score, _idx, box in scored[:limit]]

    def iter_expanded_alias_boxes(
        self,
        indices: Iterable[int],
        *,
        query_text: str = "",
        max_aliases_per_candidate: int = 2,
        order: str = "source_first",
    ) -> List[AliasBox]:
        """Expand ranked object indices into deduplicated alias-box hypotheses."""

        out: List[AliasBox] = []
        seen: set[tuple[int, int]] = set()
        for idx in indices:
            for box in self.alias_boxes(
                int(idx),
                query_text=query_text,
                max_aliases=max_aliases_per_candidate,
                order=order,
            ):
                key = (int(box.canonical_object_id), int(box.alias_index))
                if key in seen:
                    continue
                seen.add(key)
                out.append(box)
        return out
