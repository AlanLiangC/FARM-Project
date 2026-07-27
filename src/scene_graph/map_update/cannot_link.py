"""Cannot-link constraints for object identity merges.

These constraints record object pairs that have been observed as distinct
detections in the same frame. They are stored by external object id so they
survive tensor reordering and remain valid across redirect/canonicalization.
"""

from __future__ import annotations

import contextlib
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch


CannotLinkState = Dict[int, Set[int]]


def _object_ids_as_list(scene_state: Dict[str, Any]) -> List[int]:
    object_ids = scene_state.get("object_id")
    if object_ids is None:
        return []
    if isinstance(object_ids, torch.Tensor):
        return [int(x) for x in object_ids.detach().to("cpu", dtype=torch.int64).tolist()]
    out: List[int] = []
    for value in object_ids:
        with contextlib.suppress(Exception):
            out.append(int(value))
    return out


def resolve_canonical_object_id(scene_state: Dict[str, Any], object_id: int) -> int:
    redirects = scene_state.get("id_redirect") or {}
    cur = int(object_id)
    seen: set[int] = set()
    while cur not in seen:
        seen.add(cur)
        nxt = redirects.get(cur) if isinstance(redirects, dict) else None
        if nxt is None and isinstance(redirects, dict):
            nxt = redirects.get(str(cur))
        if nxt is None:
            break
        with contextlib.suppress(Exception):
            nxt = int(nxt)
            if nxt == cur:
                break
            cur = nxt
            continue
        break
    return int(cur)


def canonical_object_id_for_index(scene_state: Dict[str, Any], obj_idx: int) -> Optional[int]:
    object_ids = _object_ids_as_list(scene_state)
    if obj_idx < 0 or obj_idx >= len(object_ids):
        return None
    return resolve_canonical_object_id(scene_state, object_ids[int(obj_idx)])


def _add_pair_to_state(state: CannotLinkState, a: int, b: int) -> None:
    a = int(a)
    b = int(b)
    if a == b:
        return
    state.setdefault(a, set()).add(b)
    state.setdefault(b, set()).add(a)


def normalize_cannot_link_state(scene_state: Dict[str, Any]) -> CannotLinkState:
    raw = scene_state.get("cannot_link_object_ids") or {}
    out: CannotLinkState = {}

    if isinstance(raw, dict):
        iterator = raw.items()
    else:
        iterator = []
        if isinstance(raw, (list, tuple, set)):
            iterator = raw

    if isinstance(raw, dict):
        for key, values in iterator:
            with contextlib.suppress(Exception):
                a = resolve_canonical_object_id(scene_state, int(key))
                for value in values or []:
                    b = resolve_canonical_object_id(scene_state, int(value))
                    _add_pair_to_state(out, a, b)
    else:
        for item in iterator:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            with contextlib.suppress(Exception):
                a = resolve_canonical_object_id(scene_state, int(item[0]))
                b = resolve_canonical_object_id(scene_state, int(item[1]))
                _add_pair_to_state(out, a, b)

    scene_state["cannot_link_object_ids"] = out
    return out


def add_cannot_link_object_ids(scene_state: Dict[str, Any], pairs: Iterable[Tuple[int, int]]) -> int:
    state = normalize_cannot_link_state(scene_state)
    n_added = 0
    for a_raw, b_raw in pairs:
        a = resolve_canonical_object_id(scene_state, int(a_raw))
        b = resolve_canonical_object_id(scene_state, int(b_raw))
        if a == b:
            continue
        before = len(state.get(a, set()))
        _add_pair_to_state(state, a, b)
        if len(state.get(a, set())) > before:
            n_added += 1
    return n_added


def add_cannot_link_indices(scene_state: Dict[str, Any], pairs: Iterable[Tuple[int, int]]) -> int:
    id_pairs: List[Tuple[int, int]] = []
    for a_idx, b_idx in pairs:
        a = canonical_object_id_for_index(scene_state, int(a_idx))
        b = canonical_object_id_for_index(scene_state, int(b_idx))
        if a is None or b is None:
            continue
        id_pairs.append((a, b))
    return add_cannot_link_object_ids(scene_state, id_pairs)


def are_cannot_linked_object_ids(scene_state: Dict[str, Any], a: int, b: int) -> bool:
    state = normalize_cannot_link_state(scene_state)
    ca = resolve_canonical_object_id(scene_state, int(a))
    cb = resolve_canonical_object_id(scene_state, int(b))
    return ca != cb and cb in state.get(ca, set())


def are_cannot_linked_indices(scene_state: Dict[str, Any], a_idx: int, b_idx: int) -> bool:
    a = canonical_object_id_for_index(scene_state, int(a_idx))
    b = canonical_object_id_for_index(scene_state, int(b_idx))
    if a is None or b is None:
        return False
    return are_cannot_linked_object_ids(scene_state, a, b)


def cannot_link_index_pairs(scene_state: Dict[str, Any], *, n_objects: Optional[int] = None) -> Set[Tuple[int, int]]:
    state = normalize_cannot_link_state(scene_state)
    object_ids = _object_ids_as_list(scene_state)
    if n_objects is not None:
        object_ids = object_ids[: int(n_objects)]
    id_to_idx = {
        resolve_canonical_object_id(scene_state, oid): idx
        for idx, oid in enumerate(object_ids)
    }
    pairs: Set[Tuple[int, int]] = set()
    for a, values in state.items():
        ia = id_to_idx.get(resolve_canonical_object_id(scene_state, int(a)))
        if ia is None:
            continue
        for b in values:
            ib = id_to_idx.get(resolve_canonical_object_id(scene_state, int(b)))
            if ib is None or ia == ib:
                continue
            pairs.add((min(ia, ib), max(ia, ib)))
    return pairs


def add_same_frame_cannot_links_from_preliminary_neighbors(
    scene_state: Dict[str, Any],
    neighbors: Sequence[Any],
    detection_image_ids: Sequence[Optional[int]],
) -> int:
    """Record cannot-links from same-frame detections' preliminary winners."""

    by_image: Dict[int, Set[int]] = defaultdict(set)
    for det_idx, neighbor in enumerate(neighbors or []):
        if det_idx >= len(detection_image_ids):
            continue
        image_id = detection_image_ids[det_idx]
        if image_id is None:
            continue
        if not isinstance(neighbor, torch.Tensor) or neighbor.numel() == 0:
            continue
        winner_idx = int(torch.min(neighbor).item())
        by_image[int(image_id)].add(winner_idx)

    pairs: List[Tuple[int, int]] = []
    for indices in by_image.values():
        ordered = sorted(indices)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                pairs.append((a, b))
    return add_cannot_link_indices(scene_state, pairs)


def add_same_frame_cannot_links_from_detection_assignments(
    scene_state: Dict[str, Any],
    detection_image_ids: Sequence[Optional[int]],
    detection_object_indices: Sequence[Optional[int]],
) -> int:
    by_image: Dict[int, Set[int]] = defaultdict(set)
    for det_idx, obj_idx in enumerate(detection_object_indices):
        if obj_idx is None:
            continue
        if det_idx >= len(detection_image_ids):
            continue
        image_id = detection_image_ids[det_idx]
        if image_id is None:
            continue
        by_image[int(image_id)].add(int(obj_idx))

    pairs: List[Tuple[int, int]] = []
    for indices in by_image.values():
        ordered = sorted(indices)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                pairs.append((a, b))
    return add_cannot_link_indices(scene_state, pairs)
