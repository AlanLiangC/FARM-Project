"""
Scene graph pruning: compute which object indices should be soft-pruned (marked inactive).

Criteria are modular callables; an object is pruned only when all registered criteria
return True for that object. New criteria can be added without changing the core loop.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

# Criterion: (state, obj_idx) -> True if this object passes this criterion (eligible to prune).
PruneCriterion = Callable[[Dict[str, Any], int], bool]


def caption_keywords_criterion(
    keywords: List[str],
    case_sensitive: bool = False,
) -> PruneCriterion:
    """
    Returns a criterion that is True when the object's caption contains any of the
    given keywords (substring match). Empty keywords list yields a criterion that
    always returns False.
    """
    if not keywords:
        return lambda _state, _idx: False

    if case_sensitive:

        def criterion(state: Dict[str, Any], obj_idx: int) -> bool:
            captions = state.get("object_caption") or []
            if obj_idx < 0 or obj_idx >= len(captions):
                return False
            raw = captions[obj_idx]
            caption = str(raw).strip() if raw is not None else ""
            if not caption:
                return False
            return any(kw in caption for kw in keywords)

    else:
        _kw_lower = [k.strip().lower() for k in keywords if k.strip()]

        def criterion(state: Dict[str, Any], obj_idx: int) -> bool:
            if not _kw_lower:
                return False
            captions = state.get("object_caption") or []
            if obj_idx < 0 or obj_idx >= len(captions):
                return False
            raw = captions[obj_idx]
            caption = str(raw).strip() if raw is not None else ""
            if not caption:
                return False
            caption_lower = caption.lower()
            return any(kw in caption_lower for kw in _kw_lower)

    return criterion


def compute_indices_to_prune(
    state: Dict[str, Any],
    criteria: Sequence[PruneCriterion],
) -> List[int]:
    """
    Returns list of object indices that should be pruned: active objects for which
    every criterion returns True. Handles missing/wrong-length state defensively.
    """
    if not criteria:
        return []

    active = state.get("active")
    if active is None:
        return []

    try:
        n = len(active)
    except TypeError:
        return []

    result: List[int] = []
    for i in range(n):
        try:
            is_active = active[i]
            if hasattr(is_active, "item"):
                is_active = bool(is_active.item())
            else:
                is_active = bool(is_active)
        except (IndexError, KeyError, TypeError):
            continue
        if not is_active:
            continue
        try:
            if all(c(state, i) for c in criteria):
                result.append(i)
        except Exception:
            continue
    return result
