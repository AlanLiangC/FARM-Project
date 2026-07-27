"""Canonical-view selection for apples-to-apples visible-mask IoU scoring.

The visible-mask scorer in :mod:`scene_graph.eval.visible_mask` historically
iterates over up to ``max_views`` saved observation frames per predicted
candidate and reports ``best_iou = max IoU across frames``. This systematically
favours methods that persist many frames per object (ours typically retains
N≈50, whereas BBQ and RynnBrain effectively report N=1). To make the IoU
comparison fair, each method must commit to a single canonical view per
prediction *before* IoU is measured.

Pickers consume the per-object metadata that's already persisted in
``scene_state["object_mask_observations"]`` (per-detection records with
``image_id``, ``raw_pixels``, ``image_shape``, ``crop_bbox_xyxy``) plus the
high-quality-view pool in ``scene_state["viewpoint_image_ids"]``. None of the
pickers load mask pixel data; they decide from metadata only, which keeps the
ablation pass cheap.

Tie-breaker: the smaller ``image_id`` wins on equal scores, so picks are
deterministic across runs.

Design rationale: single-view scoring with a deterministic picker is what the
2026-05-16 ablation locked for the headline visible-mask metric.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

ViewPickerFn = Callable[..., Optional[int]]

VIEW_PICKERS: Dict[str, ViewPickerFn] = {}


def register_view_picker(name: str) -> Callable[[ViewPickerFn], ViewPickerFn]:
    def decorator(fn: ViewPickerFn) -> ViewPickerFn:
        VIEW_PICKERS[name] = fn
        return fn
    return decorator


def list_view_pickers() -> List[str]:
    return sorted(VIEW_PICKERS)


def get_view_picker(name: str) -> ViewPickerFn:
    if name not in VIEW_PICKERS:
        raise ValueError(
            f"unknown view picker {name!r}; valid: {sorted(VIEW_PICKERS)}"
        )
    return VIEW_PICKERS[name]


def _image_area(observation: Mapping[str, Any]) -> int:
    shape = observation.get("image_shape") or ()
    try:
        if len(shape) >= 2:
            return int(shape[0]) * int(shape[1])
    except TypeError:
        pass
    return 0


def _edge_touch_fraction(observation: Mapping[str, Any]) -> float:
    """Fraction in {0, 0.25, 0.5, 0.75, 1.0} of bbox edges touching the border."""
    bbox = observation.get("crop_bbox_xyxy")
    shape = observation.get("image_shape") or ()
    if bbox is None:
        return 0.0
    try:
        if len(bbox) != 4 or len(shape) < 2:
            return 0.0
        H, W = int(shape[0]), int(shape[1])
        x0, y0, x1, y1 = (int(v) for v in bbox)
    except (TypeError, ValueError):
        return 0.0
    if H <= 0 or W <= 0:
        return 0.0
    touches = 0
    if x0 <= 1:
        touches += 1
    if y0 <= 1:
        touches += 1
    if x1 >= W - 1:
        touches += 1
    if y1 >= H - 1:
        touches += 1
    return touches / 4.0


def _image_id(observation: Mapping[str, Any]) -> int:
    try:
        return int(observation.get("image_id"))
    except (TypeError, ValueError):
        return -1


def _select_max(
    observations: Sequence[Mapping[str, Any]],
    score_fn: Callable[[Mapping[str, Any]], float],
) -> Optional[Mapping[str, Any]]:
    best: Optional[Mapping[str, Any]] = None
    best_score = -1.0
    best_image_id = -1
    for obs in observations:
        s = score_fn(obs)
        iid = _image_id(obs)
        if s > best_score or (s == best_score and (best is None or iid < best_image_id)):
            best, best_score, best_image_id = obs, s, iid
    return best


@register_view_picker("v0_multiview")
def picker_v0_multiview(
    observations: Sequence[Mapping[str, Any]], **_: Any
) -> Optional[int]:
    """Return ``None`` → caller falls back to the legacy best-of-N scorer."""
    return None


@register_view_picker("v1_largest_mask")
def picker_v1_largest_mask(
    observations: Sequence[Mapping[str, Any]], **_: Any
) -> Optional[int]:
    """Pick the observation maximising ``raw_pixels / (H * W)``."""

    def score(obs: Mapping[str, Any]) -> float:
        area = _image_area(obs)
        if area <= 0:
            return -1.0
        return float(obs.get("raw_pixels", 0)) / float(area)

    pick = _select_max(observations, score)
    return _image_id(pick) if pick is not None else None


@register_view_picker("v2_largest_nonedge_mask")
def picker_v2_largest_nonedge_mask(
    observations: Sequence[Mapping[str, Any]], **_: Any
) -> Optional[int]:
    """Pick the observation maximising ``mask_ratio * (1 - edge_touch_fraction)``."""

    def score(obs: Mapping[str, Any]) -> float:
        area = _image_area(obs)
        if area <= 0:
            return -1.0
        ratio = float(obs.get("raw_pixels", 0)) / float(area)
        return ratio * (1.0 - _edge_touch_fraction(obs))

    pick = _select_max(observations, score)
    return _image_id(pick) if pick is not None else None


@register_view_picker("v6_hq_median")
def picker_v6_hq_median(
    observations: Sequence[Mapping[str, Any]],
    *,
    hq_image_ids: Sequence[int] = (),
    **_: Any,
) -> Optional[int]:
    """Median by ``raw_pixels / image_area`` within the high-quality pool.

    Falls back to all observations if the high-quality pool has no overlap with
    the saved mask observations.
    """
    hq_set = {int(x) for x in hq_image_ids}
    pool = [obs for obs in observations if _image_id(obs) in hq_set]
    if not pool:
        pool = list(observations)
    if not pool:
        return None

    def key(obs: Mapping[str, Any]):
        area = _image_area(obs)
        ratio = float(obs.get("raw_pixels", 0)) / float(area) if area > 0 else 0.0
        return (ratio, _image_id(obs))

    pool_sorted = sorted(pool, key=key)
    # Upper median for even-length lists is the deterministic choice.
    return _image_id(pool_sorted[len(pool_sorted) // 2])


def select_canonical_view(
    *,
    object_mask_observations: Sequence[Mapping[str, Any]],
    picker: str = "v0_multiview",
    hq_image_ids: Sequence[int] = (),
) -> Optional[int]:
    """Run the named picker against a single object's saved observations."""
    fn = get_view_picker(picker)
    if not object_mask_observations:
        return None
    return fn(object_mask_observations, hq_image_ids=hq_image_ids)


def resolve_chosen_view_image_id(
    mask_index: Any,
    candidate: Mapping[str, Any],
    picker_name: str = "v0_multiview",
) -> Optional[int]:
    """Compute the canonical-view image_id for a candidate.

    Resolution precedence:
      1. ``candidate["chosen_view_image_id"]`` (or aliases ``canonical_view_image_id``,
         ``view_image_id``, ``frame_idx`` for RynnBrain, ``color_image_idx`` for BBQ)
         — used as-is. This is how baselines declare their natively-single view.
      2. Otherwise, if ``picker_name != "v0_multiview"``, run the named picker
         against the candidate's ``evidence_object_id`` mask observations.
      3. Otherwise return ``None`` (legacy best-of-N back-compat).

    Helper used by per-utterance scorers in
    :mod:`scene_graph.eval.referit3d.metrics` and
    :mod:`scene_graph.eval.iref_vla.metrics`.

    ``mask_index`` must expose:
      - ``candidate_evidence_object_id(candidate) -> int``
      - ``object_id_to_index: Dict[int, int]``
      - ``object_mask_observations: Sequence[Sequence[Mapping[str, Any]]]``
      - ``viewpoint_image_ids: Sequence[Sequence[int]]``
    """

    # Precedence 1: candidate already declares its canonical view explicitly.
    # Only the canonical field name is honored — baselines (BBQ / RynnBrain) that
    # want to declare their natively-single view must run their converter to
    # write ``chosen_view_image_id`` rather than rely on alias auto-inference,
    # which would risk hijacking ours' candidate records that don't carry a
    # commitment to a specific frame.
    if isinstance(candidate, Mapping):
        explicit = candidate.get("chosen_view_image_id")
        if explicit is not None:
            try:
                return int(explicit)
            except (TypeError, ValueError):
                pass

    # Precedence 2: V0 → None (back-compat best-of-N scoring path).
    if picker_name == "v0_multiview":
        return None

    # Precedence 3: run the named picker against the candidate's evidence object.
    evidence_oid = mask_index.candidate_evidence_object_id(candidate)
    idx = mask_index.object_id_to_index.get(int(evidence_oid))
    if idx is None:
        return None

    mask_obs_list = mask_index.object_mask_observations
    if idx >= len(mask_obs_list):
        return None
    raw = mask_obs_list[idx]
    if not isinstance(raw, (list, tuple)):
        return None
    observations: List[Mapping[str, Any]] = list(raw)

    hq_image_ids: Sequence[int] = ()
    viewpoint_image_ids = getattr(mask_index, "viewpoint_image_ids", None) or []
    if idx < len(viewpoint_image_ids):
        row = viewpoint_image_ids[idx]
        if isinstance(row, (list, tuple)):
            hq_image_ids = tuple(int(x) for x in row)

    return select_canonical_view(
        object_mask_observations=observations,
        picker=picker_name,
        hq_image_ids=hq_image_ids,
    )
