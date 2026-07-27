"""Named spatial-disambiguation method profiles.

The profiles are intentionally small: they select which parts of the existing
executor are active and how predicate scores are composed. This keeps eval runs
comparable without forking the retrieval pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SpatialMethod:
    """Configuration for one spatial reasoning ablation."""

    name: str
    description: str
    evaluate_predicates: bool
    force_no_vlm: bool
    composition: str
    predicate_weight: Optional[float] = None
    require_target_class_filter: bool = False
    target_class_filter_min_candidates: int = 2
    # When set (float in (0, 1]), replace
    # the hard ``require_target_class_filter`` prune with a per-candidate
    # ``class_factor`` ∈ {class_mismatch_floor, 1.0} that multiplies the
    # candidate's target similarity BEFORE spatial-predicate scoring.
    # ``None`` means "no soft-class factor; honor require_target_class_filter
    # if set, otherwise no class scoring at all" (back-compat path).
    class_mismatch_floor: Optional[float] = None
    # Track B (2026-05-17): which candidate signal(s) to match the
    # parsed target_class against. ``"tiered"`` (default) checks YOLOE
    # ``object_category`` first and falls back to ``object_caption`` only when
    # the category is generic/uninformative. ``"yoloe"`` skips the caption.
    # ``"caption"`` skips the category. ``"both"`` matches if EITHER signal
    # contains the class term. Honored when ``class_mismatch_floor`` is set.
    class_match_source: str = "tiered"
    # Track B (2026-05-17): confidence-based fallback in tiered mode.
    # When set (float in [0, 1]), the tiered fallback condition becomes
    # "YOLOE category is generic OR ``object_detection_category_conf[idx]``
    # for the chosen category is below this threshold". Higher threshold →
    # more fallbacks to caption (catches low-confidence YOLOE detections).
    # ``None`` (default) preserves the current behavior: fallback only when
    # the YOLOE category is one of {none, object, other, target, thing,
    # unknown}. Honored only when ``class_match_source == "tiered"``.
    class_match_tiered_conf_threshold: Optional[float] = None
    # Optional local covisibility constraint for target-anchor predicates.
    # 0 disables it. When >0, pairwise spatial predicates only consider anchor
    # candidates reachable from the target by this many hops in the selected
    # covisibility graph.
    covisibility_hops: int = 0
    # Source for the optional covisibility constraint. "shared_images" matches
    # the ablation definition: one hop means two objects were visible from the
    # same frame/viewpoint. "bitset" uses the packed online mapper graph.
    covisibility_source: str = "auto"


SPATIAL_METHODS: Dict[str, SpatialMethod] = {
    "current": SpatialMethod(
        name="current",
        description="Current soft geometric predicates with optional gated VLM rerank.",
        evaluate_predicates=True,
        force_no_vlm=False,
        composition="current",
        predicate_weight=0.25,
    ),
    "semantic_only": SpatialMethod(
        name="semantic_only",
        description="Semantic retrieval only; parsed spatial predicates are ignored.",
        evaluate_predicates=False,
        force_no_vlm=True,
        composition="semantic",
    ),
    "hard_predicates": SpatialMethod(
        name="hard_predicates",
        description="Fast-path predicates as hard filters, then semantic ranking.",
        evaluate_predicates=True,
        force_no_vlm=True,
        composition="hard",
    ),
    "soft_predicates": SpatialMethod(
        name="soft_predicates",
        description="Fast-path soft predicates, no VLM.",
        evaluate_predicates=True,
        force_no_vlm=True,
        composition="current",
        predicate_weight=0.25,
    ),
    "soft_predicates_w45": SpatialMethod(
        name="soft_predicates_w45",
        description="Fast-path soft predicates with 45% predicate weight, no VLM.",
        evaluate_predicates=True,
        force_no_vlm=True,
        composition="current",
        predicate_weight=0.45,
    ),
    "soft_predicates_w50": SpatialMethod(
        name="soft_predicates_w50",
        description="Fast-path soft predicates with 50% predicate weight, no VLM.",
        evaluate_predicates=True,
        force_no_vlm=True,
        composition="current",
        predicate_weight=0.50,
    ),
    "target_class_soft_predicates_w50": SpatialMethod(
        name="target_class_soft_predicates_w50",
        description=(
            "Fast-path soft predicates with 50% predicate weight, only after "
            "a target-class lexical filter succeeds; otherwise semantic-only."
        ),
        evaluate_predicates=True,
        force_no_vlm=True,
        composition="current",
        predicate_weight=0.50,
        require_target_class_filter=True,
        target_class_filter_min_candidates=2,
    ),
    "calibrated_logprob": SpatialMethod(
        name="calibrated_logprob",
        description="Calibrated predicate probabilities composed in log space.",
        evaluate_predicates=True,
        force_no_vlm=False,
        composition="logprob",
    ),
    # "Unified spatial method": soft-class
    # variants of soft_predicates_w50: no hard target_class filter prune;
    # instead, candidates whose category/caption doesn't match the parsed
    # target class get their target similarity multiplied by class_mismatch_floor.
    # Three floors registered for the ablation (0.3, 0.5, 0.7);
    # winner becomes the locked default unified_soft_w50.
    "unified_soft_w50_f03": SpatialMethod(
        name="unified_soft_w50_f03",
        description="Unified soft predicates w50 + soft class factor (floor=0.3).",
        evaluate_predicates=True,
        force_no_vlm=True,
        composition="current",
        predicate_weight=0.50,
        require_target_class_filter=False,
        class_mismatch_floor=0.3,
    ),
    "unified_soft_w50_f05": SpatialMethod(
        name="unified_soft_w50_f05",
        description="Unified soft predicates w50 + soft class factor (floor=0.5).",
        evaluate_predicates=True,
        force_no_vlm=True,
        composition="current",
        predicate_weight=0.50,
        require_target_class_filter=False,
        class_mismatch_floor=0.5,
    ),
    "unified_soft_w50_f07": SpatialMethod(
        name="unified_soft_w50_f07",
        description="Unified soft predicates w50 + soft class factor (floor=0.7).",
        evaluate_predicates=True,
        force_no_vlm=True,
        composition="current",
        predicate_weight=0.50,
        require_target_class_filter=False,
        class_mismatch_floor=0.7,
    ),
    # Locked default (2026-05-17 ablation
    # on the 5-scene subset). class_mismatch_floor=0.3 chosen for the biggest
    # HM3D headroom gain (+0.004 absolute, +10% relative); ScanNet regression
    # is -0.0034 (within ±0.005 acceptance bar; <1σ at n=887). Replaces the
    # per-benchmark dispatch `soft_predicates_w50` (ScanNet) / `class_soft_w50`
    # (HM3D).
    "unified_soft_w50": SpatialMethod(
        name="unified_soft_w50",
        description=(
            "Unified spatial method (locked 2026-05-17). Soft predicates w50 + "
            "soft class factor at class_mismatch_floor=0.3. No hard target_class "
            "prune. Replaces soft_predicates_w50 / class_soft_w50."
        ),
        evaluate_predicates=True,
        force_no_vlm=True,
        composition="current",
        predicate_weight=0.50,
        require_target_class_filter=False,
        class_mismatch_floor=0.3,
        class_match_source="tiered",
    ),
    # Track B diagnostic siblings (class-match source ablation).
    # All share class_mismatch_floor=0.3 (locked Track A choice); only the
    # class-signal source varies.
    "unified_soft_w50_yoloe": SpatialMethod(
        name="unified_soft_w50_yoloe",
        description="unified_soft_w50 + class match against YOLOE category only (no caption fallback).",
        evaluate_predicates=True,
        force_no_vlm=True,
        composition="current",
        predicate_weight=0.50,
        require_target_class_filter=False,
        class_mismatch_floor=0.3,
        class_match_source="yoloe",
    ),
    "unified_soft_w50_caption": SpatialMethod(
        name="unified_soft_w50_caption",
        description="unified_soft_w50 + class match against VLM caption only (skip YOLOE category).",
        evaluate_predicates=True,
        force_no_vlm=True,
        composition="current",
        predicate_weight=0.50,
        require_target_class_filter=False,
        class_mismatch_floor=0.3,
        class_match_source="caption",
    ),
    "unified_soft_w50_both": SpatialMethod(
        name="unified_soft_w50_both",
        description="unified_soft_w50 + class match if EITHER YOLOE category OR VLM caption contains the class term.",
        evaluate_predicates=True,
        force_no_vlm=True,
        composition="current",
        predicate_weight=0.50,
        require_target_class_filter=False,
        class_mismatch_floor=0.3,
        class_match_source="both",
    ),
    # Track B (2026-05-17) — confidence-based fallback ablation.
    # Same as `unified_soft_w50` (tiered) but the fallback to caption ALSO
    # fires when YOLOE confidence for the chosen category is below a
    # threshold (above the streaming-mapping default of 0.20/0.25). Three
    # thresholds swept to see whether catching low-confidence YOLOE labels
    # via caption helps recall.
    "unified_soft_w50_conf030": SpatialMethod(
        name="unified_soft_w50_conf030",
        description="unified_soft_w50 + tiered fallback also fires when YOLOE conf < 0.30.",
        evaluate_predicates=True,
        force_no_vlm=True,
        composition="current",
        predicate_weight=0.50,
        require_target_class_filter=False,
        class_mismatch_floor=0.3,
        class_match_source="tiered",
        class_match_tiered_conf_threshold=0.30,
    ),
    "unified_soft_w50_conf045": SpatialMethod(
        name="unified_soft_w50_conf045",
        description="unified_soft_w50 + tiered fallback also fires when YOLOE conf < 0.45.",
        evaluate_predicates=True,
        force_no_vlm=True,
        composition="current",
        predicate_weight=0.50,
        require_target_class_filter=False,
        class_mismatch_floor=0.3,
        class_match_source="tiered",
        class_match_tiered_conf_threshold=0.45,
    ),
    "unified_soft_w50_conf060": SpatialMethod(
        name="unified_soft_w50_conf060",
        description="unified_soft_w50 + tiered fallback also fires when YOLOE conf < 0.60.",
        evaluate_predicates=True,
        force_no_vlm=True,
        composition="current",
        predicate_weight=0.50,
        require_target_class_filter=False,
        class_mismatch_floor=0.3,
        class_match_source="tiered",
        class_match_tiered_conf_threshold=0.60,
    ),
    "unified_soft_w50_covis3": SpatialMethod(
        name="unified_soft_w50_covis3",
        description=(
            "unified_soft_w50 plus a 3-hop local covisibility constraint for "
            "target-anchor spatial predicate assignments."
        ),
        evaluate_predicates=True,
        force_no_vlm=True,
        composition="current",
        predicate_weight=0.50,
        require_target_class_filter=False,
        class_mismatch_floor=0.3,
        class_match_source="tiered",
        covisibility_hops=3,
        covisibility_source="shared_images",
    ),
    "joint_v1": SpatialMethod(
        name="joint_v1",
        description=(
            "Truncation-free vectorized joint relational retrieval: calibrated "
            "anchor affinities over ALL objects x vectorized predicate matrices "
            "(execute_spatial_query dispatches to joint_executor before profile "
            "resolution, so the fields below are not consulted)."
        ),
        evaluate_predicates=True,
        force_no_vlm=True,
        composition="current",
    ),
}

_ALIASES = {
    "default": "current",
    "spatial": "current",
    "soft": "soft_predicates",
    "soft_w45": "soft_predicates_w45",
    "soft45": "soft_predicates_w45",
    "w45": "soft_predicates_w45",
    "soft_w50": "soft_predicates_w50",
    "soft50": "soft_predicates_w50",
    "w50": "soft_predicates_w50",
    "class_soft_w50": "target_class_soft_predicates_w50",
    "class_filtered_soft_w50": "target_class_soft_predicates_w50",
    "target_class_w50": "target_class_soft_predicates_w50",
    "unified_w50_f03": "unified_soft_w50_f03",
    "unified_w50_f05": "unified_soft_w50_f05",
    "unified_w50_f07": "unified_soft_w50_f07",
    "unified": "unified_soft_w50",
    "unified_w50": "unified_soft_w50",
    "unified_covis3": "unified_soft_w50_covis3",
    "covis3": "unified_soft_w50_covis3",
    "target_class_soft": "target_class_soft_predicates_w50",
    "hard": "hard_predicates",
    "semantic": "semantic_only",
    "embedding": "semantic_only",
    "calibrated": "calibrated_logprob",
    "logprob": "calibrated_logprob",
    "csd": "soft_predicates",
    "csd_soft": "soft_predicates",
    "csd_hard": "hard_predicates",
    "csd_calibrated": "calibrated_logprob",
}


def normalize_spatial_method(name: str | None) -> str:
    """Return a canonical method name or raise ``ValueError``."""

    key = str(name or "current").strip().lower().replace("-", "_")
    key = _ALIASES.get(key, key)
    if key not in SPATIAL_METHODS:
        valid = ", ".join(sorted(SPATIAL_METHODS))
        raise ValueError(f"unknown spatial method {name!r}; valid methods: {valid}")
    return key


def spatial_method_choices() -> List[str]:
    """Return all canonical method names and accepted aliases for CLIs."""

    return sorted(set(SPATIAL_METHODS) | set(_ALIASES))


def get_spatial_method(name: str | None) -> SpatialMethod:
    """Resolve a method name to a :class:`SpatialMethod`."""

    return SPATIAL_METHODS[normalize_spatial_method(name)]
