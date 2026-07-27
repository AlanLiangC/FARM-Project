"""Data models for spatial-relation based retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class Predicate:
    """A single spatial/semantic predicate parsed from a natural language query."""

    name: str
    args: List[str]
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryGraph:
    """Parsed representation of a spatial query as target + predicates.

    ``target_description`` is the LLM-extracted clean target noun phrase used
    for **semantic retrieval** (embedding similarity).

    ``target_class`` (Track B, locked 2026-05-17)
    is the LLM-extracted canonical class noun used for the
    **class-match factor**. When the LLM omits it, ``target_class`` is
    ``None`` and downstream code falls back to regex-based extraction from
    ``target_description``. For well-formed queries the two fields are
    usually identical; the field exists to robustly handle paraphrased /
    functional / multi-hop queries (e.g., "the broken one") on free-form
    benchmarks like OpenEQA target-noun grounding or real-robot trials.
    """

    target_description: str
    predicates: List[Predicate]
    reasoning: str = ""
    target_class: Optional[str] = None


@dataclass
class PredicateResult:
    """Evaluation result for a single predicate on a candidate."""

    name: str
    score: float
    status: Literal["evaluated", "fast_path", "dropped"]
    drop_reason: Optional[str] = None


@dataclass
class ScoredCandidate:
    """A candidate object scored against all predicates in a QueryGraph."""

    object_index: int
    object_id: int
    predicate_results: List[PredicateResult]
    composite_score: float
    matched_anchors: Dict[str, int] = field(default_factory=dict)
    target_similarity: Optional[float] = None
    vlm_rerank_score: Optional[float] = None
    predicate_geo_mean: Optional[float] = None
    predicate_weight: Optional[float] = None

    @property
    def predicates_evaluated(self) -> int:
        return sum(1 for r in self.predicate_results if r.status != "dropped")

    @property
    def predicates_dropped(self) -> int:
        return sum(1 for r in self.predicate_results if r.status == "dropped")
