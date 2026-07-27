"""Multi-modal scene graph retrieval.

Entry points: :class:`~scene_graph.retrieval.scene_graph_retriever.SceneGraphRetriever`
(embedding-based candidate retrieval over a saved scene state) and the
relational pipeline in :mod:`scene_graph.retrieval.spatial_reasoning`
(``parse_query`` -> ``execute_spatial_query``).
"""

from .spatial_reasoning import (
    Predicate,
    PredicateResult,
    QueryGraph,
    ScoredCandidate,
    execute_spatial_query,
    parse_query,
)

__all__ = [
    "Predicate",
    "PredicateResult",
    "QueryGraph",
    "ScoredCandidate",
    "execute_spatial_query",
    "parse_query",
]
