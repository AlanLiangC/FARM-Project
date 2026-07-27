"""Spatial-relation based retrieval via predicate decomposition and fuzzy logic."""

from .calibration import PredicateCalibration, SpatialCalibrator
from .executor import execute_spatial_query
from .methods import (
    SPATIAL_METHODS,
    SpatialMethod,
    get_spatial_method,
    normalize_spatial_method,
    spatial_method_choices,
)
from .models import Predicate, PredicateResult, QueryGraph, ScoredCandidate
from .query_parser import parse_query

__all__ = [
    "PredicateCalibration",
    "Predicate",
    "PredicateResult",
    "QueryGraph",
    "ScoredCandidate",
    "SPATIAL_METHODS",
    "SpatialCalibrator",
    "SpatialMethod",
    "execute_spatial_query",
    "get_spatial_method",
    "normalize_spatial_method",
    "parse_query",
    "spatial_method_choices",
]
