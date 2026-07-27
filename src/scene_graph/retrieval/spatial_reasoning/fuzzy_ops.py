"""Fuzzy logic operations for composing predicate scores."""

from __future__ import annotations

import math
from typing import List


def gaussian_membership(distance: float, sigma: float) -> float:
    """Monotonic decay from 1.0 at distance=0. Used for Near-type predicates."""
    return math.exp(-(distance**2) / (2 * sigma**2))


def sigmoid_membership(value: float, center: float, temperature: float) -> float:
    """Soft step function, 0.5 at value=center. Used for Above/Below."""
    exponent = -(value - center) / temperature
    exponent = max(-500.0, min(500.0, exponent))
    return 1.0 / (1.0 + math.exp(exponent))


def t_norm_product(scores: List[float]) -> float:
    """Product t-norm: AND composition. Any zero kills the result."""
    result = 1.0
    for s in scores:
        result *= s
    return result


def t_conorm_probabilistic(scores: List[float]) -> float:
    """Probabilistic sum: OR composition."""
    result = 0.0
    for s in scores:
        result = result + s - result * s
    return result


def t_norm_weighted(scores: List[float], weights: List[float]) -> float:
    """Weighted product: s1^w1 * s2^w2 * ... Allows importance weighting."""
    result = 1.0
    for s, w in zip(scores, weights):
        if s <= 0.0:
            return 0.0
        result *= s**w
    return result


def normalize_vlm_score(raw_score: int | float) -> float:
    """Convert VLM 0-100 integer score to [0, 1] fuzzy membership."""
    return max(0.0, min(1.0, float(raw_score) / 100.0))
