"""Predicate score calibration helpers for spatial reasoning."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def clamp_probability(value: float, eps: float = 1e-6) -> float:
    """Clamp a numeric score to a valid probability interval."""

    if not math.isfinite(float(value)):
        return eps
    return max(float(eps), min(1.0 - float(eps), float(value)))


@dataclass(frozen=True)
class PredicateCalibration:
    """Calibration model for one predicate.

    Supported kinds:
      - ``identity``: return the raw score.
      - ``logistic``: sigmoid(intercept + coef[0] * raw_score).
      - ``bin``: empirical bin calibration over raw scores in [0, 1].
    """

    kind: str = "identity"
    intercept: float = 0.0
    coef: List[float] | None = None
    bins: List[Dict[str, float]] | None = None
    eps: float = 1e-6

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PredicateCalibration":
        return cls(
            kind=str(data.get("kind", "identity")),
            intercept=float(data.get("intercept", 0.0)),
            coef=[float(x) for x in data.get("coef", [])],
            bins=[dict(b) for b in data.get("bins", [])],
            eps=float(data.get("eps", 1e-6)),
        )

    def calibrate(self, raw_score: float) -> float:
        x = clamp_probability(raw_score, self.eps)
        kind = self.kind.lower()
        if kind == "identity":
            return x
        if kind == "logistic":
            coef = self.coef or [1.0]
            z = float(self.intercept) + float(coef[0]) * x
            z = max(-60.0, min(60.0, z))
            return clamp_probability(1.0 / (1.0 + math.exp(-z)), self.eps)
        if kind == "bin":
            for b in self.bins or []:
                lo = float(b.get("lo", 0.0))
                hi = float(b.get("hi", 1.0))
                if lo <= x <= hi or (x == 1.0 and hi >= 1.0):
                    return clamp_probability(float(b.get("prob", x)), self.eps)
            return x
        return x


class SpatialCalibrator:
    """Collection of per-predicate calibration models."""

    def __init__(self, predicates: Optional[Mapping[str, PredicateCalibration]] = None):
        self.predicates: Dict[str, PredicateCalibration] = dict(predicates or {})

    @classmethod
    def identity(cls) -> "SpatialCalibrator":
        return cls({})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SpatialCalibrator":
        raw = data.get("predicates", data)
        predicates: Dict[str, PredicateCalibration] = {}
        if isinstance(raw, Mapping):
            for name, payload in raw.items():
                if isinstance(payload, Mapping):
                    predicates[str(name)] = PredicateCalibration.from_dict(payload)
        return cls(predicates)

    @classmethod
    def load(cls, path: str | Path | None) -> "SpatialCalibrator":
        if path is None:
            return cls.identity()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"spatial calibration file not found: {p}")
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def calibrate(self, predicate_name: str, raw_score: float) -> float:
        model = self.predicates.get(str(predicate_name))
        if model is None:
            return clamp_probability(raw_score)
        return model.calibrate(raw_score)

    def available_predicates(self) -> List[str]:
        return sorted(self.predicates)


def logprob_score(
    predicate_scores: Iterable[tuple[str, float]],
    target_score: float,
    calibrator: Optional[SpatialCalibrator] = None,
) -> float:
    """Compose semantic and predicate probabilities by multiplying in log space."""

    cal = calibrator or SpatialCalibrator.identity()
    log_p = math.log(clamp_probability(target_score))
    for name, raw in predicate_scores:
        log_p += math.log(cal.calibrate(name, raw))
    return float(math.exp(max(-60.0, min(0.0, log_p))))
