"""Deterministic 4-D query planning and evidence lookup for FARM nodes.

Time and motion are deliberately parsed without an LLM. A historical query is
executed against a lightweight temporal view whose ``means`` and ``active``
fields describe the requested instant/window; the spatial executor therefore
cannot accidentally use the latest geometry for a historical relation.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, replace
from statistics import median
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class TemporalRequest:
    """Deterministic part of a natural-language 4-D query."""

    cleaned_query: str
    mode: str = "none"  # none, elapsed, interval, timestamp, latest, earliest, history
    timestamp_ns: int | None = None
    start_ns: int | None = None
    end_ns: int | None = None
    require_motion: bool = False
    min_displacement_m: float | None = None
    ask_current_location: bool = False
    require_identity_continuity: bool = False
    original_query: str = ""

    @property
    def is_4d(self) -> bool:
        return bool(
            self.mode != "none"
            or self.require_motion
            or self.ask_current_location
            or self.require_identity_continuity
        )


_ELAPSED_PATTERNS = (
    re.compile(r"(?:\bat\s*|第\s*|在\s*)?(\d+(?:\.\d+)?)\s*(?:s(?:ec(?:ond)?s?)?|秒)(?:\b|时)?", re.I),
    re.compile(r"(?:time|时间)\s*[=:：]?\s*(\d+(?:\.\d+)?)\s*(?:s|秒)", re.I),
)
_FIRST_WINDOW_PATTERNS = (
    re.compile(
        r"(?:\bduring\s+|\bin\s+|\bwithin\s+)?(?:the\s+)?(?:first|initial)\s+"
        r"(\d+(?:\.\d+)?)\s*(?:s(?:ec(?:ond)?s?)?)\b",
        re.I,
    ),
    re.compile(r"(?:最初|起初|开始(?:的)?|前)\s*(\d+(?:\.\d+)?)\s*秒(?:内|期间|之内)?", re.I),
)
_RANGE_PATTERNS = (
    re.compile(
        r"(?:\bbetween\s+|\bfrom\s+)(\d+(?:\.\d+)?)\s*(?:s(?:ec(?:ond)?s?)?)\s*"
        r"(?:and|to|[-–—])\s*(\d+(?:\.\d+)?)\s*(?:s(?:ec(?:ond)?s?)?)\b",
        re.I,
    ),
    re.compile(r"(\d+(?:\.\d+)?)\s*秒\s*(?:到|至|[-–—])\s*(\d+(?:\.\d+)?)\s*秒(?:内|期间)?", re.I),
)
_MOTION_WORDS = re.compile(
    r"\b(?:move[ds]?|moving|relocat(?:e[ds]?|ed|ion)|displac(?:e[ds]?|ed|ement))\b|"
    r"移动|位移|搬移|换了位置|重新定位",
    re.I,
)
_MOTION_THRESHOLD_PATTERNS = (
    re.compile(
        r"(?:move[ds]?|relocat(?:e[ds]?|ed|ion)|displac(?:e[ds]?|ed|ement))"
        r"[^,;，。?]{0,36}?(?:more\s+than|over|at\s+least|>=|>)\s*"
        r"(\d+(?:\.\d+)?)\s*(?:m(?:eters?)?|metres?)\b",
        re.I,
    ),
    re.compile(
        r"(?:移动|位移|搬移|重新定位)[^,;，。？]{0,24}?(?:超过|大于|至少|不小于|>=|>)\s*"
        r"(\d+(?:\.\d+)?)\s*(?:m|米)",
        re.I,
    ),
)
_CURRENT_LOCATION = re.compile(
    r"\bwhere\s+is\s+(?:that(?:\s+same)?|the\s+same|this)\s+(?:object|person|one|target)\s+now\b|"
    r"\bwhere\s+(?:it|they)\s+is\s+now\b|"
    r"(?:这个|该|同一)(?:人|物体|目标|对象)?现在(?:在)?哪里|现在(?:在)?哪里",
    re.I,
)
_SAME_IDENTITY = re.compile(
    r"\b(?:that|the)\s+same\s+(?:object|person|one|target)\b|"
    r"\bsame\s+(?:identity|instance)\b|同一(?:个|把|张|台|只|件|位)?"
    r"(?:人|物体|目标|对象|实例|东西|椅子|桌子|箱子|沙发)|这个人|该目标",
    re.I,
)


def _remove_match(text: str, match: re.Match[str] | None) -> str:
    if match is None:
        return text
    return text[: match.start()] + " " + text[match.end() :]


def _clean_residual(text: str) -> str:
    # Answer-format and history clauses are hard constraints, not LLM spatial
    # predicates. Keep only the target and static spatial relation.
    # Insert a clause boundary before a Chinese same-object target, otherwise
    # the preceding motion-clause cleanup can accidentally consume its noun.
    text = re.sub(
        r"的\s*((?:同一)(?:个|把|张|台|只|件|位)?"
        r"(?:人|物体|目标|对象|实例|东西|椅子|桌子|箱子|沙发))",
        r"，\1", text, flags=re.I,
    )
    text = re.sub(
        r"[,，]?\s*(?:and\s+)?where\s+is\s+(?:that(?:\s+same)?|the\s+same|this)\s+"
        r"(?:object|person|one|target)\s+now\s*[?？]?",
        " ", text, flags=re.I,
    )
    text = re.sub(
        r"[,，]?\s*(?:并且|以及|然后)?\s*(?:这个|该|同一)(?:人|物体|目标|对象)?现在(?:在)?哪里\s*[?？]?",
        " ", text, flags=re.I,
    )
    text = re.sub(
        r"[,，]?\s*(?:并且|以及|然后)?\s*(?:它|他|她)现在(?:在)?哪里\s*[?？]?",
        " ", text, flags=re.I,
    )
    text = re.sub(
        r"[,，]?\s*(?:and\s+)?(?:then\s+|later\s+|afterwards\s+)?(?:was\s+)?"
        r"(?:move[ds]?|relocat(?:e[ds]?|ed)|displac(?:e[ds]?|ed))"
        r"(?:\s+by)?[^,;，。?？]*(?=$|[,;，。?？])",
        " ", text, flags=re.I,
    )
    text = re.sub(
        r"[,，]?\s*(?:之后|后来|随后|然后)?(?:又)?(?:移动|位移|搬移|重新定位)"
        r"[^,;，。?？]*(?=$|[,;，。?？])",
        " ", text, flags=re.I,
    )
    text = re.sub(r"\b(?:appeared|was\s+visible|could\s+be\s+seen)\b|出现|可见", " ", text, flags=re.I)
    text = re.sub(r"\b(?:later|afterwards|then)\b|(?:之后|后来|随后|然后)", " ", text, flags=re.I)
    text = re.sub(
        r"同一(?:个|把|张|台|只|件|位)?"
        r"(人|物体|目标|对象|实例|东西|椅子|桌子|箱子|沙发)",
        r"\1", text, flags=re.I,
    )
    text = re.sub(r"\s+([,，;；?？])", r"\1", text)
    text = re.sub(r"\s*[,，、;；]+\s*", " ", text)
    text = re.sub(r"^[\s,，;；]+|[\s,，;；]+$", "", text)
    return " ".join(text.split())


def parse_temporal_query(query: str, state: dict[str, Any]) -> TemporalRequest:
    """Parse time/motion intent and return the residual spatial query."""

    original = str(query or "").strip()
    text = original
    origin = int(state.get("temporal_origin_ns", 0) or 0)
    mode = "none"
    timestamp_ns: int | None = None
    start_ns: int | None = None
    end_ns: int | None = None

    # Explicit intervals beat generic words such as "first". Historical
    # clauses also beat the "now" in an answer request.
    for pattern in _RANGE_PATTERNS:
        match = pattern.search(text)
        if match:
            first_s, second_s = sorted((float(match.group(1)), float(match.group(2))))
            start_ns = origin + int(round(first_s * 1.0e9))
            end_ns = origin + int(round(second_s * 1.0e9))
            mode = "interval"
            text = _remove_match(text, match)
            break

    if mode == "none":
        for pattern in _FIRST_WINDOW_PATTERNS:
            match = pattern.search(text)
            if match:
                start_ns = origin
                end_ns = origin + int(round(float(match.group(1)) * 1.0e9))
                mode = "interval"
                text = _remove_match(text, match)
                break

    if mode == "none":
        for pattern in _ELAPSED_PATTERNS:
            match = pattern.search(text)
            if match:
                timestamp_ns = origin + int(round(float(match.group(1)) * 1.0e9))
                mode = "elapsed"
                text = _remove_match(text, match)
                break

    if mode == "none":
        stamp_match = re.search(r"(?:timestamp|时间戳)\s*[=:：]?\s*(\d{9,19})", text, re.I)
        if stamp_match:
            raw = int(stamp_match.group(1))
            timestamp_ns = raw if raw >= 10**14 else raw * 1_000_000_000
            mode = "timestamp"
            text = _remove_match(text, stamp_match)

    if mode == "none":
        iso_match = re.search(
            r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?", text,
        )
        if iso_match:
            value = iso_match.group(0).replace("Z", "+00:00")
            parsed = dt.datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.astimezone()
            timestamp_ns = int(parsed.timestamp() * 1.0e9)
            mode = "timestamp"
            text = _remove_match(text, iso_match)

    ask_current = bool(_CURRENT_LOCATION.search(original))
    require_identity = bool(_SAME_IDENTITY.search(original))
    require_motion = bool(_MOTION_WORDS.search(original))
    min_displacement_m: float | None = None
    for pattern in _MOTION_THRESHOLD_PATTERNS:
        match = pattern.search(original)
        if match:
            min_displacement_m = float(match.group(1))
            require_motion = True
            break

    if mode == "none":
        if re.search(r"\b(latest|current|now|last)\b|现在|当前|最后|最新", original, re.I):
            mode = "latest"
        elif re.search(r"\b(earliest|first|initial)\b|最初|起初|第一次", original, re.I):
            mode = "earliest"
        elif re.search(r"\b(when|history|trajectory|timeline)\b|什么时候|何时|历史|轨迹|时间", original, re.I):
            mode = "history"

    cleaned = _clean_residual(text)
    return TemporalRequest(
        cleaned_query=cleaned or original,
        mode=mode,
        timestamp_ns=timestamp_ns,
        start_ns=start_ns,
        end_ns=end_ns,
        require_motion=require_motion,
        min_displacement_m=min_displacement_m,
        ask_current_location=ask_current,
        require_identity_continuity=require_identity,
        original_query=original,
    )


def _observations(state: dict[str, Any], object_index: int) -> list[dict[str, Any]]:
    rows = state.get("object_temporal_observations") or []
    if not 0 <= int(object_index) < len(rows) or not isinstance(rows[object_index], list):
        return []
    return sorted(
        (value for value in rows[object_index] if isinstance(value, dict)),
        key=lambda value: int(value.get("timestamp_ns", 0)),
    )


def _representative(observations: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not observations:
        return None
    result = dict(observations[len(observations) // 2])
    positions = [value.get("position") for value in observations]
    positions = [value for value in positions if isinstance(value, (list, tuple)) and len(value) >= 3]
    if positions:
        result["position"] = [float(median(float(value[axis]) for value in positions)) for axis in range(3)]
    result["timestamp_ns"] = int(median(int(value.get("timestamp_ns", 0)) for value in observations))
    result["window_observation_count"] = len(observations)
    return result


def temporal_snapshot(state: dict[str, Any], object_index: int, request: TemporalRequest) -> dict[str, Any] | None:
    observations = _observations(state, object_index)
    if not observations:
        return None
    if request.mode == "interval" and request.start_ns is not None and request.end_ns is not None:
        inside = [
            value for value in observations
            if int(request.start_ns) <= int(value.get("timestamp_ns", 0)) <= int(request.end_ns)
        ]
        return _representative(inside)
    if request.mode == "earliest":
        return observations[0]
    if request.mode in {"latest", "none", "history"}:
        current_rows = state.get("object_current_state") or []
        if 0 <= int(object_index) < len(current_rows) and isinstance(current_rows[object_index], dict):
            current = current_rows[object_index]
            if current.get("position") is not None:
                return current
        return observations[-1]
    if request.timestamp_ns is None:
        return observations[-1]
    target = int(request.timestamp_ns)
    coverage_rows = state.get("object_time_coverage") or []
    coverage = coverage_rows[object_index] if object_index < len(coverage_rows) else []
    if coverage and not any(
        int(row.get("start_ns", 0)) <= target <= int(row.get("end_ns", 0))
        for row in coverage if isinstance(row, dict)
    ):
        # A real coverage interval is a hard visibility constraint. Only a
        # singleton observation gets sampling tolerance around its timestamp.
        if any(
            int(row.get("end_ns", 0)) > int(row.get("start_ns", 0))
            or int(row.get("observation_count", 1) or 1) > 1
            for row in coverage if isinstance(row, dict)
        ):
            return None
        stamps = [int(value.get("timestamp_ns", 0)) for value in observations]
        positive = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
        tolerance = max(100_000_000, int(median(positive) * 1.5)) if positive else 500_000_000
        if min(abs(value - target) for value in stamps) > tolerance:
            return None
    return min(observations, key=lambda value: abs(int(value.get("timestamp_ns", 0)) - target))


def _set_row(value: Any, index: int, row: Sequence[float] | bool) -> None:
    """Assign to torch/numpy/list containers without importing torch here."""
    try:
        value[index] = row
    except (TypeError, ValueError):
        item = value[index]
        if hasattr(item, "new_tensor"):
            value[index] = item.new_tensor(row)
        else:
            raise


def temporal_scene_slice(state: dict[str, Any], request: TemporalRequest) -> dict[str, Any]:
    """Return a shallow scene view with geometry materialised at query time."""

    if request.mode not in {"elapsed", "timestamp", "interval", "earliest"}:
        return state
    means_source = state.get("means")
    if means_source is None:
        return state
    try:
        object_count = len(means_source)
    except TypeError:
        return state

    view = dict(state)
    means = means_source.clone() if hasattr(means_source, "clone") else means_source.copy()
    active_source = state.get("active")
    active = (
        [True] * object_count if active_source is None
        else (active_source.clone() if hasattr(active_source, "clone") else active_source.copy())
    )
    cov_source = state.get("cov6")
    cov = cov_source.clone() if hasattr(cov_source, "clone") else (cov_source.copy() if cov_source is not None else None)

    for object_index in range(object_count):
        snapshot = temporal_snapshot(state, object_index, request)
        was_active = bool(active_source[object_index]) if active_source is not None else True
        visible = bool(was_active and snapshot is not None and snapshot.get("position") is not None)
        _set_row(active, object_index, visible)
        if not visible:
            continue
        _set_row(means, object_index, snapshot["position"][:3])
        if cov is not None and isinstance(snapshot.get("cov6"), (list, tuple)) and len(snapshot["cov6"]) == 6:
            _set_row(cov, object_index, snapshot["cov6"])
    view["means"] = means
    view["active"] = active
    if cov is not None:
        view["cov6"] = cov
    view["_temporal_query_request"] = request
    return view


def _row(rows: Any, object_index: int, default: Any) -> Any:
    return rows[object_index] if isinstance(rows, list) and 0 <= object_index < len(rows) else default


def _identity_evidence(state: dict[str, Any], object_index: int) -> str:
    tracks = _row(state.get("object_instance_track_ids"), object_index, [])
    motion = _row(state.get("object_motion_state"), object_index, {})
    if isinstance(motion, dict):
        if motion.get("identity_ambiguous"):
            return "ambiguous_identity_or_depth_jump"
        for event in motion.get("relocation_events") or []:
            if isinstance(event, dict) and event.get("identity_evidence"):
                return str(event["identity_evidence"])
    if tracks:
        return "stable_instance_track"
    if any(value.get("instance_track_id") is not None for value in _observations(state, object_index)):
        return "stable_instance_track"
    return "canonical_node_only"


def apply_4d_constraints(
    state: dict[str, Any], candidates: Iterable[Any], request: TemporalRequest,
) -> list[Any]:
    """Apply hard 4-D clauses, then evidence-aware reranking."""

    ranked: list[tuple[float, Any]] = []
    motion_rows = state.get("object_motion_state") or []
    current_rows = state.get("object_current_state") or []
    for candidate in candidates:
        object_index = int(candidate.object_index)
        snapshot = temporal_snapshot(state, object_index, request)
        if request.mode in {"elapsed", "timestamp", "interval", "earliest"} and snapshot is None:
            continue
        motion = _row(motion_rows, object_index, {})
        motion = motion if isinstance(motion, dict) else {}
        displacement = float(
            motion.get("confirmed_displacement_m", motion.get("displacement_m", 0.0)) or 0.0
        )
        has_motion = bool(
            motion.get("ever_moved") or motion.get("is_dynamic")
            or int(motion.get("relocation_count", 0) or 0) > 0
            or motion.get("relocation_events")
        )
        if request.require_motion and not has_motion:
            continue
        if request.min_displacement_m is not None and displacement + 1.0e-9 < request.min_displacement_m:
            continue
        identity = _identity_evidence(state, object_index)
        if request.require_identity_continuity and identity in {
            "canonical_node_only",
            "ambiguous_identity_or_depth_jump",
        }:
            continue
        current = _row(current_rows, object_index, {})
        if request.ask_current_location and not (isinstance(current, dict) and current.get("position") is not None):
            continue

        base = float(getattr(candidate, "composite_score", 0.0))
        evidence_terms: list[float] = []
        if snapshot is not None:
            evidence_terms.append(min(1.0, float(snapshot.get("window_observation_count", 1)) / 3.0))
        if request.require_motion:
            evidence_terms.append(1.0)
        if request.min_displacement_m is not None and request.min_displacement_m > 0:
            evidence_terms.append(min(1.0, displacement / (1.5 * request.min_displacement_m)))
        if request.require_identity_continuity:
            evidence_terms.append(1.0)
        evidence_score = sum(evidence_terms) / len(evidence_terms) if evidence_terms else 1.0
        final_score = 0.80 * base + 0.20 * evidence_score if request.is_4d else base
        try:
            candidate = replace(candidate, composite_score=final_score)
        except (TypeError, ValueError):
            pass
        ranked.append((final_score, candidate))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [candidate for _score, candidate in ranked]


def format_query_plan(state: dict[str, Any], request: TemporalRequest) -> str:
    """Compact human-readable plan shown in Viewer and the CLI."""

    if not request.is_4d:
        return "static semantic/spatial query"
    origin = int(state.get("temporal_origin_ns", 0) or 0)
    clauses: list[str] = []
    if request.mode == "interval" and request.start_ns is not None and request.end_ns is not None:
        clauses.append(
            f"historical slice t={(request.start_ns - origin) / 1e9:.2f}–{(request.end_ns - origin) / 1e9:.2f}s"
        )
    elif request.timestamp_ns is not None:
        clauses.append(f"historical slice t={(request.timestamp_ns - origin) / 1e9:.2f}s")
    elif request.mode != "none":
        clauses.append(request.mode)
    if request.require_motion:
        motion = "confirmed motion"
        if request.min_displacement_m is not None:
            motion += f" ≥ {request.min_displacement_m:g}m"
        clauses.append(motion)
    if request.require_identity_continuity:
        clauses.append("stable identity")
    if request.ask_current_location:
        clauses.append("current-state answer")
    return " → ".join(clauses)


def _position_text(snapshot: dict[str, Any] | None) -> str:
    if not isinstance(snapshot, dict):
        return "unknown"
    position = snapshot.get("position") or []
    if len(position) < 3:
        return "unknown"
    return f"({float(position[0]):.2f}, {float(position[1]):.2f}, {float(position[2]):.2f}) m"


def format_temporal_evidence(state: dict[str, Any], object_index: int, request: TemporalRequest) -> str:
    snapshot = temporal_snapshot(state, object_index, request)
    if snapshot is None:
        return ""
    origin = int(state.get("temporal_origin_ns", 0) or 0)
    stamp = int(snapshot.get("timestamp_ns", 0))
    elapsed = (stamp - origin) / 1.0e9
    motion = _row(state.get("object_motion_state"), object_index, {})
    motion = motion if isinstance(motion, dict) else {}
    identity = _identity_evidence(state, object_index)

    if request.is_4d:
        parts = [f"past t={elapsed:.2f}s {_position_text(snapshot)}"]
        if request.require_motion:
            displacement = motion.get("confirmed_displacement_m", motion.get("displacement_m", 0.0))
            parts.append(f"displacement={float(displacement or 0.0):.2f}m")
        if request.require_identity_continuity:
            parts.append(f"identity={identity}")
        if request.ask_current_location:
            current = _row(state.get("object_current_state"), object_index, {})
            current_stamp = int(current.get("timestamp_ns", 0)) if isinstance(current, dict) else 0
            parts.append(f"current t={(current_stamp - origin) / 1e9:.2f}s {_position_text(current)}")
        return " [4D: " + "; ".join(parts) + "]"

    if request.mode == "history":
        observations = _observations(state, object_index)
        episodes = _row(state.get("object_location_history"), object_index, [])
        ever_moved = bool(motion.get("ever_moved", motion.get("is_dynamic")))
        moving_now = bool(motion.get("is_currently_moving"))
        status = "moving now" if moving_now else ("moved previously" if ever_moved else "no confirmed motion")
        return (
            f" [timeline: {len(observations)} observations, {len(episodes) if isinstance(episodes, list) else 0} "
            f"location episodes; {status}; latest t={elapsed:.2f}s {_position_text(snapshot)}]"
        )
    return f" [t={elapsed:.2f}s {_position_text(snapshot)}]"


__all__ = [
    "TemporalRequest",
    "apply_4d_constraints",
    "format_query_plan",
    "format_temporal_evidence",
    "parse_temporal_query",
    "temporal_scene_slice",
    "temporal_snapshot",
]
