"""Deterministic temporal parsing and lookup for 4-D FARM nodes."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TemporalRequest:
    cleaned_query: str
    mode: str = "none"  # none, elapsed, timestamp, latest, earliest, history
    timestamp_ns: int | None = None


_ELAPSED_PATTERNS = (
    re.compile(r"(?:\bat\s*|第\s*)(\d+(?:\.\d+)?)\s*(?:s(?:ec(?:ond)?s?)?|秒)(?:\b|时)?", re.I),
    re.compile(r"(?:time|时间)\s*[=:：]?\s*(\d+(?:\.\d+)?)\s*(?:s|秒)", re.I),
)


def parse_temporal_query(query: str, state: dict[str, Any]) -> TemporalRequest:
    text = str(query or "").strip()
    origin = int(state.get("temporal_origin_ns", 0) or 0)
    for pattern in _ELAPSED_PATTERNS:
        match = pattern.search(text)
        if match:
            elapsed = float(match.group(1))
            cleaned = (text[: match.start()] + " " + text[match.end() :]).strip()
            return TemporalRequest(" ".join(cleaned.split()), "elapsed", origin + int(round(elapsed * 1.0e9)))
    stamp_match = re.search(r"(?:timestamp|时间戳)\s*[=:：]?\s*(\d{9,19})", text, re.I)
    if stamp_match:
        raw = int(stamp_match.group(1))
        timestamp_ns = raw if raw >= 10**14 else raw * 1_000_000_000
        cleaned = (text[: stamp_match.start()] + " " + text[stamp_match.end() :]).strip()
        return TemporalRequest(" ".join(cleaned.split()), "timestamp", timestamp_ns)
    iso_match = re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?", text)
    if iso_match:
        value = iso_match.group(0).replace("Z", "+00:00")
        parsed = dt.datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.astimezone()
        cleaned = (text[: iso_match.start()] + " " + text[iso_match.end() :]).strip()
        return TemporalRequest(" ".join(cleaned.split()), "timestamp", int(parsed.timestamp() * 1.0e9))
    if re.search(r"\b(latest|current|now|last)\b|现在|当前|最后|最新", text, re.I):
        return TemporalRequest(text, "latest", None)
    if re.search(r"\b(earliest|first|initial)\b|最初|起初|第一次", text, re.I):
        return TemporalRequest(text, "earliest", None)
    if re.search(r"\b(when|history|trajectory|timeline)\b|什么时候|何时|历史|轨迹|时间", text, re.I):
        return TemporalRequest(text, "history", None)
    return TemporalRequest(text, "none", None)


def temporal_snapshot(state: dict[str, Any], object_index: int, request: TemporalRequest) -> dict[str, Any] | None:
    rows = state.get("object_temporal_observations") or []
    if not 0 <= int(object_index) < len(rows) or not isinstance(rows[object_index], list):
        return None
    observations = [value for value in rows[object_index] if isinstance(value, dict)]
    if not observations:
        return None
    observations.sort(key=lambda value: int(value.get("timestamp_ns", 0)))
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
    if coverage and not any(int(row.get("start_ns", 0)) <= target <= int(row.get("end_ns", 0)) for row in coverage if isinstance(row, dict)):
        # Permit one sampling interval outside a singleton/interval endpoint.
        stamps = [int(value.get("timestamp_ns", 0)) for value in observations]
        positive = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
        tolerance = max(100_000_000, int(sorted(positive)[len(positive) // 2] * 1.5)) if positive else 500_000_000
        if min(abs(value - target) for value in stamps) > tolerance:
            return None
    return min(observations, key=lambda value: abs(int(value.get("timestamp_ns", 0)) - target))


def format_temporal_evidence(state: dict[str, Any], object_index: int, request: TemporalRequest) -> str:
    snapshot = temporal_snapshot(state, object_index, request)
    if snapshot is None:
        return ""
    origin = int(state.get("temporal_origin_ns", 0) or 0)
    stamp = int(snapshot.get("timestamp_ns", 0))
    elapsed = (stamp - origin) / 1.0e9
    position = snapshot.get("position") or []
    location = ""
    if len(position) >= 3:
        location = f" at ({float(position[0]):.2f}, {float(position[1]):.2f}, {float(position[2]):.2f}) m"
    if request.mode == "history":
        rows = state.get("object_temporal_observations") or []
        count = len(rows[object_index]) if object_index < len(rows) and isinstance(rows[object_index], list) else 0
        motion = state.get("object_motion_state") or []
        motion_state = motion[object_index] if object_index < len(motion) and isinstance(motion[object_index], dict) else {}
        ever_moved = bool(motion_state.get("ever_moved", motion_state.get("is_dynamic")))
        moving_now = bool(motion_state.get("is_currently_moving"))
        histories = state.get("object_location_history") or []
        episodes = histories[object_index] if object_index < len(histories) and isinstance(histories[object_index], list) else []
        status = "moving now" if moving_now else ("moved previously" if ever_moved else "no confirmed motion")
        return (
            f" [timeline: {count} observations, {len(episodes)} location episodes; "
            f"{status}; latest t={elapsed:.2f}s{location}]"
        )
    return f" [t={elapsed:.2f}s{location}]"
