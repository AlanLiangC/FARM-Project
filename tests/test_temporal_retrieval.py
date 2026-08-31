from __future__ import annotations

import torch

from scene_graph.retrieval.spatial_reasoning.models import ScoredCandidate
from scene_graph.retrieval.temporal import (
    apply_4d_constraints,
    format_query_plan,
    format_temporal_evidence,
    parse_temporal_query,
    temporal_scene_slice,
    temporal_snapshot,
)


QUERY_4D = (
    "Which person was near the desk during the first 3 seconds, later relocated "
    "by more than 0.8 meters, and where is that same person now?"
)


def _observation(timestamp_ns: int, position: list[float], track_id: int | None = None) -> dict:
    return {
        "timestamp_ns": timestamp_ns,
        "position": position,
        "cov6": [0.01, 0.0, 0.0, 0.01, 0.0, 0.01],
        "instance_track_id": track_id,
    }


def _state() -> dict:
    observations = [
        [
            _observation(500_000_000, [0.2, 0.0, 0.0], 7),
            _observation(2_000_000_000, [0.4, 0.0, 0.0], 7),
            _observation(10_000_000_000, [2.0, 0.0, 0.0], 7),
        ],
        [
            _observation(5_000_000_000, [0.5, 0.0, 0.0], 8),
            _observation(10_000_000_000, [0.5, 0.0, 0.0], 8),
        ],
        [
            _observation(500_000_000, [0.0, 0.0, 0.0]),
            _observation(2_000_000_000, [0.0, 0.0, 0.0]),
            _observation(10_000_000_000, [0.0, 0.0, 0.0]),
        ],
    ]
    return {
        "temporal_origin_ns": 0,
        "means": torch.tensor([[2.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "cov6": torch.zeros((3, 6)),
        "active": torch.tensor([True, True, True]),
        "object_temporal_observations": observations,
        "object_time_coverage": [
            [{"start_ns": 500_000_000, "end_ns": 10_000_000_000}],
            [{"start_ns": 5_000_000_000, "end_ns": 10_000_000_000}],
            [{"start_ns": 500_000_000, "end_ns": 10_000_000_000}],
        ],
        "object_instance_track_ids": [[7], [8], []],
        "object_motion_state": [
            {"ever_moved": True, "is_dynamic": True, "displacement_m": 1.8},
            {"ever_moved": False, "is_dynamic": False, "displacement_m": 0.0},
            {"ever_moved": False, "is_dynamic": False, "displacement_m": 0.0},
        ],
        "object_current_state": [
            {"timestamp_ns": 10_000_000_000, "position": [2.0, 0.0, 0.0]},
            {"timestamp_ns": 10_000_000_000, "position": [0.5, 0.0, 0.0]},
            {"timestamp_ns": 10_000_000_000, "position": [0.0, 0.0, 0.0]},
        ],
        "object_location_history": [[], [], []],
    }


def _candidate(index: int, score: float) -> ScoredCandidate:
    return ScoredCandidate(index, index, [], score)


def test_parse_composite_4d_query_into_hard_plan_and_clean_spatial_residual() -> None:
    request = parse_temporal_query(QUERY_4D, _state())

    assert request.cleaned_query == "Which person was near the desk"
    assert request.mode == "interval"
    assert request.start_ns == 0
    assert request.end_ns == 3_000_000_000
    assert request.require_motion is True
    assert request.min_displacement_m == 0.8
    assert request.require_identity_continuity is True
    assert request.ask_current_location is True
    assert "historical slice t=0.00–3.00s" in format_query_plan(_state(), request)


def test_chinese_4d_query_has_the_same_structured_constraints() -> None:
    request = parse_temporal_query(
        "哪一个人在最初3秒靠近桌子，之后移动超过0.8米，并且这个人现在在哪里？",
        _state(),
    )

    assert request.mode == "interval"
    assert request.end_ns == 3_000_000_000
    assert request.require_motion is True
    assert request.min_displacement_m == 0.8
    assert request.require_identity_continuity is True
    assert request.ask_current_location is True
    assert "靠近桌子" in request.cleaned_query
    assert "0.8" not in request.cleaned_query


def test_historical_scene_slice_uses_window_geometry_and_visibility() -> None:
    state = _state()
    request = parse_temporal_query(QUERY_4D, state)
    sliced = temporal_scene_slice(state, request)

    assert sliced is not state
    assert sliced["active"].tolist() == [True, False, True]
    assert torch.allclose(sliced["means"][0], torch.tensor([0.3, 0.0, 0.0]))
    assert torch.allclose(sliced["means"][2], torch.tensor([0.0, 0.0, 0.0]))
    # The current state is not mutated by materialising the historical view.
    assert torch.allclose(state["means"][0], torch.tensor([2.0, 0.0, 0.0]))


def test_4d_constraints_remove_late_and_static_distractors_and_emit_evidence() -> None:
    state = _state()
    request = parse_temporal_query(QUERY_4D, state)
    ranked = apply_4d_constraints(
        state,
        [_candidate(1, 0.99), _candidate(0, 0.80), _candidate(2, 0.95)],
        request,
    )

    assert [candidate.object_index for candidate in ranked] == [0]
    evidence = format_temporal_evidence(state, 0, request)
    assert "past t=1.25s (0.30, 0.00, 0.00) m" in evidence
    assert "displacement=1.80m" in evidence
    assert "identity=stable_instance_track" in evidence
    assert "current t=10.00s (2.00, 0.00, 0.00) m" in evidence


def test_same_identity_clause_rejects_depth_or_reid_ambiguous_node() -> None:
    state = _state()
    state["object_motion_state"][0]["identity_ambiguous"] = True
    request = parse_temporal_query(QUERY_4D, state)

    assert apply_4d_constraints(state, [_candidate(0, 0.99)], request) == []


def test_point_in_time_query_remains_backward_compatible() -> None:
    state = _state()
    request = parse_temporal_query("the person at 2s", state)

    assert request.mode == "elapsed"
    assert request.cleaned_query == "the person"
    assert temporal_snapshot(state, 0, request)["position"] == [0.4, 0.0, 0.0]
    assert temporal_snapshot(state, 1, request) is None
