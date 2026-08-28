"""Identity and time-history support for 4-D scene graph objects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import torch


def _tensor_values(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().to("cpu", dtype=torch.int64).reshape(-1).tolist()]
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return []


def apply_instance_track_correspondence(
    state: dict[str, Any],
    seg_outputs: dict[str, Any] | None,
    det_idx: torch.Tensor,
    obj_idx: torch.Tensor,
    detection_image_ids: Sequence[Optional[int]] | None = None,
    *,
    reid_similarity_threshold: float = 0.965,
    reid_margin: float = 0.015,
) -> tuple[torch.Tensor, int]:
    """Apply stable SAM identity, then conservatively stitch new tracklets.

    Exact track IDs are authoritative. If an unknown track is geometrically
    unmatched (typically after occlusion plus object motion), a same-class
    DINO match can reuse a persistent node only when the appearance score is
    high, unambiguous, and the candidate did not coexist in the same image.
    """
    if not isinstance(seg_outputs, dict) or "instance_track_ids" not in seg_outputs:
        return det_idx, 0
    detection_tracks = _tensor_values(seg_outputs.get("instance_track_ids"))
    object_tracks = state.get("object_instance_track_ids")
    active = state.get("active")
    if not detection_tracks or not isinstance(object_tracks, list) or not isinstance(active, torch.Tensor):
        return det_idx, 0
    active_values = active.detach().to("cpu", dtype=torch.bool).reshape(-1).tolist()
    lookup: dict[int, int] = {}
    for object_index, values in enumerate(object_tracks):
        if object_index >= len(active_values) or not active_values[object_index]:
            continue
        try:
            for track_id in values:
                lookup[int(track_id)] = object_index
        except TypeError:
            continue
    result = det_idx.clone()
    forced = 0
    for detection_index, track_id in enumerate(detection_tracks[: result.numel()]):
        object_index = lookup.get(track_id)
        if object_index is None or object_index >= obj_idx.numel():
            continue
        # det_idx indexes obj_idx; object_index is valid because obj_idx spans N.
        if int(result[detection_index].item()) != object_index:
            result[detection_index] = object_index
            forced += 1
    reidentified = 0
    detection_features = seg_outputs.get("features")
    detection_classes = seg_outputs.get("class_ids")
    object_features = state.get("features")
    object_classes = state.get("class_ids")
    object_images = state.get("object_image_ids") or []
    if (
        isinstance(detection_features, torch.Tensor)
        and isinstance(detection_classes, torch.Tensor)
        and isinstance(object_features, torch.Tensor)
        and isinstance(object_classes, torch.Tensor)
        and detection_features.ndim == 2
        and object_features.ndim == 2
        and detection_features.shape[1] == object_features.shape[1]
    ):
        normalized_objects = torch.nn.functional.normalize(
            object_features.detach().to("cpu", dtype=torch.float32), dim=1
        )
        normalized_detections = torch.nn.functional.normalize(
            detection_features.detach().to("cpu", dtype=torch.float32), dim=1
        )
        detection_class_values = detection_classes.detach().to("cpu", dtype=torch.int64).reshape(-1)
        object_class_values = object_classes.detach().to("cpu", dtype=torch.int64).reshape(-1)
        claimed = {
            int(result[index].item())
            for index in range(result.numel())
            if int(result[index].item()) >= 0
        }
        for detection_index, track_id in enumerate(detection_tracks[: result.numel()]):
            if int(result[detection_index].item()) >= 0 or track_id in lookup:
                continue
            if detection_index >= normalized_detections.shape[0] or detection_index >= detection_class_values.numel():
                continue
            class_id = int(detection_class_values[detection_index].item())
            image_id = (
                detection_image_ids[detection_index]
                if detection_image_ids is not None and detection_index < len(detection_image_ids)
                else None
            )
            candidates: list[int] = []
            for object_index in range(min(len(active_values), normalized_objects.shape[0])):
                if not active_values[object_index] or object_index in claimed:
                    continue
                if object_index >= object_class_values.numel() or int(object_class_values[object_index].item()) != class_id:
                    continue
                if object_index >= len(object_tracks) or not object_tracks[object_index]:
                    continue
                if image_id is not None and object_index < len(object_images):
                    try:
                        if int(image_id) in {int(value) for value in object_images[object_index]}:
                            continue
                    except (TypeError, ValueError):
                        continue
                candidates.append(object_index)
            if not candidates:
                continue
            scores = torch.mv(normalized_objects[candidates], normalized_detections[detection_index])
            order = torch.argsort(scores, descending=True)
            best_offset = int(order[0].item())
            best_score = float(scores[best_offset].item())
            second_score = float(scores[int(order[1].item())].item()) if order.numel() > 1 else -1.0
            if best_score < float(reid_similarity_threshold) or best_score - second_score < float(reid_margin):
                continue
            object_index = candidates[best_offset]
            result[detection_index] = object_index
            claimed.add(object_index)
            reidentified += 1
    state["_last_instance_reid_assignments"] = int(reidentified)
    return result, forced


def register_temporal_observations(
    state: dict[str, Any],
    seg_outputs: dict[str, Any],
    detection_image_ids: Sequence[Optional[int]],
    detection_timestamps_ns: Sequence[Optional[int]],
    det_to_obj: Sequence[Optional[int]],
    *,
    max_per_object: int = 4096,
) -> int:
    """Attach track IDs and time-indexed world-space measurements to nodes."""
    means = seg_outputs.get("means")
    cov6 = seg_outputs.get("cov6")
    track_values = _tensor_values(seg_outputs.get("instance_track_ids"))
    if not isinstance(means, torch.Tensor) or not isinstance(cov6, torch.Tensor):
        return 0
    object_count = int(state.get("means", torch.empty(0, 3)).shape[0])
    track_rows = state.get("object_instance_track_ids")
    observation_rows = state.get("object_temporal_observations")
    current_rows = state.get("object_current_state")
    if not isinstance(track_rows, list):
        track_rows = []
    if not isinstance(observation_rows, list):
        observation_rows = []
    if not isinstance(current_rows, list):
        current_rows = []
    while len(track_rows) < object_count:
        track_rows.append([])
    while len(observation_rows) < object_count:
        observation_rows.append([])
    while len(current_rows) < object_count:
        current_rows.append({})
    state["object_instance_track_ids"] = track_rows
    state["object_temporal_observations"] = observation_rows
    state["object_current_state"] = current_rows
    state["temporal_schema_version"] = max(1, int(state.get("temporal_schema_version", 1) or 1))
    added = 0
    cap = max(1, int(max_per_object))
    means_cpu = means.detach().to("cpu", dtype=torch.float32)
    cov_cpu = cov6.detach().to("cpu", dtype=torch.float32)
    for detection_index, object_index_raw in enumerate(det_to_obj):
        if object_index_raw is None or detection_index >= means_cpu.shape[0]:
            continue
        object_index = int(object_index_raw)
        if not 0 <= object_index < object_count:
            continue
        track_id = track_values[detection_index] if detection_index < len(track_values) else None
        if track_id is not None and track_id not in track_rows[object_index]:
            track_rows[object_index].append(track_id)
        image_id = detection_image_ids[detection_index] if detection_index < len(detection_image_ids) else None
        stamp = detection_timestamps_ns[detection_index] if detection_index < len(detection_timestamps_ns) else None
        timestamp_ns = int(stamp or 0)
        observation = {
            "timestamp_ns": timestamp_ns,
            "time_s": float(timestamp_ns / 1.0e9),
            "image_id": int(image_id) if image_id is not None else None,
            "position": means_cpu[detection_index].tolist(),
            "cov6": cov_cpu[detection_index].tolist(),
            "instance_track_id": int(track_id) if track_id is not None else None,
            "source": "sam3.1" if track_id is not None else "geometry",
        }
        row = observation_rows[object_index]
        if not isinstance(row, list):
            row = []
            observation_rows[object_index] = row
        # One best measurement per object/image; mapper batches can be retried.
        if image_id is not None:
            row[:] = [old for old in row if not isinstance(old, dict) or old.get("image_id") != int(image_id)]
        row.append(observation)
        row.sort(key=lambda value: (int(value.get("timestamp_ns", 0)), int(value.get("image_id") or -1)))
        if len(row) > cap:
            del row[: len(row) - cap]
        latest = row[-1]
        recent_positions = [
            value.get("position")
            for value in row[-min(3, len(row)) :]
            if isinstance(value, dict) and value.get("position") is not None
        ]
        latest_position = (
            torch.median(torch.as_tensor(recent_positions, dtype=torch.float32), dim=0).values.tolist()
            if recent_positions
            else latest.get("position")
        )
        previous_current = current_rows[object_index] if isinstance(current_rows[object_index], dict) else {}
        current_rows[object_index] = {
            **latest,
            "position": latest_position,
            "observed_at_scene_end": True,
            # Online motion is finalized from a temporal window later; do not
            # invent a motion label from one noisy frame.
            "is_currently_moving": bool(previous_current.get("is_currently_moving", False)),
        }
        if latest_position is not None and isinstance(state.get("means"), torch.Tensor):
            state["means"][object_index] = torch.as_tensor(
                latest_position,
                dtype=state["means"].dtype,
                device=state["means"].device,
            )
        if latest.get("cov6") is not None and isinstance(state.get("cov6"), torch.Tensor):
            state["cov6"][object_index] = torch.as_tensor(
                latest["cov6"],
                dtype=state["cov6"].dtype,
                device=state["cov6"].device,
            )
        added += 1
    return added
