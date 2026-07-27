"""Helpers for recording and replaying offline Viser reconstruction videos."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch

SNAPSHOT_FORMAT_VERSION = 1
ROLLING_PROCESSING_WINDOW_FRAMES = 30


def _to_cpu_detached(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {k: _to_cpu_detached(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu_detached(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_detached(v) for v in value)
    if isinstance(value, set):
        return sorted(_to_cpu_detached(v) for v in value)
    return value


def slim_scene_state_for_viser(scene_state: dict[str, Any]) -> dict[str, Any]:
    """Copy only fields used by PipelineViserVisualizer during replay."""

    tensor_keys = (
        "active",
        "means",
        "cov6",
        "object_id",
        "object_voxel_keys_flat",
        "object_voxel_keys_offsets",
        "object_voxel_levels",
        "covisibility_adj_u64",
        "covisibility_filtered_adj_u64",
    )
    list_keys = (
        "object_caption",
        "object_caption_decision",
        "object_category",
        "object_supercategory",
        "object_key_attributes",
        "object_detection_category_conf",
        "is_locked",
        "region_ids",
        "region_labels",
        "region_object_lists",
        "region_centroids",
        "region_label_confidence",
        "object_image_ids",
    )

    out: dict[str, Any] = {}
    for key in tensor_keys:
        value = scene_state.get(key)
        if value is not None:
            out[key] = _to_cpu_detached(value)
    for key in list_keys:
        value = scene_state.get(key)
        if value is not None:
            out[key] = _to_cpu_detached(value)

    for key in ("region_version", "covisibility_blocks", "covisibility_max_objects"):
        if key in scene_state:
            out[key] = _to_cpu_detached(scene_state.get(key))

    current_robot_position = scene_state.get("current_robot_position")
    if current_robot_position is not None:
        out["current_robot_position"] = _to_cpu_detached(current_robot_position)
    return out


def build_visualization_snapshot(
    *,
    scene_state: dict[str, Any],
    colors: Sequence[Any],
    depths: Sequence[Any],
    intrinsics: Sequence[Any],
    poses: Sequence[Any],
    step_index: int,
) -> dict[str, Any]:
    return {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "step_index": int(step_index),
        "colors": _to_cpu_detached(list(colors)),
        "depths": _to_cpu_detached(list(depths)),
        "intrinsics": _to_cpu_detached(list(intrinsics)),
        "poses": _to_cpu_detached(list(poses)),
        "scene_state": slim_scene_state_for_viser(scene_state),
    }


class ViserRecordingWriter:
    """Writes replay snapshots and a duration manifest outside mapper timing."""

    def __init__(self, root: Path, *, every_n: int = 1, scene_id: str = "") -> None:
        self.root = Path(root).expanduser()
        self.snapshot_dir = self.root / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.every_n = max(1, int(every_n))
        self.scene_id = str(scene_id or "")
        self.entries: list[dict[str, Any]] = []
        self._seen_batches = 0
        self._written = 0
        self._started_unix_s = time.time()
        self.manifest_path = self.root / "manifest.json"
        self._rolling_batches: list[tuple[int, float]] = []
        self._rolling_warmup: bool | None = None

    def should_record(self) -> bool:
        return (self._seen_batches % self.every_n) == 0

    def append(
        self,
        snapshot: dict[str, Any] | None,
        *,
        duration_s: float,
        batch_size: int,
        warmup: bool,
    ) -> None:
        batch_index = self._seen_batches
        self._seen_batches += 1
        rolling = self._update_rolling_processing_hz(
            duration_s=duration_s,
            batch_size=batch_size,
            warmup=warmup,
        )
        if snapshot is None or (batch_index % self.every_n) != 0:
            return

        rel_path = Path("snapshots") / f"snapshot_{self._written:06d}.pt"
        abs_path = self.root / rel_path
        torch.save(snapshot, abs_path)
        self.entries.append(
            {
                "index": int(self._written),
                "source_batch_index": int(batch_index),
                "snapshot": rel_path.as_posix(),
                "step_index": int(snapshot.get("step_index", self._written)),
                "duration_s": max(1.0e-6, float(duration_s)),
                "batch_size": max(1, int(batch_size)),
                "warmup": bool(warmup),
                **rolling,
            }
        )
        self._written += 1
        self.flush()

    def _update_rolling_processing_hz(
        self,
        *,
        duration_s: float,
        batch_size: int,
        warmup: bool,
    ) -> dict[str, Any]:
        if self._rolling_warmup is None or bool(warmup) != self._rolling_warmup:
            self._rolling_batches = []
            self._rolling_warmup = bool(warmup)

        frames = max(1, int(batch_size))
        duration = max(1.0e-6, float(duration_s))
        self._rolling_batches.append((frames, duration))

        total_frames = sum(item[0] for item in self._rolling_batches)
        while len(self._rolling_batches) > 1 and total_frames - self._rolling_batches[0][0] >= ROLLING_PROCESSING_WINDOW_FRAMES:
            total_frames -= self._rolling_batches[0][0]
            self._rolling_batches.pop(0)

        total_duration = sum(item[1] for item in self._rolling_batches)
        hz = float(total_frames / total_duration) if total_duration > 0.0 else 0.0
        return {
            "rolling_processing_hz": hz,
            "rolling_processing_window_frames": int(total_frames),
            "rolling_processing_window_target_frames": int(ROLLING_PROCESSING_WINDOW_FRAMES),
            "rolling_processing_window_duration_s": float(total_duration),
        }

    def flush(self) -> None:
        payload = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "scene_id": self.scene_id,
            "created_unix_s": self._started_unix_s,
            "updated_unix_s": time.time(),
            "every_n": self.every_n,
            "rolling_processing_window_target_frames": int(ROLLING_PROCESSING_WINDOW_FRAMES),
            "num_snapshots": len(self.entries),
            "entries": self.entries,
        }
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def close(self) -> None:
        self.flush()
