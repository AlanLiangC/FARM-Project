"""Interactive pseudo-live replay for completed RGB-D scene reconstructions."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scene_graph.offline.frame_sources.frames_json import FramesJsonFrameSource

LOGGER = logging.getLogger(__name__)


def _record_field(record: object, key: str) -> object:
    if isinstance(record, dict):
        return record.get(key)
    return getattr(record, key, None)


def _object_first_seen_indices(state: dict, frame_count: int) -> np.ndarray:
    active = state.get("active")
    means = state.get("means")
    if isinstance(active, torch.Tensor):
        object_count = int(active.numel())
    elif active is not None:
        object_count = int(np.asarray(active).size)
    elif hasattr(means, "shape") and len(means.shape) > 0:
        object_count = int(means.shape[0])
    else:
        object_count = 0

    image_order: dict[int, int] = {}
    for order, record in enumerate(state.get("images") or []):
        with contextlib.suppress(TypeError, ValueError):
            image_order[int(_record_field(record, "image_id"))] = order

    first_seen = np.zeros((object_count,), dtype=np.int64)
    observations = state.get("object_image_ids") or []
    for object_index in range(object_count):
        ids = observations[object_index] if object_index < len(observations) else []
        orders: list[int] = []
        for image_id in ids or []:
            with contextlib.suppress(TypeError, ValueError):
                value = int(image_id)
                orders.append(int(image_order.get(value, value)))
        first_seen[object_index] = min(orders) if orders else 0
    return np.clip(first_seen, 0, max(0, int(frame_count) - 1))


def _gravity_aligned_box(points: np.ndarray) -> dict[str, np.ndarray] | None:
    """Robust world-Z box for one object snapshot.

    Dynamic masks contain only the visible surface and can include a thin edge
    of the background.  Percentile trimming keeps that edge from determining
    the box, while horizontal PCA avoids a camera-axis-aligned footprint.
    """

    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 8:
        return None
    lower, upper = np.quantile(values, [0.01, 0.99], axis=0)
    trimmed = values[np.all((values >= lower) & (values <= upper), axis=1)]
    if len(trimmed) >= 8:
        values = trimmed
    xy_origin = np.median(values[:, :2], axis=0)
    centered_xy = values[:, :2] - xy_origin
    covariance = np.cov(centered_xy, rowvar=False)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        horizontal_axes = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    except np.linalg.LinAlgError:
        horizontal_axes = np.eye(2, dtype=np.float64)
    projected = centered_xy @ horizontal_axes
    projected_low, projected_high = np.quantile(projected, [0.01, 0.99], axis=0)
    z_low, z_high = np.quantile(values[:, 2], [0.01, 0.99])
    projected_center = 0.5 * (projected_low + projected_high)
    center_xy = xy_origin + horizontal_axes @ projected_center
    yaw = float(np.arctan2(horizontal_axes[1, 0], horizontal_axes[0, 0]))
    return {
        "center": np.asarray([center_xy[0], center_xy[1], 0.5 * (z_low + z_high)], dtype=np.float32),
        "dimensions": np.maximum(
            np.asarray([*(projected_high - projected_low), z_high - z_low], dtype=np.float32),
            0.08,
        ),
        "wxyz": np.asarray([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float32),
    }


class FramesJsonReplaySource:
    """Decode frames and compose a static background with a time-indexed dynamic layer."""

    def __init__(
        self,
        frames_dir: Path,
        final_state: dict,
        *,
        max_frames: int = 0,
        dynamic_cloud_path: Path | None = None,
    ) -> None:
        self.frames_dir = Path(frames_dir).expanduser().resolve()
        self.reader = FramesJsonFrameSource(
            self.frames_dir,
            end=None if int(max_frames) <= 0 else int(max_frames),
        )
        self.final_state = final_state
        self.total_frames = len(self.reader)
        self.first_seen = _object_first_seen_indices(final_state, self.total_frames)
        self.dynamic_xyz = np.empty((0, 3), dtype=np.float32)
        self.dynamic_colors = np.empty((0, 3), dtype=np.uint8)
        self.dynamic_image_ids = np.empty((0,), dtype=np.int32)
        self.dynamic_object_ids = np.empty((0,), dtype=np.int64)
        self._dynamic_by_image: dict[int, np.ndarray] = {}
        self._snapshot_boxes: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        self._template_boxes: dict[int, dict[str, np.ndarray]] = {}
        self._trajectory: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._object_index_by_id: dict[int, int] = {}
        self._load_dynamic_layer(dynamic_cloud_path)

    @property
    def has_dynamic_layer(self) -> bool:
        return bool(len(self.dynamic_xyz))

    def _load_dynamic_layer(self, path: Path | None) -> None:
        if path is None:
            return
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            return
        with np.load(source) as archive:
            self.dynamic_xyz = np.asarray(archive["xyz"], dtype=np.float32).reshape(-1, 3)
            self.dynamic_colors = np.asarray(archive["colors"], dtype=np.uint8).reshape(-1, 3)
            self.dynamic_image_ids = np.asarray(archive["image_id"], dtype=np.int32).reshape(-1)
            self.dynamic_object_ids = np.asarray(archive["object_id"], dtype=np.int64).reshape(-1)
        count = len(self.dynamic_xyz)
        if not (
            len(self.dynamic_colors) == count
            and len(self.dynamic_image_ids) == count
            and len(self.dynamic_object_ids) == count
        ):
            raise ValueError(f"inconsistent dynamic cloud arrays: {source}")
        for image_id in np.unique(self.dynamic_image_ids):
            self._dynamic_by_image[int(image_id)] = np.flatnonzero(self.dynamic_image_ids == image_id)

        object_ids = self.final_state.get("object_id")
        if isinstance(object_ids, torch.Tensor):
            object_id_values = object_ids.detach().cpu().reshape(-1).tolist()
        elif object_ids is not None:
            object_id_values = np.asarray(object_ids).reshape(-1).tolist()
        else:
            object_id_values = list(range(len(self.first_seen)))
        self._object_index_by_id = {int(value): index for index, value in enumerate(object_id_values)}

        temporal_rows = self.final_state.get("object_temporal_observations") or []
        for object_index, raw_rows in enumerate(temporal_rows):
            rows = [value for value in (raw_rows or []) if isinstance(value, dict)]
            samples: list[tuple[int, np.ndarray]] = []
            for value in rows:
                if value.get("image_id") is None:
                    continue
                position = np.asarray(value.get("position", []), dtype=np.float32).reshape(-1)
                if position.shape == (3,) and np.isfinite(position).all():
                    samples.append((int(value["image_id"]), position))
            if samples:
                samples.sort(key=lambda value: value[0])
                self._trajectory[object_index] = (
                    np.asarray([value[0] for value in samples], dtype=np.int32),
                    np.stack([value[1] for value in samples]).astype(np.float32),
                )

        # Stable object dimensions come from the most complete snapshots, not
        # the often-fragmentary last mask. Snapshot centres/orientations remain
        # time dependent.
        for object_id in np.unique(self.dynamic_object_ids):
            object_samples: list[tuple[int, int, dict[str, np.ndarray]]] = []
            object_mask = self.dynamic_object_ids == object_id
            for image_id in np.unique(self.dynamic_image_ids[object_mask]):
                indices = np.flatnonzero(object_mask & (self.dynamic_image_ids == image_id))
                box = _gravity_aligned_box(self.dynamic_xyz[indices])
                if box is None:
                    continue
                self._snapshot_boxes[(int(image_id), int(object_id))] = box
                object_samples.append((len(indices), int(image_id), box))
            if not object_samples:
                continue
            selected = sorted(object_samples, key=lambda value: (value[0], value[1]), reverse=True)[:7]
            template = dict(selected[0][2])
            template["dimensions"] = np.median(
                np.stack([value[2]["dimensions"] for value in selected]), axis=0
            ).astype(np.float32)
            self._template_boxes[int(object_id)] = template

    def dynamic_snapshot(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        indices = self._dynamic_by_image.get(int(index))
        if indices is None:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
        return self.dynamic_xyz[indices], self.dynamic_colors[indices]

    def _trajectory_position(self, object_index: int, image_id: int) -> np.ndarray | None:
        row = self._trajectory.get(int(object_index))
        if row is None:
            return None
        image_ids, positions = row
        where = int(np.searchsorted(image_ids, int(image_id)))
        if where < len(image_ids) and int(image_ids[where]) == int(image_id):
            return positions[where]
        # Fill only short tracker/detector gaps; never leave a dynamic object
        # parked indefinitely after it exits the scene.
        if 0 < where < len(image_ids) and int(image_ids[where] - image_ids[where - 1]) <= 5:
            span = float(image_ids[where] - image_ids[where - 1])
            alpha = float(image_id - image_ids[where - 1]) / max(span, 1.0)
            return (1.0 - alpha) * positions[where - 1] + alpha * positions[where]
        return None

    def _state_at(self, index: int) -> dict:
        state = dict(self.final_state)
        active = self.final_state.get("active")
        means = self.final_state.get("means")
        if isinstance(means, torch.Tensor):
            state["means"] = means.clone()
        elif means is not None:
            state["means"] = np.asarray(means).copy()
        visible = self.first_seen <= int(index)
        overrides: dict[int, dict[str, np.ndarray]] = {}
        if self.has_dynamic_layer:
            dynamic_indices = set(self._object_index_by_id.get(int(value), -1) for value in np.unique(self.dynamic_object_ids))
            dynamic_indices.discard(-1)
            for object_index in dynamic_indices:
                visible[object_index] = False
            for object_id, object_index in self._object_index_by_id.items():
                if object_index not in dynamic_indices:
                    continue
                position = self._trajectory_position(object_index, int(index))
                if position is None:
                    continue
                visible[object_index] = True
                snapshot = self._snapshot_boxes.get((int(index), int(object_id)))
                template = self._template_boxes.get(int(object_id))
                if template is None:
                    continue
                box = dict(snapshot or template)
                box["dimensions"] = template["dimensions"]
                if snapshot is None:
                    box["center"] = np.asarray(position, dtype=np.float32)
                overrides[object_index] = box
                if isinstance(state.get("means"), torch.Tensor):
                    state["means"][object_index] = torch.as_tensor(
                        box["center"],
                        dtype=state["means"].dtype,
                        device=state["means"].device,
                    )
                elif state.get("means") is not None:
                    state["means"][object_index] = box["center"]
        if isinstance(active, torch.Tensor):
            mask = torch.as_tensor(visible, dtype=torch.bool, device=active.device)
            state["active"] = active.to(dtype=torch.bool).clone() & mask
        elif active is not None:
            active_np = np.asarray(active, dtype=bool).reshape(-1)
            state["active"] = active_np & visible[: active_np.shape[0]]
        state["_object_box_overrides"] = overrides
        state["_temporal_layer_frame"] = int(index)
        return state

    def load(self, index: int) -> dict[str, Any]:
        index = max(0, min(int(index), self.total_frames - 1))
        entry = self.reader._frames[index]
        depth_size = tuple(entry.get("depth_size") or [])
        if len(depth_size) != 2:
            raise ValueError(f"replay frame {entry.get('frame_id')} has no depth_size")
        height, width = int(depth_size[0]), int(depth_size[1])
        color = self.reader._read_rgb(str(entry["rgb_path"]), height, width)
        depth = self.reader._read_depth(str(entry["depth_path"]))
        intrinsics = np.asarray(entry["K"], dtype=np.float32).reshape(3, 3)
        pose = np.asarray(entry["T_world_cam"], dtype=np.float32).reshape(4, 4)
        timestamp_ns = int(entry.get("timestamp_ns") or round(index * 1.0e9 / 5.0))
        return {
            "colors": [color],
            "depths": [depth],
            "intrinsics": [intrinsics],
            "poses": [pose],
            "scene_state": self._state_at(index),
            "timestamp_ns": timestamp_ns,
        }


class ViserReplayController:
    """Bind a frames-json replay source to a PipelineViserVisualizer GUI."""

    SPEEDS = {"0.5×": 0.5, "1×": 1.0, "2×": 2.0, "4×": 4.0}

    def __init__(
        self,
        visualizer,
        source: FramesJsonReplaySource,
        *,
        playback_fps: float = 5.0,
        title: str = "Reconstruction playback",
        preserve_accumulated_cloud: bool = False,
    ) -> None:
        if not visualizer.enabled or visualizer._server is None:
            raise RuntimeError("Viser must be enabled before adding replay controls")
        if source.total_frames <= 0:
            raise ValueError("replay source is empty")
        self.visualizer = visualizer
        self.source = source
        self.total_frames = int(source.total_frames)
        self.playback_fps = max(0.1, float(playback_fps))
        self.title = str(title)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._playing = threading.Event()
        self._current_index = self.total_frames - 1
        self._integrated_until = -1
        self._setting_slider = False
        self._setting_play = False
        self._status = None
        self._play = None
        self._frame = None
        self._speed = None
        self._loop = None
        self._restart = None
        self._full_result = None
        self._dynamic_handle = None

        if preserve_accumulated_cloud:
            points, colors = visualizer.accumulated_point_cloud()
            if points is not None and points.size:
                visualizer.add_background_point_cloud(
                    points,
                    colors,
                    point_size=visualizer._point_size,
                    name="/replay_final_cloud",
                )
                visualizer.reset_streaming_geometry()
        visualizer.set_background_visible(True)
        visualizer.prepare_replay_controls(self.total_frames)
        self._setup_gui()
        self._show_full_result()
        self._thread = threading.Thread(target=self._run, name="viser-replay", daemon=True)
        self._thread.start()

    def _setup_gui(self) -> None:
        gui = self.visualizer._server.gui
        with gui.add_folder("Playback"):
            self._status = gui.add_markdown(f"### {self.title}\n\nReady · {self.total_frames} frames")
            self._play = gui.add_checkbox("Play", False, hint="Play the reconstruction from the timeline.")
            self._frame = gui.add_slider(
                "Frame",
                min=1,
                max=self.total_frames,
                step=1,
                initial_value=self.total_frames,
            )
            self._speed = gui.add_dropdown(
                "Speed",
                tuple(self.SPEEDS),
                initial_value="1×",
            )
            self._loop = gui.add_checkbox("Loop", False)
            self._restart = gui.add_button("Restart", color="cyan")
            self._full_result = gui.add_button("Full result")

        @self._play.on_update
        def _(_event=None):
            if self._setting_play:
                return
            if bool(self._play.value):
                self._playing.set()
            else:
                self._playing.clear()

        @self._frame.on_update
        def _(_event=None):
            if self._setting_slider:
                return
            self._set_playing(False)
            self.render(int(self._frame.value) - 1)

        @self._restart.on_click
        def _(_event=None):
            self._set_playing(False)
            self.render(0)

        @self._full_result.on_click
        def _(_event=None):
            self._set_playing(False)
            self._show_full_result()

    def _set_playing(self, value: bool) -> None:
        if value:
            self._playing.set()
        else:
            self._playing.clear()
        if self._play is not None:
            self._setting_play = True
            with contextlib.suppress(Exception):
                self._play.value = bool(value)
            self._setting_play = False

    def _set_slider(self, index: int) -> None:
        self._setting_slider = True
        with contextlib.suppress(Exception):
            self._frame.value = int(index) + 1
        self._setting_slider = False

    def _set_status(self, index: int, message: str = "PLAYBACK") -> None:
        self.visualizer._gui_set_markdown(
            self._status,
            f"### {self.title}\n\n`{message}` · frame **{index + 1} / {self.total_frames}**",
        )

    def _show_dynamic_snapshot(self, index: int) -> None:
        if self._dynamic_handle is not None:
            with contextlib.suppress(Exception):
                self._dynamic_handle.remove()
            self._dynamic_handle = None
        points, colors = self.source.dynamic_snapshot(index)
        if not len(points):
            return
        self._dynamic_handle = self.visualizer._server.scene.add_point_cloud(
            "/dynamic_4d/current",
            points=np.asarray(points, dtype=np.float32),
            colors=np.asarray(colors, dtype=np.uint8),
            point_size=max(0.004, float(self.visualizer._point_size) * 1.25),
            point_shape="circle",
        )

    def _show_full_result(self) -> None:
        with self._lock:
            has_background = bool(self.visualizer._background_point_handles)
            if not has_background:
                self.render(self.total_frames - 1)
                return
            self.visualizer.reset_streaming_geometry()
            self.visualizer.set_background_visible(True)
            for handle in (
                self.visualizer._stream_ego_handle,
                self.visualizer._stream_ego_label,
            ):
                if handle is not None:
                    with contextlib.suppress(Exception):
                        handle.visible = False
            if self.source.has_dynamic_layer:
                payload = self.source.load(self.total_frames - 1)
                self.visualizer.update([], [], [], [], payload["scene_state"])
                self._show_dynamic_snapshot(self.total_frames - 1)
            else:
                self.visualizer.update([], [], [], [], self.source.final_state)
            self._integrated_until = -1
            self._current_index = self.total_frames - 1
            self._set_slider(self._current_index)
            self._set_status(self._current_index, "FULL RESULT")

    def render(self, index: int) -> None:
        index = max(0, min(int(index), self.total_frames - 1))
        with self._lock:
            if index == self._integrated_until:
                self._current_index = index
                self._set_slider(index)
                return
            if self.source.has_dynamic_layer:
                # The static cloud has already had dynamic-capable masks
                # removed. Keep it fixed and overlay only the selected time
                # slice; integrating prior raw RGB-D frames would recreate the
                # very dynamic trails this representation is meant to avoid.
                self.visualizer.set_background_visible(True)
                self.visualizer.reset_streaming_geometry()
                payload = self.source.load(index)
                empty_depths = [np.zeros_like(value) for value in payload["depths"]]
                self.visualizer.update(
                    payload["colors"],
                    empty_depths,
                    payload["intrinsics"],
                    payload["poses"],
                    payload["scene_state"],
                    frame_index=index,
                    timestamp_ns=int(payload["timestamp_ns"]),
                )
                self._show_dynamic_snapshot(index)
                self._integrated_until = index
                self._current_index = index
                self._set_slider(index)
                self._set_status(index, "STATIC + DYNAMIC t")
                return
            self.visualizer.set_background_visible(False)
            if self.visualizer._stream_ego_handle is not None:
                with contextlib.suppress(Exception):
                    self.visualizer._stream_ego_handle.visible = True
            if self._integrated_until < 0 or index < self._integrated_until:
                self.visualizer.reset_streaming_geometry()
                start = 0
            else:
                start = self._integrated_until + 1
            for prior_index in range(start, index):
                prior = self.source.load(prior_index)
                self.visualizer.integrate_replay_frame(
                    prior["colors"],
                    prior["depths"],
                    prior["intrinsics"],
                    prior["poses"],
                )
            payload = self.source.load(index)
            self.visualizer.update(
                payload["colors"],
                payload["depths"],
                payload["intrinsics"],
                payload["poses"],
                payload["scene_state"],
                frame_index=index,
                timestamp_ns=int(payload["timestamp_ns"]),
            )
            self._integrated_until = index
            self._current_index = index
            self._set_slider(index)
            self._set_status(index)

    def _run(self) -> None:
        while not self._stop.wait(0.02):
            if not self._playing.is_set():
                continue
            if self._current_index >= self.total_frames - 1:
                if bool(self._loop.value):
                    target = 0
                else:
                    # Pressing Play while parked on the final result restarts
                    # once; reaching the end during playback stops normally.
                    target = 0 if self._integrated_until < 0 else None
                    if target is None:
                        self._set_playing(False)
                        continue
            else:
                target = self._current_index + 1
            started = time.monotonic()
            try:
                self.render(target)
            except Exception as exc:  # noqa: BLE001 - keep the Viser server alive
                LOGGER.warning("Viser replay frame %s failed: %s", target, exc)
                self._set_playing(False)
                continue
            multiplier = self.SPEEDS.get(str(self._speed.value), 1.0)
            delay = max(0.0, 1.0 / (self.playback_fps * multiplier) - (time.monotonic() - started))
            self._stop.wait(delay)

    def stop(self) -> None:
        self._stop.set()
        self._playing.clear()


def attach_frames_json_replay(
    visualizer,
    frames_dir: Path,
    final_state: dict,
    *,
    playback_fps: float = 5.0,
    max_frames: int = 0,
    title: str = "Reconstruction playback",
    preserve_accumulated_cloud: bool = False,
    dynamic_cloud_path: Path | None = None,
) -> ViserReplayController:
    source = FramesJsonReplaySource(
        frames_dir,
        final_state,
        max_frames=max_frames,
        dynamic_cloud_path=dynamic_cloud_path,
    )
    return ViserReplayController(
        visualizer,
        source,
        playback_fps=playback_fps,
        title=title,
        preserve_accumulated_cloud=preserve_accumulated_cloud,
    )
