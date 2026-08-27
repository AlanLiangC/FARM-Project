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


class FramesJsonReplaySource:
    """Lazily decode frames and reveal final objects at their first observation."""

    def __init__(self, frames_dir: Path, final_state: dict, *, max_frames: int = 0) -> None:
        self.frames_dir = Path(frames_dir).expanduser().resolve()
        self.reader = FramesJsonFrameSource(
            self.frames_dir,
            end=None if int(max_frames) <= 0 else int(max_frames),
        )
        self.final_state = final_state
        self.total_frames = len(self.reader)
        self.first_seen = _object_first_seen_indices(final_state, self.total_frames)

    def _state_at(self, index: int) -> dict:
        state = dict(self.final_state)
        active = self.final_state.get("active")
        visible = self.first_seen <= int(index)
        if isinstance(active, torch.Tensor):
            mask = torch.as_tensor(visible, dtype=torch.bool, device=active.device)
            state["active"] = active.to(dtype=torch.bool).clone() & mask
        elif active is not None:
            active_np = np.asarray(active, dtype=bool).reshape(-1)
            state["active"] = active_np & visible[: active_np.shape[0]]
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
) -> ViserReplayController:
    source = FramesJsonReplaySource(frames_dir, final_state, max_frames=max_frames)
    return ViserReplayController(
        visualizer,
        source,
        playback_fps=playback_fps,
        title=title,
        preserve_accumulated_cloud=preserve_accumulated_cloud,
    )
