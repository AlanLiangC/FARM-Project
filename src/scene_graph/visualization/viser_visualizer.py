"""Minimal Viser visualization hook for the offline mapping pipeline."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

MAX_VIZUALIZATION_DEPTH = 1000.0  # meters

try:  # pragma: no cover - optional dependency
    import viser
    from viser.transforms import SO3
except Exception:  # pragma: no cover - viser is optional or may fail to import at runtime
    viser = None
    SO3 = None  # type: ignore

from scene_graph.utils.geometry import decode_voxel_keys_numpy as _decode_voxel_keys
from scene_graph.utils.geometry import voxel_cloud_aabb as _voxel_cloud_aabb


@dataclass
class _PointCloud:
    points: np.ndarray
    colors: np.ndarray


def _to_numpy(array) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _cholesky_with_jitter(
    mat: torch.Tensor,
    eye: torch.Tensor,
    base_eps: float,
    max_tries: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attempt Cholesky with progressively larger jitter; returns (L, info)."""
    L = None
    info = None
    for i in range(max_tries):
        jitter = (10**i) * base_eps
        L, info = torch.linalg.cholesky_ex(mat + jitter * eye)
        if torch.all(info == 0):
            break
    assert L is not None and info is not None
    return L, info


def _hellinger_distance_batch(
    mu1: torch.Tensor,
    cov1: torch.Tensor,
    mu2: torch.Tensor,
    cov2: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Compute Hellinger^2 distance between aligned Gaussian batches."""
    eye = torch.eye(3, device=mu1.device, dtype=mu1.dtype).unsqueeze(0)
    cov1 = cov1 + eps * eye
    cov2 = cov2 + eps * eye
    sigma = 0.5 * (cov1 + cov2)

    L1, info1 = _cholesky_with_jitter(cov1, eye, eps)
    L2, info2 = _cholesky_with_jitter(cov2, eye, eps)
    L, info_sigma = _cholesky_with_jitter(sigma, eye, eps)

    valid_mask = (info1 == 0) & (info2 == 0) & (info_sigma == 0)
    if not torch.any(valid_mask):
        return torch.ones(mu1.shape[0], device=mu1.device, dtype=mu1.dtype)

    def _logdet_from_chol(chol: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not torch.any(mask):
            return torch.zeros_like(mask, dtype=mu1.dtype)
        diag = torch.diagonal(chol[mask], dim1=-2, dim2=-1)
        return 2 * torch.log(diag).sum(-1)

    logdet_S1 = torch.zeros(mu1.shape[0], device=mu1.device, dtype=mu1.dtype)
    logdet_S2 = torch.zeros_like(logdet_S1)
    logdet_sigma = torch.zeros_like(logdet_S1)

    logdet_S1[info1 == 0] = _logdet_from_chol(L1, info1 == 0)
    logdet_S2[info2 == 0] = _logdet_from_chol(L2, info2 == 0)
    logdet_sigma[info_sigma == 0] = _logdet_from_chol(L, info_sigma == 0)

    delta = (mu1 - mu2).unsqueeze(-1)
    y = torch.zeros_like(delta)
    if torch.any(valid_mask):
        y_valid = torch.cholesky_solve(delta[valid_mask], L[valid_mask])
        y[valid_mask] = y_valid
    quad = (delta.squeeze(-1) * y.squeeze(-1)).sum(-1)

    log_bc = 0.25 * (logdet_S1 + logdet_S2) - 0.5 * logdet_sigma - 0.125 * quad
    bc = torch.zeros_like(log_bc)
    bc[valid_mask] = torch.exp(log_bc[valid_mask])
    bc[~valid_mask] = 0.0
    return 1.0 - bc


def _cov6_to_matrix_torch(cov6: torch.Tensor) -> torch.Tensor:
    """Convert packed covariance tensors (N,6) to (N,3,3)."""
    zeros = torch.zeros(cov6.shape[0], 3, 3, device=cov6.device, dtype=cov6.dtype)
    zeros[:, 0, 0] = cov6[:, 0]
    zeros[:, 0, 1] = zeros[:, 1, 0] = cov6[:, 1]
    zeros[:, 0, 2] = zeros[:, 2, 0] = cov6[:, 2]
    zeros[:, 1, 1] = cov6[:, 3]
    zeros[:, 1, 2] = zeros[:, 2, 1] = cov6[:, 4]
    zeros[:, 2, 2] = cov6[:, 5]
    return zeros


class PipelineViserVisualizer:
    """Lightweight Viser visualizer for batched mapping outputs."""

    def __init__(
        self,
        enabled: bool = True,
        voxel_size_m: float = 0.1,
        point_size_m: float = 0.01,
        host: str = "127.0.0.1",
        port: int = 8080,
        live_rgb_enabled: bool = True,
        live_rgb_max_side: int = 320,
        live_rgb_max_fps: float = 5.0,
        object_gaussians_enabled: bool = False,
        object_connections_enabled: bool = False,
        regions_enabled: bool = True,
        covisibility_connections_enabled: bool = False,
        image_pose_axes_enabled: bool = True,
        object_image_connections_enabled: bool = False,
        object_voxel_cloud_enabled: bool = False,
        object_voxel_max_points_per_object: int = 0,
        object_voxel_point_size: float = 0.025,
        object_box_from_voxels: bool = False,
        query_examples: Sequence[str] | None = None,
        object_box_max_volume_m3: float = 0.0,
        object_box_max_side_m: float = 0.0,
        object_box_max_z_m: float = 0.0,
        object_box_min_distance_m: float = 0.0,
        object_box_distance_position: Sequence[float] | None = None,
        object_box_large_side_threshold_m: float = 0.0,
        object_box_max_large_sides: int = 0,
        object_box_exclude_terms: str | Sequence[str] | None = None,
        point_min_distance_m: float = 0.0,
        point_distance_position: Sequence[float] | None = None,
        view_depth_clip_min_m: float = 0.0,
        view_depth_clip_position: Sequence[float] | None = None,
        view_depth_clip_look_at: Sequence[float] | None = None,
        on_edit_caption=None,
        on_delete_object=None,
        on_save_all=None,
        on_toggle_lock=None,
        on_add_object=None,
    ) -> None:
        self._voxel_size = float(voxel_size_m)
        self._point_size = max(1.0e-4, float(point_size_m))
        self._rng = np.random.default_rng(0)
        self._host = str(host or "127.0.0.1")
        self._port = int(port)
        self._live_rgb_enabled = bool(live_rgb_enabled)
        self._live_rgb_max_side = max(1, int(live_rgb_max_side))
        self._live_rgb_max_fps = max(0.0, float(live_rgb_max_fps))
        self._object_gaussians_enabled = bool(object_gaussians_enabled)
        self._object_connections_enabled = bool(object_connections_enabled)
        self._regions_enabled = bool(regions_enabled)
        self._covisibility_connections_enabled = bool(covisibility_connections_enabled)
        self._image_pose_axes_enabled = bool(image_pose_axes_enabled)
        self._object_image_connections_enabled = bool(object_image_connections_enabled)
        self._object_voxel_cloud_enabled = bool(object_voxel_cloud_enabled)
        self._object_voxel_max_points_per_object = max(0, int(object_voxel_max_points_per_object))
        self._object_voxel_point_size = max(1.0e-4, float(object_voxel_point_size))
        self._object_box_from_voxels = bool(object_box_from_voxels)
        self._object_box_max_volume_m3 = max(0.0, float(object_box_max_volume_m3))
        self._object_box_max_side_m = max(0.0, float(object_box_max_side_m))
        self._object_box_max_z_m = max(0.0, float(object_box_max_z_m))
        self._object_box_min_distance_m = max(0.0, float(object_box_min_distance_m))
        self._object_box_distance_position: np.ndarray | None = None
        if object_box_distance_position is not None:
            with contextlib.suppress(Exception):
                pos = np.asarray(object_box_distance_position, dtype=np.float32).reshape(3)
                if np.isfinite(pos).all():
                    self._object_box_distance_position = pos
        self._object_box_large_side_threshold_m = max(0.0, float(object_box_large_side_threshold_m))
        self._object_box_max_large_sides = max(0, int(object_box_max_large_sides))
        self._object_box_exclude_terms = self._normalize_exclude_terms(object_box_exclude_terms)
        self._point_min_distance_m = max(0.0, float(point_min_distance_m))
        self._point_distance_position: np.ndarray | None = None
        if point_distance_position is not None:
            with contextlib.suppress(Exception):
                pos = np.asarray(point_distance_position, dtype=np.float32).reshape(3)
                if np.isfinite(pos).all():
                    self._point_distance_position = pos
        self._view_depth_clip_min_m = max(0.0, float(view_depth_clip_min_m))
        self._view_depth_clip_position: np.ndarray | None = None
        self._view_depth_clip_forward: np.ndarray | None = None
        self.set_view_depth_clip(
            position=view_depth_clip_position,
            look_at=view_depth_clip_look_at,
            min_depth_m=self._view_depth_clip_min_m,
        )
        self._last_live_rgb_update_s = 0.0

        # Interactive editing callbacks
        self._on_edit_caption = on_edit_caption
        self._on_delete_object = on_delete_object
        self._on_save_all = on_save_all
        self._on_toggle_lock = on_toggle_lock
        self._on_add_object = on_add_object

        # Handles
        self._point_handle = None
        self._gaussian_handle = None
        self._object_connection_handle = None
        self._region_connection_handle = None
        self._covisibility_connection_handle = None
        self._covisibility_filtered_connection_handle = None
        self._image_pose_handle = None
        self._object_image_connection_handle = None
        self._det_connection_handle = None
        self._object_voxel_cloud_handle = None
        self._object_voxel_cloud_dim_handle = None
        self._robot_trajectory_handle = None
        self._search_path_handle = None
        self._search_highlight_box_handle = None
        self._search_highlight_edges_handle = None
        self._search_relations_handle = None
        self._query_roles: dict | None = None
        # When set (after a query), only these object ids are rendered (target /
        # confounders / anchors); every other object is hidden entirely — no box,
        # no points, and no click target. None = show everything normally.
        self._focus_object_ids: set[int] | None = None
        self._camera_frame = None
        self._server = None
        self._latest_client = None
        self._home_camera: tuple[np.ndarray, np.ndarray] | None = None
        self._query_examples_dropdown = None
        self._query_examples_ready = False
        self._query_examples_override = [
            str(q).strip() for q in (query_examples or []) if str(q).strip()
        ][: self._QUERY_EXAMPLE_COUNT]

        # Object Management
        self._id_to_color: Dict[int, np.ndarray] = {}
        self._object_cube_handles: Dict[int, viser.SceneNodeHandle] = {}
        self._detection_cube_handles: Dict[int, viser.SceneNodeHandle] = {}
        self._region_ball_handles: Dict[int, viser.SceneNodeHandle] = {}
        self._object_voxel_cache: dict[tuple[int, int, int, int, int, int, int], np.ndarray] = {}
        self._object_voxel_aabb_cache: dict[tuple[int, int, int, int, int, int], tuple[np.ndarray, np.ndarray] | None] = {}

        # Accumulators
        self._accum_points: np.ndarray | None = None
        self._accum_colors: np.ndarray | None = None
        self._accum_voxel_keys: set[tuple[int, int, int]] | None = None
        self._robot_trajectory_positions: list[np.ndarray] = []
        self._robot_trajectory_max_points: int = 20000
        self._robot_trajectory_min_step_m: float = 1e-3
        self._latest_scene_state: dict | None = None
        self._latest_poses: Sequence[torch.Tensor | np.ndarray] = []
        self._hide_unclear_object_boxes = False
        self._streaming_paused = threading.Event()

        # GUI Handles
        self._caption_display = None
        self._image_display = None
        self._live_rgb_display = None
        self._live_rgb_caption = None
        self._search_animation_cancel_event: threading.Event | None = None

        # Query / retrieval GUI handles + backend state.
        #
        # The query box runs the SAME relational pipeline the eval/paper use:
        # parse_query (LLM) -> execute_spatial_query with retrieval_mode="multi"
        # (RRF over caption-text, caption-raw, SigLIP2, and Qwen3-VL channels).
        # The engine defaults to "joint_v1" (override with VISER_SPATIAL_METHOD,
        # e.g. the paper's locked "unified_soft_w50"). That needs three vLLM
        # servers plus a local SigLIP2 model; each channel degrades gracefully
        # if its backend is down. URLs/models are env-driven so the same code
        # drives local or remote (tunnelled) servers.
        self._query_input = None
        self._query_search_button = None
        self._query_results_display = None
        self._reset_button = None
        self._retrieval_backend_button = None
        self._retrieval_status = None
        self._retrieval_ready: bool = False
        self._retrieval_procs: Dict[str, subprocess.Popen] = {}
        self._retrieval_proc_logs: Dict[str, str] = {}
        self._retrieval_lock = threading.Lock()
        self._retrieval_embedder = None  # scene_graph.llm_utils.EmbedInterface (lazy)
        self._retrieval_llm = None       # scene_graph.llm_utils.LLMInterface (lazy)
        self._retrieval_embed_base_url: str = os.getenv("VLLM_EMBED_BASE_URL", "http://localhost:8002/v1").rstrip("/")
        self._retrieval_llm_base_url: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")

        # Filter GUI handles
        self._max_side_slider = None

        # State used for picking.
        self._latest_ids: np.ndarray | None = None
        self._latest_captions: list[str] | None = None
        self._latest_caption_edit_texts: list[str] | None = None
        self._latest_images: list[object | None] | None = None
        self._latest_view_refs: list[str | None] | None = None
        self._view_image_cache: dict[str, np.ndarray] = {}
        self._latest_detection_ids: list[int] | None = None
        self._latest_detection_captions: list[str] | None = None
        self._latest_detection_images: list[object | None] | None = None

        # Interactive editing state
        self._selected_object_id: int | None = None
        self._edit_caption_input = None
        self._edit_apply_button = None
        self._delete_button = None
        self._save_all_button = None
        self._lock_toggle_button = None
        self._add_caption_input = None
        self._add_location_input = None
        self._add_image_path_input = None
        self._add_view1_input = None
        self._add_view2_input = None
        self._add_view3_input = None
        self._add_object_button = None
        self._edit_status = None

        self._enabled = bool(enabled and viser is not None)
        if not self._enabled:
            if enabled and viser is None:
                LOGGER.warning("Viser not installed; disabling visualization.")
            return

        try:
            self._server = viser.ViserServer(host=self._host, port=self._port)
            self._camera_frame = self._server.scene.add_frame(name="/camera_pose", axes_length=0.25, axes_radius=0.01)

            gui = getattr(self._server, "gui", None)
            if gui is not None:
                if self._live_rgb_enabled:
                    try:
                        with gui.add_folder("Live RGB"):
                            self._live_rgb_caption = gui.add_markdown("Waiting for RGB frames.")
                            placeholder = np.zeros((64, 64, 3), dtype=np.uint8)
                            self._live_rgb_display = gui.add_image(
                                placeholder,
                                label="Latest RGB frame",
                                format="jpeg",
                            )
                    except Exception:
                        self._live_rgb_caption = None
                        self._live_rgb_display = None
                try:
                    with gui.add_folder("Last clicked object"):
                        self._caption_display = gui.add_markdown("No object selected yet.")

                        placeholder = np.zeros((64, 64, 3), dtype=np.uint8)
                        self._image_display = gui.add_image(
                            placeholder,
                            label="Object image",
                            format="jpeg",
                        )
                except Exception:
                    self._caption_display = None
                    self._image_display = None

            @self._server.on_client_connect
            def _on_client_connect(client: "viser.ClientHandle") -> None:
                LOGGER.info("Client connected: %s", getattr(client, "client_id", "unknown"))
                self._latest_client = client
                home = self._home_camera
                if home is not None:
                    # Off-thread: a camera write can block until the client's
                    # first state message, and this callback must not stall the
                    # server's message loop (e.g. on a half-closed connection).
                    threading.Thread(
                        target=self._try_set_camera,
                        args=(client,),
                        kwargs={"position": home[0], "look_at": home[1]},
                        daemon=True,
                    ).start()

        except Exception as exc:
            LOGGER.warning("Failed to start Viser server: %s", exc)
            self._enabled = False
            return

        self._setup_filter_gui()
        self._setup_query_gui()
        self._setup_edit_gui()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def streaming_paused(self) -> bool:
        return self._streaming_paused.is_set()

    def update(
        self,
        colors: Sequence[torch.Tensor],
        depths: Sequence[torch.Tensor],
        intrinsics: Sequence[torch.Tensor],
        poses: Sequence[torch.Tensor],
        scene_state: dict,
        detection_info: dict | None = None,
        detection_neighbors: Sequence[Sequence[int]] | None = None,
    ) -> None:
        """Render the current batch + active objects."""

        if not self._enabled or self._server is None:
            return

        self._latest_scene_state = scene_state
        with contextlib.suppress(Exception):
            self._refresh_query_examples(scene_state)
        self._latest_poses = poses
        self._update_live_rgb_panel(colors)

        # OPTIMIZATION: Use atomic() to bundle all updates (points, camera, boxes)
        # into a single network packet. This prevents "lag" when updating many boxes.
        with self._server.atomic():
            self._integrate_points(colors, depths, intrinsics, poses)
            self._update_camera_pose(poses[-1] if len(poses) > 0 else None)
            self._update_robot_trajectory(poses, scene_state)
            if self._image_pose_axes_enabled:
                self._update_image_poses(scene_state)
            else:
                self._clear_image_pose_lines()
            self._update_gaussians(scene_state)
            if self._object_connections_enabled:
                self._update_object_connections(scene_state)
            else:
                self._clear_object_connections()
            if self._regions_enabled:
                self._update_regions(scene_state)
            else:
                self._clear_regions()
            if self._covisibility_connections_enabled:
                self._update_covisibility_connections(scene_state)
                self._update_covisibility_connections_filtered(scene_state)
            else:
                self._clear_covisibility_connections()
            if self._object_image_connections_enabled:
                self._update_object_image_connections(scene_state)
            else:
                self._clear_object_image_connections()
            self._update_detections(detection_info)
            self._update_detection_connections(detection_info, detection_neighbors, scene_state)

        self._server.flush()

    def add_background_point_cloud(
        self,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        *,
        point_size: float = 0.02,
        name: str = "/background",
    ) -> None:
        """Overlay a static point cloud for scene context (e.g. a dataset's
        accumulated ``cloud.npz``) when replaying a saved scene state instead
        of streaming live RGBD frames."""
        if not self._enabled or self._server is None:
            return
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] == 0:
            return
        if colors is not None:
            cols = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
            if cols.shape[0] != pts.shape[0]:
                cols = None
            elif cols.max(initial=0.0) > 1.5:  # uint8-style 0..255
                cols = cols / 255.0
        else:
            cols = None
        if cols is None:
            cols = np.full((pts.shape[0], 3), 0.55, dtype=np.float32)
        self._server.scene.add_point_cloud(
            name, points=pts, colors=cols, point_size=max(1.0e-4, float(point_size))
        )
        self._server.flush()

    def add_trajectory(self, poses: np.ndarray | None, *, name: str = "/trajectory", axes_length: float = 0.15, axes_radius: float = 0.008) -> None:
        """Draw the capture trajectory as one small coordinate frame per camera
        pose (for saved-state viewing, where no live frusta are streamed).
        *poses* is an (N, 4, 4) array of camera-to-world transforms."""
        if poses is None or self._server is None:
            return
        arr = np.asarray(poses, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[-2:] != (4, 4):
            return
        arr = arr[np.isfinite(arr).all(axis=(1, 2))]
        if arr.shape[0] == 0:
            return
        positions = np.ascontiguousarray(arr[:, :3, 3])
        wxyzs = None
        if SO3 is not None:
            with contextlib.suppress(Exception):
                wxyzs = np.asarray(SO3.from_matrix(arr[:, :3, :3]).wxyz, dtype=np.float32).reshape(-1, 4)
        if wxyzs is None or wxyzs.shape[0] != positions.shape[0]:
            wxyzs = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (positions.shape[0], 1))
        with contextlib.suppress(Exception):
            self._server.scene.add_batched_axes(
                name,
                batched_wxyzs=wxyzs,
                batched_positions=positions,
                axes_length=float(axes_length),
                axes_radius=float(axes_radius),
            )
            self._server.flush()

    def set_home_view(self, points: np.ndarray | None) -> None:
        """Frame *points* (Nx3) with an elevated 3/4 overview camera.

        Applied to already-connected clients and to every client that connects
        later, so a saved scene opens showing the whole scene instead of the
        viser default near the origin. The vertical axis is inferred as the
        AABB's smallest extent — scans are much wider than they are tall.
        """
        if points is None or self._server is None:
            return
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < 2:
            return
        # Percentile bounds so sparse range outliers don't inflate the framing.
        mins, maxs = np.percentile(pts, 2.0, axis=0), np.percentile(pts, 98.0, axis=0)
        center = (mins + maxs) / 2.0
        extent = maxs - mins
        up_axis = int(np.argmin(extent))
        horiz = [i for i in range(3) if i != up_axis]
        dist = max(float(np.linalg.norm(extent)) * 0.55, 2.0)
        offset = np.zeros(3)
        offset[horiz[0]] = 0.50 * dist
        offset[horiz[1]] = -0.40 * dist
        offset[up_axis] = 0.70 * dist
        self._home_camera = ((center + offset).astype(np.float32), center.astype(np.float32))
        clients: dict = {}
        with contextlib.suppress(Exception):
            clients = self._server.get_clients()
        for client in clients.values():
            self._try_set_camera(client, position=self._home_camera[0], look_at=self._home_camera[1])

    def set_view_depth_clip(
        self,
        *,
        position: Sequence[float] | np.ndarray | None,
        look_at: Sequence[float] | np.ndarray | None,
        min_depth_m: float | None = None,
    ) -> None:
        if min_depth_m is not None:
            self._view_depth_clip_min_m = max(0.0, float(min_depth_m))
        self._view_depth_clip_position = None
        self._view_depth_clip_forward = None
        if self._view_depth_clip_min_m <= 0.0 or position is None or look_at is None:
            return
        try:
            pos = np.asarray(position, dtype=np.float32).reshape(3)
            look = np.asarray(look_at, dtype=np.float32).reshape(3)
        except Exception:
            return
        forward = look - pos
        norm = float(np.linalg.norm(forward))
        if norm <= 1.0e-6 or not np.isfinite(forward).all():
            return
        self._view_depth_clip_position = pos
        self._view_depth_clip_forward = (forward / norm).astype(np.float32)

    def _view_depth_clip_enabled(self) -> bool:
        return (
            self._view_depth_clip_min_m > 0.0
            and self._view_depth_clip_position is not None
            and self._view_depth_clip_forward is not None
        )

    def _view_depth_keep_mask(self, points: np.ndarray) -> np.ndarray:
        points_np = np.asarray(points, dtype=np.float32)
        if points_np.ndim != 2 or points_np.shape[1] != 3:
            return np.zeros((points_np.shape[0] if points_np.ndim > 0 else 0,), dtype=bool)
        if not self._view_depth_clip_enabled():
            return np.ones((points_np.shape[0],), dtype=bool)
        assert self._view_depth_clip_position is not None
        assert self._view_depth_clip_forward is not None
        depths = (points_np - self._view_depth_clip_position) @ self._view_depth_clip_forward
        return np.isfinite(depths) & (depths >= float(self._view_depth_clip_min_m))

    def _point_distance_keep_mask(self, points: np.ndarray) -> np.ndarray:
        points_np = np.asarray(points, dtype=np.float32)
        if points_np.ndim != 2 or points_np.shape[1] != 3:
            return np.zeros((points_np.shape[0] if points_np.ndim > 0 else 0,), dtype=bool)
        if self._point_min_distance_m <= 0.0 or self._point_distance_position is None:
            return np.ones((points_np.shape[0],), dtype=bool)
        distances = np.linalg.norm(points_np - self._point_distance_position.reshape(1, 3), axis=1)
        return np.isfinite(distances) & (distances >= float(self._point_min_distance_m))

    def _box_hidden_by_view_depth(self, center: np.ndarray, dimensions: np.ndarray) -> bool:
        if not self._view_depth_clip_enabled():
            return False
        assert self._view_depth_clip_position is not None
        assert self._view_depth_clip_forward is not None
        center_np = np.asarray(center, dtype=np.float32)
        dims_np = np.asarray(dimensions, dtype=np.float32)
        if center_np.shape != (3,) or dims_np.shape != (3,):
            return False
        if not np.isfinite(center_np).all() or not np.isfinite(dims_np).all():
            return False
        center_depth = float((center_np - self._view_depth_clip_position) @ self._view_depth_clip_forward)
        half_depth_extent = float(np.abs(self._view_depth_clip_forward) @ (0.5 * np.maximum(dims_np, 0.0)))
        return (center_depth - half_depth_extent) < float(self._view_depth_clip_min_m)

    def _box_hidden_by_distance(self, center: np.ndarray, dimensions: np.ndarray) -> bool:
        if self._object_box_min_distance_m <= 0.0 or self._object_box_distance_position is None:
            return False
        center_np = np.asarray(center, dtype=np.float32)
        dims_np = np.asarray(dimensions, dtype=np.float32)
        if center_np.shape != (3,) or dims_np.shape != (3,):
            return False
        if not np.isfinite(center_np).all() or not np.isfinite(dims_np).all():
            return False
        half = 0.5 * np.maximum(dims_np, 0.0)
        delta = np.abs(self._object_box_distance_position - center_np) - half
        outside = np.maximum(delta, 0.0)
        distance = float(np.linalg.norm(outside))
        return distance < float(self._object_box_min_distance_m)

    # ------------------------------------------------------------------
    # Point cloud integration
    # ------------------------------------------------------------------
    def _integrate_points(self, colors, depths, intrinsics, poses) -> None:
        new_pc = self._build_point_cloud(colors, depths, intrinsics, poses)
        if new_pc.points.size == 0:
            return

        if self._accum_points is None:
            points = new_pc.points
            colors_np = new_pc.colors
            if self._voxel_size > 0:
                keys = np.floor(points / self._voxel_size).astype(np.int64)
                self._accum_voxel_keys = {tuple(key.tolist()) for key in keys}
        else:
            if self._voxel_size > 0:
                if self._accum_voxel_keys is None:
                    keys_existing = np.floor(self._accum_points / self._voxel_size).astype(np.int64)
                    self._accum_voxel_keys = {tuple(key.tolist()) for key in keys_existing}
                keys_new = np.floor(new_pc.points / self._voxel_size).astype(np.int64)
                keep_indices: list[int] = []
                for idx, key in enumerate(keys_new):
                    key_tuple = tuple(int(v) for v in key)
                    if key_tuple in self._accum_voxel_keys:
                        continue
                    self._accum_voxel_keys.add(key_tuple)
                    keep_indices.append(idx)
                if keep_indices:
                    keep_np = np.asarray(keep_indices, dtype=np.int64)
                    points = np.concatenate([self._accum_points, new_pc.points[keep_np]], axis=0)
                    colors_np = np.concatenate([self._accum_colors, new_pc.colors[keep_np]], axis=0)
                else:
                    points = self._accum_points
                    colors_np = self._accum_colors
            else:
                points = np.concatenate([self._accum_points, new_pc.points], axis=0)
                colors_np = np.concatenate([self._accum_colors, new_pc.colors], axis=0)

        if self._voxel_size > 0 and self._accum_voxel_keys is None:
            points, colors_np = self._voxel_downsample(points, colors_np, self._voxel_size)

        self._accum_points = points
        self._accum_colors = colors_np
        display_points = points
        display_colors = colors_np
        keep_mask = self._view_depth_keep_mask(display_points) & self._point_distance_keep_mask(display_points)
        if keep_mask.shape[0] == display_points.shape[0] and not np.all(keep_mask):
            display_points = display_points[keep_mask]
            display_colors = display_colors[keep_mask]

        # In-place update for point cloud usually implies replacing the data
        if self._point_handle is not None:
            self._point_handle.remove()
            self._point_handle = None

        if display_points.size == 0:
            return

        self._point_handle = self._server.scene.add_point_cloud(
            name="/batch_points",
            points=display_points.astype(np.float32),
            colors=display_colors.astype(np.uint8),
            point_size=self._point_size,
            point_shape="circle",
        )

    def _build_point_cloud(self, colors, depths, intrinsics, poses) -> _PointCloud:
        all_points: list[np.ndarray] = []
        all_colors: list[np.ndarray] = []

        for color, depth, K, pose in zip(colors, depths, intrinsics, poses):
            color_np = _to_numpy(color)
            depth_np = _to_numpy(depth)
            K_np = _to_numpy(K)
            pose_np = _to_numpy(pose)
            if K_np.ndim == 2 and K_np.shape[0] >= 3 and K_np.shape[1] >= 3:
                K_np = K_np[:3, :3]
            if K_np.shape != (3, 3) or pose_np.shape != (4, 4):
                continue
            if not np.all(np.isfinite(K_np)) or not np.all(np.isfinite(pose_np)):
                continue

            if depth_np.ndim == 3:
                depth_np = depth_np.squeeze()
            if color_np.ndim == 3 and color_np.shape[0] in (1, 3):
                color_np = np.transpose(color_np, (1, 2, 0))
            if color_np.shape[-1] == 1:
                color_np = np.repeat(color_np, 3, axis=-1)

            if depth_np.ndim != 2 or color_np.shape[:2] != depth_np.shape:
                continue
            valid_mask = np.isfinite(depth_np) & (depth_np > 0) & (depth_np <= MAX_VIZUALIZATION_DEPTH)
            if not np.any(valid_mask):
                continue

            fx, fy = K_np[0, 0], K_np[1, 1]
            cx, cy = K_np[0, 2], K_np[1, 2]
            if not np.isfinite(fx) or not np.isfinite(fy) or abs(float(fx)) < 1.0e-6 or abs(float(fy)) < 1.0e-6:
                continue

            ys, xs = np.meshgrid(
                np.arange(depth_np.shape[0], dtype=np.float32),
                np.arange(depth_np.shape[1], dtype=np.float32),
                indexing="ij",
            )
            z = depth_np.astype(np.float32)
            x = (xs - cx) * z / fx
            y = (ys - cy) * z / fy
            points_cam = np.stack([x, y, z], axis=-1)

            mask_flat = valid_mask.reshape(-1)
            points_cam_flat = points_cam.reshape(-1, 3)[mask_flat]
            colors_flat = color_np.reshape(-1, 3)[mask_flat]
            points_world = (points_cam_flat @ pose_np[:3, :3].T + pose_np[:3, 3]).astype(np.float32)
            finite = np.isfinite(points_world).all(axis=1)
            if not np.any(finite):
                continue
            points_world = points_world[finite]
            colors_flat = colors_flat[finite]

            all_points.append(points_world)
            all_colors.append(self._normalize_colors(colors_flat))

        if len(all_points) == 0:
            return _PointCloud(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8))

        points = np.concatenate(all_points, axis=0)
        colors_np = np.concatenate(all_colors, axis=0)
        finite = np.isfinite(points).all(axis=1)
        if not np.any(finite):
            return _PointCloud(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8))
        points = points[finite]
        colors_np = colors_np[finite]

        if self._voxel_size > 0:
            voxel_keys = np.floor(points / self._voxel_size).astype(np.int64)
            _, unique_indices = np.unique(voxel_keys, axis=0, return_index=True)
            points = points[unique_indices]
            colors_np = colors_np[unique_indices]

        return _PointCloud(points=points, colors=colors_np)

    # ------------------------------------------------------------------
    # Live RGB side panel
    # ------------------------------------------------------------------
    def _update_live_rgb_panel(self, colors: Sequence[torch.Tensor | np.ndarray]) -> None:
        if not self._live_rgb_enabled or self._live_rgb_display is None:
            return
        if colors is None or len(colors) == 0:
            return

        now = time.monotonic()
        if self._live_rgb_max_fps > 0.0:
            min_period = 1.0 / self._live_rgb_max_fps
            if (now - self._last_live_rgb_update_s) < min_period:
                return

        image = self._prepare_live_rgb(colors[-1])
        if image is None:
            return
        self._last_live_rgb_update_s = now

        display = self._live_rgb_display
        with contextlib.suppress(Exception):
            if hasattr(display, "image"):
                display.image = image
            elif hasattr(display, "value"):
                display.value = image

        caption = self._live_rgb_caption
        if caption is not None:
            h, w = image.shape[:2]
            text = f"Latest RGB frame: {w}x{h}"
            if len(colors) > 1:
                text += f" (showing camera {len(colors)} of {len(colors)})"
            with contextlib.suppress(Exception):
                if hasattr(caption, "content"):
                    caption.content = text
                elif hasattr(caption, "value"):
                    caption.value = text

    def _prepare_live_rgb(self, color: torch.Tensor | np.ndarray | object) -> np.ndarray | None:
        try:
            arr = _to_numpy(color)
        except Exception:
            return None
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.ndim != 3 or arr.shape[-1] != 3:
            return None
        if np.issubdtype(arr.dtype, np.floating):
            finite = arr[np.isfinite(arr)]
            max_val = float(finite.max()) if finite.size else 0.0
            if max_val <= 1.5:
                arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return self._resize_longest_side(arr, self._live_rgb_max_side)

    # ------------------------------------------------------------------
    # Camera + Gaussians
    # ------------------------------------------------------------------
    def _update_camera_pose(self, pose: torch.Tensor | np.ndarray | None) -> None:
        if self._camera_frame is None or pose is None:
            return
        pose_np = _to_numpy(pose)
        if pose_np.shape != (4, 4):
            return

        rotation = pose_np[:3, :3]
        translation = pose_np[:3, 3]
        try:
            quat_wxyz = SO3.from_matrix(rotation).wxyz.tolist() if SO3 is not None else None
            if hasattr(self._camera_frame, "wxyz") and quat_wxyz is not None:
                self._camera_frame.wxyz = quat_wxyz
            if hasattr(self._camera_frame, "position"):
                self._camera_frame.position = translation
        except Exception:
            pass

    @staticmethod
    def _gui_get_value(handle: object | None) -> str:
        if handle is None:
            return ""
        for attr in ("value", "content", "text", "string"):
            if hasattr(handle, attr):
                with contextlib.suppress(Exception):
                    val = getattr(handle, attr)
                    return "" if val is None else str(val)
        return ""

    @staticmethod
    def _gui_add_text(gui_obj, label: str, initial: str = ""):
        for name in ("add_text", "add_text_input", "add_string"):
            fn = getattr(gui_obj, name, None)
            if fn is None:
                continue
            try:
                return fn(initial, label=label)
            except TypeError:
                with contextlib.suppress(Exception):
                    return fn(label=label, initial_value=initial)
            except Exception:
                continue
        return None

    @staticmethod
    def _gui_add_button(gui_obj, label: str):
        fn = getattr(gui_obj, "add_button", None)
        if fn is None:
            return None
        try:
            return fn(label)
        except TypeError:
            with contextlib.suppress(Exception):
                return fn(label=label)
        except Exception:
            return None
        return None

    @staticmethod
    def _gui_add_slider(gui_obj, label: str, *, min_v: float, max_v: float, step: float, initial: float):
        fn = getattr(gui_obj, "add_slider", None)
        if fn is None:
            return None
        try:
            return fn(label, min=min_v, max=max_v, step=step, initial_value=initial)
        except TypeError:
            with contextlib.suppress(Exception):
                return fn(label, min_v, max_v, step, initial)
        except Exception:
            return None
        return None

    @staticmethod
    def _gui_set_markdown(handle: object | None, text: str) -> None:
        if handle is None:
            return
        with contextlib.suppress(Exception):
            if hasattr(handle, "content"):
                handle.content = text
            elif hasattr(handle, "value"):
                handle.value = text

    # ------------------------------------------------------------------
    # Query (language-conditioned retrieval + fly-to)

    _QUERY_EXAMPLE_PLACEHOLDER = "(choose an example)"

    def _setup_query_gui(self) -> None:
        if self._server is None:
            return
        gui = getattr(self._server, "gui", None)
        if gui is None:
            return

        try:
            with gui.add_folder("Query"):
                self._retrieval_backend_button = self._gui_add_button(gui, "Start vLLM retrieval backend")
                self._retrieval_status = gui.add_markdown("Backend: not started.")
                with contextlib.suppress(Exception):
                    self._query_examples_dropdown = gui.add_dropdown(
                        "Examples",
                        options=(self._QUERY_EXAMPLE_PLACEHOLDER,),
                        initial_value=self._QUERY_EXAMPLE_PLACEHOLDER,
                    )
                self._query_input = self._gui_add_text(gui, "Query", initial="")
                self._query_search_button = self._gui_add_button(gui, "Search")
                self._reset_button = self._gui_add_button(gui, "Reset view")
                self._query_results_display = gui.add_markdown("")
        except Exception:
            return

        dropdown = self._query_examples_dropdown
        if dropdown is not None:
            on_update = getattr(dropdown, "on_update", None)
            if callable(on_update):
                with contextlib.suppress(Exception):

                    @on_update
                    def _(_event=None):
                        try:
                            choice = str(dropdown.value)
                        except Exception:
                            return
                        if not choice or choice == self._QUERY_EXAMPLE_PLACEHOLDER:
                            return
                        with contextlib.suppress(Exception):
                            if self._query_input is not None:
                                self._query_input.value = choice
                        threading.Thread(target=self._handle_query_clicked, daemon=True).start()

        start_btn = self._retrieval_backend_button
        if start_btn is not None:
            on_click = getattr(start_btn, "on_click", None)
            if callable(on_click):
                with contextlib.suppress(Exception):

                    @on_click
                    def _(_event=None):
                        threading.Thread(target=self._start_retrieval_backend, daemon=True).start()

        search_btn = self._query_search_button
        if search_btn is not None:
            on_click = getattr(search_btn, "on_click", None)
            if callable(on_click):
                with contextlib.suppress(Exception):

                    @on_click
                    def _(_event=None):
                        threading.Thread(target=self._handle_query_clicked, daemon=True).start()

        reset_btn = self._reset_button
        if reset_btn is not None:
            on_click = getattr(reset_btn, "on_click", None)
            if callable(on_click):
                with contextlib.suppress(Exception):

                    @on_click
                    def _(_event=None):
                        self._reset_view()

    _QUERY_EXAMPLE_SKIP_CATEGORIES = frozenset(
        {"person", "warehouse", "building", "wall", "floor", "ceiling", "room", "ground", "sky"}
    )
    _QUERY_EXAMPLE_COUNT = 20

    def _refresh_query_examples(self, scene_state: dict) -> None:
        """Fill the Examples dropdown with queries derived from the scene's own
        captioned objects, so every example is answerable by construction:
        targets are distinctive captioned categories, anchors are frequent ones,
        relational examples pair each target with its geometrically *nearest*
        anchor category (so "near"/"nearest to" have a true answer), and plain
        lookups fill the rest."""
        dropdown = self._query_examples_dropdown
        if dropdown is None or self._query_examples_ready:
            return
        if self._query_examples_override:
            with contextlib.suppress(Exception):
                dropdown.options = (self._QUERY_EXAMPLE_PLACEHOLDER, *self._query_examples_override)
                dropdown.value = self._QUERY_EXAMPLE_PLACEHOLDER
                self._query_examples_ready = True
            return
        categories = scene_state.get("object_category") or []
        captions = scene_state.get("object_caption") or []
        means = scene_state.get("means")
        active = scene_state.get("active")
        try:
            means_np = _to_numpy(means) if means is not None else None
            active_np = _to_numpy(active).astype(bool) if active is not None else None
        except Exception:
            means_np, active_np = None, None

        indices_by_category: dict[str, list[int]] = {}
        for i, cat in enumerate(categories):
            if not isinstance(cat, str):
                continue
            name = cat.strip().lower()
            if len(name) < 3 or not any(ch.isalpha() for ch in name):
                continue
            if name in self._QUERY_EXAMPLE_SKIP_CATEGORIES:
                continue
            if active_np is not None and i < active_np.shape[0] and not bool(active_np[i]):
                continue
            caption = captions[i] if i < len(captions) else None
            if not (isinstance(caption, str) and caption.strip()):
                continue
            if self._object_text_is_excluded(caption, name):
                continue
            indices_by_category.setdefault(name, []).append(i)
        if not indices_by_category:
            return  # e.g. live mapping before captions arrive — retry next update

        def _min_dist(target: str, anchor: str) -> float:
            if means_np is None:
                return float("inf")
            t = means_np[indices_by_category[target]]
            a = means_np[indices_by_category[anchor]]
            d = np.linalg.norm(t[:, None, :] - a[None, :, :], axis=-1)
            return float(d.min()) if d.size else float("inf")

        # Targets must be caption-aligned: the category word appears in at
        # least one of its own instances' captions, so the caption channel can
        # actually rank them (a "broom" whose caption says "brush" won't).
        caption_aligned = {
            c
            for c, idxs in indices_by_category.items()
            if any(
                i < len(captions) and isinstance(captions[i], str) and c in captions[i].lower()
                for i in idxs
            )
        }
        # Distinctive categories, most instances first (plenty of distractors,
        # still findable); the very common ones serve as anchors instead.
        targets = sorted(
            (c for c in caption_aligned if 2 <= len(indices_by_category[c]) <= 40),
            key=lambda c: (-len(indices_by_category[c]), c),
        )
        if not targets:
            targets = sorted(caption_aligned or indices_by_category, key=lambda c: (-len(indices_by_category[c]), c))
        anchors = sorted(
            (c for c, idxs in indices_by_category.items() if len(idxs) >= 3),
            key=lambda c: (-len(indices_by_category[c]), c),
        )[:10]

        def _nearest_anchor(t: str) -> tuple[str | None, float]:
            best, best_d = None, float("inf")
            for a in anchors:
                if a == t or a in t or t in a:  # skip near-synonym pairs ("box" / "cardboard box")
                    continue
                d = _min_dist(t, a)
                if d < best_d:
                    best, best_d = a, d
            return best, best_d

        near_examples: list[str] = []
        nearest_examples: list[str] = []
        closest_examples: list[str] = []
        plain_examples: list[str] = []
        for t in targets:
            anchor, d = _nearest_anchor(t)
            if anchor is not None and d < 6.0 and len(near_examples) < 7:
                near_examples.append(f"the {t} near the {anchor}")
            elif anchor is not None and d < 12.0 and len(nearest_examples) < 3:
                nearest_examples.append(f"the {t} nearest to the {anchor}")
            elif anchor is not None and len(closest_examples) < 2:
                closest_examples.append(f"the {t} closest to the {anchor}")
            else:
                plain_examples.append(f"a {t}")
        examples = near_examples + nearest_examples + closest_examples
        for plain in plain_examples:
            if len(examples) >= self._QUERY_EXAMPLE_COUNT:
                break
            examples.append(plain)
        for t in targets:  # top up with plain lookups if templates ran short
            if len(examples) >= self._QUERY_EXAMPLE_COUNT:
                break
            candidate = f"a {t}"
            if candidate not in examples:
                examples.append(candidate)
        examples = list(dict.fromkeys(examples))[: self._QUERY_EXAMPLE_COUNT]
        if not examples:
            return
        with contextlib.suppress(Exception):
            dropdown.options = (self._QUERY_EXAMPLE_PLACEHOLDER, *examples)
            dropdown.value = self._QUERY_EXAMPLE_PLACEHOLDER
            self._query_examples_ready = True

    @staticmethod
    def _server_reachable(base_url: str) -> bool:
        try:
            import requests  # lazy: viser must not hard-depend on requests
        except Exception:
            return False
        with contextlib.suppress(Exception):
            resp = requests.get(f"{base_url.rstrip('/')}/models", timeout=2.0)
            return resp.status_code == 200
        return False

    def _retrieval_server_specs(self) -> list[dict]:
        """vLLM servers the full relational pipeline uses (matches run.sh vllm).

        Each is env-gated + env-addressable so a remote/tunnelled deployment
        just points the URLs elsewhere and leaves the servers unlaunched.
        """
        def _enabled(name: str, default: bool) -> bool:
            return str(os.getenv(name, "1" if default else "0")).strip().lower() not in {"0", "false", "no", "off"}

        return [
            {
                "name": "llm (Qwen3.5-9B, query parsing)",
                "enabled": _enabled("VISER_RETRIEVAL_LLM_ENABLED", True),
                "base_url": self._retrieval_llm_base_url,
                "hf_ckpt": os.getenv("VLLM_LLM_HF_CKPT", "Qwen/Qwen3.5-9B"),
                "served": os.getenv("VLLM_MODEL", "qwen3.5-9b"),
                "gpu": os.getenv("VISER_RETRIEVAL_LLM_GPU", ""),
                "args": ["--max-model-len", "3084", "--gpu-memory-utilization", "0.75", "--dtype", "half",
                         "--max-num-seqs", "5", "--max-num-batched-tokens", "2048", "--enable-chunked-prefill",
                         "--reasoning-parser", "qwen3", "--disable-log-stats"],
            },
            {
                "name": "embed (Qwen3-Embedding-0.6B, caption channel)",
                "enabled": True,
                "base_url": self._retrieval_embed_base_url,
                "hf_ckpt": os.getenv("VLLM_EMBED_HF_CKPT", "Qwen/Qwen3-Embedding-0.6B"),
                "served": os.getenv("VLLM_EMBED_MODEL", "qwen3-emb-0.6b"),
                "gpu": os.getenv("VISER_RETRIEVAL_EMBED_GPU", ""),
                "args": ["--runner", "pooling", "--dtype", "half", "--max-model-len", "512",
                         "--gpu-memory-utilization", "0.2", "--max-num-seqs", "5",
                         "--max-num-batched-tokens", "2048", "--enforce-eager", "--disable-log-stats"],
            },
            {
                "name": "vl_embed (Qwen3-VL-Embedding-2B, qwen3_vl channel)",
                "enabled": _enabled("QWEN3_VL_EMBED_ENABLED", True),
                "base_url": os.getenv("VLLM_QWEN3_VL_EMBED_BASE_URL", "http://localhost:8006/v1").rstrip("/"),
                "hf_ckpt": os.getenv("VLLM_QWEN3_VL_EMBED_HF_CKPT", "Qwen/Qwen3-VL-Embedding-2B"),
                "served": os.getenv("VLLM_QWEN3_VL_EMBED_MODEL", "qwen3-vl-emb-2b"),
                "gpu": os.getenv("VISER_RETRIEVAL_VL_GPU", ""),
                "args": ["--runner", "pooling", "--dtype", "half", "--max-model-len", "600",
                         "--gpu-memory-utilization", "0.7", "--max-num-seqs", "5",
                         "--max-num-batched-tokens", "2048", "--enforce-eager", "--disable-log-stats"],
            },
        ]

    def _start_retrieval_backend(self) -> None:
        """Bring up (or attach to) the vLLM servers for the relational pipeline.

        For each enabled server we attach if it is already reachable, else launch
        ``vllm serve`` as a subprocess. Channels whose server never comes up are
        skipped by ``execute_spatial_query`` (graceful degradation).
        """
        specs = [s for s in self._retrieval_server_specs() if s["enabled"]]
        with self._retrieval_lock:
            for spec in specs:
                name = spec["name"]
                if self._server_reachable(spec["base_url"]):
                    continue
                proc = self._retrieval_procs.get(name)
                if proc is not None and proc.poll() is None:
                    continue  # already launching
                port = spec["base_url"].rsplit(":", 1)[-1].split("/")[0]
                cmd = ["vllm", "serve", spec["hf_ckpt"], "--host", "127.0.0.1", "--port", str(port),
                       "--served-model-name", spec["served"], *spec["args"]]
                env = dict(os.environ)
                if spec.get("gpu"):
                    env["CUDA_VISIBLE_DEVICES"] = str(spec["gpu"])
                log_path = f"/tmp/viser_vllm_{port}.log"
                try:
                    log_file = open(log_path, "ab")  # noqa: SIM115 - owned by the subprocess
                    self._retrieval_procs[name] = subprocess.Popen(
                        cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env
                    )
                    self._retrieval_proc_logs[name] = log_path
                except Exception as exc:
                    LOGGER.warning("Failed to launch %s: %s", name, exc)

        # Poll for readiness outside the lock so the UI stays responsive.
        for waited in range(600):
            statuses = []
            for spec in specs:
                ok = self._server_reachable(spec["base_url"])
                proc = self._retrieval_procs.get(spec["name"])
                if ok:
                    statuses.append((spec["name"], "ready"))
                elif proc is not None and proc.poll() is not None:
                    log_hint = self._retrieval_proc_logs.get(spec["name"])
                    statuses.append((spec["name"], f"exited — see `{log_hint}`" if log_hint else "exited"))
                else:
                    statuses.append((spec["name"], "loading"))
            self._retrieval_ready = self._server_reachable(self._retrieval_embed_base_url)
            lines = ["**Backend**" + (f" ({waited}s)" if any(s == "loading" for _, s in statuses) else "")]
            for nm, st in statuses:
                mark = "✅" if st == "ready" else ("⏳" if st == "loading" else "❌")
                lines.append(f"- {mark} {nm}: {st}")
            self._gui_set_markdown(self._retrieval_status, "\n".join(lines))
            if all(st != "loading" for _, st in statuses):
                return
            time.sleep(2.0)

    def _ensure_retrieval_interfaces(self) -> bool:
        """Lazily build the LLM + embedder interfaces used by the eval pipeline."""
        if self._retrieval_embedder is not None and self._retrieval_llm is not None:
            return True
        try:
            from scene_graph.llm_utils import EmbedInterface, LLMInterface
            if self._retrieval_embedder is None:
                self._retrieval_embedder = EmbedInterface(verbose=False)
            if self._retrieval_llm is None:
                self._retrieval_llm = LLMInterface(verbose=False, log_dir="/tmp/viser_llm_logs")
                with contextlib.suppress(Exception):
                    self._retrieval_llm.config.max_tokens = 512
            return True
        except Exception as exc:
            LOGGER.warning("Failed to build retrieval interfaces: %s", exc)
            return False

    def _run_relational_query(self, query: str):
        """Run the eval/paper relational pipeline: parse_query -> execute_spatial_query.

        Returns ``(results, method, query_graph)`` where results is a list of
        ``(object_id, composite_score, caption)`` ranked best-first.
        """
        from scene_graph.retrieval.spatial_reasoning import execute_spatial_query, parse_query
        from scene_graph.retrieval.spatial_reasoning.models import Predicate, QueryGraph

        state = self._latest_scene_state or {}
        llm = self._retrieval_llm
        embedder = self._retrieval_embedder

        # 1. Relational decomposition (needs the LLM; falls back to semantic-only).
        query_graph = None
        with contextlib.suppress(Exception):
            query_graph = parse_query(query, llm)

        # Retrieval engine: the joint vectorized engine by default (see
        # joint_executor). Set VISER_SPATIAL_METHOD=unified_soft_w50 to run
        # the paper's locked protocol instead.
        spatial_method = (os.getenv("VISER_SPATIAL_METHOD") or "joint_v1").strip() or "joint_v1"

        if query_graph is not None and query_graph.predicates:
            method = f"spatial (relational, {spatial_method})"
            scored = execute_spatial_query(
                query_graph, state, llm, embedder,
                use_vlm=False, pre_filter_k=-1, max_output_candidates=20, raw_query=query,
                retrieval_mode="multi", candidate_pool_mode="active",
                spatial_method=spatial_method, verbose=False,
            )
        else:
            fallback_method = spatial_method if spatial_method == "joint_v1" else "semantic_only"
            method = f"semantic-only ({fallback_method})"
            target_desc = query_graph.target_description if query_graph is not None else query
            scored = execute_spatial_query(
                QueryGraph(
                    target_description=target_desc,
                    predicates=[Predicate("IsCategory", ["$target", target_desc])],
                    reasoning="semantic-only (no predicates parsed)",
                ),
                state, llm, embedder,
                use_vlm=False, pre_filter_k=-1, max_output_candidates=20, raw_query=query,
                retrieval_mode="multi", candidate_pool_mode="active",
                spatial_method=fallback_method, verbose=False,
            )

        scored = sorted(scored, key=lambda c: float(c.composite_score), reverse=True)
        captions = state.get("object_caption") or []
        object_ids_np = None
        with contextlib.suppress(Exception):
            object_ids_np = _to_numpy(state.get("object_id")).astype(int, copy=False)

        results = []
        for cand in scored[:5]:
            cap = ""
            oi = int(cand.object_index)
            if 0 <= oi < len(captions) and isinstance(captions[oi], str):
                cap = captions[oi]
            results.append((int(cand.object_id), float(cand.composite_score), cap))

        # Focus set = target + confounders (every scored same-class candidate) +
        # anchors (objects the spatial predicates reference). Everything else is
        # faded so the target stands out.
        candidate_ids: set[int] = set()
        anchor_ids: set[int] = set()
        for cand in scored:
            candidate_ids.add(int(cand.object_id))
            # Per-candidate matched anchors (populated for regular predicates).
            for anchor_idx in (getattr(cand, "matched_anchors", {}) or {}).values():
                if object_ids_np is not None and 0 <= int(anchor_idx) < object_ids_np.shape[0]:
                    anchor_ids.add(int(object_ids_np[int(anchor_idx)]))
        # Superlative predicates (e.g. Closest) leave matched_anchors empty, so
        # also resolve anchor descriptions from the query graph directly.
        resolved_all, resolved_primary = self._anchor_object_ids(query_graph, object_ids_np)
        anchor_ids |= resolved_all
        focus_ids = candidate_ids | anchor_ids

        # Role assignment for the color coding (target > anchor > distractor)
        # and the relation edges from the top match to the anchors that
        # grounded it (matched anchors; for superlatives the primary resolved
        # anchor per description).
        target_id = int(scored[0].object_id) if scored else None
        edge_anchor_ids: set[int] = set()
        if scored:
            for anchor_idx in (getattr(scored[0], "matched_anchors", {}) or {}).values():
                if object_ids_np is not None and 0 <= int(anchor_idx) < object_ids_np.shape[0]:
                    edge_anchor_ids.add(int(object_ids_np[int(anchor_idx)]))
            if not edge_anchor_ids:
                # Fall back to the semantically-resolved primary anchors only
                # when some relational predicate actually scored for the top
                # match — a dropped/unsupported constraint must not draw edges.
                top_scored_ok = any(
                    getattr(r, "status", "") != "dropped" and float(getattr(r, "score", 0.0)) > 0.05
                    for r in (scored[0].predicate_results or [])
                    if getattr(r, "name", "") not in {"IsCategory", "HasAttribute", "InRegion"}
                )
                if top_scored_ok:
                    edge_anchor_ids |= resolved_primary
        target_set = {target_id} if target_id is not None else set()
        dropped = []
        if scored:
            for r in scored[0].predicate_results or []:
                if getattr(r, "status", "") == "dropped" and getattr(r, "drop_reason", None):
                    dropped.append(f"{r.name}: {str(r.drop_reason).replace('_', ' ')}")
        roles = {
            "target": target_id,
            "anchors": anchor_ids - target_set,
            "distractors": candidate_ids - anchor_ids - target_set,
            "edges": sorted(edge_anchor_ids - target_set),
            "dropped": dropped,
        }
        return results, method, query_graph, focus_ids, roles

    def _anchor_object_ids(self, query_graph, object_ids_np, top_k_per_anchor: int = 3) -> tuple[set[int], set[int]]:
        """Resolve anchor descriptions in *query_graph* to object ids.

        Returns ``(all_ids, primary_ids)`` — the top-k matches per anchor
        description, and just the best match per description (used for the
        relation edges)."""
        out: set[int] = set()
        primary: set[int] = set()
        if query_graph is None or object_ids_np is None or self._retrieval_embedder is None:
            return out, primary
        target_desc = (getattr(query_graph, "target_description", "") or "").strip().lower()
        anchor_descs: set[str] = set()
        for p in (getattr(query_graph, "predicates", []) or []):
            # Semantic predicates (HasAttribute 'blue', IsCategory, InRegion)
            # describe the target itself — their arguments are not anchors.
            if getattr(p, "name", "") in {"IsCategory", "HasAttribute", "InRegion"}:
                continue
            for arg in (getattr(p, "args", None) or getattr(p, "arguments", []) or []):
                if isinstance(arg, str) and arg and not arg.startswith("$"):
                    if arg.strip().lower() != target_desc:
                        anchor_descs.add(arg)
        if not anchor_descs:
            return out, primary
        try:
            from scene_graph.retrieval.spatial_reasoning.semantic_retrieval import retrieve_semantic_candidates
        except Exception:
            return out, primary
        state = self._latest_scene_state or {}
        for desc in anchor_descs:
            with contextlib.suppress(Exception):
                res = retrieve_semantic_candidates(desc, state, self._retrieval_embedder, mode="multi", k=top_k_per_anchor)
                fused = getattr(res, "fused_scores", {}) or {}
                ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k_per_anchor]
                for pos, (idx, _score) in enumerate(ranked):
                    if 0 <= int(idx) < object_ids_np.shape[0]:
                        oid = int(object_ids_np[int(idx)])
                        out.add(oid)
                        if pos == 0:
                            primary.add(oid)
        return out, primary

    def _handle_query_clicked(self) -> None:
        if self._server is None:
            return
        query = self._gui_get_value(self._query_input).strip()
        if not query:
            self._gui_set_markdown(self._query_results_display, "_Enter a query._")
            return
        if not self._server_reachable(self._retrieval_embed_base_url):
            self._gui_set_markdown(
                self._query_results_display,
                "_Retrieval backend not ready — click **Start vLLM retrieval backend** first._",
            )
            return
        if not self._ensure_retrieval_interfaces():
            self._gui_set_markdown(self._query_results_display, "_Failed to build retrieval interfaces._")
            return

        self._gui_set_markdown(self._query_results_display, f"_Searching for “{query}”…_")
        try:
            results, method, query_graph, focus_ids, roles = self._run_relational_query(query)
        except Exception as exc:
            LOGGER.warning("Relational query failed: %s", exc)
            self._gui_set_markdown(self._query_results_display, f"_Query failed: {exc}_")
            return
        if not results:
            self._gui_set_markdown(self._query_results_display, "_No match found._")
            return

        lines = [f"**“{query}”** — _{method}_"]
        if query_graph is not None:
            tgt = getattr(query_graph, "target_description", "") or ""
            preds = getattr(query_graph, "predicates", []) or []
            if tgt:
                lines.append(f"target: **{tgt}**")
            for p in preds[:6]:
                name = getattr(p, "name", getattr(p, "predicate", "?"))
                pargs = getattr(p, "args", getattr(p, "arguments", []))
                lines.append(f"· `{name}({', '.join(str(a) for a in pargs)})`")
        for note in (roles or {}).get("dropped", [])[:3]:
            lines.append(f"· ⚠ _{note} — constraint not applied_")
        lines.append("")
        lines.append("**Top matches:**")
        for rank, (obj_id, score, caption) in enumerate(results, start=1):
            cap = (caption or "(no caption)").strip()
            lines.append(f"{rank}. `#{obj_id}` ({score:.3f}) — {cap}")
        lines.append("")
        lines.append("_Others hidden — click **Reset view** to restore._")
        self._gui_set_markdown(self._query_results_display, "\n\n".join(lines))

        # Hide everything except target + distractors + anchors (color-coded
        # by role), then fly to #1 and draw its relation edges.
        self._query_roles = roles
        self._apply_focus(focus_ids)
        self._jump_to_object_id(results[0][0])

    def _apply_focus(self, focus_ids: set[int] | None) -> None:
        """Hide all objects except *focus_ids* (None restores everything)."""
        self._focus_object_ids = set(focus_ids) if focus_ids else None
        if not focus_ids:
            self._query_roles = None
        if self._server is None or self._latest_scene_state is None:
            return
        with contextlib.suppress(Exception):
            with self._server.atomic():
                self._update_gaussians(self._latest_scene_state)
            self._server.flush()

    def _reset_view(self) -> None:
        """Clear query focus + search overlays and restore the original view."""
        self._focus_object_ids = None
        # Remove the search highlight box/edges and path drawn for the last query.
        for attr in ("_search_highlight_box_handle", "_search_highlight_edges_handle", "_search_relations_handle", "_search_path_handle"):
            handle = getattr(self, attr, None)
            if handle is not None:
                with contextlib.suppress(Exception):
                    handle.remove()
                setattr(self, attr, None)
        if self._server is not None and self._latest_scene_state is not None:
            with contextlib.suppress(Exception):
                with self._server.atomic():
                    self._update_gaussians(self._latest_scene_state)
                self._server.flush()
        home = self._home_camera
        if home is not None and self._server is not None:
            clients: dict = {}
            with contextlib.suppress(Exception):
                clients = self._server.get_clients()
            for client in clients.values():
                self._try_set_camera(client, position=home[0], look_at=home[1])
        self._gui_set_markdown(self._query_results_display, "_View reset._")

    def _jump_to_object_id(self, target_id: int) -> None:
        """Highlight object *target_id* and fly the camera to it."""
        if self._server is None:
            return
        scene_state = self._latest_scene_state or {}
        poses = self._latest_poses or []

        object_center, view_position = self._get_object_center_and_view(scene_state, int(target_id))
        if object_center is None:
            LOGGER.info("Target object id %s not found in current scene state.", target_id)
            return

        current_robot = (
            self._robot_trajectory_positions[-1]
            if self._robot_trajectory_positions
            else self._get_robot_position(poses, scene_state)
        )
        if current_robot is None:
            current_robot = object_center

        with self._server.atomic():
            self._draw_search_highlight(object_center, target_id=int(target_id))
            self._draw_relation_edges(int(target_id))
            self._draw_search_path(current_robot, object_center, view_position)
        self._server.flush()

        self._start_search_camera_sequence(current_robot, object_center, close_camera_position=view_position)

    # ------------------------------------------------------------------
    # Filters

    def _setup_filter_gui(self) -> None:
        if self._server is None:
            return
        gui = getattr(self._server, "gui", None)
        if gui is None:
            return

        # Slider ceiling = "off" (show everything). Lowering it hides any object
        # whose largest AABB side exceeds the value — walls, floors, big furniture.
        ceiling = 10.0
        initial = self._object_box_max_side_m if self._object_box_max_side_m > 0.0 else ceiling
        initial = float(min(max(initial, 0.1), ceiling))
        try:
            with gui.add_folder("Filters"):
                self._max_side_slider = self._gui_add_slider(
                    gui, "Max box side (m)", min_v=0.1, max_v=ceiling, step=0.1, initial=initial
                )
        except Exception:
            return

        # Apply the initial value (unless it's the ceiling, which means "off").
        self._object_box_max_side_m = 0.0 if initial >= ceiling else initial

        slider = self._max_side_slider
        if slider is None:
            return
        on_update = getattr(slider, "on_update", None)
        if callable(on_update):
            with contextlib.suppress(Exception):

                @on_update
                def _(_event=None):
                    self._handle_max_side_changed()

    def _handle_max_side_changed(self) -> None:
        if self._max_side_slider is None:
            return
        try:
            value = float(getattr(self._max_side_slider, "value"))
        except Exception:
            return
        # Ceiling (>= 10 m) means "no filter".
        self._object_box_max_side_m = 0.0 if value >= 10.0 else max(0.0, value)
        if self._server is None or self._latest_scene_state is None:
            return
        with contextlib.suppress(Exception):
            with self._server.atomic():
                self._update_gaussians(self._latest_scene_state)
            self._server.flush()

    def _setup_edit_gui(self) -> None:
        """Set up interactive editing GUI in Viser."""
        if self._server is None:
            return
        if not any([self._on_edit_caption, self._on_delete_object, self._on_save_all, self._on_add_object]):
            return
        gui = getattr(self._server, "gui", None)
        if gui is None:
            return

        def _add_text(gui_obj, label: str, initial: str = ""):
            for name in ("add_text", "add_text_input", "add_string"):
                fn = getattr(gui_obj, name, None)
                if fn is None:
                    continue
                try:
                    return fn(initial, label=label)
                except TypeError:
                    with contextlib.suppress(Exception):
                        return fn(label=label, initial_value=initial)
                except Exception:
                    continue
            return None

        def _add_button(gui_obj, label: str):
            for name in ("add_button",):
                fn = getattr(gui_obj, name, None)
                if fn is None:
                    continue
                try:
                    return fn(label)
                except TypeError:
                    with contextlib.suppress(Exception):
                        return fn(label=label)
                except Exception:
                    continue
            return None

        try:
            with gui.add_folder("Interactive Edit"):
                self._edit_caption_input = _add_text(gui, "New caption", initial="")
                self._edit_apply_button = _add_button(gui, "Apply caption")
                self._delete_button = _add_button(gui, "Delete object")
                self._lock_toggle_button = _add_button(gui, "Lock/Unlock object")
                self._save_all_button = _add_button(gui, "Save all")
                self._add_caption_input = _add_text(gui, "Object caption", initial="")
                self._add_location_input = _add_text(gui, "Object location [x,y,z]", initial="")
                self._add_image_path_input = _add_text(gui, "Object image path (optional)", initial="")
                self._add_view1_input = _add_text(gui, "View 1 [x,y,z,wx,wy,wz,w]", initial="")
                self._add_view2_input = _add_text(gui, "View 2 [x,y,z,wx,wy,wz,w]", initial="")
                self._add_view3_input = _add_text(gui, "View 3 [x,y,z,wx,wy,wz,w]", initial="")
                self._add_object_button = _add_button(gui, "Add object")
                self._edit_status = gui.add_markdown("Ready.")
        except Exception:
            return

        # Wire up buttons
        for button, callback in [
            (self._edit_apply_button, self._handle_edit_caption_clicked),
            (self._delete_button, self._handle_delete_clicked),
            (self._lock_toggle_button, self._handle_lock_toggle_clicked),
            (self._save_all_button, self._handle_save_all_clicked),
            (self._add_object_button, self._handle_add_object_clicked),
        ]:
            if button is not None:
                on_click = getattr(button, "on_click", None)
                if callable(on_click):
                    with contextlib.suppress(Exception):

                        @on_click
                        def _(_event=None, callback_func=callback):
                            callback_func()

    def _set_edit_status(self, msg: str) -> None:
        """Update the edit status display."""
        if self._edit_status is not None:
            with contextlib.suppress(Exception):
                self._edit_status.content = msg

    @staticmethod
    def _parse_pose7_input(raw: str) -> tuple[list[float] | None, str]:
        text = str(raw or "").strip()
        if not text:
            return None, "empty input"
        normalized = text
        for token in "[]()":
            normalized = normalized.replace(token, " ")
        normalized = normalized.replace(",", " ")
        parts = [piece for piece in normalized.split() if piece]
        if len(parts) != 7:
            return None, f"expected 7 numbers, got {len(parts)}"
        try:
            values = [float(piece) for piece in parts]
        except Exception:
            return None, "contains non-numeric values"
        arr = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            return None, "contains NaN or Inf"
        quat = arr[3:7]
        q_norm = float(np.linalg.norm(quat))
        if q_norm <= 1e-12:
            return None, "quaternion norm is zero"
        arr[3:7] = quat / q_norm
        return arr.astype(np.float32).tolist(), ""

    @staticmethod
    def _parse_xyz_input(raw: str) -> tuple[list[float] | None, str]:
        text = str(raw or "").strip()
        if not text:
            return None, "empty input"
        normalized = text
        for token in "[]()":
            normalized = normalized.replace(token, " ")
        normalized = normalized.replace(",", " ")
        parts = [piece for piece in normalized.split() if piece]
        if len(parts) != 3:
            return None, f"expected 3 numbers, got {len(parts)}"
        try:
            values = [float(piece) for piece in parts]
        except Exception:
            return None, "contains non-numeric values"
        arr = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            return None, "contains NaN or Inf"
        return arr.astype(np.float32).tolist(), ""

    def _handle_edit_caption_clicked(self) -> None:
        """Handle caption edit button click."""
        if self._selected_object_id is None:
            self._set_edit_status("❌ No object selected.")
            return
        new_caption = self._gui_get_value(self._edit_caption_input).strip()
        if not new_caption:
            self._set_edit_status("❌ Caption is empty.")
            return
        if self._on_edit_caption is None:
            self._set_edit_status("❌ Edit callback not configured.")
            return
        try:
            ok, msg = self._on_edit_caption(self._selected_object_id, new_caption)
            if ok:
                self._set_edit_status(f"✓ Caption updated for object {self._selected_object_id}.")
                self._edit_caption_input.value = ""
            else:
                self._set_edit_status(f"❌ {msg}")
        except Exception as exc:
            self._set_edit_status(f"❌ Error: {exc}")

    def _handle_delete_clicked(self) -> None:
        """Handle delete object button click."""
        if self._selected_object_id is None:
            self._set_edit_status("❌ No object selected.")
            return
        if self._on_delete_object is None:
            self._set_edit_status("❌ Delete callback not configured.")
            return
        try:
            ok, msg = self._on_delete_object(self._selected_object_id)
            if ok:
                self._set_edit_status(f"✓ Object {self._selected_object_id} deleted.")
                self._selected_object_id = None
            else:
                self._set_edit_status(f"❌ {msg}")
        except Exception as exc:
            self._set_edit_status(f"❌ Error: {exc}")

    def _handle_lock_toggle_clicked(self) -> None:
        """Handle lock/unlock object button click."""
        if self._selected_object_id is None:
            self._set_edit_status("❌ No object selected.")
            return
        if self._on_toggle_lock is None:
            self._set_edit_status("❌ Lock toggle callback not configured.")
            return
        try:
            ok, msg = self._on_toggle_lock(self._selected_object_id)
            if ok:
                self._set_edit_status(f"✓ {msg}")
            else:
                self._set_edit_status(f"❌ {msg}")
        except Exception as exc:
            self._set_edit_status(f"❌ Error: {exc}")

    def _handle_save_all_clicked(self) -> None:
        """Handle save all button click."""
        if self._on_save_all is None:
            self._set_edit_status("❌ Save callback not configured.")
            return
        try:
            ok, msg = self._on_save_all()
            if ok:
                self._set_edit_status("✓ Saved scene state, JSON, and snapshots.")
            else:
                self._set_edit_status(f"❌ {msg}")
        except Exception as exc:
            self._set_edit_status(f"❌ Error: {exc}")

    def _handle_add_object_clicked(self) -> None:
        """Handle add object button click."""
        if self._on_add_object is None:
            self._set_edit_status("❌ Add callback not configured.")
            return

        caption = self._gui_get_value(self._add_caption_input).strip()
        if not caption:
            self._set_edit_status("❌ Object caption is empty.")
            return

        location, location_err = self._parse_xyz_input(self._gui_get_value(self._add_location_input))
        if location is None:
            self._set_edit_status(f"❌ Object location: {location_err}.")
            return
        image_path = self._gui_get_value(self._add_image_path_input).strip()

        raw_views = [
            self._gui_get_value(self._add_view1_input),
            self._gui_get_value(self._add_view2_input),
            self._gui_get_value(self._add_view3_input),
        ]
        views: list[list[float]] = []
        for idx, raw_view in enumerate(raw_views, start=1):
            pose7, err = self._parse_pose7_input(raw_view)
            if pose7 is None:
                self._set_edit_status(f"❌ View {idx}: {err}.")
                return
            views.append(pose7)

        try:
            try:
                ok, msg = self._on_add_object(caption, location, views, image_path)
            except TypeError:
                try:
                    ok, msg = self._on_add_object(caption, location, views)
                except TypeError:
                    ok, msg = self._on_add_object(caption, views)
            if ok:
                self._set_edit_status(msg if msg else "✓ Object added.")
                if self._add_caption_input is not None:
                    with contextlib.suppress(Exception):
                        self._add_caption_input.value = ""
                if self._add_location_input is not None:
                    with contextlib.suppress(Exception):
                        self._add_location_input.value = ""
                if self._add_image_path_input is not None:
                    with contextlib.suppress(Exception):
                        self._add_image_path_input.value = ""
                for handle in (self._add_view1_input, self._add_view2_input, self._add_view3_input):
                    if handle is not None:
                        with contextlib.suppress(Exception):
                            handle.value = ""
            else:
                self._set_edit_status(f"❌ {msg}")
        except Exception as exc:
            self._set_edit_status(f"❌ Error: {exc}")

    def _get_object_center_and_view(
        self, scene_state: dict, target_object_id: int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        means = scene_state.get("means")
        object_ids = scene_state.get("object_id")
        if means is None or object_ids is None:
            return None, None

        means_np = _to_numpy(means)
        ids_np = _to_numpy(object_ids).astype(int, copy=False)
        if means_np.ndim != 2 or means_np.shape[0] == 0 or means_np.shape[-1] != 3:
            return None, None
        if ids_np.ndim != 1 or ids_np.shape[0] != means_np.shape[0]:
            return None, None

        match = np.nonzero(ids_np == int(target_object_id))[0]
        if match.size == 0:
            return None, None

        active = scene_state.get("active")
        if active is not None:
            with contextlib.suppress(Exception):
                active_mask = _to_numpy(active).astype(bool, copy=False)
                if active_mask.shape[0] == ids_np.shape[0]:
                    active_match = match[active_mask[match]]
                    if active_match.size:
                        match = active_match

        obj_idx = int(match[0])
        center = np.asarray(means_np[obj_idx], dtype=np.float32)
        if not np.all(np.isfinite(center)):
            return None, None

        view_position = None
        object_image_ids = scene_state.get("viewpoint_image_ids") or scene_state.get("object_image_ids")
        images = scene_state.get("images") or []

        image_ids: list[int] = []
        if isinstance(object_image_ids, (list, tuple)) and obj_idx < len(object_image_ids):
            entry = object_image_ids[obj_idx]
            if isinstance(entry, (list, tuple)):
                image_ids = [int(x) for x in entry if isinstance(x, int) and x >= 0]
            elif isinstance(entry, int) and entry >= 0:
                image_ids = [int(entry)]

        if image_ids and images:
            pose_by_id: dict[int, np.ndarray] = {}
            for record in images:
                rec_id = getattr(record, "image_id", None)
                pose = getattr(record, "pose", None)
                if rec_id is None or pose is None:
                    continue
                with contextlib.suppress(Exception):
                    pose_np = _to_numpy(pose)
                    if isinstance(pose_np, np.ndarray) and pose_np.shape == (4, 4):
                        pose_by_id[int(rec_id)] = pose_np

            candidates: list[np.ndarray] = []
            for img_id in image_ids:
                pose_np = pose_by_id.get(int(img_id))
                if pose_np is None:
                    continue
                candidate = pose_np[:3, 3].astype(np.float32)
                if np.all(np.isfinite(candidate)):
                    candidates.append(candidate)

            if candidates:
                candidate_positions = np.stack(candidates, axis=0).astype(np.float32, copy=False)
                d2 = np.sum((candidate_positions - center[None, :]) ** 2, axis=1)
                order = np.argsort(d2)
                median_idx = int(order[len(order) // 2])
                view_position = candidate_positions[median_idx]

        return center, view_position

    def _draw_search_highlight(self, center: np.ndarray, target_id: int | None = None) -> None:
        """Highlight the search target by restyling its own box — bright green
        and near-opaque. No overlay geometry, so the target's box stays fully
        visible and clickable; the next scene refresh (new query, Reset view,
        filter change) restores the normal style."""
        if self._server is None:
            return
        # Clear overlays from any earlier highlight style.
        if self._search_highlight_box_handle is not None:
            with contextlib.suppress(Exception):
                self._search_highlight_box_handle.remove()
            self._search_highlight_box_handle = None
        if self._search_highlight_edges_handle is not None:
            with contextlib.suppress(Exception):
                self._search_highlight_edges_handle.remove()
            self._search_highlight_edges_handle = None

        if target_id is None:
            return
        handle = self._object_cube_handles.get(int(target_id))
        if handle is None:
            return
        with contextlib.suppress(Exception):
            handle.color = np.array([255, 212, 90], dtype=np.uint8)  # target #FFD45A
        with contextlib.suppress(Exception):
            handle.opacity = 0.95

    def _draw_relation_edges(self, target_id: int) -> None:
        """Draw the grounding relations as #FDB515 lines from the top match to
        the anchor objects its predicates matched against."""
        if self._search_relations_handle is not None:
            with contextlib.suppress(Exception):
                self._search_relations_handle.remove()
            self._search_relations_handle = None
        if self._server is None:
            return
        roles = self._query_roles or {}
        edge_ids = [int(a) for a in (roles.get("edges") or [])]
        target_handle = self._object_cube_handles.get(int(target_id))
        if not edge_ids or target_handle is None:
            return
        try:
            t_ctr = np.asarray(target_handle.position, dtype=np.float32).reshape(3)
        except Exception:
            return
        segments = []
        for aid in edge_ids:
            anchor_handle = self._object_cube_handles.get(aid)
            if anchor_handle is None:
                continue
            with contextlib.suppress(Exception):
                a_ctr = np.asarray(anchor_handle.position, dtype=np.float32).reshape(3)
                if np.isfinite(a_ctr).all():
                    segments.append(np.stack([t_ctr, a_ctr], axis=0))
        if not segments:
            return
        seg = np.stack(segments, axis=0)  # (K, 2, 3)
        colors = np.tile(np.array([253, 181, 21], dtype=np.uint8).reshape(1, 1, 3), (seg.shape[0], 2, 1))
        with contextlib.suppress(Exception):
            self._search_relations_handle = self._server.scene.add_line_segments(
                "/search/relations", points=seg, colors=colors, line_width=6.0
            )

    def _draw_search_path(
        self, current_robot: np.ndarray, object_center: np.ndarray, view_position: np.ndarray | None
    ) -> None:
        if self._server is None:
            return
        green = np.array([0, 255, 0], dtype=np.uint8)

        if self._search_path_handle is not None:
            with contextlib.suppress(Exception):
                self._search_path_handle.remove()
            self._search_path_handle = None

        current_robot = np.asarray(current_robot, dtype=np.float32)
        object_center = np.asarray(object_center, dtype=np.float32)
        if current_robot.shape != (3,) or object_center.shape != (3,):
            return
        if not np.all(np.isfinite(current_robot)) or not np.all(np.isfinite(object_center)):
            return

        points: np.ndarray | None = None
        if self._robot_trajectory_positions and view_position is not None:
            view_position = np.asarray(view_position, dtype=np.float32)
            traj = np.stack(self._robot_trajectory_positions, axis=0).astype(np.float32, copy=False)
            if traj.ndim == 2 and traj.shape[0] >= 2 and view_position.shape == (3,):
                d2 = np.sum((traj - view_position[None, :]) ** 2, axis=1)
                start_idx = int(np.argmin(d2))
                hist = traj[start_idx:]
                if hist.shape[0] >= 2:
                    points = np.concatenate([hist[::-1], object_center[None, :]], axis=0)

        if points is None:
            points = np.stack([current_robot, object_center], axis=0).astype(np.float32)

        if points.shape[0] < 2:
            return

        segments = np.stack([points[:-1], points[1:]], axis=1).astype(np.float32)
        colors = np.tile(green, (segments.shape[0], 2, 1)).astype(np.uint8)
        with contextlib.suppress(Exception):
            self._search_path_handle = self._server.scene.add_line_segments(
                name="/search/path",
                points=segments,
                colors=colors,
                line_width=4.0,
            )

    def _start_search_camera_sequence(
        self,
        current_robot: np.ndarray,
        object_center: np.ndarray,
        *,
        close_camera_position: np.ndarray | None,
    ) -> None:
        client = self._latest_client
        if client is None:
            return

        if self._search_animation_cancel_event is not None:
            self._search_animation_cancel_event.set()
        cancel_event = threading.Event()
        self._search_animation_cancel_event = cancel_event

        current_robot = np.asarray(current_robot, dtype=np.float32)
        object_center = np.asarray(object_center, dtype=np.float32)
        if current_robot.shape != (3,) or object_center.shape != (3,):
            return

        high_pos, high_look = self._compute_high_angle_view(current_robot, object_center)
        close_pos, close_look = self._compute_close_view(object_center, close_camera_position)
        if high_pos is None or high_look is None or close_pos is None or close_look is None:
            return

        def _run() -> None:
            start_pos, start_look = self._try_get_camera_state(client)
            # If we can't read the current camera, fall back to a snap to high-angle first.
            if start_pos is None or start_look is None:
                self._try_set_camera(client, position=high_pos, look_at=high_look)
            else:
                self._animate_camera(
                    client,
                    start_pos=start_pos,
                    start_look=start_look,
                    end_pos=high_pos,
                    end_look=high_look,
                    duration_s=2.0,
                    cancel_event=cancel_event,
                )

            if cancel_event.is_set():
                return
            # Wait between the two transitions.
            t0 = time.time()
            while not cancel_event.is_set() and (time.time() - t0) < 10.0:
                time.sleep(0.05)
            if cancel_event.is_set():
                return

            self._animate_camera(
                client,
                start_pos=high_pos,
                start_look=high_look,
                end_pos=close_pos,
                end_look=close_look,
                duration_s=2.0,
                cancel_event=cancel_event,
            )

        threading.Thread(target=_run, daemon=True).start()

    @staticmethod
    def _compute_high_angle_view(
        current_robot: np.ndarray, object_center: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        try:
            current_robot = np.asarray(current_robot, dtype=np.float32)
            object_center = np.asarray(object_center, dtype=np.float32)
        except Exception:
            return None, None
        if current_robot.shape != (3,) or object_center.shape != (3,):
            return None, None
        focus = 0.5 * (current_robot + object_center)
        dist = float(np.linalg.norm(current_robot - object_center))
        dist = max(2.0, 1.5 * dist)
        position = focus + np.array([0.0, -0.8 * dist, 1.2 * dist], dtype=np.float32)
        return position, focus

    @staticmethod
    def _compute_close_view(
        object_center: np.ndarray, camera_position: np.ndarray | None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        try:
            focus = np.asarray(object_center, dtype=np.float32)
        except Exception:
            return None, None
        if focus.shape != (3,) or not np.all(np.isfinite(focus)):
            return None, None
        if camera_position is not None:
            with contextlib.suppress(Exception):
                cam_pos = np.asarray(camera_position, dtype=np.float32)
                if cam_pos.shape == (3,) and np.all(np.isfinite(cam_pos)):
                    return cam_pos, focus
        position = focus + np.array([0.0, -0.8, 0.5], dtype=np.float32)
        return position, focus

    @staticmethod
    def _smoothstep(t: float) -> float:
        t = max(0.0, min(1.0, float(t)))
        return t * t * (3.0 - 2.0 * t)

    def _animate_camera(
        self,
        client: object,
        *,
        start_pos: np.ndarray,
        start_look: np.ndarray,
        end_pos: np.ndarray,
        end_look: np.ndarray,
        duration_s: float,
        cancel_event: threading.Event,
        fps: float = 30.0,
    ) -> None:
        duration_s = max(0.01, float(duration_s))
        steps = max(2, int(math.ceil(duration_s * float(fps))))
        dt = duration_s / float(steps - 1)

        start_pos = np.asarray(start_pos, dtype=np.float32)
        start_look = np.asarray(start_look, dtype=np.float32)
        end_pos = np.asarray(end_pos, dtype=np.float32)
        end_look = np.asarray(end_look, dtype=np.float32)

        if start_pos.shape != (3,) or start_look.shape != (3,) or end_pos.shape != (3,) or end_look.shape != (3,):
            return

        for i in range(steps):
            if cancel_event.is_set():
                return
            alpha = self._smoothstep(i / float(steps - 1))
            pos = (1.0 - alpha) * start_pos + alpha * end_pos
            look = (1.0 - alpha) * start_look + alpha * end_look
            self._try_set_camera(client, position=pos, look_at=look)
            time.sleep(dt)

    @staticmethod
    def _try_get_camera_state(client: object) -> tuple[np.ndarray | None, np.ndarray | None]:
        cam = getattr(client, "camera", None)
        if cam is None:
            return None, None

        pos = None
        for attr in ("position", "xyz"):
            if hasattr(cam, attr):
                with contextlib.suppress(Exception):
                    cand = np.asarray(getattr(cam, attr), dtype=np.float32)
                    if cand.shape == (3,) and np.all(np.isfinite(cand)):
                        pos = cand
                        break

        look = None
        for attr in ("look_at", "target"):
            if hasattr(cam, attr):
                with contextlib.suppress(Exception):
                    cand = getattr(cam, attr)
                    if callable(cand):
                        continue
                    cand_np = np.asarray(cand, dtype=np.float32)
                    if cand_np.shape == (3,) and np.all(np.isfinite(cand_np)):
                        look = cand_np
                        break

        return pos, look

    @staticmethod
    def _try_set_camera(client: object, *, position: np.ndarray, look_at: np.ndarray) -> None:
        cam = getattr(client, "camera", None)
        if cam is None:
            return
        # Best-effort across viser versions.
        with contextlib.suppress(Exception):
            if hasattr(cam, "position"):
                cam.position = position
            elif hasattr(cam, "xyz"):
                cam.xyz = position
        with contextlib.suppress(Exception):
            if hasattr(cam, "look_at"):
                la = getattr(cam, "look_at")
                if callable(la):
                    la(look_at)
                else:
                    cam.look_at = look_at
            elif hasattr(cam, "target"):
                cam.target = look_at

    def _get_robot_position(
        self,
        poses: Sequence[torch.Tensor | np.ndarray],
        scene_state: dict,
    ) -> np.ndarray | None:
        robot_pose = (
            scene_state.get("robot_pose")
            or scene_state.get("T_world_robot")
            or scene_state.get("T_world_base")
            or scene_state.get("base_pose")
        )
        if robot_pose is not None:
            try:
                robot_pose_np = _to_numpy(robot_pose)
            except Exception:
                robot_pose_np = None
            if isinstance(robot_pose_np, np.ndarray):
                if robot_pose_np.shape == (4, 4):
                    position = robot_pose_np[:3, 3].astype(np.float32)
                    if np.all(np.isfinite(position)):
                        return position
                if robot_pose_np.shape == (3,):
                    position = robot_pose_np.astype(np.float32)
                    if np.all(np.isfinite(position)):
                        return position

        if not poses:
            return None

        positions: list[np.ndarray] = []
        for pose in poses:
            try:
                pose_np = _to_numpy(pose)
            except Exception:
                continue
            if pose_np.shape != (4, 4):
                continue
            position = pose_np[:3, 3].astype(np.float32)
            if not np.all(np.isfinite(position)):
                continue
            positions.append(position)

        if not positions:
            return None
        return np.mean(np.stack(positions, axis=0), axis=0).astype(np.float32, copy=False)

    def _update_robot_trajectory(self, poses: Sequence[torch.Tensor | np.ndarray], scene_state: dict) -> None:
        if self._server is None:
            return

        position = self._get_robot_position(poses, scene_state)
        if position is None:
            return

        if self._robot_trajectory_positions:
            delta = position - self._robot_trajectory_positions[-1]
            if float(np.linalg.norm(delta)) < self._robot_trajectory_min_step_m:
                return
        self._robot_trajectory_positions.append(position)

        if len(self._robot_trajectory_positions) > self._robot_trajectory_max_points:
            overflow = len(self._robot_trajectory_positions) - self._robot_trajectory_max_points
            if overflow > 0:
                self._robot_trajectory_positions = self._robot_trajectory_positions[overflow:]

        if len(self._robot_trajectory_positions) < 2:
            return

        positions = np.stack(self._robot_trajectory_positions, axis=0).astype(np.float32, copy=False)
        segments = np.stack([positions[:-1], positions[1:]], axis=1)
        colors = np.tile(np.array([255, 0, 0], dtype=np.uint8), (segments.shape[0], 2, 1))

        if self._robot_trajectory_handle is not None:
            with contextlib.suppress(Exception):
                self._robot_trajectory_handle.remove()

        self._robot_trajectory_handle = self._server.scene.add_line_segments(
            name="/robot_trajectory",
            points=segments,
            colors=colors,
            line_width=4.0,
        )

    @staticmethod
    def _row_value(rows: object, idx: int, default: object = None) -> object:
        if isinstance(rows, (list, tuple)) and 0 <= idx < len(rows):
            return rows[idx]
        return default

    @staticmethod
    def _string_value(value: object) -> str:
        return str(value or "").strip()

    @staticmethod
    def _string_list_value(value: object) -> list[str]:
        if not isinstance(value, (list, tuple)):
            return []
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out

    @staticmethod
    def _normalize_exclude_terms(value: str | Sequence[str] | None) -> tuple[str, ...]:
        if value is None:
            return ()
        raw_items: list[str] = []
        if isinstance(value, str):
            raw_items = value.replace(";", ",").split(",")
        elif isinstance(value, (list, tuple)):
            raw_items = [str(item) for item in value]
        terms: list[str] = []
        for item in raw_items:
            term = str(item or "").strip().lower()
            if term:
                terms.append(term)
        return tuple(dict.fromkeys(terms))

    def _object_text_is_excluded(self, *texts: object) -> bool:
        if not self._object_box_exclude_terms:
            return False
        haystack = " ".join(str(text or "").lower() for text in texts if text is not None)
        if not haystack:
            return False
        return any(
            re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", haystack) is not None
            for term in self._object_box_exclude_terms
        )

    def _format_caption_json_markdown(
        self,
        *,
        description: str,
        category: str,
        supercategory: str,
        attributes: list[str],
        decision: str,
    ) -> str:
        payload = {
            "category": category,
            "supercategory": supercategory,
            "attributes": attributes,
            "description": description,
            "decision": decision,
        }
        lines = [
            "**Caption JSON**",
            "",
            "```json",
            json.dumps(payload, indent=2, ensure_ascii=False),
            "```",
        ]
        return "\n".join(lines)

    def _object_voxel_arrays(self, scene_state: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        flat_t = scene_state.get("object_voxel_keys_flat")
        offsets_t = scene_state.get("object_voxel_keys_offsets")
        levels_t = scene_state.get("object_voxel_levels")
        if flat_t is None or offsets_t is None or levels_t is None:
            return None
        flat = _to_numpy(flat_t).astype(np.int64, copy=False)
        offsets = _to_numpy(offsets_t).astype(np.int64, copy=False)
        levels = _to_numpy(levels_t).astype(np.int64, copy=False)
        if flat.ndim != 1 or offsets.ndim != 1 or levels.ndim != 1:
            return None
        return flat, offsets, levels

    def _cached_object_voxel_aabb(
        self,
        flat: np.ndarray,
        *,
        obj_idx: int,
        start: int,
        end: int,
        level: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if end <= start:
            return None
        first_key = int(flat[start]) if start < end else 0
        last_key = int(flat[end - 1]) if start < end else 0
        cache_key = (int(obj_idx), int(start), int(end), int(level), first_key, last_key)
        if cache_key not in self._object_voxel_aabb_cache:
            self._object_voxel_aabb_cache[cache_key] = _voxel_cloud_aabb(flat[start:end], int(level))
        return self._object_voxel_aabb_cache[cache_key]

    def _object_box_geometry_from_voxels(
        self,
        scene_state: dict,
        *,
        active_indices_all: np.ndarray,
        fallback_centers: np.ndarray,
        fallback_dimensions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._object_box_from_voxels:
            return fallback_centers, fallback_dimensions
        arrays = self._object_voxel_arrays(scene_state)
        if arrays is None:
            return fallback_centers, fallback_dimensions
        flat, offsets, levels = arrays
        centers = np.asarray(fallback_centers, dtype=np.float32).copy()
        dimensions = np.asarray(fallback_dimensions, dtype=np.float32).copy()
        count = min(len(active_indices_all), centers.shape[0], dimensions.shape[0])
        for idx in range(count):
            obj_idx = int(active_indices_all[idx])
            if obj_idx + 1 >= offsets.shape[0] or obj_idx >= levels.shape[0]:
                continue
            start = int(offsets[obj_idx])
            end = int(offsets[obj_idx + 1])
            if start < 0 or end > flat.shape[0] or end <= start:
                continue
            box = self._cached_object_voxel_aabb(
                flat,
                obj_idx=obj_idx,
                start=start,
                end=end,
                level=int(levels[obj_idx]),
            )
            if box is None:
                continue
            mn, mx = box
            mn = np.asarray(mn, dtype=np.float32)
            mx = np.asarray(mx, dtype=np.float32)
            dims = np.maximum(mx - mn, 0.08)
            if np.isfinite(mn).all() and np.isfinite(mx).all() and np.isfinite(dims).all():
                centers[idx] = 0.5 * (mn + mx)
                dimensions[idx] = dims
        return centers, dimensions

    def _update_gaussians(self, scene_state: dict) -> None:
        if self._server is None:
            return
        active = scene_state.get("active")
        means = scene_state.get("means")
        cov6 = scene_state.get("cov6")
        object_ids = scene_state.get("object_id")
        object_captions = scene_state.get("object_caption") or []
        object_decisions = scene_state.get("object_caption_decision") or []
        object_categories = scene_state.get("object_category") or []
        object_supercategories = scene_state.get("object_supercategory") or []
        object_key_attributes = scene_state.get("object_key_attributes") or []
        object_det_maps = scene_state.get("object_detection_category_conf") or []
        object_images_all = scene_state.get("rgb_observations") or []

        if active is None or means is None or cov6 is None:
            return

        active_mask = _to_numpy(active).astype(bool)
        if active_mask.size == 0 or not active_mask.any():
            if self._gaussian_handle is not None:
                self._gaussian_handle.remove()
                self._gaussian_handle = None

            # Cleanup cubes
            for handle in self._object_cube_handles.values():
                with contextlib.suppress(Exception):
                    handle.remove()
            self._object_cube_handles = {}
            if self._object_voxel_cloud_handle is not None:
                with contextlib.suppress(Exception):
                    self._object_voxel_cloud_handle.remove()
                self._object_voxel_cloud_handle = None

            self._latest_ids = None
            self._latest_caption_edit_texts = None
            return

        means_np = _to_numpy(means)[active_mask]
        cov_np = self._cov6_to_covariance(_to_numpy(cov6)[active_mask])
        box_dimensions = self._object_box_dimensions_from_covariances(cov_np)

        if object_ids is None:
            ids_np = np.arange(means_np.shape[0], dtype=int)
        else:
            ids_np = _to_numpy(object_ids)[active_mask].astype(int)

        # Metadata preparation
        active_indices_all = np.nonzero(active_mask)[0]
        box_centers, box_dimensions = self._object_box_geometry_from_voxels(
            scene_state,
            active_indices_all=active_indices_all,
            fallback_centers=means_np.astype(np.float32),
            fallback_dimensions=box_dimensions,
        )
        captions = []
        caption_edit_texts = []
        is_clear_by_idx = []
        is_visible_by_idx = []
        has_caption_by_idx = []
        images = []

        # Map image_id -> on-disk reference so click-to-inspect can fall back
        # to the object's anchor view when live crops are absent (saved graphs).
        id_to_image_ref: dict[int, str] = {}
        image_records = scene_state.get("images")
        if isinstance(image_records, (list, tuple)):
            for rec in image_records:
                with contextlib.suppress(Exception):
                    if isinstance(rec, dict):
                        ref = str(rec.get("storage_path") or rec.get("source_ref") or "")
                        image_id = rec.get("image_id")
                    else:  # ImageRecord dataclass after load_scene_state
                        ref = str(getattr(rec, "storage_path", "") or getattr(rec, "source_ref", "") or "")
                        image_id = getattr(rec, "image_id", None)
                    if ref and image_id is not None:
                        id_to_image_ref[int(image_id)] = ref
        view_ids_all = scene_state.get("viewpoint_image_ids") or scene_state.get("object_image_ids") or []
        view_refs: list[str | None] = []

        for idx in range(means_np.shape[0]):
            obj_idx_all = int(active_indices_all[idx])
            view_refs.append(self._first_view_ref(view_ids_all, id_to_image_ref, obj_idx_all))
            caption_text = (
                str(object_captions[obj_idx_all])
                if 0 <= obj_idx_all < len(object_captions)
                else ""
            )
            category = self._string_value(self._row_value(object_categories, obj_idx_all, ""))
            supercategory = self._string_value(self._row_value(object_supercategories, obj_idx_all, ""))
            key_attributes = self._string_list_value(self._row_value(object_key_attributes, obj_idx_all, []))
            decision = self._string_value(self._row_value(object_decisions, obj_idx_all, "")).lower()
            if decision not in {"keep", "drop"}:
                decision = "keep" if (caption_text.strip() or category or supercategory or key_attributes) else ""
            is_clear = decision != "drop"
            is_visible = True
            det_map = object_det_maps[obj_idx_all] if 0 <= obj_idx_all < len(object_det_maps) else {}
            det_keys = " ".join(str(k) for k in det_map.keys()) if isinstance(det_map, dict) else ""
            if self._object_text_is_excluded(caption_text, category, supercategory, " ".join(key_attributes), det_keys):
                is_visible = False
            caption_json = self._format_caption_json_markdown(
                description=caption_text,
                category=category,
                supercategory=supercategory,
                attributes=key_attributes,
                decision=decision,
            )
            captions.append(caption_json)
            caption_edit_texts.append(caption_text)
            is_clear_by_idx.append(is_clear)
            is_visible_by_idx.append(is_visible)
            has_caption_by_idx.append(bool(caption_text.strip()))
            images.append(object_images_all[obj_idx_all] if 0 <= obj_idx_all < len(object_images_all) else None)

        self._latest_ids = ids_np
        self._latest_captions = captions
        self._latest_caption_edit_texts = caption_edit_texts
        self._latest_images = images
        self._latest_view_refs = view_refs

        # Colors
        colors = np.zeros((means_np.shape[0], 3), dtype=np.uint8)
        for idx, obj_id in enumerate(ids_np):
            if int(obj_id) not in self._id_to_color:
                self._id_to_color[int(obj_id)] = self._rng.integers(low=0, high=255, size=(3,), dtype=np.uint8)
            colors[idx] = self._id_to_color[int(obj_id)]

        is_locked_list = scene_state.get("is_locked") or []
        is_locked_by_idx = [
            bool(is_locked_list[obj_idx]) if obj_idx < len(is_locked_list) else False for obj_idx in active_indices_all
        ]

        if self._object_gaussians_enabled:
            finite_gaussian = np.isfinite(means_np).all(axis=1) & np.isfinite(cov_np).all(axis=(1, 2))
            if np.any(finite_gaussian):
                opacities = np.full((int(finite_gaussian.sum()),), 0.5, dtype=np.float32).reshape(-1, 1)
                if self._gaussian_handle is not None:
                    self._gaussian_handle.remove()
                self._gaussian_handle = self._server.scene.add_gaussian_splats(
                    name="/object_gaussians",
                    centers=means_np[finite_gaussian].astype(np.float32),
                    rgbs=colors[finite_gaussian],
                    opacities=opacities,
                    covariances=cov_np[finite_gaussian].astype(np.float32),
                )
        elif self._gaussian_handle is not None:
            self._gaussian_handle.remove()
            self._gaussian_handle = None

        # --- Update Boxes with Optimization ---
        self._update_object_cubes(
            box_centers,
            colors,
            ids_np,
            is_locked_by_idx,
            is_clear_by_idx,
            dimensions=box_dimensions,
            has_caption_by_idx=has_caption_by_idx,
            visible_by_idx=is_visible_by_idx,
        )
        self._update_object_voxel_cloud(
            scene_state,
            active_indices_all=active_indices_all,
            ids=ids_np,
            colors=colors,
            dimensions=box_dimensions,
            is_clear_by_idx=is_clear_by_idx,
            visible_by_idx=is_visible_by_idx,
        )

    def _object_box_dimensions_from_covariances(self, covariances: np.ndarray) -> np.ndarray:
        if covariances.ndim != 3 or covariances.shape[1:] != (3, 3) or covariances.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)
        diag = np.diagonal(covariances, axis1=1, axis2=2).astype(np.float32, copy=False)
        diag = np.clip(diag, 1.0e-4, None)
        # A 5-sigma box is wide enough to read as an object extent without
        # letting noisy covariance tails dominate the scene view.
        dims = 5.0 * np.sqrt(diag)
        return np.maximum(dims, 0.08).astype(np.float32, copy=False)

    def _update_image_poses(self, scene_state: dict) -> None:
        if self._server is None:
            return

        images = scene_state.get("images")
        if not images:
            self._clear_image_pose_lines()
            return

        axis_colors = (
            np.array([255, 0, 0], dtype=np.uint8),
            np.array([0, 255, 0], dtype=np.uint8),
            np.array([0, 128, 255], dtype=np.uint8),
        )
        segments: list[np.ndarray] = []
        colors: list[np.ndarray] = []
        axis_length = 0.1

        for record in images:
            pose = getattr(record, "pose", None)
            if pose is None:
                continue
            try:
                pose_np = _to_numpy(pose)
            except Exception:
                continue
            if pose_np.shape != (4, 4):
                continue
            origin = pose_np[:3, 3]
            rot = pose_np[:3, :3]
            if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(rot)):
                continue

            for axis_idx, axis_color in enumerate(axis_colors):
                direction = rot[:, axis_idx]
                endpoint = origin + axis_length * direction
                segment = np.stack([origin, endpoint], axis=0)
                segments.append(segment.astype(np.float32))
                colors.append(np.tile(axis_color, (2, 1)))

        if not segments:
            self._clear_image_pose_lines()
            return

        points = np.stack(segments, axis=0)
        colors_np = np.stack(colors, axis=0).astype(np.uint8)

        if self._image_pose_handle is not None:
            with contextlib.suppress(Exception):
                self._image_pose_handle.remove()
        self._image_pose_handle = self._server.scene.add_line_segments(
            name="/image_pose_axes",
            points=points,
            colors=colors_np,
            line_width=1.0,
        )

    def _clear_image_pose_lines(self) -> None:
        if self._image_pose_handle is not None:
            with contextlib.suppress(Exception):
                self._image_pose_handle.remove()
            self._image_pose_handle = None

    def _update_object_connections(self, scene_state: dict) -> None:
        if self._server is None:
            return
        segments = self._compute_object_connections(scene_state)
        if segments is None or segments.size == 0:
            self._clear_object_connections()
            return

        points = segments.astype(np.float32)
        colors = np.tile(np.array([[255, 255, 0]], dtype=np.uint8), (points.shape[0], 2, 1))

        if self._object_connection_handle is not None:
            with contextlib.suppress(Exception):
                self._object_connection_handle.remove()

        self._object_connection_handle = self._server.scene.add_line_segments(
            name="/object_connections",
            points=points,
            colors=colors,
            line_width=2.0,
        )

    def _clear_object_connections(self) -> None:
        if self._object_connection_handle is not None:
            with contextlib.suppress(Exception):
                self._object_connection_handle.remove()
            self._object_connection_handle = None

    def _update_regions(self, scene_state: dict) -> None:
        if self._server is None:
            return

        means = scene_state.get("means")
        if means is None:
            self._clear_regions()
            return

        means_np = _to_numpy(means)
        if means_np.ndim != 2 or means_np.shape[1] != 3 or means_np.shape[0] == 0:
            self._clear_regions()
            return

        active = scene_state.get("active")
        if active is not None:
            active_mask = _to_numpy(active).astype(bool)
            if active_mask.size < means_np.shape[0]:
                padded = np.zeros((means_np.shape[0],), dtype=bool)
                padded[: active_mask.size] = active_mask
                active_mask = padded
            else:
                active_mask = active_mask[: means_np.shape[0]]
        else:
            active_mask = np.ones((means_np.shape[0],), dtype=bool)

        raw_regions = scene_state.get("region_object_lists") or []
        if not isinstance(raw_regions, (list, tuple)) or len(raw_regions) == 0:
            self._clear_regions()
            return

        labels = scene_state.get("region_labels") or []
        seen_region_ids: set[int] = set()
        line_segments: list[np.ndarray] = []
        line_colors: list[np.ndarray] = []

        for region_id, members_raw in enumerate(raw_regions):
            if not isinstance(members_raw, (list, tuple)) or len(members_raw) == 0:
                continue

            member_indices: list[int] = []
            for idx_raw in members_raw:
                try:
                    idx = int(idx_raw)
                except Exception:
                    continue
                if idx < 0 or idx >= means_np.shape[0] or idx >= active_mask.size:
                    continue
                if not active_mask[idx]:
                    continue
                center = means_np[idx]
                if center.shape[-1] != 3 or not np.all(np.isfinite(center)):
                    continue
                member_indices.append(idx)

            if not member_indices:
                continue

            member_points = means_np[member_indices].astype(np.float32, copy=False)
            centroid = member_points.mean(axis=0)
            if not np.all(np.isfinite(centroid)):
                continue

            color = self._region_color(region_id)
            seen_region_ids.add(region_id)
            handle = self._region_ball_handles.get(region_id)
            if handle is not None:
                try:
                    handle.position = centroid
                    handle.color = color
                except Exception:
                    with contextlib.suppress(Exception):
                        handle.remove()
                    handle = None

            if handle is None:
                try:
                    handle = self._server.scene.add_icosphere(
                        name=f"/regions/region_{region_id}",
                        radius=0.22,
                        subdivisions=2,
                        position=centroid,
                        color=color,
                        opacity=0.9,
                    )
                    self._region_ball_handles[region_id] = handle
                except Exception:
                    continue

            if labels and region_id < len(labels):
                label = str(labels[region_id] or "").strip()
                if label and hasattr(handle, "label"):
                    with contextlib.suppress(Exception):
                        handle.label = label

            for member in member_points:
                line_segments.append(np.stack([centroid, member], axis=0).astype(np.float32))
                line_colors.append(np.tile(color, (2, 1)))

        stale = [rid for rid in self._region_ball_handles if rid not in seen_region_ids]
        for rid in stale:
            handle = self._region_ball_handles.pop(rid, None)
            if handle is not None:
                with contextlib.suppress(Exception):
                    handle.remove()

        if not line_segments:
            self._clear_region_connections()
            return

        points = np.stack(line_segments, axis=0).astype(np.float32, copy=False)
        colors = np.stack(line_colors, axis=0).astype(np.uint8, copy=False)
        if self._region_connection_handle is not None:
            with contextlib.suppress(Exception):
                self._region_connection_handle.remove()
        self._region_connection_handle = self._server.scene.add_line_segments(
            name="/region_connections",
            points=points,
            colors=colors,
            line_width=2.5,
        )

    @staticmethod
    def _region_color(region_id: int) -> np.ndarray:
        palette = np.array(
            [
                [255, 122, 24],
                [64, 164, 223],
                [124, 204, 82],
                [231, 84, 128],
                [247, 206, 70],
                [143, 112, 219],
                [45, 190, 162],
                [229, 105, 64],
            ],
            dtype=np.uint8,
        )
        return palette[int(region_id) % len(palette)]

    def _clear_region_connections(self) -> None:
        if self._region_connection_handle is not None:
            with contextlib.suppress(Exception):
                self._region_connection_handle.remove()
            self._region_connection_handle = None

    def _clear_regions(self) -> None:
        for handle in list(self._region_ball_handles.values()):
            with contextlib.suppress(Exception):
                handle.remove()
        self._region_ball_handles = {}
        self._clear_region_connections()

    def _update_covisibility_connections(self, scene_state: dict, *, max_edges: int = 10_000) -> None:
        if self._server is None:
            return

        segments = self._compute_covisibility_connections(scene_state, max_edges=max_edges)
        if segments is None or segments.size == 0:
            self._clear_covisibility_connections(raw=True, filtered=False)
            return

        points = segments.astype(np.float32, copy=False)
        colors = np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (points.shape[0], 2, 1))

        if self._covisibility_connection_handle is not None:
            with contextlib.suppress(Exception):
                self._covisibility_connection_handle.remove()

        self._covisibility_connection_handle = self._server.scene.add_line_segments(
            name="/covisibility_connections",
            points=points,
            colors=colors,
            line_width=1.0,
        )

    def _update_covisibility_connections_filtered(self, scene_state: dict, *, max_edges: int = 10_000) -> None:
        if self._server is None:
            return

        segments = self._compute_covisibility_connections(
            scene_state,
            max_edges=max_edges,
            adj_key="covisibility_filtered_adj_u64",
        )
        if segments is None or segments.size == 0:
            self._clear_covisibility_connections(raw=False, filtered=True)
            return

        points = segments.astype(np.float32, copy=False)
        colors = np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (points.shape[0], 2, 1))

        if self._covisibility_filtered_connection_handle is not None:
            with contextlib.suppress(Exception):
                self._covisibility_filtered_connection_handle.remove()

        self._covisibility_filtered_connection_handle = self._server.scene.add_line_segments(
            name="/covisibility_connections_filtered",
            points=points,
            colors=colors,
            line_width=3.0,
        )

    def _clear_covisibility_connections(self, *, raw: bool = True, filtered: bool = True) -> None:
        if raw and self._covisibility_connection_handle is not None:
            with contextlib.suppress(Exception):
                self._covisibility_connection_handle.remove()
            self._covisibility_connection_handle = None
        if filtered and self._covisibility_filtered_connection_handle is not None:
            with contextlib.suppress(Exception):
                self._covisibility_filtered_connection_handle.remove()
            self._covisibility_filtered_connection_handle = None

    def _update_object_image_connections(self, scene_state: dict) -> None:
        if self._server is None:
            return

        object_image_ids = scene_state.get("object_image_ids")
        means = scene_state.get("means")
        images = scene_state.get("images")
        if object_image_ids is None or means is None or images is None:
            self._clear_object_image_connections()
            return

        if len(object_image_ids) == 0 or means.shape[0] == 0:
            self._clear_object_image_connections()
            return

        means_np = _to_numpy(means)
        if means_np.ndim != 2:
            self._clear_object_image_connections()
            return

        active_mask = None
        if scene_state.get("active") is not None:
            active_mask = _to_numpy(scene_state["active"]).astype(bool)

        segments: list[np.ndarray] = []
        num_objects = means_np.shape[0]

        for obj_idx, image_ids in enumerate(object_image_ids):
            if obj_idx >= num_objects:
                break
            if active_mask is not None and (obj_idx >= len(active_mask) or not active_mask[obj_idx]):
                continue
            if image_ids is None:
                continue
            obj_center = means_np[obj_idx]
            if obj_center.shape[-1] != 3 or not np.all(np.isfinite(obj_center)):
                continue
            if not isinstance(image_ids, (list, tuple)):
                continue
            for image_id in image_ids:
                if not isinstance(image_id, int):
                    continue
                if image_id < 0 or image_id >= len(images):
                    continue
                pose = getattr(images[image_id], "pose", None)
                if pose is None:
                    continue
                try:
                    pose_np = _to_numpy(pose)
                except Exception:
                    continue
                if pose_np.shape != (4, 4):
                    continue
                cam_pos = pose_np[:3, 3]
                if not np.all(np.isfinite(cam_pos)):
                    continue
                segments.append(np.stack([obj_center, cam_pos], axis=0).astype(np.float32))

        if not segments:
            self._clear_object_image_connections()
            return

        points = np.stack(segments, axis=0)
        colors = np.tile(np.array([200, 50, 50], dtype=np.uint8), (points.shape[0], 2, 1)).astype(np.uint8)

        if self._object_image_connection_handle is not None:
            with contextlib.suppress(Exception):
                self._object_image_connection_handle.remove()

        self._object_image_connection_handle = self._server.scene.add_line_segments(
            name="/object_image_connections",
            points=points,
            colors=colors,
            line_width=0.2,
        )

    def _clear_object_image_connections(self) -> None:
        if self._object_image_connection_handle is not None:
            with contextlib.suppress(Exception):
                self._object_image_connection_handle.remove()
            self._object_image_connection_handle = None

    def _update_detections(self, detection_info: dict | None) -> None:
        for handle in list(self._detection_cube_handles.values()):
            with contextlib.suppress(Exception):
                handle.remove()
        self._detection_cube_handles = {}
        self._latest_detection_ids = None
        self._latest_detection_captions = None
        self._latest_detection_images = None

        if detection_info is None:
            return

        means = detection_info.get("means")
        if means is None:
            return

        means_np = _to_numpy(means)
        if means_np.ndim != 2 or means_np.shape[0] == 0:
            return

        captions = detection_info.get("captions") or []
        images = detection_info.get("images") or []

        self._latest_detection_ids = list(range(means_np.shape[0]))
        self._latest_detection_captions = captions
        self._latest_detection_images = images

        dimensions = np.array([0.05, 0.05, 0.05], dtype=np.float32)
        color = np.array([255, 64, 64], dtype=np.uint8)

        for det_id, center in enumerate(means_np):
            if center.shape[-1] != 3 or not np.all(np.isfinite(center)):
                continue
            name = f"/detections/det_{det_id}"
            try:
                handle = self._server.scene.add_box(
                    name=name,
                    dimensions=dimensions,
                    position=center.astype(np.float32),
                    color=color,
                    opacity=0.7,
                )

                @handle.on_click
                def _(_, captured_id=det_id):
                    self._handle_detection_click(captured_id)

                self._detection_cube_handles[det_id] = handle
            except Exception:
                continue

    def _update_detection_connections(
        self,
        detection_info: dict | None,
        detection_neighbors: Sequence[Sequence[int]] | None,
        scene_state: dict,
    ) -> None:
        if self._server is None:
            return
        if detection_info is None or detection_neighbors is None:
            self._clear_detection_connections()
            return

        means = detection_info.get("means")
        if means is None:
            self._clear_detection_connections()
            return

        det_means_np = _to_numpy(means)
        if det_means_np.ndim != 2 or det_means_np.shape[0] == 0:
            self._clear_detection_connections()
            return

        obj_means = scene_state.get("means")
        if obj_means is None or obj_means.shape[0] == 0:
            self._clear_detection_connections()
            return
        obj_means_np = _to_numpy(obj_means)
        obj_active = scene_state.get("active")
        if obj_active is not None:
            obj_active_mask = _to_numpy(obj_active).astype(bool)
        else:
            obj_active_mask = None

        segments: list[np.ndarray] = []
        max_objects = obj_means_np.shape[0]
        for det_idx, obj_indices in enumerate(detection_neighbors):
            if det_idx >= det_means_np.shape[0]:
                break
            if obj_indices is None:
                continue
            det_center = det_means_np[det_idx]
            if not np.all(np.isfinite(det_center)):
                continue
            for obj_idx in obj_indices:
                if 0 <= obj_idx < max_objects:
                    if obj_active_mask is not None and not obj_active_mask[obj_idx]:
                        continue
                    obj_center = obj_means_np[obj_idx]
                    if not np.all(np.isfinite(obj_center)):
                        continue
                    segments.append(np.stack([det_center, obj_center], axis=0))

        if not segments:
            self._clear_detection_connections()
            return

        points = np.stack(segments, axis=0).astype(np.float32)
        colors = np.tile(np.array([[0, 200, 255]], dtype=np.uint8), (points.shape[0], 2, 1))

        if self._det_connection_handle is not None:
            with contextlib.suppress(Exception):
                self._det_connection_handle.remove()

        self._det_connection_handle = self._server.scene.add_line_segments(
            name="/detection_connections",
            points=points,
            colors=colors,
            line_width=1.5,
        )

    def _clear_detection_connections(self) -> None:
        if self._det_connection_handle is not None:
            with contextlib.suppress(Exception):
                self._det_connection_handle.remove()
            self._det_connection_handle = None

    def _compute_object_connections(self, scene_state: dict, hellinger_thresh: float = 0.7) -> np.ndarray | None:
        active = scene_state.get("active")
        means = scene_state.get("means")
        cov6 = scene_state.get("cov6")
        if active is None or means is None or cov6 is None:
            return None
        if active.numel() == 0:
            return None

        active_mask = active.bool()
        if active_mask.sum().item() < 2:
            return None

        means_active = means[active_mask]
        cov_active = _cov6_to_matrix_torch(cov6[active_mask])

        num = means_active.shape[0]
        idx_i = []
        idx_j = []
        for i in range(num):
            for j in range(i + 1, num):
                idx_i.append(i)
                idx_j.append(j)

        if not idx_i:
            return None

        mu1 = means_active[idx_i]
        mu2 = means_active[idx_j]
        cov1 = cov_active[idx_i]
        cov2 = cov_active[idx_j]

        with torch.amp.autocast("cuda", enabled=False):
            h2 = _hellinger_distance_batch(mu1.float(), cov1.float(), mu2.float(), cov2.float())

        valid_mask = h2 < hellinger_thresh
        if not torch.any(valid_mask):
            return None

        mu1_np = mu1[valid_mask].detach().cpu().numpy().astype(np.float32)
        mu2_np = mu2[valid_mask].detach().cpu().numpy().astype(np.float32)
        return np.stack([mu1_np, mu2_np], axis=1)

    @staticmethod
    def _iter_set_bits_u64(word: int):
        while word:
            lsb = word & -word
            yield lsb.bit_length() - 1
            word ^= lsb

    def _compute_covisibility_connections(
        self, scene_state: dict, *, max_edges: int, adj_key: str = "covisibility_adj_u64"
    ) -> np.ndarray | None:
        active = scene_state.get("active")
        means = scene_state.get("means")
        adj_u64 = scene_state.get(adj_key)

        if means is None or adj_u64 is None:
            return None

        means_np = _to_numpy(means)
        if means_np.ndim != 2 or means_np.shape[1] != 3 or means_np.shape[0] == 0:
            return None
        num_objects = int(means_np.shape[0])

        try:
            adj_np = _to_numpy(adj_u64)
        except Exception:
            return None
        if adj_np.ndim != 2 or adj_np.shape[0] < num_objects or adj_np.shape[1] <= 0:
            return None

        blocks = int(adj_np.shape[1])

        if active is not None:
            active_mask = _to_numpy(active).astype(bool)
            if active_mask.size < num_objects:
                padded = np.zeros((num_objects,), dtype=bool)
                padded[: active_mask.size] = active_mask
                active_mask = padded
            else:
                active_mask = active_mask[:num_objects]
        else:
            active_mask = np.ones((num_objects,), dtype=bool)

        active_indices = np.nonzero(active_mask)[0]
        if active_indices.size < 2:
            return None

        segments: list[np.ndarray] = []
        max_edges_int = max(0, int(max_edges))
        if max_edges_int == 0:
            return None

        # Collect edges as undirected (min(i,j), max(i,j)) so we visualize:
        # - symmetric adjacencies (typical undirected bitsets)
        # - upper-triangular-only adjacencies (space-saving)
        # - directed/partial adjacencies (show union, still undirected visually)
        seen_edges: set[tuple[int, int]] = set()

        adj_u = np.asarray(adj_np[:num_objects, :blocks]).astype(np.uint64, copy=False)
        for i in active_indices.tolist():
            if len(segments) >= max_edges_int:
                break
            row = adj_u[i]
            src = means_np[i]
            if not np.all(np.isfinite(src)):
                continue
            for block_idx in range(blocks):
                if len(segments) >= max_edges_int:
                    break
                word = int(row[block_idx])
                if word == 0:
                    continue
                for bit in self._iter_set_bits_u64(word):
                    j = block_idx * 64 + bit
                    if j == i or j >= num_objects:
                        continue
                    if not active_mask[j]:
                        continue
                    a, b = (i, j) if i < j else (j, i)
                    key = (int(a), int(b))
                    if key in seen_edges:
                        continue
                    dst = means_np[j]
                    if not np.all(np.isfinite(dst)):
                        continue
                    seen_edges.add(key)
                    segments.append(np.stack([src, dst], axis=0).astype(np.float32))
                    if len(segments) >= max_edges_int:
                        break

        if not segments:
            return None
        return np.stack(segments, axis=0)

    def _object_dims_hidden_by_size(self, dims: np.ndarray) -> bool:
        raw_dims = np.maximum(np.asarray(dims, dtype=np.float32), 0.0)
        if raw_dims.shape != (3,) or not np.all(np.isfinite(raw_dims)):
            return False
        if self._object_box_max_volume_m3 > 0.0 and float(np.prod(raw_dims)) > self._object_box_max_volume_m3:
            return True
        if self._object_box_max_side_m > 0.0 and float(np.max(raw_dims)) > self._object_box_max_side_m:
            return True
        return False

    def _sample_object_voxels(
        self,
        flat: np.ndarray,
        *,
        obj_idx: int,
        start: int,
        end: int,
        level: int,
        object_id: int,
    ) -> np.ndarray:
        if end <= start:
            return np.zeros((0, 3), dtype=np.float32)
        max_points = int(self._object_voxel_max_points_per_object)
        first_key = int(flat[start]) if start < end else 0
        last_key = int(flat[end - 1]) if start < end else 0
        cache_key = (int(obj_idx), int(start), int(end), int(level), max_points, first_key, last_key)
        cached = self._object_voxel_cache.get(cache_key)
        if cached is not None:
            return cached

        points = np.asarray(_decode_voxel_keys(flat[start:end], int(level)), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            sampled = np.zeros((0, 3), dtype=np.float32)
        else:
            finite = np.isfinite(points).all(axis=1)
            points = points[finite]
            if max_points > 0 and points.shape[0] > max_points:
                rng = np.random.default_rng(int(object_id) * 9176 + 13)
                keep = rng.choice(points.shape[0], size=max_points, replace=False)
                keep.sort()
                points = points[keep]
            sampled = points.astype(np.float32, copy=False)
        self._object_voxel_cache[cache_key] = sampled
        return sampled

    def _update_object_voxel_cloud(
        self,
        scene_state: dict,
        *,
        active_indices_all: np.ndarray,
        ids: np.ndarray,
        colors: np.ndarray,
        dimensions: np.ndarray,
        is_clear_by_idx: list[bool],
        visible_by_idx: list[bool],
    ) -> None:
        if self._server is None:
            return
        def _clear_voxel_handles() -> None:
            for attr in ("_object_voxel_cloud_handle", "_object_voxel_cloud_dim_handle"):
                handle = getattr(self, attr, None)
                if handle is not None:
                    with contextlib.suppress(Exception):
                        handle.remove()
                    setattr(self, attr, None)

        if not self._object_voxel_cloud_enabled:
            _clear_voxel_handles()
            return
        arrays = self._object_voxel_arrays(scene_state)
        if arrays is None:
            _clear_voxel_handles()
            return
        flat, offsets, levels = arrays

        focus = self._focus_object_ids  # None => everything is "in focus"

        # After a query, only focused objects (target / confounders / anchors) are
        # rendered at all; everything else is dropped entirely (hidden, and — since
        # its geometry is gone — non-interactive).
        all_points: list[np.ndarray] = []
        all_colors: list[np.ndarray] = []
        count = min(len(active_indices_all), len(ids), colors.shape[0], dimensions.shape[0])
        for idx in range(count):
            obj_idx = int(active_indices_all[idx])
            if obj_idx + 1 >= offsets.shape[0] or obj_idx >= levels.shape[0]:
                continue
            if idx < len(visible_by_idx) and not bool(visible_by_idx[idx]):
                continue
            if idx < len(is_clear_by_idx) and self._hide_unclear_object_boxes and not bool(is_clear_by_idx[idx]):
                continue
            if self._object_dims_hidden_by_size(dimensions[idx]):
                continue

            object_id = int(ids[idx])
            if focus is not None and object_id not in focus:
                continue

            start = int(offsets[obj_idx])
            end = int(offsets[obj_idx + 1])
            if start < 0 or end > flat.shape[0] or end <= start:
                continue
            points = self._sample_object_voxels(
                flat,
                obj_idx=obj_idx,
                start=start,
                end=end,
                level=int(levels[obj_idx]),
                object_id=object_id,
            )
            if points.shape[0] == 0:
                continue
            keep_mask = self._view_depth_keep_mask(points)
            keep_mask = keep_mask & self._point_distance_keep_mask(points)
            if keep_mask.shape[0] == points.shape[0] and not np.all(keep_mask):
                points = points[keep_mask]
            if points.shape[0] == 0:
                continue
            color = np.tile(np.clip(colors[idx], 0, 255).astype(np.uint8), (points.shape[0], 1))
            all_points.append(points)
            all_colors.append(color)

        _clear_voxel_handles()

        if all_points:
            points_np = np.concatenate(all_points, axis=0).astype(np.float32, copy=False)
            colors_np = np.concatenate(all_colors, axis=0).astype(np.uint8, copy=False)
            self._object_voxel_cloud_handle = self._server.scene.add_point_cloud(
                name="/voxel_cloud",
                points=points_np,
                colors=colors_np,
                point_size=self._object_voxel_point_size,
                point_shape="circle",
            )

    def _update_object_cubes(
        self,
        centers: np.ndarray,
        colors: np.ndarray,
        ids: np.ndarray,
        is_locked_by_idx: list[bool] | None = None,
        is_clear_by_idx: list[bool] | None = None,
        dimensions: np.ndarray | None = None,
        has_caption_by_idx: list[bool] | None = None,
        visible_by_idx: list[bool] | None = None,
    ) -> None:
        if self._server is None:
            return

        # We rely on the self._server.atomic() in the parent update() method
        # to ensure this loop doesn't cause network lag.

        default_dimensions = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        if dimensions is None or dimensions.shape != centers.shape:
            dimensions = np.tile(default_dimensions, (len(centers), 1))
        seen_ids: set[int] = set()
        if is_locked_by_idx is None:
            is_locked_by_idx = [False] * len(ids)
        if is_clear_by_idx is None:
            is_clear_by_idx = [True] * len(ids)
        if has_caption_by_idx is None:
            has_caption_by_idx = [False] * len(ids)
        if visible_by_idx is None:
            visible_by_idx = [True] * len(ids)

        for idx, (center, color, obj_id) in enumerate(zip(centers, colors, ids)):
            if center.shape[-1] != 3 or not np.all(np.isfinite(center)):
                continue
            obj_int = int(obj_id)
            # Focus membership decides which boxes stay visible after a query;
            # the display filters (max box side etc.) apply uniformly to
            # focused and unfocused boxes alike.
            in_focus = self._focus_object_ids is not None and obj_int in self._focus_object_ids
            if idx < len(visible_by_idx) and not bool(visible_by_idx[idx]):
                handle = self._object_cube_handles.pop(obj_int, None)
                if handle is not None:
                    with contextlib.suppress(Exception):
                        handle.remove()
                continue
            dims = np.asarray(dimensions[idx], dtype=np.float32) if idx < len(dimensions) else default_dimensions
            if dims.shape != (3,) or not np.all(np.isfinite(dims)):
                dims = default_dimensions
            raw_dims = np.maximum(dims.astype(np.float32, copy=False), 0.0)
            hide_for_size = False
            if self._object_box_max_volume_m3 > 0.0:
                volume_m3 = float(np.prod(raw_dims))
                if volume_m3 > self._object_box_max_volume_m3:
                    hide_for_size = True
            if self._object_box_max_side_m > 0.0:
                side_m = float(np.max(raw_dims))
                if side_m > self._object_box_max_side_m:
                    hide_for_size = True
            if self._object_box_large_side_threshold_m > 0.0:
                large_sides = int(np.count_nonzero(raw_dims > self._object_box_large_side_threshold_m))
                if large_sides > self._object_box_max_large_sides:
                    hide_for_size = True
            if self._object_box_max_z_m > 0.0:
                max_z_m = float(center[2] + 0.5 * raw_dims[2])
                if max_z_m > self._object_box_max_z_m:
                    hide_for_size = True
            if hide_for_size:
                handle = self._object_cube_handles.pop(obj_int, None)
                if handle is not None:
                    with contextlib.suppress(Exception):
                        handle.remove()
                continue
            if self._box_hidden_by_distance(center, raw_dims):
                handle = self._object_cube_handles.pop(obj_int, None)
                if handle is not None:
                    with contextlib.suppress(Exception):
                        handle.remove()
                continue
            if self._box_hidden_by_view_depth(center, raw_dims):
                handle = self._object_cube_handles.pop(obj_int, None)
                if handle is not None:
                    with contextlib.suppress(Exception):
                        handle.remove()
                continue
            dims = np.clip(raw_dims, 0.08, 6.0).astype(np.float32, copy=False)
            is_clear = idx >= len(is_clear_by_idx) or bool(is_clear_by_idx[idx])
            if self._hide_unclear_object_boxes and not is_clear:
                handle = self._object_cube_handles.pop(obj_int, None)
                if handle is not None:
                    with contextlib.suppress(Exception):
                        handle.remove()
                continue

            # Focus mode (after a query): hide every non-relevant box. Toggle
            # visibility instead of removing the handle — viser removes scene
            # nodes by name-prefix match, so removing cube_17 would also take
            # out cube_170..cube_179 (including focused ones). Invisible boxes
            # are not raycast, so they are non-interactive too.
            if self._focus_object_ids is not None and not in_focus:
                handle = self._object_cube_handles.get(obj_int)
                if handle is not None:
                    seen_ids.add(obj_int)
                    with contextlib.suppress(Exception):
                        handle.visible = False
                continue

            seen_ids.add(obj_int)
            rgb = np.clip(color, 0, 255).astype(np.uint8)
            is_locked = idx < len(is_locked_by_idx) and is_locked_by_idx[idx]
            has_caption = idx < len(has_caption_by_idx) and bool(has_caption_by_idx[idx])
            if is_locked:
                rgb = np.clip((rgb.astype(np.float32) * 0.6 + 128 * 0.4), 0, 255).astype(np.uint8)
            opacity = 0.8 if is_locked else (0.58 if has_caption else 0.30)
            # Query-role color coding: target #FFD45A, anchors #7FA8C6,
            # distractors #AA98BA.
            roles = self._query_roles if self._focus_object_ids is not None else None
            if roles:
                if obj_int == roles.get("target"):
                    rgb, opacity = np.array([255, 212, 90], dtype=np.uint8), 0.95
                elif obj_int in (roles.get("anchors") or ()):
                    rgb, opacity = np.array([127, 168, 198], dtype=np.uint8), 0.75
                elif obj_int in (roles.get("distractors") or ()):
                    rgb, opacity = np.array([170, 152, 186], dtype=np.uint8), 0.55

            handle = self._object_cube_handles.get(obj_int)

            # --- 1. In-Place Update (Fastest) ---
            if handle is not None:
                try:
                    # Directly updating attributes avoids object destruction/creation
                    handle.position = center
                    handle.color = rgb
                    handle.opacity = opacity
                    if hasattr(handle, "dimensions"):
                        handle.dimensions = dims
                    with contextlib.suppress(Exception):
                        handle.visible = True  # clear any earlier focus-hide
                    continue
                except Exception:
                    # If handle became invalid on server side, remove and recreate
                    with contextlib.suppress(Exception):
                        handle.remove()

            # --- 2. Creation (Only when new) ---
            box_name = f"/object_cubes/cube_{obj_int}"
            try:
                handle = self._server.scene.add_box(
                    name=box_name,
                    dimensions=dims,
                    position=center,
                    color=rgb,
                    opacity=opacity,
                    wireframe=False,
                )

                # Attach Click Listener to the BOX (This works, PointCloud does not)
                @handle.on_click
                def _(_, captured_id=obj_int):
                    self._handle_object_click(captured_id)

                self._object_cube_handles[obj_int] = handle
            except Exception:
                continue

        # --- 3. Cleanup Stale Objects ---
        # Identify objects that exist in handles but not in current frame
        stale_ids = [oid for oid in self._object_cube_handles if oid not in seen_ids]
        for oid in stale_ids:
            handle = self._object_cube_handles.pop(oid, None)
            if handle:
                with contextlib.suppress(Exception):
                    handle.remove()

    # ------------------------------------------------------------------
    # Interaction Handling
    # ------------------------------------------------------------------
    def _handle_object_click(self, obj_id: int) -> None:
        if self._latest_ids is None:
            return

        matches = np.where(self._latest_ids == obj_id)[0]
        if matches.size == 0:
            return

        idx = int(matches[0])

        caption = self._latest_captions[idx] if idx < len(self._latest_captions) else ""
        img_raw = self._latest_images[idx] if idx < len(self._latest_images) else None
        no_crop = img_raw is None or (isinstance(img_raw, (list, tuple, dict)) and len(img_raw) == 0)
        if no_crop and self._latest_view_refs is not None and idx < len(self._latest_view_refs):
            img_raw = self._load_image_ref(self._latest_view_refs[idx])
        edit_caption = (
            self._latest_caption_edit_texts[idx]
            if self._latest_caption_edit_texts is not None and idx < len(self._latest_caption_edit_texts)
            else caption
        )

        img = self._prepare_image_gallery(img_raw)
        self._set_clicked_caption(obj_id, caption)
        self._set_clicked_image(img)

        # Track selected object for editing
        self._selected_object_id = obj_id
        if self._edit_caption_input is not None:
            with contextlib.suppress(Exception):
                self._edit_caption_input.value = edit_caption

    def _handle_detection_click(self, det_id: int) -> None:
        if self._latest_detection_ids is None:
            return
        if det_id < 0 or det_id >= len(self._latest_detection_ids):
            return
        captions = self._latest_detection_captions or []
        images = self._latest_detection_images or []
        caption = captions[det_id] if det_id < len(captions) else ""
        img_raw = images[det_id] if det_id < len(images) else None
        img = self._prepare_image_gallery(img_raw)
        safe_caption = caption if caption else "N/A"
        self._set_caption_text(f"**Detection {det_id}:** {safe_caption}")
        self._set_clicked_image(img)

    def _set_clicked_caption(self, obj_id: int, caption: str) -> None:
        safe_caption = caption if caption else "N/A"
        text = f"**Object {obj_id}:** {safe_caption}"
        self._set_caption_text(text)

    def _set_caption_text(self, text: str) -> None:
        display = self._caption_display
        if display is None:
            return
        if hasattr(display, "content"):
            display.content = text
        elif hasattr(display, "value"):
            display.value = text
        elif hasattr(display, "markdown"):
            display.markdown = text
        elif hasattr(display, "set_markdown"):
            display.set_markdown(text)

    def _prepare_image(self, image: object | None) -> np.ndarray | None:
        bbox: Sequence[float] | None = None
        mask: object | None = None
        size: Sequence[float] | None = None
        if image is None:
            return None
        if isinstance(image, dict):
            if image.get("image_caption") is not None:
                bbox = image.get("bbox_caption", image.get("bbox"))
                mask = image.get("mask_caption", image.get("mask"))
                size = image.get("size_caption", image.get("size"))
                image = image.get("image_caption")
            else:
                bbox = image.get("bbox")
                mask = image.get("mask")
                size = image.get("size")
                image = image.get("image")
            if image is None:
                return None
        try:
            arr = _to_numpy(image)
        except Exception:
            return None

        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.ndim != 3 or arr.shape[-1] != 3:
            return None

        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        if mask is not None:
            arr = self._draw_mask_overlay(arr.copy(), mask, size=size)
        if bbox is not None:
            arr = arr.copy()
            arr = self._draw_bbox(arr, bbox, size=size)
        return self._resize_longest_side(arr, 200)

    @staticmethod
    def _draw_mask_overlay(
        arr: np.ndarray,
        mask: object,
        size: Sequence[float] | None = None,
        color: Sequence[int] = (0, 255, 0),
        alpha: float = 0.35,
    ) -> np.ndarray:
        del size  # Mask arrays are resized from their actual crop dimensions.
        if arr.ndim != 3 or arr.shape[-1] != 3:
            return arr
        h, w = arr.shape[:2]
        if h <= 0 or w <= 0:
            return arr
        try:
            mask_np = _to_numpy(mask)
        except Exception:
            return arr
        if mask_np.ndim == 3:
            mask_np = np.squeeze(mask_np)
        if mask_np.ndim != 2 or mask_np.size == 0:
            return arr

        mask_bool = mask_np.astype(bool, copy=False)
        src_h, src_w = mask_bool.shape[:2]
        if src_h <= 0 or src_w <= 0:
            return arr

        if mask_bool.shape != (h, w):
            src_h, src_w = mask_bool.shape[:2]
            y_idx = np.minimum((np.arange(h) * src_h / h).astype(int), src_h - 1)
            x_idx = np.minimum((np.arange(w) * src_w / w).astype(int), src_w - 1)
            mask_bool = mask_bool[y_idx[:, None], x_idx[None, :]]

        if not np.any(mask_bool):
            return arr

        alpha_f = float(np.clip(alpha, 0.0, 1.0))
        color_arr = np.asarray(color, dtype=np.float32).reshape(1, 3)
        blended = arr.astype(np.float32, copy=True)
        blended[mask_bool] = (1.0 - alpha_f) * blended[mask_bool] + alpha_f * color_arr
        return np.clip(blended, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _draw_bbox(
        arr: np.ndarray,
        bbox: Sequence[float],
        size: Sequence[float] | None = None,
        color: Sequence[int] = (255, 0, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        if arr.ndim != 3 or arr.shape[-1] != 3:
            return arr
        if bbox is None or len(bbox) != 4:
            return arr
        h, w = arr.shape[:2]
        if h <= 0 or w <= 0:
            return arr
        try:
            x0, y0, x1, y1 = [float(v) for v in bbox]
        except Exception:
            return arr

        if size is not None and len(size) == 2:
            try:
                src_w = float(size[0])
                src_h = float(size[1])
            except Exception:
                src_w = 0.0
                src_h = 0.0
            if src_w > 0.0 and src_h > 0.0 and (src_w != w or src_h != h):
                scale_x = w / src_w
                scale_y = h / src_h
                x0 *= scale_x
                x1 *= scale_x
                y0 *= scale_y
                y1 *= scale_y

        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        x0_i = int(round(x0))
        y0_i = int(round(y0))
        x1_i = int(round(x1))
        y1_i = int(round(y1))

        x0_i = max(0, min(w - 1, x0_i))
        x1_i = max(0, min(w - 1, x1_i))
        y0_i = max(0, min(h - 1, y0_i))
        y1_i = max(0, min(h - 1, y1_i))
        if x1_i < x0_i or y1_i < y0_i:
            return arr

        thickness = max(1, int(thickness))
        color_arr = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
        for t in range(thickness):
            x0_t = max(0, x0_i - t)
            x1_t = min(w - 1, x1_i + t)
            y0_t = max(0, y0_i - t)
            y1_t = min(h - 1, y1_i + t)
            arr[y0_t, x0_t : x1_t + 1] = color_arr
            arr[y1_t, x0_t : x1_t + 1] = color_arr
            arr[y0_t : y1_t + 1, x0_t] = color_arr
            arr[y0_t : y1_t + 1, x1_t] = color_arr
        return arr

    def _prepare_image_gallery(self, images: object | Sequence[object] | None) -> np.ndarray | None:
        if images is None:
            return None
        if not isinstance(images, (list, tuple)):
            images = [images]

        prepared: list[np.ndarray] = []
        for img in images:
            arr = self._prepare_image(img)
            if arr is not None:
                prepared.append(arr)

        if not prepared:
            return None

        cols = min(3, len(prepared))
        rows = int(math.ceil(len(prepared) / cols))
        max_h = max(img.shape[0] for img in prepared)
        max_w = max(img.shape[1] for img in prepared)
        tile_h = max(1, max_h)
        tile_w = max(1, max_w)
        gallery = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

        for idx, img in enumerate(prepared):
            r = idx // cols
            c = idx % cols
            canvas = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            h, w = img.shape[:2]
            canvas[:h, :w] = img
            gallery[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = canvas

        return gallery

    @staticmethod
    def _first_view_ref(view_ids_all: object, id_to_ref: dict[int, str], obj_idx: int) -> str | None:
        """Best-view image reference for object *obj_idx*, if one is on record."""
        try:
            if not id_to_ref or not isinstance(view_ids_all, (list, tuple)) or obj_idx >= len(view_ids_all):
                return None
            entry = view_ids_all[obj_idx]
            if isinstance(entry, (list, tuple)):
                entry = entry[0] if entry else None
            if entry is None:
                return None
            return id_to_ref.get(int(entry))
        except Exception:
            return None

    def _load_image_ref(self, ref: str | None) -> np.ndarray | None:
        """Load an object's anchor frame from disk — the click-to-inspect
        fallback for saved scene states, where live crops are unavailable."""
        if not ref:
            return None
        cached = self._view_image_cache.get(ref)
        if cached is not None:
            return cached
        try:
            from PIL import Image

            with Image.open(ref) as im:
                arr = np.asarray(im.convert("RGB"))
        except Exception as exc:
            LOGGER.info("Anchor view unavailable (%s): %s", ref, exc)
            return None
        arr = self._resize_longest_side(arr, 480)
        self._view_image_cache[ref] = arr
        if len(self._view_image_cache) > 64:
            self._view_image_cache.pop(next(iter(self._view_image_cache)))
        return arr

    def _set_clicked_image(self, image: np.ndarray | None) -> None:
        display = self._image_display
        if display is None:
            return
        img = image if image is not None else np.zeros((64, 64, 3), dtype=np.uint8)

        if hasattr(display, "image"):
            display.image = img
        elif hasattr(display, "value"):
            display.value = img

    @staticmethod
    def _resize_longest_side(arr: np.ndarray, longest_side: int) -> np.ndarray:
        if longest_side <= 0:
            return arr
        height, width = arr.shape[:2]
        current_longest = max(height, width)
        if current_longest == 0 or current_longest == longest_side:
            return arr
        if height >= width:
            new_h = longest_side
            new_w = max(1, int(round(width * longest_side / height)))
        else:
            new_w = longest_side
            new_h = max(1, int(round(height * longest_side / width)))

        y_idx = np.linspace(0, height - 1, new_h)
        x_idx = np.linspace(0, width - 1, new_w)
        y_idx = np.clip(np.round(y_idx).astype(int), 0, height - 1)
        x_idx = np.clip(np.round(x_idx).astype(int), 0, width - 1)
        return arr[y_idx[:, None], x_idx[None, :], :]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_colors(self, colors: np.ndarray) -> np.ndarray:
        colors = colors.astype(np.float32)
        max_val = colors.max() if colors.size > 0 else 1.0
        if max_val > 1.0:
            colors = np.clip(colors, 0.0, 255.0) / 255.0
        colors = np.clip(colors, 0.0, 1.0)
        return (colors * 255.0).astype(np.uint8)

    @staticmethod
    def _cov6_to_covariance(cov6: np.ndarray) -> np.ndarray:
        if cov6.ndim == 1:
            cov6 = cov6[None, :]
        xx, xy, xz, yy, yz, zz = np.split(cov6, 6, axis=-1)
        row0 = np.concatenate([xx, xy, xz], axis=-1)
        row1 = np.concatenate([xy, yy, yz], axis=-1)
        row2 = np.concatenate([xz, yz, zz], axis=-1)
        return np.stack([row0, row1, row2], axis=-2)

    @staticmethod
    def _voxel_downsample(points, colors, voxel_size):
        if points.size > 0:
            finite = np.isfinite(points).all(axis=1)
            points = points[finite]
            colors = colors[finite]
        if voxel_size <= 0.0 or points.size == 0:
            return points, colors
        voxel_keys = np.floor(points / voxel_size).astype(np.int64)
        _, unique_indices = np.unique(voxel_keys, axis=0, return_index=True)
        return points[unique_indices], colors[unique_indices]
