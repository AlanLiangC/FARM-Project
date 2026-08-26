#!/usr/bin/env python3

"""ROS2 node that streams synchronized RGB-D batches into mapping/scripts/pipeline components."""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple, Union

from mapping.lib.bbox_utils import assign_bbox, mask_to_xyxy
from mapping.lib.embedding_io import (  # noqa: E402
    bbox_area_xyxy,
    bbox_area_xyxy_clamped,
    observation_to_hwc_uint8,
    pack_embedding_history_matrix,
    pack_embedding_matrix,
    pack_text_history,
)
from mapping.lib.covisibility import (  # noqa: E402
    compute_covisibility_filtered_neighbors_indices,
    hellinger2_pairs,
    save_covisibility_visualization,
    update_covisibility_filtered_adjacency,
    update_covisibility_from_batch,
)
from mapping.lib.filtering import (  # noqa: E402
    filter_detections_by_distance,
    filter_detections_by_num_pixels,
    filter_detections_duplicates_iou,
    filter_detections_touching_image_border,
    filter_uninformative_yoloe_labels,
    mask_seg_outputs,
    normalize_seg_outputs,
)
from mapping.lib.geometric_transforms import (  # noqa: E402
    matrix_from_xyz_quat,
    normalize_camera_id,
    pose_from_matrix,
    rotate_image_array_cw,
    rotate_image_tensor_cw,
    rotz_homogeneous,
    transform_to_matrix,
    xyz_quat_from_matrix,
)
from mapping.lib.image_decoding import (  # noqa: E402
    decode_compressed,
    decode_depth,
    decode_image,
    decode_rgb,
    has_compressed_rgb_payload,
    has_raw_rgb_payload,
)
from mapping.lib.scene_graph_io import (  # noqa: E402
    build_detected_objects,
    build_timed_persist_checkpoints,
    take_scene_graph_snapshot,
)
from mapping.lib.viser_integration import (  # noqa: E402
    build_detection_captions,
    depth_image_from_array,
    encode_np_image_to_compressed,
    flatten_neighbors,
    format_detection_summary,
    label_from_id,
    resolve_object_index_by_id,
    rgb_image_from_array,
)
from mapping.lib.captioning import (  # noqa: E402
    build_local_caption_embeddings,
    build_local_caption_loser_ids,
    build_local_caption_object_ids,
    build_local_caption_texts,
    maybe_process_captions,
)
from mapping.lib.batch_processing import (  # noqa: E402
    log_queue_depths,
    oldest_frame_age,
)
from mapping.lib.frame_sync import (  # noqa: E402
    assemble_frame_payload,
    drain_ready_batch,
    do_log_queue_depths,
    get_oldest_frame_age,
    try_match_pending,
)
from mapping.lib.parameters import declare_mapper_parameters  # noqa: E402
from mapping.lib.viser_edit import (  # noqa: E402
    add_object_to_scene_state,
    delete_object,
    edit_caption,
    toggle_lock,
    validate_add_object_inputs,
)
from mapping.lib.persistence import (  # noqa: E402
    do_save_scene_state,
    maybe_persist_by_distance,
    maybe_persist_by_time,
    persist_all_artifacts,
    wait_for_save_idle,
    write_snapshot_and_json,
)

import cv2

try:  # h5py is optional; snapshot saving is disabled if unavailable.
    import h5py
except Exception:  # pragma: no cover - optional dependency
    h5py = None
import numpy as np
import rclpy
import torch
import torch.nn.functional as F
import yaml
from builtin_interfaces.msg import Time as BuiltinTimeMsg
from geometry_msgs.msg import Pose, Transform
from loguru import logger
from scene_graph.camera_config import CAMERA_CONFIG
from scene_graph.captioning.crop_util import compute_caption_observations
from scene_graph.captioning.services import CaptionManager
from scene_graph.debug import DebugTracer, resolve_trace_path
from scene_graph.debug.diagnostics import (
    state_digest,
    summarize_correspondence,
    summarize_filtering,
    summarize_gaussian_update,
    summarize_neighbors,
    summarize_segmentation,
    summarize_voxel_cloud,
)
from scene_graph.map_update.covisibility import (
    update_covisibility_active_bitset,
    update_covisibility_from_visible_indices,
)
from scene_graph.map_update.cannot_link import add_same_frame_cannot_links_from_detection_assignments
from scene_graph.map_update.get_neighbors import get_neighbors
from scene_graph.map_update.mask_observations import register_detection_mask_observations
from scene_graph.map_update.object_update import update_scene_graph_state
from scene_graph.map_update.pruning import (
    caption_keywords_criterion,
    compute_indices_to_prune,
)
from scene_graph.map_update.union_find import find_object_correspondence
from scene_graph.pipeline.steps import compute_detection_image_ids, resolve_correspondence
from mapping.lib.paths import (
    default_scene_graph_json_save_path,
    default_scene_graph_snapshot_dir,
    default_scene_state_path,
    default_storage_image_dir,
)
from scene_graph.runtime_paths import find_model_dir, find_package_file
from scene_graph.map_update.models import initialize_scene_graph_state
from scene_graph.storage.models import ImageRecord
from scene_graph.utils.geometry import transform_segmentation_to_world
from scene_graph.scene_state_io import load_scene_state
from scene_graph.segmentation import DINOFeaturesExtractor, YOLOESegmenter
from scene_graph.storage.image_save_worker import (
    ImageSaveWorker,
    mark_image_saved,
    register_batch_images,
)
from mapping_msgs.msg import (
    DetectedObject,
    DetectedObjects,
    FrameMetadata,
    LocalCaption,
    LocalCaptionArray,
    RGBDFrame,
    SceneGraphSnapshotSimple,
)
from mapping_msgs.srv import Siglip2TextEmbed
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.logging import LoggingSeverity
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Header, String
from std_srvs.srv import Trigger

try:
    from scene_graph.visualization.viser_util import update_viser_visualization
    from scene_graph.visualization.viser_visualizer import PipelineViserVisualizer
except Exception:  # pragma: no cover - optional dependency
    PipelineViserVisualizer = None

    def update_viser_visualization(*args, **kwargs):
        return kwargs.get("existing_rgb_observations")


try:  # matplotlib is optional; covisibility plotting is disabled if it is unavailable.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None


import contextlib


def prepare_for_debugging(value, deep_copy: bool = True):
    # Handle Tensors first (The most important part for data safety)
    if isinstance(value, torch.Tensor):
        out = value.detach()
        if deep_copy:
            out = out.clone()
        return out.cpu().numpy()

    # Handle standard collections recursively
    if isinstance(value, dict):
        return {k: prepare_for_debugging(v, deep_copy) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        converted = [prepare_for_debugging(v, deep_copy) for v in value]
        return tuple(converted) if isinstance(value, tuple) else converted

    # Fallback for primitive types (int, float, str, None, or numpy arrays)
    import copy

    return copy.deepcopy(value) if deep_copy else value


class StreamingMapper(Node):
    """ROS2 node that maps incoming color (RGB) + depth frames using the mapping package."""

    # Camera-specific image rotations to apply before segmentation/mapping.
    # Degrees are clockwise and must be multiples of 90.
    _CAMERA_IMAGE_ROTATION_CW: Dict[str, int] = {
        "head_left": 90,
        "head_right": 90,
        "right": 180,
    }

    # Camera neighborhood used for local covisibility updates.
    # Example locality: an object visible in head_left is compared against objects visible in head_left/head_right/left.
    _COVISIBILITY_CAMERA_NEIGHBORS: Dict[str, Tuple[str, ...]] = {
        "head_left": ("head_left", "head_right"),
        "head_right": ("head_right", "head_left"),
        "left": ("left", "head_left"),
        "right": ("right", "head_right"),
        "rear": ("rear",),
    }
    _REAR_LEG_MASK_HEIGHT_PX = 320
    _REAR_LEG_MASK_WIDTH_PX = 150

    _UNINFORMATIVE_LABELS = {
        "alley",
        "asphalt",
        "asphalt road",
        "avenue",
        "badlands",
        "beach",
        "bike path",
        "boardwalk",
        "canyon",
        "carpet",
        "cave",
        "ceiling",
        "city street",
        "cliff",
        "country lane",
        "crater lake",
        "crossroad",
        "crosswalk",
        "curb",
        "deck",
        "desert",
        "driveway",
        "dune",
        "earth",
        "embankment",
        "estuary",
        "fjord",
        "floor",
        "forest",
        "forest road",
        "glacier",
        "grass",
        "grassland",
        "gravel",
        "ground",
        "gulf",
        "harbor",
        "headland",
        "highway",
        "hill",
        "hillside",
        "hot spring",
        "inlet",
        "intersection",
        "island",
        "islet",
        "lagoon",
        "lake",
        "lakeshore",
        "land",
        "landfill",
        "lawn",
        "moor",
        "mound",
        "mountain",
        "mountain range",
        "mountain stream",
        "oasis",
        "outcrop",
        "overpass",
        "pasture",
        "path",
        "pavement",
        "peak",
        "peninsula",
        "plain",
        "plateau",
        "quarry",
        "race track",
        "raceway",
        "railroad",
        "railway line",
        "ravine",
        "reef",
        "reservoir",
        "ridge",
        "river",
        "road",
        "salt lake",
        "salt marsh",
        "sand",
        "savanna",
        "sea",
        "sea ice",
        "seabed",
        "shore",
        "shoreline",
        "sky",
        "slope",
        "snowfield",
        "strait",
        "stream",
        "summit",
        "swamp",
        "tarmac",
        "terrain",
        "thicket",
        "tide pool",
        "track",
        "trail",
        "train track",
        "trench",
        "tributary",
        "tundra",
        "valley",
        "volcano",
        "wall",
        "waterfall",
        "waterway",
        "wetland",
        "yard",
        "zebra crossing",
        "grove",  # Removes grass in isaac
        "corn field",  # Removes grass in isaac
        "footprint",
        "spear",  # Removes grass in isaac
        "seaweed",  # Removes grass in isaac
        "flare",  # Removes grass in isaac
        "reed",  # Removes grass in isaac
        "weed",  # Removes grass in isaac
        "plantation",  # Removes large patches of grass in isaac
        "hedge",  # Removes large patches of grass in isaac
    }

    # Additional neighbor pseudo-detections for local captions are restricted to this
    # Hellinger^2 threshold (matches `mapping.map_update.get_neighbors` convention).
    _LOCAL_CAPTION_NEIGHBOR_HELLINGER_THRESH = 0.6

    def __init__(self) -> None:
        super().__init__("mapping_streaming_mapper")
        declare_mapper_parameters(self)

        logger_level = self.get_parameter("logger_level").get_parameter_value().string_value
        level_map = {
            "DEBUG": LoggingSeverity.DEBUG,
            "INFO": LoggingSeverity.INFO,
            "WARN": LoggingSeverity.WARN,
            "WARNING": LoggingSeverity.WARN,
            "ERROR": LoggingSeverity.ERROR,
            "CRITICAL": LoggingSeverity.FATAL,
        }
        requested_severity = level_map.get(logger_level.upper(), LoggingSeverity.INFO)
        ros_logger = self.get_logger()
        ros_logger.set_level(requested_severity)
        logger.remove()
        logger.add(sys.stderr, level=logger_level, colorize=True)

        self._offline_debug = bool(self.get_parameter("offline_debug").value)
        self._timing_enabled = bool(self.get_parameter("timing_enabled").value)

        # Optional config for storage overrides
        self._config_path = self.get_parameter("config_path").get_parameter_value().string_value
        self._storage_cfg = {}
        if self._config_path and os.path.isfile(self._config_path):
            try:
                with open(self._config_path, "r", encoding="utf-8") as file:
                    cfg = yaml.safe_load(file) or {}
                    self._storage_cfg = dict(cfg.get("storage", {}) or {})
            except Exception as exc:
                ros_logger.warn(f"Failed to load config at {self._config_path}: {exc}")

        # Mapping components
        self._segmenter_flush_every_n_messages = int(self.get_parameter("segmenter_flush_every_n_messages").value)
        self._segmenter_flush_count = 0
        model_id = self.get_parameter("segmenter_model_id").value
        vocab_file = self.get_parameter("segmenter_vocab_file").value
        imgsz = int(self.get_parameter("segmenter_imgsz").value)
        conf_thres = float(self.get_parameter("segmenter_conf").value)
        iou_thres = float(self.get_parameter("segmenter_iou").value)
        segmenter_device = self.get_parameter("segmenter_device").value or None
        scene_state_device_raw = self.get_parameter("scene_state_device").value or "cpu"
        try:
            self._scene_state_device = torch.device(scene_state_device_raw)
        except Exception:
            self.get_logger().warn(f"Invalid scene_state_device={scene_state_device_raw!r}; defaulting to CPU.")
            self._scene_state_device = torch.device("cpu")
        use_dino = bool(self.get_parameter("segmenter_use_dino").value)
        dino_model = str(self.get_parameter("segmenter_dino_model").value or "").strip()
        dino_weights = self.get_parameter("segmenter_dino_weights_path").value
        dino_load_size = int(self.get_parameter("segmenter_dino_load_size").value)
        dino_facet = str(self.get_parameter("segmenter_dino_facet").value or "token").strip()
        dino_stride = int(self.get_parameter("segmenter_dino_stride").value)
        vis_segmentation_dir = self.get_parameter("vis_segmentation_dir").value or None
        mask_erosion_px = int(self.get_parameter("segmenter_mask_erosion_px").value)
        mahalanobis_thresh = float(self.get_parameter("segmenter_mahalanobis_thresh").value)
        depth_mode_filter_enabled = bool(self.get_parameter("segmenter_depth_mode_filter_enabled").value)
        depth_mode_k_mad = float(self.get_parameter("segmenter_depth_mode_k_mad").value)
        depth_mode_min_mad_m = float(self.get_parameter("segmenter_depth_mode_min_mad_m").value)

        dino_extractor = None
        if use_dino:
            dino_extractor = DINOFeaturesExtractor(
                model=dino_model or "facebook/dinov3-vits16-pretrain-lvd1689m",
                load_size=dino_load_size if dino_load_size > 0 else 512,
                facet=dino_facet or "token",
                stride=dino_stride if dino_stride > 0 else None,
                weights_path=str(dino_weights) if dino_weights else None,
                device=segmenter_device,
            )
        self._segmenter = YOLOESegmenter(
            model_id=model_id,
            vocab_file=vocab_file,
            imgsz=imgsz,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            device=segmenter_device,
            use_dino_features=use_dino,
            dino_extractor=dino_extractor,
            mask_erosion_px=mask_erosion_px,
            mahalanobis_thresh=mahalanobis_thresh,
            depth_mode_filter_enabled=depth_mode_filter_enabled,
            depth_mode_k_mad=depth_mode_k_mad,
            depth_mode_min_mad_m=depth_mode_min_mad_m,
            vis_segmentation_dir=vis_segmentation_dir,
            timing_enabled=self._timing_enabled,
        )

        self._covisibility_enabled = bool(self.get_parameter("covisibility_enabled").value)
        self._covisibility_max_objects = max(1, int(self.get_parameter("covisibility_max_objects").value))
        covisibility_alloc_max_objects = self._covisibility_max_objects if self._covisibility_enabled else 1

        self._filter_detections_touching_image_border_enabled = bool(
            self.get_parameter("filter_detections_touching_image_border_enabled").value
        )
        self._filter_touching_image_border_margin_px = int(
            self.get_parameter("filter_touching_image_border_margin_px").value
        )
        self._filter_touching_image_border_min_kept_num_pixels = int(
            self.get_parameter("filter_touching_image_border_min_kept_num_pixels").value
        )
        self._filter_touching_image_border_max_area_fraction = float(
            self.get_parameter("filter_touching_image_border_max_area_fraction").value
        )
        self._filter_detections_by_distance_enabled = bool(
            self.get_parameter("filter_detections_by_distance_enabled").value
        )
        self._filter_by_distance_min_m = float(self.get_parameter("filter_by_distance_min_m").value)
        self._filter_by_distance_max_m = float(self.get_parameter("filter_by_distance_max_m").value)
        self._filter_detections_by_num_pixels_enabled = bool(
            self.get_parameter("filter_detections_by_num_pixels_enabled").value
        )
        self._filter_by_num_pixels_min = int(self.get_parameter("filter_by_num_pixels_min").value)
        self._filter_uninformative_yoloe_labels_enabled = bool(
            self.get_parameter("filter_uninformative_yoloe_labels_enabled").value
        )
        self._filter_detections_duplicates_iou_enabled = bool(
            self.get_parameter("filter_detections_duplicates_iou_enabled").value
        )
        self._filter_duplicates_iou_min = float(self.get_parameter("filter_duplicates_iou_min").value)
        self._correspondence_feature_sim_thresh = float(
            self.get_parameter("correspondence_feature_sim_thresh").value
        )
        self._correspondence_hellinger_thresh = float(
            self.get_parameter("correspondence_hellinger_thresh").value
        )
        self._correspondence_hellinger_match_floor_m2 = float(
            self.get_parameter("correspondence_hellinger_match_floor_m2").value
        )
        self._correspondence_max_merge_distance_m = float(
            self.get_parameter("correspondence_max_merge_distance_m").value
        )
        self._correspondence_same_image_one_to_one = bool(
            self.get_parameter("correspondence_same_image_one_to_one").value
        )
        self._correspondence_use_class_gate = bool(
            self.get_parameter("correspondence_use_class_gate").value
        )
        self._correspondence_assignment_mode = str(
            self.get_parameter("correspondence_assignment_mode").value or "best_only"
        ).strip()

        self._caption_pad_ratio = float(self.get_parameter("caption_pad_ratio").value)
        self._caption_min_bbox_side = int(self.get_parameter("caption_min_bbox_side").value)
        self._caption_visual_prompt_mode = str(
            self.get_parameter("caption_visual_prompt_mode").value or "bbox_crop"
        ).strip().lower()
        self._caption_context_thumbnail_width = int(self.get_parameter("caption_context_thumbnail_width").value)
        self._caption_target_long_side = int(self.get_parameter("caption_target_long_side").value)
        self._caption_mask_fill_alpha = float(self.get_parameter("caption_mask_fill_alpha").value)
        self._caption_mask_outline_px = int(self.get_parameter("caption_mask_outline_px").value)

        self._scene_state = initialize_scene_graph_state(
            self._segmenter.feature_dim,
            self._scene_state_device,
            covisibility_max_objects=covisibility_alloc_max_objects,
        )
        covis_viz_enabled_param = bool(self.get_parameter("covisibility_viz_enabled").value)
        if self._covisibility_enabled and covis_viz_enabled_param and plt is None:
            self.get_logger().warn("matplotlib unavailable; covisibility graph visualization is disabled.")
        self._covisibility_viz_enabled = bool(
            self._covisibility_enabled and covis_viz_enabled_param and plt is not None
        )
        covis_viz_path_raw = str(self.get_parameter("covisibility_viz_path").value or "").strip()
        self._covisibility_viz_path: Optional[Path] = Path(covis_viz_path_raw) if covis_viz_path_raw else None
        self._covisibility_history_batches = max(1, int(self.get_parameter("covisibility_history_batches").value))
        self._covisibility_prev_per_camera: dict[str, set[int]] = {}
        self._covisibility_prev_unknown_visible: set[int] = set()
        self._scene_state_load_path = str(self.get_parameter("scene_state_load_path").value or "").strip()
        self._scene_state_save_path = str(self.get_parameter("scene_state_save_path").value or "").strip()
        if not self._scene_state_save_path:
            self._scene_state_save_path = self._scene_state_load_path
        self._scene_state_save_on_shutdown = bool(self.get_parameter("scene_state_save_on_shutdown").value)

        # ── Debug trace (per-frame JSONL) ────────────────────────────────────
        # Resolved from the ROS param, env var, or env dir + save_path stem.
        # Disabled if neither is set.
        debug_trace_path_param = str(self.get_parameter("debug_trace_path").value or "").strip()
        save_path_obj = Path(self._scene_state_save_path) if self._scene_state_save_path else None
        trace_path = resolve_trace_path(debug_trace_path_param, save_path=save_path_obj)
        self._debug_tracer: Optional[DebugTracer] = DebugTracer.create_if_path(trace_path)
        self._tracer_scene_started: bool = False
        if self._debug_tracer is not None:
            self._tracer_scene_id = (
                save_path_obj.stem if save_path_obj is not None and save_path_obj.suffix
                else (save_path_obj.name if save_path_obj is not None else "unknown_scene")
            )
            self.get_logger().info(f"Debug trace enabled: {trace_path}  scene_id={self._tracer_scene_id}")
        else:
            self._tracer_scene_id = ""
        self._scene_state_save_observations = bool(self.get_parameter("scene_state_save_observations").value)
        self._scene_state_save_wait_busy_timeout_sec = float(
            self.get_parameter("scene_state_save_wait_busy_timeout_sec").value
        )
        self._scene_state_save_observation_view_limit = int(
            self.get_parameter("scene_state_save_observation_view_limit").value
        )
        self._image_save_close_timeout_sec = float(self.get_parameter("image_save_close_timeout_sec").value)
        self._image_save_drain_on_shutdown = bool(self.get_parameter("image_save_drain_on_shutdown").value)
        self._caption_drain_timeout_sec = max(0.0, float(self.get_parameter("caption_drain_timeout_sec").value))

        self._scene_state_save_lock = threading.Lock()
        self._covisibility_lock = threading.Lock()
        self._viser_edit_lock = threading.Lock()
        self._latest_visualization_payload: Optional[Dict[str, Any]] = None
        self._worker = None
        self._shutdown_requested_event = threading.Event()
        self._shutdown_reason = ""
        self._shutdown_started = False
        self._shutdown_lock = threading.Lock()
        self._shutdown_timer = self.create_timer(0.2, self._maybe_shutdown)

        if bool(self.get_parameter("scene_state_save_service_enabled").value):
            self._save_scene_state_service = self.create_service(
                Trigger, "~/save_scene_state", self._handle_save_scene_state
            )
        else:
            self._save_scene_state_service = None
        if bool(self.get_parameter("scene_state_shutdown_service_enabled").value):
            self._save_and_shutdown_service = self.create_service(
                Trigger, "~/save_and_shutdown", self._handle_save_and_shutdown
            )
        else:
            self._save_and_shutdown_service = None

        if self._scene_state_load_path:
            try:
                loaded = load_scene_state(
                    self._scene_state_load_path,
                    feature_dim=self._segmenter.feature_dim,
                    device=self._scene_state_device,
                )
                self._scene_state = loaded
                loaded_objects = int(getattr(self._scene_state.get("means"), "shape", [0])[0])
                ros_logger.info(f"Loaded scene state from {self._scene_state_load_path} (objects={loaded_objects})")
            except Exception as exc:
                ros_logger.warn(f"Failed to load scene state from {self._scene_state_load_path}: {exc}")
        if self._covisibility_enabled:
            with contextlib.suppress(Exception):
                num_objects = int(getattr(self._scene_state.get("means"), "shape", [0])[0])
                update_covisibility_active_bitset(self._scene_state, num_objects=num_objects)

        lock_initial = bool(self.get_parameter("lock_initial_scene_state").value)
        if lock_initial and self._scene_state_load_path:
            is_locked = self._scene_state.get("is_locked")
            if isinstance(is_locked, list):
                means = self._scene_state.get("means")
                N = int(means.shape[0]) if means is not None and hasattr(means, "shape") else 0
                for i in range(N):
                    while len(is_locked) <= i:
                        is_locked.append(False)
                    is_locked[i] = True
                ros_logger.info(f"Locked {N} objects (lock_initial_scene_state=true)")

        self._image_saving_enabled = bool(self.get_parameter("image_saving_enabled").value)
        self._image_save_worker: Optional[ImageSaveWorker] = None
        self._image_storage_dir: Optional[Path] = None
        image_format = str(self.get_parameter("storage_image_format").value or "jpg").strip().lower()
        if image_format not in {"h5", "jpg", "jpeg"}:
            ros_logger.warn(f"Unsupported storage_image_format={image_format!r}; using 'jpg'.")
            image_format = "jpg"
        if image_format == "jpeg":
            image_format = "jpg"
        self._image_storage_format = image_format
        self._image_preview_max_width = max(0, int(self.get_parameter("storage_preview_max_width").value))
        self._image_preview_jpeg_quality = max(
            1, min(100, int(self.get_parameter("storage_preview_jpeg_quality").value))
        )
        self._image_save_queue_size = max(1, int(self.get_parameter("image_save_queue_size").value))
        self._image_save_max_per_batch = max(1, int(self.get_parameter("image_save_max_per_batch").value))
        self._image_save_dropped_last = 0
        if self._image_saving_enabled:
            self._image_save_worker = ImageSaveWorker(max_queue_size=self._image_save_queue_size)
            snapshot_dir_hint_raw = str(self.get_parameter("scene_graph_snapshot_dir").value or "").strip()
            if not snapshot_dir_hint_raw:
                snapshot_dir_hint_raw = str(
                    self._storage_cfg.get(
                        "snapshot_dir",
                        default_scene_graph_snapshot_dir(),
                    )
                    or ""
                ).strip()
            snapshot_root_dir: Optional[Path] = None
            if snapshot_dir_hint_raw:
                snapshot_root_dir = Path(snapshot_dir_hint_raw).expanduser().parent

            # Keep backward compatibility for legacy default "log/image_store":
            # when unset/legacy, anchor image_store next to snapshots.
            image_dir_raw = str(
                self.get_parameter("storage_image_dir").get_parameter_value().string_value or ""
            ).strip()
            image_dir_default = default_storage_image_dir()
            if not image_dir_raw or image_dir_raw == image_dir_default:
                image_dir_raw = str(self._storage_cfg.get("image_dir", "") or "").strip()
            if image_dir_raw == image_dir_default:
                image_dir_raw = ""
            if image_dir_raw:
                self._image_storage_dir = Path(image_dir_raw).expanduser()
            elif snapshot_root_dir is not None:
                self._image_storage_dir = snapshot_root_dir / "image_store"
            else:
                self._image_storage_dir = Path(default_storage_image_dir())
            self._image_storage_dir.mkdir(parents=True, exist_ok=True)

        self._object_mask_saving_enabled = bool(self.get_parameter("object_mask_saving_enabled").value)
        self._object_mask_observation_max_per_object = max(
            1, int(self.get_parameter("object_mask_observation_max_per_object").value)
        )
        self._object_mask_observation_save_crops = bool(
            self.get_parameter("object_mask_observation_save_crops").value
        )
        self._object_mask_observation_crop_jpeg_quality = max(
            1, min(100, int(self.get_parameter("object_mask_observation_crop_jpeg_quality").value))
        )
        self._object_mask_storage_dir: Optional[Path] = None
        if self._object_mask_saving_enabled:
            mask_dir_raw = str(
                self.get_parameter("object_mask_storage_dir").get_parameter_value().string_value or ""
            ).strip()
            if mask_dir_raw:
                self._object_mask_storage_dir = Path(mask_dir_raw).expanduser()
            elif self._scene_state_save_path:
                self._object_mask_storage_dir = (
                    Path(self._scene_state_save_path).expanduser().parent
                    / f"{Path(self._scene_state_save_path).stem}_masks"
                )
            elif self._image_storage_dir is not None:
                self._object_mask_storage_dir = self._image_storage_dir.parent / f"{self._image_storage_dir.name}_masks"
            else:
                self._object_mask_storage_dir = Path(default_storage_image_dir()).expanduser().parent / "object_masks"
            self._object_mask_storage_dir.mkdir(parents=True, exist_ok=True)

        caption_enabled = bool(self.get_parameter("caption_enabled").value)
        caption_device = self.get_parameter("caption_device").value
        caption_server = str(self.get_parameter("caption_server").value or "ollama").strip().lower()
        caption_version = int(self.get_parameter("caption_version").value)
        caption_batch_size = max(1, int(self.get_parameter("caption_batch_size").value))
        caption_spatial_context = bool(self.get_parameter("caption_spatial_context").value)
        caption_spatial_context_include_position = bool(
            self.get_parameter("caption_spatial_context_include_position").value
        )
        recaption_time_threshold_sec = float(self.get_parameter("recaption_time_threshold_sec").value)
        caption_step_interval = max(1, int(self.get_parameter("caption_step_interval").value))
        caption_start_step = max(0, int(self.get_parameter("caption_start_step").value))
        caption_merge_log_path = str(self.get_parameter("caption_merge_log_path").value or "").strip()
        if not caption_merge_log_path:
            caption_merge_log_path = ""
        caption_merge_enabled = bool(self.get_parameter("caption_merge_enabled").value)
        caption_merge_hellinger_thresh = float(self.get_parameter("caption_merge_hellinger_thresh").value)
        caption_merge_caption_thresh = float(self.get_parameter("caption_merge_caption_thresh").value)
        caption_merge_siglip2_thresh = float(self.get_parameter("caption_merge_siglip2_thresh").value)
        caption_merge_require_visual = bool(self.get_parameter("caption_merge_require_visual").value)
        caption_merge_require_category_compat = bool(
            self.get_parameter("caption_merge_require_category_compat").value
        )
        caption_prompt_variant = str(self.get_parameter("caption_prompt_variant").value or "default").strip().lower()
        caption_deactivate_unclear_enabled = bool(
            self.get_parameter("caption_deactivate_unclear_enabled").value
        )
        self._caption_step_interval = caption_step_interval
        self._caption_start_step = caption_start_step
        self._pending_caption_indices: List[int] = []
        # Start CaptionWorker if we need interactive edit embeddings, even when auto-captioning is off.
        self._auto_caption_enabled = bool(caption_enabled)
        # If Viser editing is enabled we need the worker for editing embeddings.
        worker_needed = bool(self.get_parameter("viser_enabled").value) or bool(caption_enabled)

        self._caption_manager = CaptionManager(
            scene_state=self._scene_state,
            enabled=worker_needed,
            debug=False,
            caption_batch_size=caption_batch_size,
            caption_device=caption_device,
            caption_server=caption_server,
            merge_log_path=caption_merge_log_path or None,
            merge_enabled=caption_merge_enabled,
            caption_spatial_context=caption_spatial_context,
            caption_spatial_context_include_position=caption_spatial_context_include_position,
            recaption_time_threshold_sec=recaption_time_threshold_sec,
            caption_merge_hellinger_thresh=caption_merge_hellinger_thresh,
            caption_merge_caption_thresh=caption_merge_caption_thresh,
            caption_merge_siglip2_thresh=caption_merge_siglip2_thresh,
            caption_merge_require_visual=caption_merge_require_visual,
            caption_merge_require_category_compat=caption_merge_require_category_compat,
            caption_visual_prompt_mode=self._caption_visual_prompt_mode,
            caption_prompt_variant=caption_prompt_variant,
            caption_version=caption_version,
            deactivate_unclear_objects=caption_deactivate_unclear_enabled,
        )
        if self._caption_manager.enabled:
            self._caption_manager.maybe_start_worker()
            # Expose worker shortcut for interactive edits
            self._worker = getattr(self._caption_manager, "worker", None)
            try:
                self._caption_manager.warm_up_model()
                ros_logger.info("Caption model warm-up successful")
            except Exception as exc:
                ros_logger.warn(f"Caption model warm-up failed: {exc}")

        # Region clustering
        self._region_enabled = bool(self.get_parameter("region_enabled").value)
        self._region_step_interval = max(1, int(self.get_parameter("region_step_interval").value))
        self._region_start_step = max(0, int(self.get_parameter("region_start_step").value))
        self._region_min_objects = max(1, int(self.get_parameter("region_min_objects").value))
        self._region_manager = None
        if self._region_enabled:
            from scene_graph.regions import RegionLabeler, RegionManager
            labeler = RegionLabeler()
            self._region_manager = RegionManager(
                self._scene_state,
                labeler=labeler,
                distance_threshold_m=float(self.get_parameter("region_distance_threshold_m").value),
                min_cluster_size=max(1, int(self.get_parameter("region_min_cluster_size").value)),
                max_region_diameter_m=float(self.get_parameter("region_max_diameter_m").value),
            )
            ros_logger.info("Region clustering enabled")

        self._siglip2_text_embed_timeout_sec = max(
            0.05, float(self.get_parameter("siglip2_text_embed_timeout_sec").value)
        )
        self._siglip2_text_embed_max_texts = max(1, int(self.get_parameter("siglip2_text_embed_max_texts").value))
        self._siglip2_text_embed_service = None
        self._siglip2_text_embed_cb_group = ReentrantCallbackGroup()
        if bool(self.get_parameter("siglip2_text_embed_service_enabled").value):
            service_name = str(self.get_parameter("siglip2_text_embed_service_name").value or "").strip()
            if not service_name:
                service_name = "/spot/mapping/siglip2_text_embed"
            worker_obj = getattr(self._caption_manager, "worker", None)
            if (
                self._caption_manager.enabled
                and worker_obj is not None
                and hasattr(worker_obj, "request_siglip2_text_embeddings")
            ):
                self._siglip2_text_embed_service = self.create_service(
                    Siglip2TextEmbed,
                    service_name,
                    self._handle_siglip2_text_embed,
                    callback_group=self._siglip2_text_embed_cb_group,
                )
                ros_logger.info(f"SigLIP2 text embedding service ready on {service_name}")
            else:
                ros_logger.warn("SigLIP2 text embedding service disabled: caption worker unavailable")

        viser_enabled = bool(self.get_parameter("viser_enabled").value)
        self._viser_visualizer = (
            PipelineViserVisualizer(
                enabled=True,
                host=str(self.get_parameter("viser_host").value or "127.0.0.1"),
                port=int(self.get_parameter("viser_port").value),
                live_rgb_enabled=bool(self.get_parameter("viser_live_rgb_enabled").value),
                live_rgb_max_side=int(self.get_parameter("viser_live_rgb_max_side").value),
                live_rgb_max_fps=float(self.get_parameter("viser_live_rgb_max_fps").value),
                on_edit_caption=self._viser_edit_caption,
                on_delete_object=self._viser_delete_object,
                on_save_all=self._viser_save_all,
                on_toggle_lock=self._viser_toggle_lock,
                on_add_object=self._viser_add_object,
            )
            if viser_enabled and PipelineViserVisualizer
            else None
        )

        self._global_frame = str(self.get_parameter("global_frame").value or "map")
        self._step_index = 0

        self._timing_log_interval_sec = float(self.get_parameter("timing_log_interval_sec").value)
        if self._timing_log_interval_sec <= 0.0:
            self._timing_log_interval_sec = 5.0
        self._timing_sum_s = 0.0
        self._timing_image_count = 0
        self._timing_batch_count = 0
        self._timing_last_log_mono = time.monotonic()
        if self._timing_enabled:
            self._timing_batch_done_pub = self.create_publisher(Header, "mapping/timing/batch_done", 10)
        else:
            self._timing_batch_done_pub = None
        self._max_time_sec = max(0.0, float(self.get_parameter("max_time_sec").value or 0.0))
        self._sim_time_remaining_sec: Optional[float] = None
        self._sim_time_remaining_active = False
        self._timed_persist_state: Dict[str, Any] = {"missing_warned": False}
        self._timed_persist_checkpoints: List[Dict[str, object]] = build_timed_persist_checkpoints(
            self._max_time_sec
        )
        if self._max_time_sec > 0.0 and self._timed_persist_checkpoints:
            checkpoint_tokens = []
            for checkpoint in self._timed_persist_checkpoints:
                labels = checkpoint.get("labels", [])
                if isinstance(labels, list):
                    checkpoint_tokens.extend(str(label) for label in labels if label)
            if checkpoint_tokens:
                ros_logger.info(
                    "Timed persist checkpoints enabled at "
                    f"max_time_sec={self._max_time_sec:.1f}s ({', '.join(checkpoint_tokens)})"
                )
            if not self._scene_state_save_path:
                ros_logger.warn(
                    "max_time_sec > 0 but scene_state_save_path is empty; timed checkpoints skip scene_state"
                )
                self._timed_persist_state["missing_warned"] = True
        self._sim_tracker_time_sub = self.create_subscription(
            BuiltinTimeMsg,
            "/sim_status_tracker_node/time",
            self._sim_time_remaining_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )

        # Subscriptions (best-practice path): one RGBDFrame topic per camera.
        camera_names_param = list(self.get_parameter("camera_names").get_parameter_value().string_array_value)
        if not camera_names_param:
            camera_names_param = ["head_left", "head_right", "left", "right", "rear"]

        self._camera_order = list(camera_names_param)
        self._rgbd_topics = [f"mapping/rgbd_frame/{name}" for name in self._camera_order]
        expected_param = int(self.get_parameter("expected_batch").get_parameter_value().integer_value)
        if expected_param <= 0:
            expected_param = len(self._camera_order)
        elif expected_param > len(self._camera_order):
            ros_logger.warn(
                f"expected_batch parameter ({expected_param}) exceeds number of cameras ({len(self._camera_order)});"
                " clamping."
            )
            expected_param = len(self._camera_order)
        self._expected_batch = expected_param

        queue_depth_param = max(
            10,
            int(self.get_parameter("queue_depth").get_parameter_value().integer_value),
        )
        self._latest_only = bool(self.get_parameter("latest_only").value)
        frame_queue_depth = 1 if self._latest_only else queue_depth_param
        self._frame_queues: Dict[str, Deque[dict]] = {
            name: deque(maxlen=frame_queue_depth) for name in self._camera_order
        }
        self._batch_timeout = float(self.get_parameter("batch_timeout").value)
        self._partial_min_cameras = 1
        max_pair_skew_sec = float(self.get_parameter("max_pair_skew_sec").value)
        if max_pair_skew_sec <= 0.0:
            ros_logger.warn("max_pair_skew_sec must be positive; using 0.05s default.")
            max_pair_skew_sec = 0.05
        self._max_pair_skew_sec = max_pair_skew_sec
        self._max_pair_skew_ns = int(max_pair_skew_sec * 1e9)

        self._debug_queue_status = bool(self.get_parameter("debug_queue_status").value)
        self._queue_debug_interval = 2.0
        self._queue_debug_state: Dict[str, float] = {"last": 0.0, "last_partial": 0.0, "last_full": 0.0}

        rgb_decode_mode = str(self.get_parameter("rgb_decode_mode").value or "auto").strip().lower()
        if rgb_decode_mode not in {"auto", "bgr2rgb", "passthrough"}:
            ros_logger.warn(f"Unsupported rgb_decode_mode={rgb_decode_mode!r}; using 'auto'.")
            rgb_decode_mode = "auto"
        self._rgb_decode_mode = rgb_decode_mode

        sensor_qos = QoSProfile(
            depth=queue_depth_param,
            history=HistoryPolicy.KEEP_LAST,
            # RGBDFrame contains both full-resolution images. Reliable QoS is
            # required for fragmented local samples on hosts whose UDP receive
            # buffer is smaller than one composite frame.
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # Per-camera throttling is no longer used; keep a guard for optional re-introduction.
        self._min_camera_period_ns = 0
        self._last_accepted_ns = {name: 0 for name in self._camera_order}
        self._min_local_caption_motion_m = -0.1  # skipping the skip logic
        self._last_local_caption_positions: Dict[str, np.ndarray] = {}

        self._rgbd_subs: Dict[str, object] = {}
        for name, rgbd_topic in zip(self._camera_order, self._rgbd_topics):
            sub = self.create_subscription(
                RGBDFrame,
                rgbd_topic,
                lambda msg, camera=name: self._rgbd_frame_callback(camera, msg),
                sensor_qos,
            )
            self._rgbd_subs[name] = sub

        topics_summary = ", ".join(f"{name}:rgbd={topic}" for name, topic in zip(self._camera_order, self._rgbd_topics))
        ros_logger.info(
            "Streaming mapping node initialized. Subscribed to "
            f"{topics_summary}. (latest_only={self._latest_only} max_pair_skew={self._max_pair_skew_sec:.3f}s)"
        )

        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._busy_event = threading.Event()
        self._psb_timeout_s = 1.0

        self._scene_graph_json_save_enabled = bool(self.get_parameter("scene_graph_json_save_enabled").value)
        self._scene_graph_json_save_path = str(self.get_parameter("scene_graph_json_save_path").value or "").strip()
        if self._scene_graph_json_save_enabled and not self._scene_graph_json_save_path:
            self.get_logger().warn("scene_graph_json_save_enabled is true but scene_graph_json_save_path is empty.")
            self._scene_graph_json_save_enabled = False
        self._scene_graph_snapshot_save_enabled = bool(self.get_parameter("scene_graph_snapshot_save_enabled").value)
        snapshot_dir_raw = str(self.get_parameter("scene_graph_snapshot_dir").value or "").strip()
        if not snapshot_dir_raw:
            snapshot_dir_raw = str(
                self._storage_cfg.get(
                    "snapshot_dir",
                    default_scene_graph_snapshot_dir(),
                )
                or ""
            ).strip()
        self._scene_graph_snapshot_dir: Optional[Path] = (
            Path(snapshot_dir_raw).expanduser() if snapshot_dir_raw else None
        )
        if self._scene_graph_snapshot_save_enabled and not self._scene_graph_snapshot_dir:
            self.get_logger().warn("scene_graph_snapshot_save_enabled is true but scene_graph_snapshot_dir is empty.")
            self._scene_graph_snapshot_save_enabled = False
        if self._scene_graph_snapshot_save_enabled and h5py is None:
            self.get_logger().warn("h5py unavailable; scene graph snapshot saving is disabled.")
            self._scene_graph_snapshot_save_enabled = False
        self._scene_graph_snapshot_lock = threading.Lock()
        self._scene_graph_snapshot_version = 0
        self._scene_graph_persist_distance_m = max(
            0.0, float(self.get_parameter("scene_graph_persist_distance_m").value)
        )
        self._persist_position_state: Dict[str, Any] = {}
        self._last_scene_graph_snapshot_ptr: Optional[Dict[str, object]] = None
        self._prune_enabled = bool(self.get_parameter("prune_enabled").value)
        self._prune_step_interval = max(1, int(self.get_parameter("prune_step_interval").value))
        keywords_str = str(self.get_parameter("prune_caption_keywords").value or "")
        keyword_list = [s.strip() for s in keywords_str.split(",") if s.strip()]
        if self._prune_enabled and keyword_list:
            self._prune_criteria: List[Any] = [caption_keywords_criterion(keyword_list)]
        else:
            self._prune_criteria = []
        if self._scene_graph_snapshot_save_enabled and self._scene_graph_snapshot_dir is not None:
            with contextlib.suppress(Exception):
                self._scene_graph_snapshot_dir.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(Exception):
                max_version = -1
                for entry in self._scene_graph_snapshot_dir.iterdir():
                    name = entry.name
                    if not entry.is_dir() or not name.startswith("v"):
                        continue
                    suffix = name[1:]
                    if not suffix.isdigit():
                        continue
                    max_version = max(max_version, int(suffix))
                if max_version >= 0:
                    self._scene_graph_snapshot_version = max_version + 1

        # Scene graph JSON publisher
        interval = float(self.get_parameter("scene_graph_publish_interval").value)
        interval = max(interval, 0.0)
        self._scene_graph_publish_interval = interval
        self._scene_graph_pub = self.create_publisher(String, "/spot/mapping/scene_graph_json", QoSProfile(depth=10))
        if interval > 0.0:
            self.create_timer(interval, self._publish_scene_graph_json)

        # Scene graph snapshot publisher (binary)
        snapshot_interval = float(self.get_parameter("scene_graph_snapshot_publish_interval").value)
        snapshot_interval = max(snapshot_interval, 0.0)
        self._scene_graph_snapshot_publish_interval = snapshot_interval
        snapshot_topic = self.get_parameter("scene_graph_snapshot_topic").get_parameter_value().string_value
        snapshot_topic = snapshot_topic or "/spot/mapping/scene_graph_snapshot"
        snapshot_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._scene_graph_snapshot_pub = self.create_publisher(SceneGraphSnapshotSimple, snapshot_topic, snapshot_qos)
        self._scene_graph_snapshot_seq = 0
        if snapshot_interval > 0.0:
            self.create_timer(snapshot_interval, self._publish_scene_graph_snapshot_simple)

        publish_local_captions_enabled = bool(self.get_parameter("publish_local_captions_enabled").value)
        if publish_local_captions_enabled:
            self._local_captions_pub = self.create_publisher(
                LocalCaptionArray, "/spot/mapping/local_captions", QoSProfile(depth=10)
            )
        else:
            self._local_captions_pub = None

        publish_detected_objects_enabled = bool(self.get_parameter("publish_detected_objects_enabled").value)
        if publish_detected_objects_enabled:
            publish_detected_objects_topic = str(
                self.get_parameter("publish_detected_objects_topic").value or "/mapping/detected_objects"
            )
            self._detected_objects_pub = self.create_publisher(
                DetectedObjects, publish_detected_objects_topic, QoSProfile(depth=10)
            )
        else:
            self._detected_objects_pub = None

        self._stop_adding_object_event = threading.Event()
        self._stop_adding_object_sub = self.create_subscription(
            Bool,
            "/spot/mapping/stop_adding_object",
            self._stop_adding_object_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )

    # ------------------------------------------------------------------
    # Helpers

    def _now_sec(self) -> float:
        """Return current node time in seconds (ROS time when use_sim_time is enabled)."""
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _sim_time_remaining_callback(self, msg: BuiltinTimeMsg) -> None:
        remaining_sec = max(0.0, float(msg.sec) + float(msg.nanosec) * 1e-9)
        # The tracker publishes 0 before a scenario starts; ignore those so we don't
        # immediately fire every "remaining <= X" checkpoint at startup.
        if remaining_sec <= 0.0 and not self._sim_time_remaining_active:
            return

        self._sim_time_remaining_sec = remaining_sec
        self._scene_state["sim_time_remaining_sec"] = remaining_sec

        if not self._sim_time_remaining_active:
            self._sim_time_remaining_active = True
            self.get_logger().info("Using /sim_status_tracker_node/time for timed checkpoint persistence.")
            # If max_time_sec was left unset, infer it from the first non-zero remaining time.
            if self._max_time_sec <= 0.0 and remaining_sec > 0.0:
                self._max_time_sec = remaining_sec
                self._timed_persist_checkpoints = build_timed_persist_checkpoints(self._max_time_sec)

    def _time_from_msg(self, stamp_msg) -> Time:
        clock_type = self.get_clock().clock_type
        try:
            return Time.from_msg(stamp_msg, clock_type=clock_type)
        except TypeError:
            return Time(
                seconds=int(getattr(stamp_msg, "sec", 0)),
                nanoseconds=int(getattr(stamp_msg, "nanosec", 0)),
                clock_type=clock_type,
            )

    def _assemble_frame_payload(
        self,
        camera: str,
        bundle: RGBDFrame,
    ) -> Optional[Dict[str, np.ndarray]]:
        return assemble_frame_payload(
            camera,
            bundle,
            time_from_msg=self._time_from_msg,
            now_sec=self._now_sec,
            logger_warn=self.get_logger().warn,
            logger_debug=self.get_logger().debug,
            logger_error=self.get_logger().error,
        )

    @staticmethod
    def _maybe_to_device(tensor: object, device: torch.device) -> object:
        if isinstance(tensor, torch.Tensor) and tensor.device != device:
            try:
                return tensor.to(device=device, non_blocking=True)
            except Exception:
                return tensor
        return tensor

    def _build_neighbor_view(self, device: torch.device) -> dict:
        # Only move the tensors needed for neighbor search; keep the CPU-resident
        # scene_state as the authoritative copy to save VRAM.
        state = self._scene_state
        return {
            "features": self._maybe_to_device(state.get("features"), device),
            "means": self._maybe_to_device(state.get("means"), device),
            "cov6": self._maybe_to_device(state.get("cov6"), device),
            "class_ids": self._maybe_to_device(state.get("class_ids"), device),
            "active": self._maybe_to_device(state.get("active"), device),
        }

    def _prepare_frames(self, decoded_batch: List[Dict[str, np.ndarray]]) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        try:
            colors: List[torch.Tensor] = []
            depths: List[torch.Tensor] = []
            rgb_intrinsics: List[torch.Tensor] = []
            depth_intrinsics: List[torch.Tensor] = []
            poses_world: List[torch.Tensor] = []

            target_device = getattr(self._segmenter, "device", torch.device("cpu"))
            if not isinstance(target_device, torch.device):
                target_device = torch.device(target_device)

            for frame_data in decoded_batch:
                rgb = frame_data["rgb"]
                depth_f32 = frame_data["depth_f32"]
                if depth_f32.ndim == 3 and depth_f32.shape[-1] == 1:
                    depth_f32 = depth_f32[..., 0]

                T_world_cam = frame_data["T_world_cam"]

                camera_name = str(frame_data.get("camera", "")).strip()
                rot_cw = int(self._CAMERA_IMAGE_ROTATION_CW.get(camera_name, 0))

                rgb_height, rgb_width = int(rgb.shape[0]), int(rgb.shape[1])
                rgb_fx = float(frame_data["rgb_instrinsics"]["fx"])
                rgb_fy = float(frame_data["rgb_instrinsics"]["fy"])
                rgb_cx = float(frame_data["rgb_instrinsics"]["cx"])
                rgb_cy = float(frame_data["rgb_instrinsics"]["cy"])

                depth_height_native = int(frame_data["depth_instrinsics"]["height"])
                depth_width_native = int(frame_data["depth_instrinsics"]["width"])
                depth_fx = float(frame_data["depth_instrinsics"]["fx"])
                depth_fy = float(frame_data["depth_instrinsics"]["fy"])
                depth_cx = float(frame_data["depth_instrinsics"]["cx"])
                depth_cy = float(frame_data["depth_instrinsics"]["cy"])

                # Resize depth on GPU to RGB size and scale depth intrinsics accordingly.
                depth_tensor = torch.from_numpy(depth_f32).to(
                    device=target_device, dtype=torch.float32, non_blocking=False
                )
                if depth_height_native != rgb_height or depth_width_native != rgb_width:
                    depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
                    depth_tensor = F.interpolate(depth_tensor, size=(rgb_height, rgb_width), mode="nearest")
                    depth_tensor = depth_tensor.squeeze(0).squeeze(0)
                scale_factor_width = rgb_width / depth_width_native
                scale_factor_height = rgb_height / depth_height_native
                depth_fx = depth_fx * scale_factor_width
                depth_fy = depth_fy * scale_factor_height
                depth_cx = depth_cx * scale_factor_width
                depth_cy = depth_cy * scale_factor_height
                depth_height, depth_width = rgb_height, rgb_width

                pose_world = torch.from_numpy(T_world_cam.astype(np.float32, copy=False))
                if rot_cw:
                    if rot_cw == 90:
                        rgb_fx, rgb_fy, rgb_cx, rgb_cy = (
                            rgb_fy,
                            rgb_fx,
                            (rgb_height - 1.0) - rgb_cy,
                            rgb_cx,
                        )
                        depth_fx, depth_fy, depth_cx, depth_cy = (
                            depth_fy,
                            depth_fx,
                            (depth_height - 1.0) - depth_cy,
                            depth_cx,
                        )
                    elif rot_cw == 180:
                        rgb_cx, rgb_cy = (
                            (rgb_width - 1.0) - rgb_cx,
                            (rgb_height - 1.0) - rgb_cy,
                        )
                        depth_cx, depth_cy = (
                            (depth_width - 1.0) - depth_cx,
                            (depth_height - 1.0) - depth_cy,
                        )
                    else:
                        raise ValueError(f"Unsupported rotation {rot_cw}° for camera '{camera_name}'")

                    # Keep pixel axes aligned with the camera frame by applying the inverse
                    # camera-frame roll to the world pose (reparameterization of the same physical camera pose).
                    pose_world = pose_world @ rotz_homogeneous(-rot_cw)

                # Move color to the segmenter device; depth already on device and resized.
                color_tensor = torch.from_numpy(rgb.astype(np.uint8, copy=False)).to(
                    device=target_device, non_blocking=False
                )

                if rot_cw:
                    color_tensor = rotate_image_tensor_cw(color_tensor, rot_cw)
                    depth_tensor = rotate_image_tensor_cw(depth_tensor, rot_cw)

                rgb_intr = torch.eye(4, dtype=torch.float32, device=target_device)
                rgb_intr[0, 0] = rgb_fx
                rgb_intr[1, 1] = rgb_fy
                rgb_intr[0, 2] = rgb_cx
                rgb_intr[1, 2] = rgb_cy

                depth_intr = torch.eye(4, dtype=torch.float32, device=target_device)
                depth_intr[0, 0] = depth_fx
                depth_intr[1, 1] = depth_fy
                depth_intr[0, 2] = depth_cx
                depth_intr[1, 2] = depth_cy

                colors.append(color_tensor)
                depths.append(depth_tensor)
                rgb_intrinsics.append(rgb_intr)
                depth_intrinsics.append(depth_intr)
                poses_world.append(pose_world)
            return colors, depths, rgb_intrinsics, depth_intrinsics, poses_world

        except Exception as exc:
            self.get_logger().error(f"Failed to prepare frames from queues: {exc}")
            logger.opt(exception=exc).error("Failed to prepare frames from queues")
            return None, None, None, None, None

    # ------------------------------------------------------------------
    # Scene graph publication

    def _object_reference_bbox_area(self, object_index: int) -> Optional[float]:
        if object_index < 0:
            return None
        rgb_obs = self._scene_state.get("rgb_observations", []) or []
        if object_index >= len(rgb_obs):
            return None
        obs_list = rgb_obs[object_index]
        obs_candidates = list(obs_list) if isinstance(obs_list, (list, tuple)) else [obs_list]
        for obs in reversed(obs_candidates):
            if not isinstance(obs, dict):
                continue
            area_raw = obs.get("bbox_area_source")
            with contextlib.suppress(Exception):
                area_val = float(area_raw)
                if np.isfinite(area_val) and area_val > 0.0:
                    return area_val
            bbox_raw = obs.get("bbox_source")
            if bbox_raw is None:
                bbox_raw = obs.get("bbox")
            area_fallback = bbox_area_xyxy(bbox_raw)
            if area_fallback is not None:
                return area_fallback
        return None

    def _publish_scene_graph_json(self) -> None:
        if self._scene_graph_pub is None:
            return
        objects, _snapshot_ctx = take_scene_graph_snapshot(
            self._scene_state, logger_info=self.get_logger().info
        )
        if not objects:
            return
        payload = {"objects": objects}
        snapshot_ptr = getattr(self, "_last_scene_graph_snapshot_ptr", None)
        if snapshot_ptr:
            payload["snapshot"] = snapshot_ptr
        msg = String()
        try:
            msg.data = json.dumps(payload)
        except Exception:
            return
        self._scene_graph_pub.publish(msg)
        # Disabled: co-visibility graph visualization/saving.
        # save_covisibility_visualization(state=self._scene_state, viz_path=self._covisibility_viz_path,
        #     logger_info=self.get_logger().info, logger_warn=self.get_logger().warn)

    def _persist_scene_graph_now(self) -> bool:
        """Persist scene_graph.json and/or a snapshot immediately (best-effort)."""
        if not bool(getattr(self, "_scene_graph_json_save_enabled", False)) and not bool(
            getattr(self, "_scene_graph_snapshot_save_enabled", False)
        ):
            return False

        new_version, snapshot_ptr, saved_any = write_snapshot_and_json(
            scene_state=self._scene_state,
            snapshot_dir=self._scene_graph_snapshot_dir,
            snapshot_version=self._scene_graph_snapshot_version,
            snapshot_lock=self._scene_graph_snapshot_lock,
            snapshot_save_enabled=bool(getattr(self, "_scene_graph_snapshot_save_enabled", False)),
            json_save_enabled=bool(getattr(self, "_scene_graph_json_save_enabled", False)),
            json_save_path=str(getattr(self, "_scene_graph_json_save_path", "") or ""),
            covisibility_enabled=self._covisibility_enabled,
            covisibility_lock=self._covisibility_lock,
            hellinger_thresh=self._LOCAL_CAPTION_NEIGHBOR_HELLINGER_THRESH,
            get_stamp_msg=lambda: self.get_clock().now().to_msg(),
            logger_info=self.get_logger().info,
            logger_warn=self.get_logger().warn,
        )
        self._scene_graph_snapshot_version = new_version
        if snapshot_ptr:
            self._last_scene_graph_snapshot_ptr = snapshot_ptr
        return saved_any

    def _persist_all_artifacts_now(self, *, reason: str) -> bool:
        return persist_all_artifacts(
            reason=reason,
            persist_scene_graph=self._persist_scene_graph_now,
            save_scene_state_fn=self._save_scene_state_now,
            scene_state_save_path=str(getattr(self, "_scene_state_save_path", "") or "").strip(),
            scene_state_save_lock=self._scene_state_save_lock,
            logger_warn=self.get_logger().warn,
            timed_persist_state=self._timed_persist_state,
        )

    def _maybe_persist_scene_graph_by_time(self) -> None:
        maybe_persist_by_time(
            checkpoints=getattr(self, "_timed_persist_checkpoints", []),
            sim_time_remaining_sec=getattr(self, "_sim_time_remaining_sec", None),
            max_time_sec=self._max_time_sec,
            persist_all=lambda reason: self._persist_all_artifacts_now(reason=reason),
            logger_info=self.get_logger().info,
        )

    def _maybe_persist_scene_graph_by_distance(self) -> None:
        maybe_persist_by_distance(
            threshold_m=float(getattr(self, "_scene_graph_persist_distance_m", 0.0) or 0.0),
            scene_state=self._scene_state,
            json_save_enabled=bool(getattr(self, "_scene_graph_json_save_enabled", False)),
            snapshot_save_enabled=bool(getattr(self, "_scene_graph_snapshot_save_enabled", False)),
            scene_state_save_path=str(getattr(self, "_scene_state_save_path", "") or "").strip(),
            persist_all=lambda reason: self._persist_all_artifacts_now(reason=reason),
            persist_position_state=self._persist_position_state,
        )

    def _maybe_update_regions(self) -> None:
        """Periodically re-cluster objects into spatial regions."""
        if self._region_manager is None:
            return
        if self._step_index < self._region_start_step:
            return
        if (self._step_index - self._region_start_step) % self._region_step_interval != 0:
            return
        n_captioned = sum(1 for c in self._scene_state.get("object_caption", []) if c)
        if n_captioned < self._region_min_objects:
            return
        try:
            changed = self._region_manager.update_regions()
            if changed:
                n_regions = len(self._scene_state.get("region_labels", []))
                self.get_logger().info(f"Regions updated: {n_regions} regions from {n_captioned} objects")
        except Exception as exc:
            self.get_logger().warn(f"Region clustering failed: {exc}")

    def _maybe_prune_scene_graph(self) -> None:
        """Soft-prune objects that pass all registered criteria (under _covisibility_lock)."""
        if not getattr(self, "_prune_enabled", False) or not getattr(self, "_prune_criteria", []):
            return
        if (self._step_index % self._prune_step_interval) != 0:
            return
        with self._covisibility_lock:
            indices = compute_indices_to_prune(self._scene_state, self._prune_criteria)
            active = self._scene_state.get("active")
            if active is None:
                return
            captions_state = self._scene_state.get("object_caption") or []
            object_ids = self._scene_state.get("object_id")
            means = self._scene_state.get("means")
            captions_pruned = []
            for i in indices:
                cap = ""
                obj_id = -1
                pos = None
                if i < len(captions_state) and captions_state[i] is not None:
                    cap = str(captions_state[i]).strip()
                if object_ids is not None and i < object_ids.numel():
                    with contextlib.suppress(Exception):
                        obj_id = int(object_ids[i].item())
                if means is not None and i < means.shape[0]:
                    with contextlib.suppress(Exception):
                        pos = means[i].detach().cpu().tolist()
                captions_pruned.append((i, obj_id, cap, pos))
                try:
                    active[i] = False
                    self.get_logger().debug(
                        "Object marked INACTIVE due to PRUNING: "
                        f"idx={i} object_id={obj_id} "
                        f"caption={repr(cap)} position={pos} "
                        f"(step={self._step_index})"
                    )
                except (IndexError, KeyError, TypeError):
                    continue
            if indices:
                caption_parts = [f"idx={i} id={obj_id} cap={repr(c)}" for i, obj_id, c, _ in captions_pruned]
                self.get_logger().info(f"Pruned {len(indices)} objects (indices: {indices}); details: {caption_parts}")

    def _build_scene_graph_snapshot_simple(self) -> Optional[SceneGraphSnapshotSimple]:
        state = self._scene_state
        active = state.get("active")
        means = state.get("means")
        cov6 = state.get("cov6")
        obj_ids = state.get("object_id")
        features = state.get("features")
        captions: List[str] = state.get("object_caption", [])
        viewpoint_image_ids = state.get("viewpoint_image_ids", [])
        images_meta: List[ImageRecord] = state.get("images", [])

        if (
            not isinstance(active, torch.Tensor)
            or not isinstance(means, torch.Tensor)
            or not isinstance(cov6, torch.Tensor)
            or not isinstance(obj_ids, torch.Tensor)
        ):
            return None
        if active.numel() == 0 or means.numel() == 0 or cov6.numel() == 0 or obj_ids.numel() == 0:
            return None

        try:
            active_cpu = active.detach().to("cpu", copy=False).to(torch.bool)
            means_cpu = means.detach().to("cpu", copy=False)
            cov6_cpu = cov6.detach().to("cpu", copy=False)
            obj_ids_cpu = obj_ids.detach().to("cpu", copy=False)
        except Exception:
            return None

        limit = min(
            active_cpu.numel(),
            means_cpu.shape[0],
            cov6_cpu.shape[0],
            obj_ids_cpu.shape[0],
        )
        if limit <= 0:
            return None
        finite_mask = torch.isfinite(means_cpu[:limit]).all(dim=1)
        include_mask = active_cpu[:limit] & finite_mask
        include_idx = torch.nonzero(include_mask, as_tuple=False).view(-1).tolist()
        if not include_idx:
            return None

        msg = SceneGraphSnapshotSimple()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._global_frame
        msg.schema_version = 2
        msg.seq = int(self._scene_graph_snapshot_seq)
        self._scene_graph_snapshot_seq += 1

        msg.object_ids = [int(obj_ids_cpu[i].item()) for i in include_idx]
        means_selected = means_cpu[include_idx].to(dtype=torch.float32).contiguous().view(-1)
        msg.mean_xyz = [float(x) for x in means_selected.tolist()]
        msg.object_captions = [str(captions[i] if i < len(captions) else "") for i in include_idx]

        camera_names = sorted(CAMERA_CONFIG.keys())
        camera_to_id = {name: idx for idx, name in enumerate(camera_names)}
        unknown_camera_id = 0xFFFFFFFF

        vp_row_ptr: List[int] = [0]
        vp_se3: List[float] = []
        vp_camera_ids: List[int] = []
        total_vp = 0
        for obj_i in include_idx:
            view_ids = viewpoint_image_ids[obj_i] if obj_i < len(viewpoint_image_ids) else []
            added = 0
            for image_id in view_ids:
                try:
                    image_id_int = int(image_id)
                except Exception:
                    continue
                if image_id_int < 0 or image_id_int >= len(images_meta):
                    continue
                pose = getattr(images_meta[image_id_int], "pose", None)
                if pose is None:
                    continue
                try:
                    pose_np = (
                        pose.detach().to("cpu", copy=False).numpy()
                        if isinstance(pose, torch.Tensor)
                        else np.asarray(pose)
                    )
                except Exception:
                    continue
                if pose_np.shape != (4, 4):
                    continue
                payload = xyz_quat_from_matrix(pose_np)
                if payload is None:
                    continue
                xyz, quat = payload
                if len(xyz) != 3 or len(quat) != 4:
                    continue
                vp_se3.extend([float(v) for v in (xyz + quat)])
                cam_name = str(getattr(images_meta[image_id_int], "camera_id", "") or "").strip()
                cam_name = normalize_camera_id(cam_name)
                vp_camera_ids.append(int(camera_to_id.get(cam_name, unknown_camera_id)))
                added += 1
            total_vp += added
            vp_row_ptr.append(total_vp)
        msg.vp_row_ptr = [int(v) for v in vp_row_ptr]
        msg.vp_se3 = vp_se3
        msg.vp_camera_ids = vp_camera_ids

        if isinstance(features, torch.Tensor) and features.ndim == 2 and features.shape[0] >= limit:
            feat_dim = int(features.shape[1])
            if feat_dim < 0:
                feat_dim = 0
            if feat_dim > 0xFFFF:
                feat_dim = 0xFFFF
            msg.embedding_dim = int(feat_dim)
            try:
                feat_cpu = features.detach().to("cpu", copy=False).to(torch.float32)
                feat_selected = feat_cpu[include_idx].contiguous().view(-1)
                msg.embeddings = [float(x) for x in feat_selected.tolist()]
            except Exception:
                msg.embedding_dim = 0
                msg.embeddings = []
        else:
            msg.embedding_dim = 0
            msg.embeddings = []

        siglip2_state: List[List[float]] = state.get("object_siglip2_embedding", []) or []
        siglip2_dim = 0
        for i in include_idx:
            vec = siglip2_state[i] if i < len(siglip2_state) else []
            if isinstance(vec, (list, tuple)) and vec:
                siglip2_dim = len(vec)
                break
        if siglip2_dim < 0:
            siglip2_dim = 0
        if siglip2_dim > 0xFFFF:
            siglip2_dim = 0xFFFF
        msg.siglip2_embedding_dim = int(siglip2_dim)
        if siglip2_dim > 0:
            zero_vec = [0.0] * int(siglip2_dim)
            packed: List[float] = []
            for i in include_idx:
                vec = siglip2_state[i] if i < len(siglip2_state) else []
                if isinstance(vec, (list, tuple)) and len(vec) == siglip2_dim:
                    packed.extend([float(x) for x in vec])
                else:
                    packed.extend(zero_vec)
            msg.siglip2_embeddings = packed
        else:
            msg.siglip2_embeddings = []

        covis_row_ptr: List[int] = [0]
        covis_neighbors: List[int] = []

        neighbors_by_obj = compute_covisibility_filtered_neighbors_indices(
            state=self._scene_state,
            covisibility_lock=self._covisibility_lock,
            include_idx=[int(x) for x in include_idx],
            include_mask=include_mask,
            means_cpu=means_cpu,
            cov6_cpu=cov6_cpu,
            limit=limit,
            hellinger_thresh=self._LOCAL_CAPTION_NEIGHBOR_HELLINGER_THRESH,
        )
        for obj_i in include_idx:
            obj_i_int = int(obj_i)
            for neighbor_idx in neighbors_by_obj.get(obj_i_int, []):
                if 0 <= neighbor_idx < limit and bool(include_mask[int(neighbor_idx)].item()):
                    covis_neighbors.append(int(obj_ids_cpu[int(neighbor_idx)].item()))
            covis_row_ptr.append(len(covis_neighbors))

        if len(covis_row_ptr) != (len(include_idx) + 1):
            covis_row_ptr = [0] * (len(include_idx) + 1)
            covis_neighbors = []
        msg.covis_row_ptr = [int(v) for v in covis_row_ptr]
        msg.covis_neighbors = [int(v) for v in covis_neighbors]
        return msg

    def _publish_scene_graph_snapshot_simple(self) -> None:
        pub = getattr(self, "_scene_graph_snapshot_pub", None)
        if pub is None:
            return
        t0 = time.perf_counter()
        msg = self._build_scene_graph_snapshot_simple()
        if msg is None:
            return
        pub.publish(msg)
        t1 = time.perf_counter()
        self.get_logger().debug(f"scene graph snapshot publish dt_ms={(t1 - t0) * 1000.0:.2f}")

    # ------------------------------------------------------------------
    # Viser publication helpers (static utilities moved to lib.viser_integration)

    @staticmethod
    def _copy_compressed_image(src: CompressedImage, header: Header) -> CompressedImage:
        msg = CompressedImage()
        msg.header = header
        msg.format = src.format
        msg.data = bytes(src.data)
        return msg

    @staticmethod
    def _to_plain_list(value: object) -> List[object]:
        if isinstance(value, torch.Tensor):
            with contextlib.suppress(Exception):
                return value.detach().to("cpu", copy=False).view(-1).tolist()
            return []
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            with contextlib.suppress(Exception):
                return list(value)
        return []

    def _update_object_detection_category_conf(
        self,
        *,
        seg_outputs: dict,
        det_idx: object,
        obj_idx: object,
        prev_object_count: int,
        allow_new_objects: bool,
    ) -> None:
        state = self._scene_state
        means_state = state.get("means")
        object_count = int(getattr(means_state, "shape", [0])[0]) if means_state is not None else 0
        if object_count <= 0:
            state["object_detection_category"] = []
            state["object_detection_category_conf"] = []
            return

        raw_list = state.get("object_detection_category_conf")
        det_list_raw = (
            list(raw_list)
            if isinstance(raw_list, Sequence) and not isinstance(raw_list, (str, bytes, bytearray))
            else []
        )
        det_list: List[Dict[str, float]] = []
        for idx in range(object_count):
            entry = det_list_raw[idx] if idx < len(det_list_raw) else None
            if not isinstance(entry, dict):
                det_list.append({})
                continue
            out: Dict[str, float] = {}
            for k, v in entry.items():
                if k is None:
                    continue
                try:
                    key = str(k)
                except Exception:
                    continue
                if not key:
                    continue
                try:
                    score = float(v)
                except Exception:
                    continue
                if not math.isfinite(score):
                    continue
                out[key] = score
            det_list.append(out)

        obj_winner_list = self._to_plain_list(obj_idx)
        old_object_count = min(int(prev_object_count), len(obj_winner_list), object_count)
        for old_idx in range(old_object_count):
            winner_idx = old_idx
            try:
                winner_candidate = int(obj_winner_list[old_idx])
                if 0 <= winner_candidate < object_count:
                    winner_idx = winner_candidate
            except Exception:
                winner_idx = old_idx
            if winner_idx == old_idx:
                continue
            loser_map = det_list[old_idx]
            if not loser_map:
                continue
            winner_map = det_list[winner_idx]
            for category, score in loser_map.items():
                prev = winner_map.get(category)
                if prev is None or float(score) > float(prev):
                    winner_map[category] = float(score)
            det_list[old_idx] = {}

        det_winner_list = self._to_plain_list(det_idx)
        if not det_winner_list:
            state["object_detection_category_conf"] = det_list
            self._sync_detection_categories(det_list)
            return

        scores_list = self._to_plain_list(seg_outputs.get("scores"))
        class_ids_list = self._to_plain_list(seg_outputs.get("class_ids"))
        names = getattr(self._segmenter, "names", None)

        new_object_rank = 0
        for det_i, det_target_raw in enumerate(det_winner_list):
            try:
                det_target = int(det_target_raw)
            except Exception:
                continue

            if det_target >= 0:
                obj_target = det_target
                if det_target < len(obj_winner_list):
                    with contextlib.suppress(Exception):
                        obj_target = int(obj_winner_list[det_target])
            elif allow_new_objects:
                obj_target = int(prev_object_count) + new_object_rank
                new_object_rank += 1
            else:
                continue

            if obj_target < 0 or obj_target >= object_count:
                continue
            if det_i >= len(scores_list):
                continue
            try:
                score = float(scores_list[det_i])
            except Exception:
                continue
            if not math.isfinite(score) or score < 0.0:
                continue

            cls_id = -1
            if det_i < len(class_ids_list):
                with contextlib.suppress(Exception):
                    cls_id = int(class_ids_list[det_i])
            category = label_from_id(names, cls_id)
            if not category:
                continue
            obj_map = det_list[obj_target]
            prev = obj_map.get(category)
            if prev is None or float(score) > float(prev):
                obj_map[category] = float(score)

        state["object_detection_category_conf"] = det_list
        self._sync_detection_categories(det_list)

    def _sync_detection_categories(self, det_list: Sequence[Dict[str, float]]) -> None:
        """Persist the detector winner without modifying VLM-owned semantics."""
        state = self._scene_state
        detected: List[str] = []
        for entry in det_list:
            best = ""
            if isinstance(entry, dict) and entry:
                with contextlib.suppress(Exception):
                    best = str(max(entry.items(), key=lambda item: float(item[1]))[0]).strip()
            detected.append(best)
        state["object_detection_category"] = detected

    # _build_local_caption_* methods extracted to lib.captioning.

    def _publish_local_captions(
        self,
        batch_frames: List[Dict[str, np.ndarray]],
        poses_world: List[torch.Tensor],
        seg_outputs: dict,
        det_idx: object,
        neighbors: Sequence[Sequence[int] | torch.Tensor] | None = None,
    ) -> None:
        if self._local_captions_pub is None:
            return

        num_frames = min(len(batch_frames), len(poses_world))
        if num_frames == 0:
            return

        batch_ids = seg_outputs.get("batch_ids")
        if isinstance(batch_ids, torch.Tensor):
            try:
                batch_ids_list = batch_ids.detach().cpu().tolist()
            except Exception:
                batch_ids_list = []
        elif isinstance(batch_ids, Sequence):
            batch_ids_list = list(batch_ids)
        else:
            batch_ids_list = []

        num_det = len(batch_ids_list)
        boxes_list: List[object] = []
        scores_list: List[object] = []
        class_ids_list: List[object] = []
        means_list: List[object] = []
        masks_list: List[object] = []

        boxes_xyxy = seg_outputs.get("boxes_xyxy")
        scores = seg_outputs.get("scores")
        class_ids = seg_outputs.get("class_ids")
        means = seg_outputs.get("means")
        masks = seg_outputs.get("masks")

        if isinstance(boxes_xyxy, torch.Tensor):
            try:
                boxes_list = boxes_xyxy.detach().cpu().tolist()
            except Exception:
                boxes_list = []
        elif isinstance(boxes_xyxy, Sequence):
            boxes_list = list(boxes_xyxy)

        if isinstance(scores, torch.Tensor):
            try:
                scores_list = scores.detach().cpu().tolist()
            except Exception:
                scores_list = []
        elif isinstance(scores, Sequence):
            scores_list = list(scores)

        if isinstance(class_ids, torch.Tensor):
            try:
                class_ids_list = class_ids.detach().cpu().tolist()
            except Exception:
                class_ids_list = []
        elif isinstance(class_ids, Sequence):
            class_ids_list = list(class_ids)

        if isinstance(means, torch.Tensor):
            try:
                means_list = means.detach().cpu().tolist()
            except Exception:
                means_list = []
        elif isinstance(means, Sequence):
            means_list = list(means)

        if isinstance(masks, torch.Tensor):
            try:
                masks_list = list(masks)
            except Exception:
                masks_list = []
        elif isinstance(masks, Sequence):
            masks_list = list(masks)

        det_captions = build_local_caption_texts(self._scene_state, det_idx, num_det)
        det_embeddings = build_local_caption_embeddings(self._scene_state, det_idx, num_det)
        det_object_ids = build_local_caption_object_ids(self._scene_state, det_idx, num_det)
        det_loser_ids = build_local_caption_loser_ids(self._scene_state, det_idx, num_det)
        names = getattr(self._segmenter, "names", None)

        det_idx_list: List[int] = []
        if det_idx is None:
            det_idx_list = [-1] * num_det
        elif isinstance(det_idx, torch.Tensor):
            with contextlib.suppress(Exception):
                det_idx_list = det_idx.detach().to("cpu", copy=False).view(-1).tolist()
        else:
            with contextlib.suppress(Exception):
                det_idx_list = list(det_idx)
        if len(det_idx_list) < num_det:
            det_idx_list.extend([-1] * (num_det - len(det_idx_list)))

        det_indices_by_frame: List[List[int]] = [[] for _ in range(num_frames)]
        for det_i, batch_id in enumerate(batch_ids_list):
            try:
                batch_idx = int(batch_id)
            except Exception:
                continue
            if 0 <= batch_idx < num_frames:
                det_indices_by_frame[batch_idx].append(det_i)

        for frame_idx in range(num_frames):
            frame = batch_frames[frame_idx]
            msg = LocalCaptionArray()

            header = Header()
            stamp_ns = int(frame.get("stamp_ns", 0) or 0)
            if stamp_ns > 0:
                header.stamp = Time(nanoseconds=stamp_ns).to_msg()
            else:
                header.stamp = self.get_clock().now().to_msg()
            header.frame_id = self._global_frame
            msg.header = header

            msg.camera_name = str(frame.get("camera", ""))

            pose_np: Optional[np.ndarray] = None
            with contextlib.suppress(Exception):
                pose_np = poses_world[frame_idx].detach().to("cpu", copy=False).numpy()
            # NOTE: Motion-based suppression is intentionally disabled so we always publish
            # `/spot/mapping/local_captions` whenever frames are processed.
            # (Previously: skipped publishing if the robot moved less than `min_local_caption_motion_m`.)

            try:
                if pose_np is None:
                    raise ValueError("pose missing")
                msg.robot_pose = pose_from_matrix(pose_np)
            except Exception:
                msg.robot_pose = Pose()

            image_header = Header()
            image_header.stamp = header.stamp
            image_header.frame_id = str(frame.get("frame_id", "")) or msg.camera_name

            image_msg: Optional[Image] = None
            rgb = frame.get("rgb")
            if isinstance(rgb, np.ndarray):
                rot_cw = int(self._CAMERA_IMAGE_ROTATION_CW.get(msg.camera_name, 0))
                if rot_cw:
                    rgb = rotate_image_array_cw(rgb, rot_cw)
                image_msg = rgb_image_from_array(rgb, image_header)
            if image_msg is None:
                image_msg = Image()
                image_msg.header = image_header
            msg.image = image_msg

            local_caption_msgs: List[LocalCaption] = []
            for det_i in det_indices_by_frame[frame_idx]:
                det_msg = LocalCaption()
                # Use the YOLO-E segmentation mask bbox (pixel coords in the full image) for visualization.
                bbox_xyxy: Optional[Tuple[float, float, float, float]] = None
                if det_i < len(masks_list):
                    bbox_xyxy = mask_to_xyxy(masks_list[det_i])
                if bbox_xyxy is None and det_i < len(boxes_list):
                    with contextlib.suppress(Exception):
                        bbox_xyxy = (
                            float(boxes_list[det_i][0]),
                            float(boxes_list[det_i][1]),
                            float(boxes_list[det_i][2]),
                            float(boxes_list[det_i][3]),
                        )
                if bbox_xyxy is not None:
                    assign_bbox(det_msg.bbox, bbox_xyxy)
                if det_i < len(scores_list):
                    with contextlib.suppress(Exception):
                        det_msg.confidence = float(scores_list[det_i])
                if det_i < len(class_ids_list):
                    try:
                        det_msg.category = label_from_id(names, int(class_ids_list[det_i]))
                    except Exception:
                        det_msg.category = label_from_id(names, -1)
                if det_i < len(means_list):
                    with contextlib.suppress(Exception):
                        mean_vec = means_list[det_i]
                        if isinstance(mean_vec, Sequence) and len(mean_vec) >= 3:
                            det_msg.position_3d.x = float(mean_vec[0])
                            det_msg.position_3d.y = float(mean_vec[1])
                            det_msg.position_3d.z = float(mean_vec[2])
                if det_i < len(det_captions):
                    det_msg.caption = det_captions[det_i] or ""
                if det_i < len(det_embeddings):
                    with contextlib.suppress(Exception):
                        det_msg.caption_embedding = det_embeddings[det_i] or []
                if det_i < len(det_object_ids):
                    with contextlib.suppress(Exception):
                        det_msg.matched_object_id = int(det_object_ids[det_i])
                if det_i < len(det_loser_ids) and hasattr(det_msg, "matched_object_loser_ids"):
                    with contextlib.suppress(Exception):
                        det_msg.matched_object_loser_ids = det_loser_ids[det_i] or []
                local_caption_msgs.append(det_msg)

            if neighbors:
                try:
                    means_state = self._scene_state.get("means")
                    cov6_state = self._scene_state.get("cov6")
                    det_means_state = seg_outputs.get("means")
                    det_cov6_state = seg_outputs.get("cov6")
                    if not isinstance(means_state, torch.Tensor) or not isinstance(cov6_state, torch.Tensor):
                        raise TypeError("scene state missing torch means/cov6")
                    if not isinstance(det_means_state, torch.Tensor) or not isinstance(det_cov6_state, torch.Tensor):
                        raise TypeError("seg outputs missing torch means/cov6")
                    if means_state.ndim != 2 or means_state.shape[1] < 3:
                        raise ValueError("scene state means has unexpected shape")

                    # Collect candidate (detection, object) neighbor pairs for detections in this frame,
                    # then filter them down to a stricter Hellinger^2 threshold for local captions.
                    det_ids_parts: List[torch.Tensor] = []
                    obj_ids_parts: List[torch.Tensor] = []
                    matched_obj_indices: set[int] = set()
                    for det_i in det_indices_by_frame[frame_idx]:
                        if 0 <= det_i < len(det_idx_list):
                            obj_idx = int(det_idx_list[det_i])
                            if obj_idx >= 0:
                                matched_obj_indices.add(obj_idx)
                        if det_i >= len(neighbors):
                            continue
                        neigh = neighbors[det_i]
                        if neigh is None:
                            continue
                        if isinstance(neigh, torch.Tensor):
                            neigh_tensor = neigh.to(device=means_state.device, dtype=torch.long, copy=False).view(-1)
                        else:
                            try:
                                neigh_tensor = torch.as_tensor(
                                    list(neigh),
                                    device=means_state.device,
                                    dtype=torch.long,
                                )
                            except Exception:
                                continue

                        if neigh_tensor.numel() == 0:
                            continue
                        # Clamp to valid scene object index range.
                        N_obj = int(means_state.shape[0])
                        neigh_tensor = neigh_tensor[(neigh_tensor >= 0) & (neigh_tensor < N_obj)]
                        if neigh_tensor.numel() == 0:
                            continue

                        det_ids_parts.append(
                            torch.full(
                                (int(neigh_tensor.numel()),),
                                int(det_i),
                                device=means_state.device,
                                dtype=torch.long,
                            )
                        )
                        obj_ids_parts.append(neigh_tensor)

                    if det_ids_parts and obj_ids_parts:
                        det_ids_cat = torch.cat(det_ids_parts, dim=0)
                        obj_ids_cat = torch.cat(obj_ids_parts, dim=0)

                        mu_det = det_means_state.index_select(0, det_ids_cat)
                        cov6_det = det_cov6_state.index_select(0, det_ids_cat)
                        mu_obj = means_state.index_select(0, obj_ids_cat)
                        cov6_obj = cov6_state.index_select(0, obj_ids_cat)

                        thresh = float(self._LOCAL_CAPTION_NEIGHBOR_HELLINGER_THRESH)
                        H2 = hellinger2_pairs(mu_det, cov6_det, mu_obj, cov6_obj)
                        keep_pairs = H2 < thresh
                        if not bool(keep_pairs.any()):
                            raise ValueError("no neighbors under local caption threshold")

                        obj_idx_tensor = torch.unique(obj_ids_cat[keep_pairs])
                        if matched_obj_indices and obj_idx_tensor.numel():
                            matched_tensor = torch.as_tensor(
                                sorted(matched_obj_indices),
                                device=means_state.device,
                                dtype=torch.long,
                            )
                            if hasattr(torch, "isin"):
                                obj_idx_tensor = obj_idx_tensor[~torch.isin(obj_idx_tensor, matched_tensor)]
                            else:
                                eq = obj_idx_tensor.unsqueeze(1) == matched_tensor.unsqueeze(0)
                                obj_idx_tensor = obj_idx_tensor[~eq.any(dim=1)]

                        if obj_idx_tensor.numel() == 0:
                            raise ValueError("no caption neighbors after removing matched objects")

                        obj_idx_tensor, _ = torch.sort(obj_idx_tensor)

                        # Compute camera intrinsics matching the rotated published image.
                        K_src = frame.get("K")
                        rgb_src = frame.get("rgb")
                        if not isinstance(K_src, np.ndarray) or K_src.shape[0] < 2 or K_src.shape[1] < 3:
                            raise TypeError("frame missing intrinsics matrix K")
                        if not isinstance(rgb_src, np.ndarray) or rgb_src.ndim < 2:
                            raise TypeError("frame missing RGB array")

                        rot_cw = int(self._CAMERA_IMAGE_ROTATION_CW.get(msg.camera_name, 0))
                        height_raw, width_raw = (
                            int(rgb_src.shape[0]),
                            int(rgb_src.shape[1]),
                        )
                        fx = float(K_src[0, 0])
                        fy = float(K_src[1, 1])
                        cx = float(K_src[0, 2])
                        cy = float(K_src[1, 2])
                        if rot_cw:
                            if rot_cw == 90:
                                fx, fy, cx, cy = fy, fx, (height_raw - 1.0) - cy, cx
                            elif rot_cw == 180:
                                cx, cy = (width_raw - 1.0) - cx, (height_raw - 1.0) - cy

                        height_img = int(msg.image.height) if getattr(msg, "image", None) is not None else 0
                        width_img = int(msg.image.width) if getattr(msg, "image", None) is not None else 0
                        if height_img <= 0 or width_img <= 0:
                            raise ValueError("published image has invalid dimensions")

                        device = means_state.device

                        mu_w = means_state.index_select(0, obj_idx_tensor).detach().to(dtype=torch.float32, copy=False)
                        cov6_w = cov6_state.index_select(0, obj_idx_tensor).detach().to(dtype=torch.float32, copy=False)

                        # cov6 -> full covariance matrix (world frame).
                        cov_w = torch.zeros((cov6_w.shape[0], 3, 3), device=device, dtype=torch.float32)
                        cov_w[:, 0, 0] = cov6_w[:, 0]
                        cov_w[:, 0, 1] = cov_w[:, 1, 0] = cov6_w[:, 1]
                        cov_w[:, 0, 2] = cov_w[:, 2, 0] = cov6_w[:, 2]
                        cov_w[:, 1, 1] = cov6_w[:, 3]
                        cov_w[:, 1, 2] = cov_w[:, 2, 1] = cov6_w[:, 4]
                        cov_w[:, 2, 2] = cov6_w[:, 5]

                        T_world_cam = poses_world[frame_idx].detach().to(device=device, dtype=torch.float32)
                        R_wc = T_world_cam[:3, :3]
                        t_wc = T_world_cam[:3, 3]

                        # World -> camera (row-vector convention): x_c = (x_w - t) @ R.
                        mu_c = (mu_w - t_wc.unsqueeze(0)) @ R_wc
                        cov_c = R_wc.t().unsqueeze(0) @ cov_w @ R_wc.unsqueeze(0)

                        z_cam = mu_c[:, 2]
                        valid_z = z_cam > 1e-6
                        z = z_cam.clamp(min=1e-6)
                        x = mu_c[:, 0]
                        y = mu_c[:, 1]
                        u = fx * (x / z) + cx
                        v = fy * (y / z) + cy

                        # Axis-aligned 2σ bbox extents from projected covariance.
                        inv_z = 1.0 / z
                        inv_z2 = inv_z * inv_z
                        j_u = torch.stack(
                            (fx * inv_z, torch.zeros_like(inv_z), -fx * x * inv_z2),
                            dim=1,
                        )
                        j_v = torch.stack(
                            (torch.zeros_like(inv_z), fy * inv_z, -fy * y * inv_z2),
                            dim=1,
                        )
                        var_u = torch.einsum("bi,bij,bj->b", j_u, cov_c, j_u)
                        var_v = torch.einsum("bi,bij,bj->b", j_v, cov_c, j_v)
                        ext_u = 2.0 * torch.sqrt(torch.clamp(var_u, min=0.0))
                        ext_v = 2.0 * torch.sqrt(torch.clamp(var_v, min=0.0))

                        x1 = u - ext_u
                        y1 = v - ext_v
                        x2 = u + ext_u
                        y2 = v + ext_v

                        finite = torch.isfinite(x1) & torch.isfinite(y1) & torch.isfinite(x2) & torch.isfinite(y2)
                        inside = (
                            (x1 > 0.0) & (y1 > 0.0) & (x2 < (float(width_img) - 1.0)) & (y2 < (float(height_img) - 1.0))
                        )
                        keep = valid_z & finite & inside

                        if bool(keep.any()):
                            keep_idx = torch.nonzero(keep, as_tuple=False).view(-1)

                            keep_obj_indices = obj_idx_tensor.index_select(0, keep_idx).detach().to("cpu").tolist()
                            x1_keep = x1.index_select(0, keep_idx).detach().to("cpu").tolist()
                            y1_keep = y1.index_select(0, keep_idx).detach().to("cpu").tolist()
                            x2_keep = x2.index_select(0, keep_idx).detach().to("cpu").tolist()
                            y2_keep = y2.index_select(0, keep_idx).detach().to("cpu").tolist()
                            mu_w_keep = mu_w.index_select(0, keep_idx).detach().to("cpu").tolist()

                            neighbor_count = len(keep_obj_indices)
                            neighbor_captions = build_local_caption_texts(self._scene_state, keep_obj_indices, neighbor_count)
                            neighbor_embeddings = build_local_caption_embeddings(self._scene_state, keep_obj_indices, neighbor_count)
                            neighbor_object_ids = build_local_caption_object_ids(self._scene_state, keep_obj_indices, neighbor_count)
                            neighbor_loser_ids = build_local_caption_loser_ids(self._scene_state, keep_obj_indices, neighbor_count)

                            for out_i in range(neighbor_count):
                                neighbor_msg = LocalCaption()
                                assign_bbox(
                                    neighbor_msg.bbox,
                                    (
                                        float(x1_keep[out_i]),
                                        float(y1_keep[out_i]),
                                        float(x2_keep[out_i]),
                                        float(y2_keep[out_i]),
                                    ),
                                )
                                neighbor_msg.confidence = 0.0
                                neighbor_msg.category = ""
                                mean_vec = mu_w_keep[out_i] if out_i < len(mu_w_keep) else None
                                if isinstance(mean_vec, Sequence) and len(mean_vec) >= 3:
                                    neighbor_msg.position_3d.x = float(mean_vec[0])
                                    neighbor_msg.position_3d.y = float(mean_vec[1])
                                    neighbor_msg.position_3d.z = float(mean_vec[2])
                                if out_i < len(neighbor_captions):
                                    neighbor_msg.caption = neighbor_captions[out_i] or ""
                                if out_i < len(neighbor_embeddings):
                                    with contextlib.suppress(Exception):
                                        neighbor_msg.caption_embedding = neighbor_embeddings[out_i] or []
                                if out_i < len(neighbor_object_ids):
                                    with contextlib.suppress(Exception):
                                        neighbor_msg.matched_object_id = int(neighbor_object_ids[out_i])
                                if out_i < len(neighbor_loser_ids) and hasattr(
                                    neighbor_msg, "matched_object_loser_ids"
                                ):
                                    with contextlib.suppress(Exception):
                                        neighbor_msg.matched_object_loser_ids = neighbor_loser_ids[out_i] or []
                                local_caption_msgs.append(neighbor_msg)
                except Exception:
                    pass

            msg.local_captions = local_caption_msgs

            with contextlib.suppress(Exception):
                self._local_captions_pub.publish(msg)

    # ------------------------------------------------------------------
    # ROS callbacks and batching

    def _stop_adding_object_callback(self, msg: Bool) -> None:
        stop = bool(getattr(msg, "data", False))
        currently_stopped = self._stop_adding_object_event.is_set()
        if stop == currently_stopped:
            return
        if stop:
            self._stop_adding_object_event.set()
            self.get_logger().info("stop_adding_object=True; new objects will not be added to the scene graph.")
        else:
            self._stop_adding_object_event.clear()
            self.get_logger().info("stop_adding_object=False; resuming adding new objects to the scene graph.")

    def _rgbd_frame_callback(self, camera: str, rgbd_msg: RGBDFrame) -> None:
        # Keep this callback as light as possible; decoding happens in the mapping worker thread.
        meta = rgbd_msg.meta
        stamp_ns = int(self._time_from_msg(meta.header.stamp).nanoseconds)
        if stamp_ns == 0:
            return

        last = self._last_accepted_ns.get(camera, 0)
        if self._min_camera_period_ns > 0 and stamp_ns - last < self._min_camera_period_ns:
            return
        queue = self._frame_queues.get(camera)
        if queue is None:
            self.get_logger().warn(f"Received frame for unknown camera '{camera}'")
            return
        queue.append({
            "camera": camera,
            "rgbd_msg": rgbd_msg,
            "received_time": self._now_sec(),
        })
        self._last_accepted_ns[camera] = stamp_ns
        self._segmenter_flush_count += 1
        if self._segmenter_flush_count >= self._segmenter_flush_every_n_messages:
            self._segmenter_flush_count = 0
            self._maybe_flush_partial()

    def _rgb_callback(self, camera: str, rgb_msg: CompressedImage) -> None:
        stamp_ns = int(self._time_from_msg(rgb_msg.header.stamp).nanoseconds)
        if stamp_ns == 0:
            return

        now_sec = self._now_sec()
        pending = self._pending_rgb.get(camera)
        if pending is None:
            return
        pending[stamp_ns] = (rgb_msg, now_sec)
        self._pending_rgb_order[camera].append(stamp_ns)
        self._prune_pending(camera, now_sec)
        if stamp_ns in self._pending_meta.get(camera, {}):
            self._try_match_pending(camera, stamp_ns)

    def _depth_callback(self, camera: str, depth_msg: Image) -> None:
        stamp_ns = int(self._time_from_msg(depth_msg.header.stamp).nanoseconds)
        if stamp_ns == 0:
            return

        now_sec = self._now_sec()
        pending = self._pending_depth.get(camera)
        if pending is None:
            return
        pending[stamp_ns] = (depth_msg, now_sec)
        self._pending_depth_order[camera].append(stamp_ns)
        self._prune_pending(camera, now_sec)

        pending_meta = self._pending_meta.get(camera) or {}
        for rgb_stamp_ns, (_, depth_stamp_ns, _) in list(pending_meta.items()):
            if depth_stamp_ns == stamp_ns:
                self._try_match_pending(camera, rgb_stamp_ns)

    def _meta_callback(self, camera: str, meta_msg: FrameMetadata) -> None:
        rgb_stamp_ns = int(self._time_from_msg(meta_msg.header.stamp).nanoseconds)
        if rgb_stamp_ns == 0:
            return

        depth_stamp_ns = rgb_stamp_ns
        try:
            if meta_msg.depth_stamp.sec or meta_msg.depth_stamp.nanosec:
                depth_stamp_ns = int(self._time_from_msg(meta_msg.depth_stamp).nanoseconds)
        except Exception:
            depth_stamp_ns = rgb_stamp_ns

        now_sec = self._now_sec()
        pending = self._pending_meta.get(camera)
        if pending is None:
            return
        pending[rgb_stamp_ns] = (meta_msg, depth_stamp_ns, now_sec)
        self._pending_meta_order[camera].append(rgb_stamp_ns)
        self._prune_pending(camera, now_sec)
        self._try_match_pending(camera, rgb_stamp_ns)

    def _try_match_pending(self, camera: str, rgb_stamp_ns: int) -> None:
        try_match_pending(
            camera,
            rgb_stamp_ns,
            pending_meta=self._pending_meta,
            pending_rgb=self._pending_rgb,
            pending_depth=self._pending_depth,
            camera_callback=self._camera_callback,
        )

    def _camera_callback(
        self,
        camera: str,
        rgb_msg: CompressedImage,
        depth_msg: Image,
        meta_msg: FrameMetadata,
    ) -> None:
        stamp_ns = int(self._time_from_msg(meta_msg.header.stamp).nanoseconds)
        if stamp_ns == 0:
            return

        last = self._last_accepted_ns.get(camera, 0)
        if self._min_camera_period_ns > 0 and stamp_ns - last < self._min_camera_period_ns:
            return

        frame_payload = self._assemble_frame_payload(camera, rgb_msg, depth_msg, meta_msg)
        if frame_payload is None:
            return
        queue = self._frame_queues.get(camera)
        if queue is None:
            self.get_logger().warn(f"Received frame for unknown camera '{camera}'")
            return

        frame_payload["received_time"] = self._now_sec()
        queue.append(frame_payload)
        self._last_accepted_ns[camera] = stamp_ns

    def _maybe_flush_partial(self) -> None:
        self._process_available_batches(allow_partial=True)

    def _oldest_frame_age(self) -> Optional[float]:
        return get_oldest_frame_age(self._frame_queues, self._now_sec())

    def _log_queue_depths(self, reason: str) -> None:
        do_log_queue_depths(
            self._frame_queues,
            reason,
            debug_queue_status=self._debug_queue_status,
            logger_debug=self.get_logger().debug,
        )

    def _drain_ready_batch(self, *, allow_partial: bool) -> bool:
        return drain_ready_batch(
            frame_queues=self._frame_queues,
            camera_order=self._camera_order,
            expected_batch=self._expected_batch,
            batch_timeout=self._batch_timeout,
            partial_min_cameras=self._partial_min_cameras,
            debug_queue_status=self._debug_queue_status,
            now_sec=self._now_sec,
            logger_debug=self.get_logger().debug,
            process_frame_batch=self._process_frame_batch,
            allow_partial=allow_partial,
            queue_debug_state=self._queue_debug_state,
            queue_debug_interval=self._queue_debug_interval,
        )

    def _process_available_batches(self, *, allow_partial: bool) -> None:
        if self._busy_event.is_set():
            return
        _ = self._drain_ready_batch(allow_partial=allow_partial)

    # ------------------------------------------------------------------
    # Mapping pipeline execution

    def _maybe_process_captions(self) -> None:
        # Worker may be enabled for Viser edits only; auto-captioning is separately controlled.
        maybe_process_captions(
            caption_manager=self._caption_manager,
            auto_caption_enabled=getattr(self, "_auto_caption_enabled", False),
            step_index=self._step_index,
            caption_start_step=self._caption_start_step,
            caption_step_interval=self._caption_step_interval,
            pending_caption_indices=self._pending_caption_indices,
        )

    def _process_frame_batch(self, batch_frames) -> None:
        if not batch_frames:
            return
        if self._busy_event.is_set():
            self.get_logger().debug("Mapping worker busy; deferring batch.")
            return

        self._busy_event.set()

        def _work():
            try:
                return self._run_mapping_batch(batch_frames)
            finally:
                self._busy_event.clear()

        fut = self._pool.submit(_work)
        try:
            fut.result(timeout=self._psb_timeout_s)
        except concurrent.futures.TimeoutError:
            self.get_logger().warn(f"Mapping batch exceeded {self._psb_timeout_s:.1f}s; skipping this batch.")
            fut.cancel()
            self._busy_event.clear()

    def _decode_batch(self, batch_frames) -> List[Dict[str, np.ndarray]]:
        decoded_batch: List[Dict[str, np.ndarray]] = []
        for frame in batch_frames:
            if not isinstance(frame, dict):
                continue
            bundle = frame.get("rgbd_msg")
            if bundle is None:
                decoded_batch.append(frame)
                continue
            if not isinstance(bundle, RGBDFrame):
                continue

            camera = str(frame.get("camera") or bundle.meta.camera or "").strip()
            if not camera:
                camera = normalize_camera_id(bundle.meta.camera) or "unknown"

            payload = self._assemble_frame_payload(camera, bundle)
            if payload is None:
                continue
            received_time = frame.get("received_time")
            if received_time is not None:
                payload["received_time"] = received_time
            decoded_batch.append(payload)

        return decoded_batch

    def _get_batch_info(self, decoded_batch, poses_world) -> Tuple[int, List[int], Dict[int, int]]:
        batch_size = len(decoded_batch)
        batch_camera_ids = [normalize_camera_id(frame.get("camera")) for frame in decoded_batch]
        # Offline frame sources (sens / npz / frames-json) populate ``source_ref``
        # so the eval viser can recover RGB without needing a saved JPEG copy.
        # Live ROS callbacks leave it empty (no notion of frame-id-in-source).
        batch_source_refs = [str(frame.get("source_ref") or "") for frame in decoded_batch]
        batch_image_ids = register_batch_images(
            self._scene_state,
            poses_world,
            camera_ids=batch_camera_ids,
            source_refs=batch_source_refs,
        )
        batch_image_lookup = {img_id: idx for idx, img_id in enumerate(batch_image_ids)}
        return batch_size, batch_image_ids, batch_image_lookup

    def _update_current_robot_position(self, poses_world: List[torch.Tensor]) -> None:
        if not poses_world:
            return
        positions: List[torch.Tensor] = []
        for pose in poses_world:
            if not isinstance(pose, torch.Tensor) or pose.numel() < 16:
                continue
            with contextlib.suppress(Exception):
                pos = pose[:3, 3].detach().to("cpu", dtype=torch.float32, copy=False).view(-1)[:3]
                if pos.numel() == 3 and torch.isfinite(pos).all():
                    positions.append(pos)
        if not positions:
            return
        if len(positions) == 1:
            robot_pos = positions[0]
        else:
            robot_pos = torch.stack(positions, dim=0).mean(dim=0)
        self._scene_state["current_robot_position"] = robot_pos

    def _segment_batch(self, colors, depths, depth_intrinsics) -> Optional[dict]:
        try:
            seg_outputs = self._segmenter(
                colors,
                depths,
                depth_intrinsics,
                offline_debug=self._offline_debug,
            )
        except Exception as exc:
            self.get_logger().error(f"Segmentation failed: {exc}")
            logger.opt(exception=exc).error("Segmentation failed")
            return None

        if isinstance(seg_outputs, list):
            self.get_logger().warn("Segmenter returned list output; flattening unsupported.")
            return None

        return seg_outputs

    def _normalize_and_transform_segmentation_to_world(self, seg_outputs, poses_world) -> Optional[dict]:
        seg_outputs = normalize_seg_outputs(
            seg_outputs, fallback_device=getattr(self._segmenter, "device", None)
        )
        transform_segmentation_to_world(seg_outputs, poses_world)
        return seg_outputs

    def _filter_segmentation_outputs(
        self,
        seg_outputs,
        colors,
        poses_world,
        debug_info,
        filter_events_log: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[dict]:
        def _n(seg) -> int:
            m = seg.get("means") if isinstance(seg, dict) else None
            return int(getattr(m, "shape", [0])[0]) if m is not None else 0

        # When tracing, attach a stable per-detection 'orig_idx' that survives
        # mask_seg_outputs (it preserves any tensor field of shape [N]). After
        # each filter we compute dropped_orig_idx via set-diff and look the
        # actual bbox / class / score / batch_id back up from the unfiltered
        # seg_outputs for the drop-heatmap visualizer.
        unfiltered_for_trace: Optional[dict] = None
        if filter_events_log is not None:
            n_raw = _n(seg_outputs)
            if n_raw > 0:
                # Build a dict view of the unfiltered seg_outputs that's stable
                # across filters; we look up bbox / class / score by orig_idx.
                unfiltered_for_trace = dict(seg_outputs)
                if "_orig_idx" not in seg_outputs:
                    seg_outputs = dict(seg_outputs)
                    means_dev = seg_outputs.get("means")
                    device = means_dev.device if isinstance(means_dev, torch.Tensor) else torch.device("cpu")
                    seg_outputs["_orig_idx"] = torch.arange(n_raw, device=device, dtype=torch.long)

        def _record(
            name: str,
            n_in: int,
            seg_out: dict,
            config: Dict[str, Any],
            seg_in: Optional[dict] = None,
        ) -> None:
            if filter_events_log is None:
                return
            n_out = _n(seg_out)
            event: Dict[str, Any] = {
                "name": name,
                "n_in": int(n_in),
                "n_out": int(n_out),
                "n_dropped": int(max(0, n_in - n_out)),
                "config": config,
            }
            # Drop details: who got dropped at this step? Diff orig_idx
            # before-vs-after this filter, then look up bbox/class/score
            # from the unfiltered batch.
            try:
                in_idx = (seg_in or {}).get("_orig_idx") if seg_in is not None else None
                out_idx = seg_out.get("_orig_idx")
                if isinstance(in_idx, torch.Tensor) and isinstance(out_idx, torch.Tensor):
                    in_set = set(in_idx.detach().to("cpu", dtype=torch.long).tolist())
                    out_set = set(out_idx.detach().to("cpu", dtype=torch.long).tolist())
                    dropped = sorted(in_set - out_set)
                    cap = 64
                    event["dropped_orig_idx"] = dropped[:cap]
                    if dropped and unfiltered_for_trace is not None:
                        names_seq = getattr(self._segmenter, "names", None) or []
                        boxes = unfiltered_for_trace.get("boxes_xyxy")
                        cids = unfiltered_for_trace.get("class_ids")
                        scores = unfiltered_for_trace.get("scores")
                        bids = unfiltered_for_trace.get("batch_ids")
                        npx = unfiltered_for_trace.get("num_pixels")
                        details = []
                        for o in dropped[:cap]:
                            d: Dict[str, Any] = {"orig_idx": int(o)}
                            if isinstance(boxes, torch.Tensor) and 0 <= o < boxes.shape[0]:
                                d["bbox"] = [float(x) for x in boxes[o].detach().to("cpu").tolist()[:4]]
                            if isinstance(cids, torch.Tensor) and 0 <= o < cids.shape[0]:
                                ci = int(cids[o].item())
                                d["class_id"] = ci
                                if 0 <= ci < len(names_seq):
                                    d["label"] = str(names_seq[ci])
                            if isinstance(scores, torch.Tensor) and 0 <= o < scores.shape[0]:
                                d["score"] = float(scores[o].item())
                            if isinstance(bids, torch.Tensor) and 0 <= o < bids.shape[0]:
                                d["batch_id"] = int(bids[o].item())
                            if isinstance(npx, torch.Tensor) and 0 <= o < npx.shape[0]:
                                d["num_pixels"] = int(npx[o].item())
                            details.append(d)
                        event["dropped_details"] = details
            except Exception as exc:  # noqa: BLE001 — never let tracing break the run
                event["drop_capture_error"] = repr(exc)
            filter_events_log.append(event)

        if self._filter_detections_touching_image_border_enabled:
            n_in = _n(seg_outputs)
            seg_in = seg_outputs
            seg_outputs = filter_detections_touching_image_border(
                seg_outputs,
                colors,
                margin_px=self._filter_touching_image_border_margin_px,
                min_kept_num_pixels=self._filter_touching_image_border_min_kept_num_pixels,
                max_area_fraction=self._filter_touching_image_border_max_area_fraction,
            )
            _record("border", n_in, seg_outputs,
                    {
                        "margin_px": int(self._filter_touching_image_border_margin_px),
                        "min_kept_num_pixels": int(self._filter_touching_image_border_min_kept_num_pixels),
                        "max_area_fraction": float(self._filter_touching_image_border_max_area_fraction),
                    },
                    seg_in=seg_in)
        self._add_debug_info(debug_info, "seg_outputs_filtered_touching_image_border", seg_outputs)
        if self._filter_detections_by_num_pixels_enabled:
            n_in = _n(seg_outputs)
            seg_in = seg_outputs
            seg_outputs = filter_detections_by_num_pixels(
                seg_outputs, min_num_pixels=self._filter_by_num_pixels_min
            )
            _record("num_pixels", n_in, seg_outputs,
                    {"min_pixels": int(self._filter_by_num_pixels_min)},
                    seg_in=seg_in)
        self._add_debug_info(debug_info, "seg_outputs_filtered_by_num_pixels", seg_outputs)
        if self._filter_detections_by_distance_enabled:
            n_in = _n(seg_outputs)
            seg_in = seg_outputs
            seg_outputs = filter_detections_by_distance(
                seg_outputs,
                poses_world,
                min_distance_m=self._filter_by_distance_min_m,
                max_distance_m=self._filter_by_distance_max_m,
            )
            _record(
                "distance",
                n_in,
                seg_outputs,
                {"min_m": float(self._filter_by_distance_min_m), "max_m": float(self._filter_by_distance_max_m)},
                seg_in=seg_in,
            )
        self._add_debug_info(debug_info, "seg_outputs_filtered_by_distance", seg_outputs)
        if self._filter_uninformative_yoloe_labels_enabled:
            n_in = _n(seg_outputs)
            seg_in = seg_outputs
            seg_outputs = filter_uninformative_yoloe_labels(
                seg_outputs,
                names=getattr(self._segmenter, "names", None) or [],
                uninformative_labels=self._UNINFORMATIVE_LABELS,
            )
            _record(
                "uninformative_labels",
                n_in,
                seg_outputs,
                {"n_blocked_labels": int(len(self._UNINFORMATIVE_LABELS))},
                seg_in=seg_in,
            )
        self._add_debug_info(debug_info, "seg_outputs_filtered_uninformative_yoloe_labels", seg_outputs)
        if self._filter_detections_duplicates_iou_enabled:
            n_in = _n(seg_outputs)
            seg_in = seg_outputs
            seg_outputs = filter_detections_duplicates_iou(seg_outputs, min_iou=self._filter_duplicates_iou_min)
            _record("duplicates_iou", n_in, seg_outputs,
                    {"min_iou": float(self._filter_duplicates_iou_min)},
                    seg_in=seg_in)
        self._add_debug_info(debug_info, "seg_outputs_filtered_duplicates_iou", seg_outputs)
        # Strip the trace-only field before returning so downstream code (caption
        # observations, neighbor lookup, fusion) doesn't see an unexpected key.
        if isinstance(seg_outputs, dict) and "_orig_idx" in seg_outputs:
            seg_outputs = {k: v for k, v in seg_outputs.items() if k != "_orig_idx"}
        return seg_outputs

    def _compute_caption_observations(self, seg_outputs, colors) -> Optional[list]:
        try:
            pad_ratio = self._caption_pad_ratio
            min_bbox_side = self._caption_min_bbox_side
            return compute_caption_observations(
                seg_outputs,
                colors,
                pad_ratio=pad_ratio,
                min_bbox_side=min_bbox_side,
                visual_prompt_mode=self._caption_visual_prompt_mode,
                context_thumbnail_width=self._caption_context_thumbnail_width,
                target_long_side=self._caption_target_long_side,
                mask_fill_alpha=self._caption_mask_fill_alpha,
                mask_outline_px=self._caption_mask_outline_px,
            )
        except Exception:
            return [None] * len(seg_outputs.get("means", []))

    def _add_debug_info(self, debug_info, key, value) -> None:
        if self._offline_debug:
            debug_info[key] = prepare_for_debugging(value)

    def _publish_detected_objects(self, detected_objects: Optional[DetectedObjects]) -> None:
        if detected_objects is None:
            return
        try:
            self._detected_objects_pub.publish(detected_objects)
        except Exception as exc:
            self.get_logger().warn(f"Failed to publish detected objects: {exc}")

    def _run_mapping_batch(self, batch_frames) -> Optional[dict]:
        debug_info = {}
        step_times: Dict[str, float] = {}

        # Tracer setup: emit scene_start on first batch, gather per-frame buckets.
        tracer = self._debug_tracer
        tracing = tracer is not None
        filter_events_log: List[Dict[str, Any]] = [] if tracing else None  # type: ignore[assignment]
        if tracing and not self._tracer_scene_started:
            try:
                tracer.start_scene(
                    self._tracer_scene_id or "unknown",
                    config={
                        "save_path": str(self._scene_state_save_path),
                        "feature_dim": int(getattr(self._segmenter, "feature_dim", 0) or 0),
                        "covisibility_enabled": bool(self._covisibility_enabled),
                        "filtering": {
                            "border_enabled": bool(self._filter_detections_touching_image_border_enabled),
                            "border_margin_px": int(self._filter_touching_image_border_margin_px),
                            "num_pixels_enabled": bool(self._filter_detections_by_num_pixels_enabled),
                            "num_pixels_min": int(self._filter_by_num_pixels_min),
                            "distance_enabled": bool(self._filter_detections_by_distance_enabled),
                            "distance_min_m": float(self._filter_by_distance_min_m),
                            "distance_max_m": float(self._filter_by_distance_max_m),
                            "uninformative_labels_enabled": bool(self._filter_uninformative_yoloe_labels_enabled),
                            "duplicates_iou_enabled": bool(self._filter_detections_duplicates_iou_enabled),
                            "duplicates_iou_min": float(self._filter_duplicates_iou_min),
                        },
                        "correspondence": {
                            "feature_sim_thresh": float(self._correspondence_feature_sim_thresh),
                            "hellinger_thresh": float(self._correspondence_hellinger_thresh),
                            "hellinger_match_floor_m2": float(self._correspondence_hellinger_match_floor_m2),
                            "max_merge_distance_m": float(self._correspondence_max_merge_distance_m),
                            "same_image_one_to_one": bool(self._correspondence_same_image_one_to_one),
                            "use_class_gate": bool(self._correspondence_use_class_gate),
                            "assignment_mode": str(self._correspondence_assignment_mode),
                        },
                    },
                )
                self._tracer_scene_started = True
            except Exception as exc:
                self.get_logger().warn(f"DebugTracer.start_scene failed: {exc}")

        # 1: decode RGBDFrame msgs → numpy dicts
        t0 = time.perf_counter()
        decoded_batch = self._decode_batch(batch_frames)
        if not decoded_batch:
            return None
        step_times["decode_batch"] = time.perf_counter() - t0

        # 2: prepare frames — unpack colors, depths, intrinsics, poses; register image IDs
        t0 = time.perf_counter()
        colors, depths, _, depth_intrinsics, poses_world = self._prepare_frames(decoded_batch)
        if not colors:
            return None
        self._update_current_robot_position(poses_world)
        batch_size, batch_image_ids, batch_image_lookup = self._get_batch_info(decoded_batch, poses_world)
        step_times["prepare_frames_and_batch_info"] = time.perf_counter() - t0

        if self._offline_debug:
            self._add_debug_info(debug_info, "colors", colors)
            self._add_debug_info(debug_info, "depths", depths)
            self._add_debug_info(debug_info, "poses_world", poses_world)
            if depths and depth_intrinsics:
                self._add_debug_info(debug_info, "depth_intrinsics", depth_intrinsics[0])
            self._add_debug_info(debug_info, "segmenter_mask_erosion_px", getattr(self._segmenter, "mask_erosion_px", 3))
            self._add_debug_info(debug_info, "segmenter_mahalanobis_thresh", getattr(self._segmenter, "mahalanobis_thresh", 2.0))

        # 3: segmentation — YOLOE/DINO → {means, cov6, features, masks, labels, batch_ids}
        t0 = time.perf_counter()
        seg_outputs = self._segment_batch(colors, depths, depth_intrinsics)
        if not seg_outputs:
            return None
        step_times["segment_batch"] = time.perf_counter() - t0

        # 4: project detections from camera space → world frame; filter noise
        t0 = time.perf_counter()
        seg_outputs_unfiltered = self._normalize_and_transform_segmentation_to_world(seg_outputs, poses_world)
        seg_outputs = self._filter_segmentation_outputs(
            seg_outputs_unfiltered,
            colors,
            poses_world,
            debug_info,
            filter_events_log=filter_events_log,
        )
        step_times["normalize_and_filter"] = time.perf_counter() - t0
        if self._offline_debug:
            self._add_debug_info(debug_info, "seg_outputs", seg_outputs_unfiltered)
            self._add_debug_info(debug_info, "seg_outputs_filtered", seg_outputs)

        # 5: publish per-frame detected objects (optional downstream consumers)
        t0 = time.perf_counter()
        detected_objects = build_detected_objects(seg_outputs)
        if self._detected_objects_pub:
            self._publish_detected_objects(detected_objects)
        step_times["detected_objects"] = time.perf_counter() - t0
        if self._offline_debug:
            self._add_debug_info(debug_info, "detected_objects", detected_objects)

        # 6: find candidate existing objects near each detection (Hellinger + feature similarity)
        t0 = time.perf_counter()
        db_for_neighbors = self._scene_state
        det_feats = seg_outputs.get("features")
        state_feats = self._scene_state.get("features")
        if isinstance(det_feats, torch.Tensor) and isinstance(state_feats, torch.Tensor):
            det_device = det_feats.device
            if state_feats.device != det_device:
                db_for_neighbors = self._build_neighbor_view(det_device)

        # Request the merge-funnel diagnostic when tracing — it's a couple of
        # extra reductions on tensors get_neighbors already builds, no extra
        # GPU passes. When not tracing, returns the legacy 2-tuple.
        neighbors_diag: Optional[Dict[str, int]] = None
        if tracing:
            neighbors, _, neighbors_diag = get_neighbors(
                seg_outputs,
                db_for_neighbors,
                active_mask=None,
                feature_sim_thresh=self._correspondence_feature_sim_thresh,
                hellinger_thresh=self._correspondence_hellinger_thresh,
                eps_cov=self._correspondence_hellinger_match_floor_m2,
                use_class_gate=self._correspondence_use_class_gate,
                return_diagnostics=True,
            )
        else:
            neighbors, _ = get_neighbors(
                seg_outputs,
                db_for_neighbors,
                active_mask=None,
                feature_sim_thresh=self._correspondence_feature_sim_thresh,
                hellinger_thresh=self._correspondence_hellinger_thresh,
                eps_cov=self._correspondence_hellinger_match_floor_m2,
                use_class_gate=self._correspondence_use_class_gate,
            )
        step_times["get_neighbors"] = time.perf_counter() - t0
        if self._offline_debug:
            self._add_debug_info(debug_info, "neighbors", neighbors)

        # 7: crop RGB patches around detections for caption worker
        t0 = time.perf_counter()
        num_detections = len(seg_outputs.get("means", []))
        detection_rgb_observations = self._compute_caption_observations(seg_outputs, colors)
        for obs in detection_rgb_observations:
            if isinstance(obs, dict) and "encoding" not in obs:
                obs["encoding"] = "rgb8"
        if not detection_rgb_observations:
            detection_rgb_observations = [None] * num_detections
        step_times["caption_observations"] = time.perf_counter() - t0
        if self._offline_debug:
            self._add_debug_info(debug_info, "detection_rgb_observations", detection_rgb_observations)

        # 8: update interactive viser visualization (optional)
        t0 = time.perf_counter()
        if self._viser_visualizer and self._viser_visualizer.enabled:
            try:
                with contextlib.suppress(Exception):
                    if self._covisibility_enabled:
                        means_state = self._scene_state.get("means")
                        if isinstance(means_state, torch.Tensor):
                            update_covisibility_filtered_adjacency(state=self._scene_state, covisibility_lock=self._covisibility_lock, hellinger_thresh=self._LOCAL_CAPTION_NEIGHBOR_HELLINGER_THRESH, limit=int(means_state.shape[0]))

                detection_rgb_observations = update_viser_visualization(
                    viser_visualizer=self._viser_visualizer,
                    colors=colors,
                    depths=depths,
                    intrinsics_batch=depth_intrinsics,
                    poses_world=poses_world,
                    scene_state=self._scene_state,
                    seg_outputs=seg_outputs,
                    neighbors=neighbors,
                    names=getattr(self._segmenter, "names", None),
                    existing_rgb_observations=detection_rgb_observations,
                )
            except Exception as exc:
                self.get_logger().warn(f"Viser visualization failed: {exc}")
        step_times["viser_visualization"] = time.perf_counter() - t0

        # 9: resolve neighbors → hard assignment: det_idx[i] = matched object OR -1 (new)
        t0 = time.perf_counter()
        detection_image_ids = compute_detection_image_ids(
            seg_outputs, batch_image_ids, num_detections
        )
        for det_i, image_id in enumerate(detection_image_ids):
            if image_id is None or det_i >= len(detection_rgb_observations):
                continue
            obs = detection_rgb_observations[det_i]
            if isinstance(obs, dict):
                obs["image_id"] = int(image_id)
        det_idx, obj_idx = resolve_correspondence(
            neighbors,
            self._scene_state["count"].shape[0],
            scene_state=self._scene_state,
            detection_image_ids=detection_image_ids,
            seg_outputs=seg_outputs,
            same_image_one_to_one=self._correspondence_same_image_one_to_one,
            assignment_mode=self._correspondence_assignment_mode,
        )
        same_image_conflicts = int(self._scene_state.pop("_last_same_image_assignment_conflicts", 0) or 0)
        step_times["correspondence"] = time.perf_counter() - t0
        if self._offline_debug:
            self._add_debug_info(debug_info, "det_idx", det_idx)
            self._add_debug_info(debug_info, "obj_idx", obj_idx)
            debug_info["same_image_assignment_conflicts"] = same_image_conflicts

        # 10: send caption crop requests to the async CaptionManager
        t0 = time.perf_counter()
        try:
            self._publish_local_captions(
                batch_frames=decoded_batch,
                poses_world=poses_world,
                seg_outputs=seg_outputs,
                det_idx=det_idx,
                neighbors=neighbors,
            )
        except Exception as exc:
            self.get_logger().warn(f"Failed to publish local captions: {exc}")
        step_times["local_captions"] = time.perf_counter() - t0

        # 11: save source frames for newly detected objects
        t0 = time.perf_counter()
        new_detection_mask = det_idx < 0
        frames_to_save: set[int] = set()
        if new_detection_mask.numel() and detection_image_ids:
            new_detection_indices = (
                torch.nonzero(new_detection_mask, as_tuple=False).view(-1).detach().to("cpu", copy=False).tolist()
            )
            for det_ind in new_detection_indices:
                image_id = detection_image_ids[det_ind] if det_ind < len(detection_image_ids) else None
                if image_id is not None:
                    frames_to_save.add(image_id)

        if self._image_saving_enabled and self._image_save_worker is not None and self._image_storage_dir is not None:
            max_saves = self._image_save_max_per_batch
            image_ext = ".h5" if self._image_storage_format == "h5" else ".jpg"
            for image_id in sorted(frames_to_save)[:max_saves]:
                batch_idx = batch_image_lookup.get(image_id)
                if batch_idx is None or batch_idx >= len(colors):
                    continue
                images_meta: List[ImageRecord] = self._scene_state.get("images", [])
                if image_id < 0 or image_id >= len(images_meta):
                    continue
                if images_meta[image_id].storage_path:
                    continue
                save_path = self._image_storage_dir / f"frame_{image_id:06d}{image_ext}"
                submitted = self._image_save_worker.submit(
                    colors[batch_idx],
                    save_path,
                    fmt=self._image_storage_format,
                    jpeg_max_width=self._image_preview_max_width,
                    jpeg_quality=self._image_preview_jpeg_quality,
                    on_success=lambda p, _image_id=image_id: mark_image_saved(self._scene_state, _image_id, p),
                    drop_if_full=True,
                )
                if not submitted:
                    dropped_now = self._image_save_worker.dropped_count()
                    if dropped_now != self._image_save_dropped_last:
                        self._image_save_dropped_last = dropped_now
                        self.get_logger().warn(
                            f"Image save queue full; dropped requests total={dropped_now} "
                            f"(queue_size={self._image_save_queue_size})"
                        )
        step_times["image_save"] = time.perf_counter() - t0

        # 12: update scene graph — fuse matched detections, append new objects
        prev_object_count = int(getattr(self._scene_state.get("means"), "shape", [0])[0])
        allow_new_objects = not self._stop_adding_object_event.is_set()

        # Snapshot pre-update means/cov6 for the tracer's numerical-health diff.
        means_before_np = None
        cov6_before_np = None
        matched_object_indices_for_trace: List[int] = []
        if tracing and prev_object_count > 0:
            try:
                means_before_np = (
                    self._scene_state["means"][:prev_object_count].detach().to("cpu", dtype=torch.float32).numpy().copy()
                )
                cov6_before_np = (
                    self._scene_state["cov6"][:prev_object_count].detach().to("cpu", dtype=torch.float32).numpy().copy()
                )
            except Exception:
                means_before_np = None
                cov6_before_np = None
            try:
                if isinstance(det_idx, torch.Tensor) and det_idx.numel() > 0:
                    matched_raw = det_idx[det_idx >= 0]
                    if matched_raw.numel() > 0 and isinstance(obj_idx, torch.Tensor) and obj_idx.numel() > 0:
                        valid = matched_raw < obj_idx.numel()
                        canonical = obj_idx[matched_raw[valid]]
                        matched_object_indices_for_trace = sorted({int(x) for x in canonical.detach().to("cpu").tolist()})
            except Exception:
                matched_object_indices_for_trace = []

        # Capture existing-objects snapshot before update_scene_graph_state mutates state.
        if self._offline_debug and prev_object_count > 0:
            self._add_debug_info(debug_info, "existing_objects_means", self._scene_state["means"][:prev_object_count].clone())
            self._add_debug_info(debug_info, "existing_objects_active", self._scene_state["active"][:prev_object_count].clone())
            self._add_debug_info(debug_info, "existing_objects_ids", self._scene_state["object_id"][:prev_object_count].clone())
            matched_obj_set: set[int] = set()
            if isinstance(det_idx, torch.Tensor) and det_idx.numel() > 0:
                matched_raw = det_idx[det_idx >= 0]
                if matched_raw.numel() > 0 and isinstance(obj_idx, torch.Tensor) and obj_idx.numel() > 0:
                    valid = matched_raw < obj_idx.numel()
                    canonical = obj_idx[matched_raw[valid]]
                    matched_obj_set = set(canonical.cpu().tolist())
            debug_info["matched_object_indices"] = sorted(matched_obj_set)
            if depth_intrinsics:
                self._add_debug_info(debug_info, "depth_intrinsics", depth_intrinsics[0])

        t0 = time.perf_counter()
        try:
            with self._covisibility_lock:
                _det_points_flat = seg_outputs.get("det_points_flat")
                _det_points_offsets = seg_outputs.get("det_points_offsets")
                update_info = update_scene_graph_state(
                    self._scene_state,
                    seg_outputs.get("means", torch.empty(0, 3)),
                    seg_outputs.get("cov6", torch.empty(0, 6)),
                    seg_outputs.get("features", torch.empty(0, self._segmenter.feature_dim)),
                    [""] * num_detections,
                    det_idx,
                    obj_idx,
                    rgb_observations=detection_rgb_observations,
                    detection_image_ids=detection_image_ids,
                    allow_new_objects=allow_new_objects,
                    det_points_flat=_det_points_flat if isinstance(_det_points_flat, torch.Tensor) else None,
                    det_points_offsets=_det_points_offsets if isinstance(_det_points_offsets, torch.Tensor) else None,
                    class_ids_d=seg_outputs.get("class_ids") if isinstance(seg_outputs.get("class_ids"), torch.Tensor) else None,
                    max_merge_distance_m=self._correspondence_max_merge_distance_m,
                )
                if update_info is not None:
                    update_info["same_image_assignment_conflicts"] = int(same_image_conflicts)
                if update_info is not None and detection_image_ids:
                    det_to_obj: List[Optional[int]] = [None] * num_detections
                    if det_idx.numel() > 0 and obj_idx.numel() > 0:
                        det_idx_cpu = det_idx.detach().to("cpu", dtype=torch.long)
                        obj_idx_cpu = obj_idx.detach().to("cpu", dtype=torch.long)
                        for det_i, raw_obj_idx in enumerate(det_idx_cpu.tolist()):
                            if 0 <= int(raw_obj_idx) < obj_idx_cpu.numel():
                                det_to_obj[det_i] = int(obj_idx_cpu[int(raw_obj_idx)].item())

                    new_det_indices = [
                        int(i)
                        for i, value in enumerate(det_idx.detach().to("cpu", dtype=torch.long).tolist())
                        if value < 0
                    ]
                    new_object_indices = [int(x) for x in (update_info.get("new_object_indices", []) or [])]
                    for det_i, obj_i in zip(new_det_indices, new_object_indices):
                        if 0 <= det_i < len(det_to_obj):
                            det_to_obj[det_i] = int(obj_i)

                    n_added = add_same_frame_cannot_links_from_detection_assignments(
                        self._scene_state,
                        detection_image_ids,
                        det_to_obj,
                    )
                    update_info["same_frame_cannot_links_added"] = int(n_added)
                    if self._object_mask_saving_enabled and self._object_mask_storage_dir is not None:
                        n_masks = register_detection_mask_observations(
                            self._scene_state,
                            seg_outputs,
                            detection_image_ids,
                            det_to_obj,
                            self._object_mask_storage_dir,
                            max_per_object=self._object_mask_observation_max_per_object,
                            detection_rgb_observations=detection_rgb_observations,
                            save_crops=self._object_mask_observation_save_crops,
                            crop_jpeg_quality=self._object_mask_observation_crop_jpeg_quality,
                        )
                        update_info["mask_observations_added"] = int(n_masks)

                # Log merged objects (objects marked inactive due to neighbor matching/union-find)
                merged_objects = update_info.get("merged_objects", [])
                if merged_objects:
                    for merge_info in merged_objects:
                        self.get_logger().debug(
                            "Object marked INACTIVE due to MERGE (neighbors/union-find): "
                            f"loser_idx={merge_info['loser_idx']} loser_id={merge_info['loser_id']} "
                            f"loser_caption={repr(merge_info['loser_caption'])} "
                            f"loser_pos={merge_info['loser_pos']} -> "
                            f"winner_idx={merge_info['winner_idx']} winner_id={merge_info['winner_id']} "
                            f"winner_caption={repr(merge_info['winner_caption'])} "
                            f"winner_pos={merge_info['winner_pos']} "
                            f"(step={self._step_index})"
                        )

                if self._covisibility_enabled:
                    (
                        self._covisibility_prev_per_camera,
                        self._covisibility_prev_unknown_visible,
                    ) = update_covisibility_from_batch(
                        state=self._scene_state,
                        det_idx=det_idx,
                        obj_idx=obj_idx,
                        detection_image_ids=detection_image_ids,
                        prev_object_count=prev_object_count,
                        allow_new_objects=allow_new_objects,
                        camera_neighbors=self._COVISIBILITY_CAMERA_NEIGHBORS,
                        history_batches=self._covisibility_history_batches,
                        prev_per_camera=self._covisibility_prev_per_camera,
                        prev_unknown_visible=self._covisibility_prev_unknown_visible,
                        logger_info=self.get_logger().info,
                    )
        except Exception as exc:
            self.get_logger().error(f"Failed to update scene graph: {exc}")
            logger.opt(exception=exc).error("Failed to update scene graph")
            return None
        step_times["update_scene_graph"] = time.perf_counter() - t0

        if self._offline_debug:
            post_object_count = int(getattr(self._scene_state.get("means"), "shape", [0])[0])
            self._add_debug_info(debug_info, "prev_object_count", prev_object_count)
            self._add_debug_info(debug_info, "post_object_count", post_object_count)
            self._add_debug_info(debug_info, "update_info", update_info)
            self._add_debug_info(debug_info, "allow_new_objects", allow_new_objects)
            det_idx_cpu = det_idx.detach().cpu() if isinstance(det_idx, torch.Tensor) else det_idx
            means_det = seg_outputs.get("means")
            fusion_summary = []
            for d_i in range(num_detections):
                matched = int(det_idx_cpu[d_i].item()) if hasattr(det_idx_cpu[d_i], "item") else int(det_idx_cpu[d_i])
                n_neighbors = (
                    int(neighbors[d_i].numel())
                    if d_i < len(neighbors) and isinstance(neighbors[d_i], torch.Tensor)
                    else 0
                )
                is_new = matched < 0
                mean_pos = (
                    means_det[d_i].detach().cpu().tolist()
                    if isinstance(means_det, torch.Tensor) and d_i < means_det.shape[0]
                    else None
                )
                fusion_summary.append({
                    "detection_idx": d_i,
                    "matched_object_idx": matched,
                    "num_neighbors": n_neighbors,
                    "is_new": is_new,
                    "detection_mean": mean_pos,
                })
            debug_info["fusion_summary"] = fusion_summary

        # 13: accumulate per-object detection category confidence scores
        t0 = time.perf_counter()
        try:
            self._update_object_detection_category_conf(
                seg_outputs=seg_outputs,
                det_idx=det_idx,
                obj_idx=obj_idx,
                prev_object_count=prev_object_count,
                allow_new_objects=allow_new_objects,
            )
        except Exception as exc:
            self.get_logger().warn(f"Failed to update per-object detection results: {exc}")
        step_times["update_category_conf"] = time.perf_counter() - t0

        # 14: trigger caption inference for new objects; persist scene graph snapshot
        t0 = time.perf_counter()
        new_indices = update_info.get("new_object_indices", []) if update_info else []
        if self._caption_manager.enabled and new_indices:
            self._pending_caption_indices.extend(self._canonical_caption_indices(new_indices))

        self._step_index += 1
        self._maybe_process_captions()
        self._maybe_persist_scene_graph_by_distance()
        self._maybe_persist_scene_graph_by_time()
        step_times["captions_and_persist"] = time.perf_counter() - t0

        # 14.5: update spatial regions
        t0 = time.perf_counter()
        self._maybe_update_regions()
        step_times["update_regions"] = time.perf_counter() - t0

        # 15: prune inactive / redundant objects from the scene graph
        t0 = time.perf_counter()
        self._maybe_prune_scene_graph()
        step_times["prune_scene_graph"] = time.perf_counter() - t0

        self._latest_visualization_payload = {
            "colors": colors,
            "depths": depths,
            "intrinsics": depth_intrinsics,
            "poses": poses_world,
            "step_index": int(self._step_index),
        }

        if self._timing_enabled and batch_size > 0:
            duration_s = sum(step_times.values())
            self._timing_sum_s += duration_s
            self._timing_image_count += batch_size
            self._timing_batch_count += 1
            now_mono = time.monotonic()
            if (now_mono - self._timing_last_log_mono) >= self._timing_log_interval_sec:
                avg_ms = (self._timing_sum_s / self._timing_image_count) * 1000.0
                batch_ms = (duration_s / batch_size) * 1000.0
                parts = " ".join(f"{k}={v * 1000:.1f}ms" for k, v in sorted(step_times.items()))
                segmentor_parts = " ".join(
                    f"{k}={v * 1000:.1f}ms" for k, v in sorted(seg_outputs.get("timings", {}).items())
                )
                self.get_logger().info(
                    f"Mapping timing: batch_images={batch_size} "
                    f"batch_ms_per_image={batch_ms:.2f} avg_ms_per_image={avg_ms:.2f} "
                    f"images={self._timing_image_count} | {parts} | {segmentor_parts}"
                )
                self._timing_last_log_mono = now_mono

            if getattr(self, "_timing_batch_done_pub", None) is not None:
                msg = Header()
                msg.stamp = self.get_clock().now().to_msg()
                self._timing_batch_done_pub.publish(msg)

        # ── Emit a per-frame trace event ────────────────────────────────────
        if tracing and tracer is not None:
            try:
                seg_summary = summarize_segmentation(
                    seg_outputs_unfiltered, names=getattr(self._segmenter, "names", None) or None,
                )
                seg_summary_filtered = summarize_segmentation(
                    seg_outputs, names=getattr(self._segmenter, "names", None) or None,
                )
                filt_summary = summarize_filtering(filter_events_log or [])
                neigh_summary = summarize_neighbors(neighbors, seg_outputs, self._scene_state)
                # Attach the merge-funnel counters from get_neighbors so the
                # trace can show (per frame) "how many dets had no candidate
                # at all" vs "had candidates but lost to Hellinger".
                if neighbors_diag is not None:
                    neigh_summary["funnel"] = dict(neighbors_diag)
                corr_summary = summarize_correspondence(
                    det_idx,
                    obj_idx,
                    prev_object_count=prev_object_count,
                    update_info=update_info,
                )
                corr_summary["same_image_assignment_conflicts"] = int(same_image_conflicts)
                gauss_summary = summarize_gaussian_update(
                    means_before=means_before_np,
                    cov6_before=cov6_before_np,
                    state_after=self._scene_state,
                    update_info=update_info,
                    matched_object_indices=matched_object_indices_for_trace,
                )
                voxel_summary = summarize_voxel_cloud(self._scene_state)
                state_summary = state_digest(self._scene_state)
                # Record per-camera image dims so the drop-heatmap visualizer
                # can render bbox centers in pixel space without guessing.
                image_dims: List[Dict[str, int]] = []
                try:
                    for img in (colors or []):
                        if isinstance(img, torch.Tensor) and img.ndim >= 2:
                            shape = tuple(int(s) for s in img.shape)
                            if len(shape) == 2:
                                image_dims.append({"h": shape[0], "w": shape[1]})
                            elif len(shape) == 3 and shape[-1] in (1, 3, 4):
                                image_dims.append({"h": shape[0], "w": shape[1]})
                            elif len(shape) == 3 and shape[0] in (1, 3, 4):
                                image_dims.append({"h": shape[1], "w": shape[2]})
                except Exception:
                    image_dims = []
                tracer.record_frame({
                    "step_idx": int(self._step_index - 1),  # _step_index was incremented before this point
                    "batch_size": int(batch_size),
                    "batch_image_ids": list(batch_image_ids) if batch_image_ids else [],
                    "image_dims": image_dims,
                    "timing_ms": {k: float(v) * 1000.0 for k, v in step_times.items()},
                    "state_after": state_summary,
                    "segmentation": seg_summary,
                    "segmentation_post_filter": seg_summary_filtered,
                    "filtering": filt_summary,
                    "neighbors": neigh_summary,
                    "correspondence": corr_summary,
                    "gaussian_update": gauss_summary,
                    "voxel_cloud": voxel_summary,
                })
            except Exception as exc:
                self.get_logger().warn(f"DebugTracer.record_frame failed: {exc}")

        return debug_info

    def export_visualization_snapshot(self) -> Optional[dict]:
        """Build a CPU replay snapshot for offline Viser video capture."""

        payload = getattr(self, "_latest_visualization_payload", None)
        if not payload:
            return None
        try:
            from scene_graph.offline.viser_recording import build_visualization_snapshot

            return build_visualization_snapshot(
                scene_state=self._scene_state,
                colors=payload.get("colors") or [],
                depths=payload.get("depths") or [],
                intrinsics=payload.get("intrinsics") or [],
                poses=payload.get("poses") or [],
                step_index=int(payload.get("step_index", self._step_index)),
            )
        except Exception as exc:
            self.get_logger().warn(f"Failed to export visualization snapshot: {exc}")
            return None

    # ------------------------------------------------------------------
    # Viser interactive editing callbacks

    def _resolve_object_index_by_id(self, object_id: int) -> Optional[int]:
        """Map an object_id to its row index in scene_state."""
        return resolve_object_index_by_id(self._scene_state, object_id)

    def _viser_edit_caption(self, object_id: int, new_caption: str) -> tuple[bool, str]:
        try:
            with self._viser_edit_lock:
                if self._worker is None or not hasattr(self._worker, "request_caption_text_embeddings"):
                    return False, "Caption embedding backend unavailable on this node"
                ok, msg = edit_caption(
                    self._scene_state,
                    object_id,
                    new_caption,
                    request_embeddings=lambda texts: self._worker.request_caption_text_embeddings(
                        texts, timeout_s=30.0, normalize=True
                    ),
                )
                if not ok:
                    return False, msg
                self._force_publish_and_snapshot()
                with contextlib.suppress(Exception):
                    self._save_scene_state_now(reason="viser_edit_caption")
                return True, ""
        except Exception as exc:
            self.get_logger().error(f"Failed to edit caption for object {object_id}: {exc}")
            return False, str(exc)

    def _viser_delete_object(self, object_id: int) -> tuple[bool, str]:
        try:
            with self._viser_edit_lock:
                ok, msg = delete_object(
                    self._scene_state,
                    object_id,
                    step_index=getattr(self, "_step_index", -1),
                    logger_info=self.get_logger().info,
                )
                if not ok:
                    return False, msg
                self._force_publish_and_snapshot()
                with contextlib.suppress(Exception):
                    self._save_scene_state_now(reason="viser_delete_object")
                return True, ""
        except Exception as exc:
            self.get_logger().error(f"Failed to delete object {object_id}: {exc}")
            return False, str(exc)

    def _viser_toggle_lock(self, object_id: int) -> tuple[bool, str]:
        try:
            with self._viser_edit_lock:
                ok, msg = toggle_lock(
                    self._scene_state,
                    object_id,
                    logger_info=self.get_logger().info,
                )
                if not ok:
                    return False, msg
                self._force_publish_and_snapshot()
                with contextlib.suppress(Exception):
                    self._save_scene_state_now(reason="viser_toggle_lock")
                return ok, msg
        except Exception as exc:
            self.get_logger().error(f"Failed to toggle lock for object {object_id}: {exc}")
            return False, str(exc)

    def _viser_save_all(self) -> tuple[bool, str]:
        """Save scene_state.pt, scene_graph.json, and snapshots."""
        try:
            # Save pt
            self._save_scene_state_now(reason="viser_save_all")
            # Save scene_graph.json + snapshot
            self._force_publish_and_snapshot()
            return True, ""
        except Exception as exc:
            self.get_logger().error(f"Failed to save scene state: {exc}")
            return False, str(exc)

    def _viser_add_object(
        self,
        caption: str,
        location_xyz: Sequence[float],
        views: Sequence[Sequence[float]],
        image_path: Optional[str] = None,
    ) -> tuple[bool, str]:
        try:
            with self._viser_edit_lock:
                error, location_arr, caption_text, parsed_views, uploaded_image_rgb = (
                    validate_add_object_inputs(caption, location_xyz, views, image_path)
                )
                if error is not None:
                    return False, error

                state = self._scene_state
                for key in ("means", "cov6", "features", "count", "active", "class_ids", "object_id"):
                    if not isinstance(state.get(key), torch.Tensor):
                        return False, "Scene state tensors are not initialized."
                means = state["means"]
                if means.ndim != 2 or means.shape[1] != 3:
                    return False, "Scene state means tensor has invalid shape."
                if state["cov6"].ndim != 2 or state["cov6"].shape[1] != 6:
                    return False, "Scene state cov6 tensor has invalid shape."
                if state["features"].ndim != 2:
                    return False, "Scene state features tensor has invalid shape."

                caption_embedding: List[float] = []
                embedding_error: Optional[str] = None
                if self._worker is not None and hasattr(self._worker, "request_caption_text_embeddings"):
                    try:
                        ok, err, vecs = self._worker.request_caption_text_embeddings(
                            [caption_text], timeout_s=30.0, normalize=True
                        )
                        if ok and vecs and isinstance(vecs[0], list):
                            caption_embedding = [float(x) for x in vecs[0]]
                        elif not ok:
                            embedding_error = str(err or "unknown_error")
                    except Exception as exc:
                        embedding_error = str(exc)
                else:
                    embedding_error = "backend unavailable"

                image_path_text = str(image_path or "").strip()
                uploaded_image_resolved_path = ""
                if image_path_text:
                    cand = Path(image_path_text).expanduser()
                    if not cand.is_absolute():
                        cand = (Path.cwd() / cand).resolve()
                    uploaded_image_resolved_path = str(cand)

                n_old, next_object_id, manual_image_ids = add_object_to_scene_state(
                    state,
                    caption_text=caption_text,
                    location_arr=location_arr,
                    parsed_views=parsed_views,
                    uploaded_image_rgb=uploaded_image_rgb,
                    uploaded_image_resolved_path=uploaded_image_resolved_path,
                    caption_embedding=caption_embedding,
                    embedding_error=embedding_error,
                )

                self.get_logger().info(
                    "Object added from Viser UI: "
                    f"idx={n_old} object_id={next_object_id} caption={caption_text!r} "
                    f"mean={location_arr.astype(np.float32).tolist()} viewpoints={manual_image_ids}"
                )

                self._force_publish_and_snapshot()
                with contextlib.suppress(Exception):
                    self._save_scene_state_now(reason="viser_add_object")

                image_note = " + uploaded image" if uploaded_image_rgb is not None else ""
                if embedding_error:
                    return (
                        True,
                        (
                            f"✓ Added object {next_object_id}{image_note} "
                            f"(caption embedding unavailable: {embedding_error})."
                        ),
                    )
                return True, f"✓ Added object {next_object_id}{image_note}."
        except Exception as exc:
            self.get_logger().error(f"Failed to add object from Viser UI: {exc}")
            return False, str(exc)

    def _force_publish_and_snapshot(self) -> None:
        """Immediately publish scene graph JSON and take a snapshot."""
        try:
            new_version, snapshot_ptr, _ = write_snapshot_and_json(
                scene_state=self._scene_state,
                snapshot_dir=self._scene_graph_snapshot_dir,
                snapshot_version=self._scene_graph_snapshot_version,
                snapshot_lock=self._scene_graph_snapshot_lock,
                snapshot_save_enabled=self._scene_graph_snapshot_save_enabled,
                json_save_enabled=bool(getattr(self, "_scene_graph_json_save_enabled", False)),
                json_save_path=str(getattr(self, "_scene_graph_json_save_path", "") or ""),
                covisibility_enabled=self._covisibility_enabled,
                covisibility_lock=self._covisibility_lock,
                hellinger_thresh=self._LOCAL_CAPTION_NEIGHBOR_HELLINGER_THRESH,
                get_stamp_msg=lambda: self.get_clock().now().to_msg(),
                logger_info=self.get_logger().info,
                logger_warn=self.get_logger().warn,
            )
            self._scene_graph_snapshot_version = new_version
            if snapshot_ptr:
                self._last_scene_graph_snapshot_ptr = snapshot_ptr
            self.get_logger().info("✓ Scene graph JSON published/saved")
        except Exception as exc:
            self.get_logger().warn(f"✗ Failed to publish/snapshot: {exc}")

    # ------------------------------------------------------------------
    # Shutdown/cleanup

    def _request_shutdown(self, reason: str) -> None:
        if self._shutdown_requested_event.is_set():
            return
        self._shutdown_reason = str(reason or "requested")
        self._shutdown_requested_event.set()

    def _maybe_shutdown(self) -> None:
        if not self._shutdown_requested_event.is_set():
            return
        with self._shutdown_lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True

        with contextlib.suppress(Exception):
            self.get_logger().info(f"Shutdown requested ({self._shutdown_reason}); requesting rclpy shutdown (async)")

        def _do_shutdown() -> None:
            fn = getattr(rclpy, "try_shutdown", None) or getattr(rclpy, "shutdown", None)
            if fn is None:
                return
            with contextlib.suppress(Exception):
                fn()

        threading.Thread(target=_do_shutdown, daemon=True, name="StreamingMapperShutdown").start()
        with contextlib.suppress(Exception):
            self._shutdown_timer.cancel()

    def _wait_for_save_idle(self, timeout_sec: float) -> None:
        wait_for_save_idle(getattr(self, "_busy_event", None), timeout_sec)

    def _save_scene_state_now(self, *, reason: str, wait_for_idle: bool = True) -> Path:
        return do_save_scene_state(
            reason=reason,
            wait_for_idle=wait_for_idle,
            scene_state_save_path=self._scene_state_save_path,
            scene_state=self._scene_state,
            segmenter_feature_dim=self._segmenter.feature_dim,
            save_observations=self._scene_state_save_observations,
            observation_view_limit_raw=self._scene_state_save_observation_view_limit,
            busy_event=getattr(self, "_busy_event", None),
            busy_timeout_sec=self._scene_state_save_wait_busy_timeout_sec,
        )

    def _handle_siglip2_text_embed(
        self,
        request: Siglip2TextEmbed.Request,
        response: Siglip2TextEmbed.Response,
    ) -> Siglip2TextEmbed.Response:
        texts = [str(text or "").strip() for text in list(request.texts or [])]
        if not texts:
            response.ok = False
            response.error = "bad_request_empty_texts"
            response.dim = 0
            response.n_texts = 0
            response.embeddings_flat = []
            return response
        if len(texts) > self._siglip2_text_embed_max_texts:
            response.ok = False
            response.error = f"bad_request_too_many_texts:{len(texts)}>{self._siglip2_text_embed_max_texts}"
            response.dim = 0
            response.n_texts = int(len(texts))
            response.embeddings_flat = []
            return response
        if any(not text for text in texts):
            response.ok = False
            response.error = "bad_request_empty_text_item"
            response.dim = 0
            response.n_texts = int(len(texts))
            response.embeddings_flat = []
            return response

        worker_obj = getattr(self._caption_manager, "worker", None)
        if worker_obj is None or not hasattr(worker_obj, "request_siglip2_text_embeddings"):
            response.ok = False
            response.error = "temporarily_unavailable"
            response.dim = 0
            response.n_texts = int(len(texts))
            response.embeddings_flat = []
            return response

        try:
            ok, error, vectors = worker_obj.request_siglip2_text_embeddings(
                texts,
                normalize=bool(request.normalize),
                timeout_s=self._siglip2_text_embed_timeout_sec,
                client_id=int(request.client_id),
            )
        except Exception as exc:
            ok = False
            error = f"embed_failed:{exc}"
            vectors = []

        if not ok:
            response.ok = False
            response.error = str(error or "embed_failed")
            response.dim = 0
            response.n_texts = int(len(texts))
            response.embeddings_flat = []
            return response

        dim = 0
        for vec in vectors:
            if isinstance(vec, (list, tuple)) and len(vec) > 0:
                dim = int(len(vec))
                break
        if dim <= 0:
            response.ok = False
            response.error = "embed_failed_empty_vectors"
            response.dim = 0
            response.n_texts = int(len(texts))
            response.embeddings_flat = []
            return response

        matrix = np.zeros((len(texts), dim), dtype=np.float32)
        for row_idx, vec in enumerate(vectors):
            if not isinstance(vec, (list, tuple)):
                continue
            if len(vec) == dim:
                matrix[row_idx] = np.asarray(vec, dtype=np.float32)

        response.ok = True
        response.error = ""
        response.dim = int(dim)
        response.n_texts = int(len(texts))
        response.embeddings_flat = matrix.reshape(-1).astype(np.float32).tolist()
        return response

    def _handle_save_scene_state(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        with self._scene_state_save_lock:
            try:
                saved_path = self._save_scene_state_now(reason="service_save_scene_state")
                response.success = True
                response.message = str(saved_path)
            except Exception as exc:
                response.success = False
                response.message = str(exc)
        return response

    def _handle_save_and_shutdown(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        with self._scene_state_save_lock:
            try:
                saved_path = self._save_scene_state_now(reason="service_save_and_shutdown")
                response.success = True
                response.message = str(saved_path)
            except Exception as exc:
                response.success = False
                response.message = str(exc)
        self._request_shutdown("service_save_and_shutdown")
        return response

    def destroy_node(self) -> bool:
        if self._timing_enabled and self._timing_image_count > 0:
            avg_ms = (self._timing_sum_s / self._timing_image_count) * 1000.0
            self.get_logger().info(
                f"Mapping timing summary: images={self._timing_image_count} "
                f"batches={self._timing_batch_count} avg_ms_per_image={avg_ms:.2f}"
            )
        with contextlib.suppress(Exception):
            if self._caption_manager and self._caption_manager.enabled:
                try:
                    if self._pending_caption_indices:
                        self._caption_manager.enqueue_objects(self._pending_caption_indices)
                        self._pending_caption_indices.clear()
                    self._caption_manager.wait_until_idle(
                        timeout=self._caption_drain_timeout_sec, poll_interval=0.05,
                    )
                    self._caption_manager.drain_results()
                finally:
                    self._caption_manager.shutdown_worker()
                    self._caption_manager.drain_results()
        with contextlib.suppress(Exception):
            if self._scene_state_save_on_shutdown and self._scene_state_save_path:
                with self._scene_state_save_lock:
                    try:
                        saved_path = self._save_scene_state_now(reason="shutdown")
                        self.get_logger().info(f"Saved scene state to {saved_path}")
                    except Exception as exc:
                        self.get_logger().warn(f"Failed to save scene state to {self._scene_state_save_path}: {exc}")
        with contextlib.suppress(Exception):
            if self._image_save_worker is not None:
                self._image_save_worker.close(
                    timeout_sec=self._image_save_close_timeout_sec,
                    drain=self._image_save_drain_on_shutdown,
                )

        # Close debug tracer (write scene_end + flush JSONL).
        with contextlib.suppress(Exception):
            tracer = getattr(self, "_debug_tracer", None)
            if tracer is not None and getattr(self, "_tracer_scene_started", False):
                try:
                    tracer.end_scene({
                        "final_state": state_digest(self._scene_state),
                        "final_voxel_cloud": summarize_voxel_cloud(self._scene_state),
                        "final_step_index": int(getattr(self, "_step_index", 0)),
                    })
                finally:
                    tracer.close()

        return super().destroy_node()

    def _canonical_caption_indices(self, object_indices: List[int]) -> List[int]:
        """
        Resolve requested caption indices onto canonical (post-merge) active objects, and
        avoid enqueueing caption work for objects that already have captions.
        """
        state = self._scene_state
        object_ids = state.get("object_id")
        if object_ids is None:
            return []
        id_redirect = state.get("id_redirect") or {}
        active_flags = state.get("active")
        captions = state.get("object_caption", []) or []

        def _resolve_oid(oid: int) -> int:
            seen: set[int] = set()
            cur = int(oid)
            while True:
                if cur in seen:
                    break
                seen.add(cur)
                nxt = id_redirect.get(cur)
                if nxt is None:
                    break
                try:
                    nxt_int = int(nxt)
                except Exception:
                    break
                if nxt_int == cur:
                    break
                cur = nxt_int
            return cur

        def _find_index(oid: int) -> Optional[int]:
            if hasattr(object_ids, "nonzero"):
                matches = (object_ids == oid).nonzero(as_tuple=False)
                if matches is not None and matches.numel() > 0:
                    return int(matches.view(-1)[0].item())
            try:
                return list(object_ids).index(oid)
            except ValueError:
                return None

        out: List[int] = []
        seen_idx: set[int] = set()
        for idx in object_indices:
            try:
                idx_int = int(idx)
            except Exception:
                continue
            if idx_int < 0 or idx_int >= len(object_ids):
                continue
            raw_oid = object_ids[idx_int]
            try:
                oid_int = int(raw_oid.item()) if hasattr(raw_oid, "item") else int(raw_oid)
            except Exception:
                continue
            canonical_oid = _resolve_oid(oid_int)
            canonical_idx = _find_index(canonical_oid)
            if canonical_idx is None:
                continue
            if active_flags is not None and canonical_idx < len(active_flags):
                try:
                    is_active = (
                        bool(active_flags[canonical_idx].item())
                        if hasattr(active_flags[canonical_idx], "item")
                        else bool(active_flags[canonical_idx])
                    )
                except Exception:
                    is_active = False
                if not is_active:
                    continue
            existing_caption = ""
            if canonical_idx < len(captions):
                try:
                    existing_caption = str(captions[canonical_idx] or "").strip()
                except Exception:
                    existing_caption = ""
            if existing_caption:
                continue
            if canonical_idx in seen_idx:
                continue
            seen_idx.add(canonical_idx)
            out.append(canonical_idx)
        return out


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = StreamingMapper()

    def _handle_signal(signum: int, _frame: object) -> None:
        with contextlib.suppress(Exception):
            node.get_logger().info(f"Received signal {signum}; shutting down.")
        with contextlib.suppress(Exception):
            node._request_shutdown(f"signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(Exception):
            signal.signal(sig, _handle_signal)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        with contextlib.suppress(KeyboardInterrupt):
            executor.spin()
    finally:
        with contextlib.suppress(Exception):
            executor.shutdown()
        node.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.shutdown()


if __name__ == "__main__":
    main()
