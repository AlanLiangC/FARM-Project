#!/usr/bin/env python3
"""Run text-prompt YOLOE visual search on Spot cameras."""

from __future__ import annotations

import contextlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Transform
from scene_graph.camera_config import CAMERA_CONFIG
from mapping.lib.paths import default_adaptation_results_dir, default_scene_graph_snapshot_dir
from scene_graph.runtime_paths import find_model_file
from mapping_msgs.msg import (
    FrameMetadata,
    LocalCaption,
    RGBDFrame,
    VisualSearchResult,
    VisualSearchResultArray,
)
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Header, String
from ultralytics import YOLOE

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for device probing
    torch = None


@dataclass
class GoalQuery:
    object_id: int
    description: str


@dataclass
class GoalState:
    goals: List[GoalQuery]
    conf: float
    iou: float
    imgsz: int


@dataclass
class AdaptationCrop:
    confidence: float
    image_bgr: np.ndarray
    source_camera: str
    stamp_sec: int
    stamp_nanosec: int


@dataclass
class AdaptationObject:
    detection_id: int
    label: str
    confidence_mean: float
    confidence_count: int
    position_mean: np.ndarray
    top_crops: List[AdaptationCrop] = field(default_factory=list)
    viewpoints: List[List[float]] = field(default_factory=list)
    viewpoint_camera_ids: List[str] = field(default_factory=list)
    top_crop_paths: List[str] = field(default_factory=list)
    full_image_path: str = ""
    full_image_camera_id: str = ""
    best_full_conf: float = -1.0
    # Used to avoid repeated disk writes when top crops did not change.
    last_saved_top_sig: Tuple[Tuple[float, int, int], ...] = field(default_factory=tuple)


class VisualSearchYOLOETextPromptNode(Node):
    """ROS2 node that publishes VisualSearchResultArray from text-prompt YOLOE."""

    _MODE_VALIDATION = "validation"
    _MODE_ADAPTATION = "adaptation"
    _SUPPORTED_MODES = (_MODE_VALIDATION, _MODE_ADAPTATION)
    _CAMERA_IMAGE_ROTATION_CW = {
        "head_left": 90,
        "head_right": 90,
        "right": 180,
    }
    _REAR_LEG_MASK_HEIGHT_PX = 320
    _REAR_LEG_MASK_WIDTH_PX = 150
    _DEFAULT_BBOX_BORDER_FILTER_PX = 3.0
    _DEFAULT_ADAPTATION_MERGE_DISTANCE_M = 0.5
    _DEFAULT_ADAPTATION_CROP_PAD_RATIO = 0.3
    _DEFAULT_ADAPTATION_CROP_MIN_SIDE_PX = 96
    _DEFAULT_ADAPTATION_MAX_MASK_POINTS = 4096
    _DEFAULT_ADAPTATION_MANIFEST_INTERVAL_SEC = 1.0
    _DEFAULT_ADAPTATION_MAX_VIEWPOINTS = 8
    _DISTRACTOR_TEXTS = [
        "wall",
        "floor",
        "ceiling",
        "ground",
        "road",
        "sidewalk",
        "grass",
        "tree",
        "sky",
        "building facade",
        "stone",
        "shadow",
        "reflection",
        "glare",
        "other object",
    ]

    _HIKING_OBJECTS = [
        "an umbrella",
        "rain boots",
        "hiking boots",
        "a raincoat",
        "a jacket",
        "leather shoes",
        "lettuce",
        "eggplant",
        "black tux",
        "an apple",
        "a peach",
        "a pear",
        "a tomato",
        "sneakers",
        "flip-flops",
        "a measuring tape",
        "a protractor",
        "a ruler",
        "a hammer",
        "a wrench",
        "a screwdriver",
        "a drawing compass",
        "an electric drill",
        "a toolbox",
        "pliers",
        "a refrigerator",
        "a wallet",
        "a backpack",
        "a square panel on the ground",
        "a safe",
        "an ironing board",
        "an electric iron",
        "stacks of folded clothes",
        "a banana",
        "a water bottle",
        "a soda can",
        "a dish",
        "reading glasses",
        "sunglasses",
        "bread",
        "a key",
        "a paint bucket",
        "a spray paint can",
        "a bowl of fruit",
        "a basket of strawberries",
        "a step stool",
        "an igloo lunchbox",
        "a food container",
        "a juice bottle",
        "a jar",
        "a coffee machine",
        "a mug",
        "a coffee bag",
        "a mountaineering bag",
        "a toaster",
        "a treasure box",
        "a phone charger",
        "pants",
        "a clothes hanger",
        "a tie",
        "a medicine bottle",
        "pills",
        "a tool chest",
        "a toothbrush",
        "a watch",
        "a piece of cheese",
        "a phone",
        "a camera",
        "socks",
        "a whistle",
        "a sunscreen bottle",
        "a duck",
        "a trash can",
        "a cup",
        "a first-aid kit",
        "a dog",
        "a cat",
        "a hospital",
        "a warehouse",
        "a police station",
        "a pallet",
        "a pallet jack",
        "a delivery truck",
        "a safety cone",
        "a breaker box",
        "a switch panel",
        "a fire station",
        "a fire hydrant",
        "a park gazebo",
        "a vending machine",
        "a mailbox",
        "a restroom",
        "a restaurant",
        "a church",
        "a bank",
        "a school",
        "a bench",
        "a bird",
        "a sleeping bag",
        "a sleeping pad",
        "a water jug",
        "paper towels",
        "a yurt",
        "a hammock",
        "a trail sign",
        "a battery",
        "a cabin",
        "a shelter",
        "a tarp",
        "a stone fireplace",
        "a teapot",
        "a cooler box",
        "ice cubes",
        "a brownie",
        "a compass",
        "firewood",
        "a bollard",
        "a roadside post",
        "a wall button",
        "a colored button",
        "a painting",
        "a hat",
        "a headlamp",
        "a lamp",
        "a hiking pole",
        "telegraph pole",
        "a notebook",
        "a wood block",
        "a fork",
        "a barber's pole",
        "a water dispenser",
        "a sponge",
        "a chair",
        "a tent",
        "an oven",
        "a microwave",
        "a stove",
        "an outdoor grill",
        "a kettle",
        "a pan",
        "a plate",
        "a bowl",
        "utensils",
        "a food can",
        "a towel",
        "a hair dryer",
        "a computer",
        "a shelf",
        "a laundry basket",
        "a heater",
        "a fan",
        "scissors",
        "a stapler",
        "a folder",
        "a calculator",
        "a paper clip",
        "a ladder",
        "a helmet",
        "a safety vest",
        "goggles",
        "a campfire",
        "a fire pit",
        "a canteen",
        "a thermos bottle",
        "a flask",
        "a picnic basket",
        "a picnic blanket",
        "a picnic table",
        "a bicycle",
        "a map",
        "a lantern",
        "a flashlight",
        "an axe",
        "a shovel",
        "a rope",
        "a knife",
        "a matchbox",
        "a walking cane",
        "a glove",
        "a lake",
        "a radio",
        "leaf",
        "grove",
        "corn field",
        "spear",
        "seaweed",
        "flare",
        "reed",
        "weed",
        "plantation",
        "hedge",
        "green grass",
        "green grass straw",
        "yellow grass straw",
        "yellow corn stalk",
        "green corn stalk",
    ]

    _INSPECTION_OBJECTS = [
        # --- Provided Items ---
        "backpack",
        "barrier",
        "berries",
        "berry tree",
        "button apparatus",
        "cable reel",
        "car key",
        "desk",
        "dirty clothes",
        "drill",
        "laptop",
        "measuring tape",
        "pallet",
        "pipes",
        "safety goggles",
        "stool",
        "strawberries",
        "water tank",
        "toolbox",
        "truck",
        "car",
        # --- Pipeline & Facility Infrastructure (From Map) ---
        "spherical storage tank",
        "cylindrical silo",
        "cooling tower",
        "smoke stack",
        "refinery column",
        "heat exchanger",
        "overhead pipe rack",
        "pressure vessel",
        "containment wall",
        "catwalk",
        "industrial staircase",
        "vent stack",
        "flange",
        "valve wheel",
        "pressure gauge",
        "elbow joint",
        # --- Power & Control Systems ---
        "control panel",
        "breaker box",
        "junction box",
        "transformer substation",
        "electric motor",
        "generator",
        "extension cord",
        "battery pack",
        "solar panel",
        "antenna",
        "utility pole",
        "emergency stop button",
        # --- Maintenance Tools & Equipment ---
        "pipe wrench",
        "adjustable wrench",
        "multimeter",
        "thermal camera",
        "flashlight",
        "headlamp",
        "grease gun",
        "oil can",
        "wire brush",
        "torque wrench",
        "air compressor",
        "sledgehammer",
        "spirit level",
        "calipers",
        "marking paint can",
        "ladder",
        "scaffolding",
        # --- Safety & Logistics ---
        "hard hat",
        "safety vest",
        "work gloves",
        "ear muffs",
        "respirator",
        "fire extinguisher",
        "first aid kit",
        "safety cone",
        "caution tape",
        "hazmat placard",
        "bollard",
        "pallet jack",
        "forklift",
        "oil drum",
        "gas cylinder",
        "shipping container",
        "security fence",
        "sliding gate",
        # --- Site Documentation & Personal Items ---
        "clipboard",
        "permanent marker",
        "notebook",
        "tablet",
        "radio transceiver",
        "blueprint",
        "map",
        "id badge",
        "key card",
        "phone charger",
        "water bottle",
        "coffee mug",
        "lunchbox",
        "paper towels",
        # --- Ground & Environmental Elements ---
        "concrete pavement",
        "gravel pit",
        "manhole cover",
        "drainage grate",
        "metal grating",
        "dirt pile",
        "sandbag",
        "puddle",
        "oil slick",
        "weeds",
        "green grass",
        "bird",
        "stray cat",
        "blueberry",
    ]

    _MAP_DERIVED_OBJECTS = [
        # --- Large Scale Infrastructure ---
        "spherical storage tank",
        "cylindrical silo",
        "cooling tower",
        "smoke stack",
        "chimney",
        "refinery column",
        "heat exchanger",
        "storage vessel",
        "containment wall",
        "overhead pipe rack",
        "catwalk",
        "industrial staircase",
        "loading dock",
        # --- Processing & Mechanical ---
        "centrifugal pump",
        "turbine",
        "electric motor",
        "compressor skid",
        "conveyor belt",
        "ventilation fan unit",
        "HVAC system",
        "filtration unit",
        "pressure vessel",
        "boiler",
        # --- Structural & Ground Elements ---
        "concrete pavement",
        "gravel pit",
        "retaining wall",
        "security fence",
        "barbed wire",
        "sliding gate",
        "manhole cover",
        "drainage grate",
        "metal grating",
        "utility pole",
        "transformer substation",
        # --- Navigation Markers from the Map ---
        "waypoint marker",
        "inspection point",
        "entry gate",
        "exit gate",
        "yellow safety line",
        "parking bay",
        "loading zone",
    ]

    _PRIOR_OBJECTS = _HIKING_OBJECTS + _INSPECTION_OBJECTS + _MAP_DERIVED_OBJECTS

    def __init__(self) -> None:
        super().__init__("visual_search_yoloe_text_prompt")
        with contextlib.suppress(Exception):
            self.declare_parameter("use_sim_time", True)

        default_model_path = str(find_model_file("yoloe-v8l-seg.pt", "yoloe") or "")

        self.declare_parameter("model_path", default_model_path)
        self.declare_parameter("device", "")
        self.declare_parameter("default_conf", 0.25)
        self.declare_parameter("default_iou", 0.7)
        self.declare_parameter("default_imgsz", 640)
        self.declare_parameter("run_when_goal_empty", False)
        self.declare_parameter("mode", self._MODE_ADAPTATION)
        self.declare_parameter("mode_topic", "/spot/mapping/visual_search_mode")
        self.declare_parameter("goal_topic", "/spot/mapping/visual_search_goal")
        self.declare_parameter("results_topic", "/spot/mapping/visual_search_results")
        self.declare_parameter("stopped_for_vlm_validation_topic", "/spot/mapping/stopped_for_vlm_validation")
        self.declare_parameter("debug_save_enable", True)
        self.declare_parameter("debug_save_dir", "log/visual_search_results_yoloe")
        self.declare_parameter("debug_save_jpeg_quality", 95)
        self.declare_parameter("adaptation_results_save_enable", True)
        # Full-frame adaptation image saves are optional and can create large disk churn.
        self.declare_parameter("adaptation_save_full_images", False)
        self.declare_parameter(
            "adaptation_results_dir",
            default_adaptation_results_dir(),
        )
        self.declare_parameter(
            "scene_graph_snapshot_dir",
            default_scene_graph_snapshot_dir(),
        )
        self.declare_parameter("camera_names", ["head_left", "head_right", "left", "right", "rear"])
        self.declare_parameter("camera_meta_topics", [])
        self.declare_parameter("min_publish_movement_m", 0.1)
        # Processing rate:
        # - <= 0: process on every incoming message (no rate limiting)
        # - > 0: process latest inputs on a timer at this rate (Hz)
        self.declare_parameter("process_rate_hz", 0.0)
        # Filter out detections whose bbox touches the image border within this many pixels.
        # Set to 0 to disable.
        self.declare_parameter("bbox_border_filter_px", float(self._DEFAULT_BBOX_BORDER_FILTER_PX))
        # Extra gating to avoid "forced choice" mislabels when using text prompts.
        self.declare_parameter("min_target_confidence", 0.2)
        self.declare_parameter(
            "adaptation_merge_distance_m",
            float(self._DEFAULT_ADAPTATION_MERGE_DISTANCE_M),
        )
        self.declare_parameter(
            "adaptation_crop_pad_ratio",
            float(self._DEFAULT_ADAPTATION_CROP_PAD_RATIO),
        )
        self.declare_parameter(
            "adaptation_crop_min_side_px",
            int(self._DEFAULT_ADAPTATION_CROP_MIN_SIDE_PX),
        )
        self.declare_parameter(
            "adaptation_max_mask_points",
            int(self._DEFAULT_ADAPTATION_MAX_MASK_POINTS),
        )
        # Throttle manifest writes; disk IO is not real-time friendly.
        self.declare_parameter(
            "adaptation_manifest_interval_sec",
            float(self._DEFAULT_ADAPTATION_MANIFEST_INTERVAL_SEC),
        )
        self.declare_parameter(
            "adaptation_max_viewpoints",
            int(self._DEFAULT_ADAPTATION_MAX_VIEWPOINTS),
        )

        camera_names_param = list(self.get_parameter("camera_names").get_parameter_value().string_array_value)
        if not camera_names_param:
            camera_names_param = ["head_left", "head_right", "left", "right", "rear"]

        default_topics = []
        for name in camera_names_param:
            cfg = CAMERA_CONFIG.get(name)
            default_topics.append(cfg.rgb_topic if cfg is not None else "")

        self.declare_parameter("camera_topics", default_topics)
        self.declare_parameter("rgbd_topics", [f"mapping/rgbd_frame/{name}" for name in camera_names_param])

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        if not model_path:
            model_path = default_model_path

        device_param = self.get_parameter("device").get_parameter_value().string_value
        device = device_param.strip()
        if not device:
            if torch is not None and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        if torch is not None:
            # Prefer full FP32 precision on Ampere/Tensor Cores.
            with contextlib.suppress(Exception):
                torch.backends.cuda.matmul.allow_tf32 = False
            with contextlib.suppress(Exception):
                torch.backends.cudnn.allow_tf32 = False
            with contextlib.suppress(Exception):
                torch.set_float32_matmul_precision("highest")

        self._default_conf = float(self.get_parameter("default_conf").value)
        self._default_iou = float(self.get_parameter("default_iou").value)
        self._default_imgsz = int(self.get_parameter("default_imgsz").value)
        self._run_when_goal_empty = bool(self.get_parameter("run_when_goal_empty").value)
        self._adaptation_merge_distance_m = max(0.0, float(self.get_parameter("adaptation_merge_distance_m").value))
        self._adaptation_crop_pad_ratio = max(0.0, float(self.get_parameter("adaptation_crop_pad_ratio").value))
        self._adaptation_crop_min_side_px = max(0, int(self.get_parameter("adaptation_crop_min_side_px").value))
        self._adaptation_max_mask_points = max(0, int(self.get_parameter("adaptation_max_mask_points").value))
        self._adaptation_manifest_interval_sec = max(
            0.0, float(self.get_parameter("adaptation_manifest_interval_sec").value)
        )
        self._adaptation_max_viewpoints = max(1, int(self.get_parameter("adaptation_max_viewpoints").value))

        mode_param_raw = str(self.get_parameter("mode").value or "").strip().lower()
        initial_mode = self._normalize_mode(mode_param_raw)

        mode_topic = (self.get_parameter("mode_topic").get_parameter_value().string_value or "").strip()
        if not mode_topic:
            mode_topic = "/spot/mapping/visual_search_mode"
        goal_topic = self.get_parameter("goal_topic").get_parameter_value().string_value
        results_topic = self.get_parameter("results_topic").get_parameter_value().string_value
        stopped_for_vlm_validation_topic = (
            self.get_parameter("stopped_for_vlm_validation_topic").get_parameter_value().string_value
        )
        debug_save_enable = bool(self.get_parameter("debug_save_enable").value)
        debug_save_dir = self.get_parameter("debug_save_dir").get_parameter_value().string_value
        debug_save_quality = int(self.get_parameter("debug_save_jpeg_quality").value)
        adaptation_save_enable = bool(self.get_parameter("adaptation_results_save_enable").value)
        adaptation_save_full_images = bool(self.get_parameter("adaptation_save_full_images").value)
        adaptation_results_dir = self.get_parameter("adaptation_results_dir").get_parameter_value().string_value
        if debug_save_quality <= 0 or debug_save_quality > 100:
            debug_save_quality = 95

        camera_topics_param = list(self.get_parameter("camera_topics").get_parameter_value().string_array_value)
        if not camera_topics_param or all(not topic for topic in camera_topics_param):
            camera_topics_param = []
            for name in camera_names_param:
                cfg = CAMERA_CONFIG.get(name)
                if cfg is None:
                    raise ValueError(f"No camera configuration found for '{name}'")
                camera_topics_param.append(cfg.rgb_topic)
        else:
            if len(camera_topics_param) != len(camera_names_param):
                raise ValueError(
                    f"camera_topics length {len(camera_topics_param)} does not match camera_names length"
                    f" {len(camera_names_param)}"
                )
            filled_topics = []
            for name, topic in zip(camera_names_param, camera_topics_param):
                topic = (topic or "").strip()
                if topic:
                    filled_topics.append(topic)
                    continue
                cfg = CAMERA_CONFIG.get(name)
                if cfg is None:
                    raise ValueError(f"No camera configuration found for '{name}' and camera_topics is empty")
                filled_topics.append(cfg.rgb_topic)
            camera_topics_param = filled_topics
        if len(camera_topics_param) != len(camera_names_param):
            raise ValueError(
                f"camera_topics length {len(camera_topics_param)} does not match camera_names length"
                f" {len(camera_names_param)}"
            )

        rgbd_topics_param = list(self.get_parameter("rgbd_topics").get_parameter_value().string_array_value)
        if not rgbd_topics_param or len(rgbd_topics_param) != len(camera_names_param):
            rgbd_topics_param = [f"mapping/rgbd_frame/{name}" for name in camera_names_param]
        else:
            rgbd_topics_param = [
                (topic or f"mapping/rgbd_frame/{name}").strip() or f"mapping/rgbd_frame/{name}"
                for name, topic in zip(camera_names_param, rgbd_topics_param)
            ]

        self._camera_order = list(camera_names_param)
        self._camera_topics = list(camera_topics_param)
        self._rgbd_topics = list(rgbd_topics_param)
        self._camera_meta_topics: List[str] = []
        self._min_publish_movement_m = max(0.0, float(self.get_parameter("min_publish_movement_m").value))
        process_rate_hz = float(self.get_parameter("process_rate_hz").value)
        self._use_process_timer = bool(np.isfinite(process_rate_hz) and process_rate_hz > 0.0)
        self._process_period_s = 0.0 if not self._use_process_timer else (1.0 / float(process_rate_hz))
        self._bbox_border_filter_px = max(0.0, float(self.get_parameter("bbox_border_filter_px").value))
        self._min_target_confidence = max(0.0, float(self.get_parameter("min_target_confidence").value))

        meta_topics_param = list(self.get_parameter("camera_meta_topics").get_parameter_value().string_array_value)
        if not meta_topics_param or all(not topic for topic in meta_topics_param):
            meta_topics_param = [f"mapping/frame_stream/{name}" for name in self._camera_order]
        else:
            if len(meta_topics_param) != len(self._camera_order):
                raise ValueError(
                    f"camera_meta_topics length {len(meta_topics_param)} does not match camera_names length"
                    f" {len(self._camera_order)}"
                )
            filled_meta_topics = []
            for name, topic in zip(self._camera_order, meta_topics_param):
                topic = (topic or "").strip()
                filled_meta_topics.append(topic or f"mapping/frame_stream/{name}")
            meta_topics_param = filled_meta_topics
        self._camera_meta_topics = list(meta_topics_param)

        self.get_logger().info(f"Loading YOLOE model: {model_path}")
        self._model = YOLOE(model_path)
        self._device = device

        # Force FP32 inference where possible (avoid AMP/FP16).
        with contextlib.suppress(Exception):
            overrides = getattr(self._model, "overrides", None)
            if isinstance(overrides, dict):
                overrides["half"] = False
                overrides["amp"] = False
        self._model.to(device)
        self._model.eval()
        if torch is not None:
            with contextlib.suppress(Exception):
                torch_model = getattr(self._model, "model", None)
                if torch_model is not None:
                    torch_model.float()
            with contextlib.suppress(Exception):
                predictor = getattr(self._model, "predictor", None)
                predictor_model = getattr(predictor, "model", None)
                if predictor_model is not None:
                    predictor_model.float()
            with contextlib.suppress(Exception):
                predictor = getattr(self._model, "predictor", None)
                args = getattr(predictor, "args", None)
                if args is not None:
                    with contextlib.suppress(Exception):
                        setattr(args, "half", False)
                    with contextlib.suppress(Exception):
                        setattr(args, "amp", False)
            with contextlib.suppress(Exception):
                torch_model = getattr(self._model, "model", None)
                param = next(torch_model.parameters(), None) if torch_model is not None else None
                dtype = param.dtype if param is not None else "n/a"
                self.get_logger().info(f"YOLOE param dtype: {dtype}")

        self._goal_lock = threading.Lock()
        self._mode_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._goal_state: Optional[GoalState] = None
        self._mode = initial_mode
        self._applied_key: Optional[tuple[str, ...]] = None
        self._class_goal_map: dict[int, GoalQuery] = {}
        self._num_target_classes = 0
        self._debug_save_enabled = debug_save_enable
        self._debug_save_dir: Optional[Path] = None
        self._adaptation_results_enabled = adaptation_save_enable
        self._adaptation_save_full_images = adaptation_save_full_images
        self._adaptation_debug_dir: Optional[Path] = None
        self._adaptation_full_dir: Optional[Path] = None
        self._adaptation_manifest_path: Optional[Path] = None
        self._debug_save_quality = debug_save_quality
        self._save_seq = 0
        self._latest_meta_positions: dict[str, np.ndarray] = {}
        self._last_processed_positions: dict[str, np.ndarray] = {}
        self._stopped_for_vlm_lock = threading.Lock()
        self._stopped_for_vlm_validation = False
        self._image_lock = threading.Lock()
        self._latest_image_seq = 0
        self._latest_images: dict[str, tuple[int, Image]] = {}
        self._last_processed_image_seq: dict[str, int] = {}
        self._rgbd_lock = threading.Lock()
        self._latest_rgbd_seq = 0
        self._latest_rgbd_frames: Dict[str, Tuple[int, RGBDFrame]] = {}
        self._last_processed_rgbd_seq: Dict[str, int] = {}
        self._adaptation_lock = threading.Lock()
        self._adaptation_objects: Dict[int, AdaptationObject] = {}
        self._next_adaptation_detection_id = 1
        self._adaptation_manifest_dirty = False
        self._adaptation_last_manifest_write_mono = 0.0

        if self._debug_save_enabled:
            if debug_save_dir:
                save_dir = Path(debug_save_dir).expanduser()
                try:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    self._debug_save_dir = save_dir
                except Exception as exc:
                    self.get_logger().warning(f"Failed to create debug_save_dir '{save_dir}': {exc}")
            if self._debug_save_dir is None:
                self.get_logger().warning("Debug save enabled but no valid debug_save_dir; skipping saves.")
            else:
                resolved_dir = str(self._debug_save_dir)
                with contextlib.suppress(Exception):
                    resolved_dir = str(self._debug_save_dir.resolve())
                self.get_logger().info(f"Saving YOLOE debug detections to: {resolved_dir}")
                with contextlib.suppress(Exception):
                    self.get_logger().info(f"YOLOE debug save cwd: {os.getcwd()}")

        if self._adaptation_results_enabled:
            results_dir = Path(adaptation_results_dir or default_adaptation_results_dir()).expanduser()
            try:
                results_dir.mkdir(parents=True, exist_ok=True)
                self._adaptation_debug_dir = results_dir / "cropped_images"
                self._adaptation_debug_dir.mkdir(parents=True, exist_ok=True)
                if self._adaptation_save_full_images:
                    self._adaptation_full_dir = results_dir / "full_images"
                    self._adaptation_full_dir.mkdir(parents=True, exist_ok=True)
                self._adaptation_manifest_path = results_dir / "objects.json"
                self.get_logger().info(f"Saving adaptation results to: {results_dir}")
            except Exception as exc:
                self._adaptation_results_enabled = False
                self.get_logger().warning(f"Failed to create adaptation_results_dir '{results_dir}': {exc}")

        sensor_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        goal_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        results_qos = QoSProfile(depth=5)
        stopped_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)

        self._mode_sub = self.create_subscription(String, mode_topic, self._on_mode, goal_qos)
        self._goal_sub = self.create_subscription(String, goal_topic, self._on_goal, goal_qos)
        self._stopped_for_vlm_sub = self.create_subscription(
            Bool,
            stopped_for_vlm_validation_topic,
            self._on_stopped_for_vlm_validation,
            stopped_qos,
        )
        self._results_pub = self.create_publisher(VisualSearchResultArray, results_topic, results_qos)

        self._image_subs = {}
        for name, topic in zip(self._camera_order, self._camera_topics):
            self._image_subs[name] = self.create_subscription(
                Image,
                topic,
                lambda msg, camera=name: self._on_image(camera, msg),
                sensor_qos,
            )
            self._last_processed_image_seq[name] = 0

        self._rgbd_subs = {}
        for name, topic in zip(self._camera_order, self._rgbd_topics):
            self._rgbd_subs[name] = self.create_subscription(
                RGBDFrame,
                topic,
                lambda msg, camera=name: self._on_rgbd(camera, msg),
                sensor_qos,
            )
            self._last_processed_rgbd_seq[name] = 0

        self._process_timer = None
        if self._use_process_timer:
            self._process_timer = self.create_timer(self._process_period_s, self._process_latest_inputs)

        meta_qos = QoSProfile(
            depth=5,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._meta_subs = {}
        for name, topic in zip(self._camera_order, self._camera_meta_topics):
            if not topic:
                continue
            self._meta_subs[name] = self.create_subscription(
                FrameMetadata,
                topic,
                lambda msg, camera=name: self._on_meta(camera, msg),
                meta_qos,
            )

        topics_summary = ", ".join(f"{name}={topic}" for name, topic in zip(self._camera_order, self._camera_topics))
        rgbd_summary = ", ".join(f"{name}={topic}" for name, topic in zip(self._camera_order, self._rgbd_topics))
        if mode_param_raw and mode_param_raw != initial_mode:
            self.get_logger().warning(
                f"Unsupported mode parameter {mode_param_raw!r}; using {initial_mode!r}. "
                f"Supported={list(self._SUPPORTED_MODES)}"
            )
        self.get_logger().info(f"Visual search mode: {self._mode}")
        self.get_logger().info(f"Subscribed mode topic: {mode_topic}")
        self.get_logger().info(f"Subscribed goal:   {goal_topic}")
        self.get_logger().info(f"Subscribed images: {topics_summary}")
        self.get_logger().info(f"Subscribed RGBD:   {rgbd_summary}")
        self.get_logger().info(f"Publishing result: {results_topic} (mapping_msgs/VisualSearchResultArray)")
        meta_summary = ", ".join(
            f"{name}={topic}" for name, topic in zip(self._camera_order, self._camera_meta_topics) if topic
        )
        if meta_summary:
            self.get_logger().info(f"Subscribed metadata: {meta_summary}")
        self.get_logger().info(f"Subscribed stopped-for-VLM: {stopped_for_vlm_validation_topic}")
        if self._use_process_timer:
            self.get_logger().info(f"Processing/publishing at {1.0 / self._process_period_s:.1f} Hz.")
        else:
            self.get_logger().info("Processing/publishing on incoming messages (no rate limit).")

    def _on_stopped_for_vlm_validation(self, msg: Bool) -> None:
        is_stopped = bool(getattr(msg, "data", False))
        with self._stopped_for_vlm_lock:
            self._stopped_for_vlm_validation = is_stopped

    @classmethod
    def _normalize_mode(cls, raw_mode: str) -> str:
        mode = (raw_mode or "").strip().lower()
        return mode if mode in cls._SUPPORTED_MODES else cls._MODE_VALIDATION

    def _get_mode(self) -> str:
        with self._mode_lock:
            return self._mode

    def _set_mode(self, mode: str, *, reason: str) -> None:
        normalized = self._normalize_mode(mode)
        with self._mode_lock:
            old_mode = self._mode
            if old_mode == normalized:
                return
            self._mode = normalized
        with self._model_lock:
            self._applied_key = None
            self._class_goal_map = {}
            self._num_target_classes = 0
        if normalized == self._MODE_ADAPTATION:
            with self._adaptation_lock:
                self._adaptation_objects.clear()
                self._next_adaptation_detection_id = 1
                self._adaptation_manifest_dirty = True
            self._save_adaptation_manifest()
        self.get_logger().info(f"Switched visual search mode {old_mode} -> {normalized} ({reason})")

    def _on_mode(self, msg: String) -> None:
        raw = str(getattr(msg, "data", "") or "")
        mode = self._normalize_mode(raw)
        if mode != raw.strip().lower():
            self.get_logger().warning(
                f"Unsupported visual search mode {raw!r}; keeping/using {mode!r}. "
                f"Supported={list(self._SUPPORTED_MODES)}"
            )
        self._set_mode(mode, reason="topic")

    @staticmethod
    def _as_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int,)):
            return int(value)
        if isinstance(value, float):
            if value != value:  # NaN
                return None
            return int(value)
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            with contextlib.suppress(Exception):
                return int(s)
        return None

    def _parse_goals(self, raw: str) -> List[GoalQuery]:
        raw = (raw or "").strip()
        if not raw:
            return []

        parsed: Any = None
        try:
            parsed = json.loads(raw)
        except Exception:
            sanitized = raw
            sanitized = re.sub(r'("(?:(?:object_)?id)")\s*:\s*(?=[,}])', r"\1: null", sanitized)
            sanitized = re.sub(r",\s*([}\]])", r"\1", sanitized)
            with contextlib.suppress(Exception):
                parsed = json.loads(sanitized)

        if isinstance(parsed, list):
            items: Sequence[Any] = parsed
        elif isinstance(parsed, dict):
            if isinstance(parsed.get("goals"), list):
                items = parsed["goals"]
            elif isinstance(parsed.get("objects"), list):
                items = parsed["objects"]
            else:
                items = [parsed]
        else:
            return [GoalQuery(object_id=-1, description=raw)]

        out: List[GoalQuery] = []
        for item in items:
            if isinstance(item, str):
                desc = item.strip()
                if desc:
                    out.append(GoalQuery(object_id=-1, description=desc))
                continue
            if not isinstance(item, dict):
                continue
            obj_id = self._as_int(item.get("id"))
            if obj_id is None:
                obj_id = self._as_int(item.get("object_id"))
            if obj_id is None:
                obj_id = -1

            desc = ""
            for key in ("description", "text", "query", "name"):
                val = item.get(key)
                if isinstance(val, str) and val.strip():
                    desc = val.strip()
                    break
            if desc:
                out.append(GoalQuery(object_id=int(obj_id), description=desc))
        return out

    def _parse_goal_msg(self, msg: String) -> Optional[GoalState]:
        raw = (msg.data or "").strip()
        if not raw:
            return None

        goals = self._parse_goals(raw)
        if not goals:
            return None

        conf = self._default_conf
        iou = self._default_iou
        imgsz = self._default_imgsz
        with contextlib.suppress(Exception):
            obj = json.loads(raw)
            if isinstance(obj, dict):
                conf = float(obj.get("conf", conf))
                iou = float(obj.get("iou", iou))
                imgsz = int(obj.get("imgsz", imgsz))

        return GoalState(goals=goals, conf=conf, iou=iou, imgsz=imgsz)

    def _on_goal(self, msg: String) -> None:
        goal = self._parse_goal_msg(msg)
        with self._goal_lock:
            self._goal_state = goal
        if self._get_mode() != self._MODE_VALIDATION:
            return

        if goal is None:
            with self._model_lock:
                if not self._run_when_goal_empty:
                    self._class_goal_map = {}
                    self._applied_key = None
            self.get_logger().info("Visual search goals cleared.")
            return

        goal_descs = [f"{g.object_id}:{g.description}" for g in goal.goals]
        self.get_logger().info(
            f"New visual search goals: {goal_descs} (conf={goal.conf:.3f}, iou={goal.iou:.3f}, imgsz={goal.imgsz})"
        )

    def _on_meta(self, camera: str, msg: FrameMetadata) -> None:
        with contextlib.suppress(Exception):
            t = msg.transform_world_cam.translation
            pos = np.array([float(t.x), float(t.y), float(t.z)], dtype=np.float32)
            self._latest_meta_positions[camera] = pos

    def _ensure_validation_text_prompt_classes(self, goals: Sequence[GoalQuery]) -> None:
        text_goals: List[GoalQuery] = []
        texts: List[str] = []
        for g in goals:
            desc = str(getattr(g, "description", "") or "").strip()
            if not desc:
                continue
            text_goals.append(g)
            texts.append(desc)
        class_texts = list(texts) + list(self._DISTRACTOR_TEXTS)
        key = tuple(class_texts)
        if not texts:
            self._class_goal_map = {}
            self._applied_key = None
            self._num_target_classes = 0
            return
        if key == self._applied_key:
            self._class_goal_map = {idx: goal for idx, goal in enumerate(text_goals)}
            self._num_target_classes = len(text_goals)
            return
        text_pe = self._model.get_text_pe(class_texts)
        self._model.set_classes(class_texts, text_pe)
        self._applied_key = key
        self._class_goal_map = {idx: goal for idx, goal in enumerate(text_goals)}
        self._num_target_classes = len(text_goals)

    def _ensure_adaptation_text_prompt_classes(self) -> None:
        targets = list(self._PRIOR_OBJECTS)
        class_texts = list(targets) + list(self._DISTRACTOR_TEXTS)
        key = tuple(class_texts)
        if key == self._applied_key:
            self._num_target_classes = len(targets)
            self._class_goal_map = {idx: GoalQuery(object_id=-1, description=text) for idx, text in enumerate(targets)}
            return
        text_pe = self._model.get_text_pe(class_texts)
        self._model.set_classes(class_texts, text_pe)
        self._applied_key = key
        self._num_target_classes = len(targets)
        self._class_goal_map = {idx: GoalQuery(object_id=-1, description=text) for idx, text in enumerate(targets)}

    def _log_predictor_state(self) -> None:
        if torch is None:
            return
        predictor = getattr(self._model, "predictor", None)
        args = getattr(predictor, "args", None)
        if args is None:
            return
        half = getattr(args, "half", None)
        amp = getattr(args, "amp", None)
        self.get_logger().debug(f"YOLOE predictor args: half={half} amp={amp}")

    @staticmethod
    def _image_msg_to_bgr(msg: Image) -> Optional[np.ndarray]:
        encoding = (msg.encoding or "").strip().lower()
        if encoding in ("bgr8", "rgb8", "mono8", "bgra8", "rgba8"):
            h = int(msg.height)
            w = int(msg.width)
            if h <= 0 or w <= 0:
                return None
            if encoding in ("bgra8", "rgba8"):
                channels = 4
            elif encoding == "mono8":
                channels = 1
            else:
                channels = 3
            step = int(msg.step) if int(msg.step) > 0 else w * channels
            expected = h * step
            if len(msg.data) < expected:
                return None
            if channels == 1:
                row_elems = step
                arr = np.frombuffer(msg.data, dtype=np.uint8, count=expected).reshape((h, row_elems))
                arr = arr[:, :w]
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            row_elems = step // channels
            arr = np.frombuffer(msg.data, dtype=np.uint8, count=expected).reshape((h, row_elems, channels))
            arr = arr[:, :w, :]
            if encoding == "rgb8":
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            if encoding == "rgba8":
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            if encoding == "bgra8":
                return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            return np.ascontiguousarray(arr)
        return None

    @staticmethod
    def _decode_bgr(msg: Union[CompressedImage, Image]) -> Optional[np.ndarray]:
        if isinstance(msg, CompressedImage):
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return bgr if bgr is not None else None

        if isinstance(msg, Image):
            return VisualSearchYOLOETextPromptNode._image_msg_to_bgr(msg)

        return None

    @staticmethod
    def _decode_depth(msg: Image, color_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
        if int(msg.height) <= 0 or int(msg.width) <= 0:
            return None

        encoding_upper = (msg.encoding or "").strip().upper()
        if encoding_upper in {"16UC1", "MONO16"}:
            dtype = np.dtype(np.uint16)
            bytes_per_pixel = 2
            needs_float_to_u16 = False
        elif encoding_upper == "32FC1":
            dtype = np.dtype(np.float32)
            bytes_per_pixel = 4
            needs_float_to_u16 = True
        else:
            return None

        step_bytes = int(msg.step) if int(msg.step) > 0 else int(msg.width) * bytes_per_pixel
        if step_bytes % bytes_per_pixel != 0:
            return None

        tight_step_bytes = int(msg.width) * bytes_per_pixel
        if step_bytes < tight_step_bytes and len(msg.data) >= int(msg.height) * tight_step_bytes:
            step_bytes = tight_step_bytes

        expected_min_bytes = int(msg.height) * step_bytes
        if len(msg.data) < expected_min_bytes:
            return None

        row_elems = step_bytes // bytes_per_pixel
        count = int(msg.height) * row_elems
        dtype = dtype.newbyteorder(">" if int(getattr(msg, "is_bigendian", 0)) else "<")
        depth_raw = np.frombuffer(msg.data, dtype=dtype, count=count).reshape(int(msg.height), row_elems)[
            :, : int(msg.width)
        ]

        if needs_float_to_u16:
            depth_f32 = depth_raw.astype(np.float32, copy=False)
            finite_positive = np.isfinite(depth_f32) & (depth_f32 > 0.0)
            depth_mm = np.where(finite_positive, depth_f32 * 1000.0, 0.0)
            depth_u16 = np.clip(depth_mm, 0.0, float(np.iinfo(np.uint16).max)).astype(np.uint16)
        else:
            depth_u16 = depth_raw.astype(np.uint16, copy=False)

        if len(color_shape) < 2:
            return None
        h, w = color_shape[:2]
        if depth_u16.shape != (h, w):
            return cv2.resize(depth_u16, (w, h), interpolation=cv2.INTER_NEAREST)
        return depth_u16

    @staticmethod
    def _select_rgb_from_rgbd(rgbd_msg: RGBDFrame) -> Optional[Union[Image, CompressedImage]]:
        rgb = getattr(rgbd_msg, "rgb", None)
        if rgb is not None:
            with contextlib.suppress(Exception):
                if (
                    int(getattr(rgb, "height", 0)) > 0
                    and int(getattr(rgb, "width", 0)) > 0
                    and len(getattr(rgb, "data", b"")) > 0
                ):
                    return rgb
        rgb_c = getattr(rgbd_msg, "rgb_compressed", None)
        if rgb_c is not None and len(getattr(rgb_c, "data", b"")) > 0:
            return rgb_c
        return None

    @staticmethod
    def _rotate_image_array_cw(image: np.ndarray, deg_cw: int) -> np.ndarray:
        deg_cw = int(deg_cw) % 360
        if deg_cw == 0:
            return image
        if deg_cw == 90:
            return np.ascontiguousarray(np.rot90(image, k=-1))
        if deg_cw == 180:
            return np.ascontiguousarray(np.rot90(image, k=2))
        if deg_cw == 270:
            return np.ascontiguousarray(np.rot90(image, k=1))
        raise ValueError(f"Rotation must be a multiple of 90 degrees, got {deg_cw}.")

    @classmethod
    def _mask_rear_leg_regions(cls, bgr: np.ndarray) -> None:
        if bgr is None or bgr.ndim < 2:
            return
        height, width = int(bgr.shape[0]), int(bgr.shape[1])
        mask_h = min(int(cls._REAR_LEG_MASK_HEIGHT_PX), height)
        mask_w = min(int(cls._REAR_LEG_MASK_WIDTH_PX), width)
        if mask_h <= 0 or mask_w <= 0:
            return
        y0 = height - mask_h
        bgr[y0:height, 0:mask_w] = 0
        bgr[y0:height, width - mask_w : width] = 0

    @classmethod
    def _mask_rear_leg_regions_depth(cls, depth_u16: np.ndarray) -> None:
        if depth_u16 is None or depth_u16.ndim < 2:
            return
        height, width = int(depth_u16.shape[0]), int(depth_u16.shape[1])
        mask_h = min(int(cls._REAR_LEG_MASK_HEIGHT_PX), height)
        mask_w = min(int(cls._REAR_LEG_MASK_WIDTH_PX), width)
        if mask_h <= 0 or mask_w <= 0:
            return
        y0 = height - mask_h
        depth_u16[y0:height, 0:mask_w] = 0
        depth_u16[y0:height, width - mask_w : width] = 0

    @staticmethod
    def _transform_to_matrix(transform: Transform) -> np.ndarray:
        q = np.array(
            [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(q))
        if norm <= 0.0:
            x = y = z = 0.0
            w = 1.0
        else:
            x, y, z, w = (q / norm).tolist()

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        rotation = np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = rotation
        out[0, 3] = float(transform.translation.x)
        out[1, 3] = float(transform.translation.y)
        out[2, 3] = float(transform.translation.z)
        return out

    @staticmethod
    def _viewpoint_payload_from_transform(transform: Transform) -> Optional[List[float]]:
        if transform is None:
            return None
        q = np.array(
            [
                float(transform.rotation.x),
                float(transform.rotation.y),
                float(transform.rotation.z),
                float(transform.rotation.w),
            ],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(q))
        if not np.isfinite(norm) or norm <= 1e-8:
            q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        else:
            q = q / norm
        xyz = np.array(
            [
                float(transform.translation.x),
                float(transform.translation.y),
                float(transform.translation.z),
            ],
            dtype=np.float64,
        )
        if not np.isfinite(xyz).all():
            return None
        return [
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
            float(q[0]),
            float(q[1]),
            float(q[2]),
            float(q[3]),
        ]

    @staticmethod
    def _append_adaptation_viewpoint(
        obj: AdaptationObject,
        *,
        viewpoint: Optional[Sequence[float]],
        camera_id: str,
        max_viewpoints: int,
    ) -> None:
        if viewpoint is None:
            return
        try:
            vp = [
                float(viewpoint[0]),
                float(viewpoint[1]),
                float(viewpoint[2]),
                float(viewpoint[3]),
                float(viewpoint[4]),
                float(viewpoint[5]),
                float(viewpoint[6]),
            ]
        except Exception:
            return
        if not np.isfinite(vp).all():
            return

        cam = str(camera_id or "")
        if obj.viewpoints:
            last_vp = obj.viewpoints[-1]
            last_cam = obj.viewpoint_camera_ids[-1] if obj.viewpoint_camera_ids else ""
            if len(last_vp) >= 7 and last_cam == cam:
                pos_delta = float(
                    np.linalg.norm(np.asarray(last_vp[:3], dtype=np.float32) - np.asarray(vp[:3], dtype=np.float32))
                )
                quat_dot = float(
                    abs(np.dot(np.asarray(last_vp[3:7], dtype=np.float32), np.asarray(vp[3:7], dtype=np.float32)))
                )
                if pos_delta <= 0.05 and quat_dot >= 0.9995:
                    return

        obj.viewpoints.append(vp)
        obj.viewpoint_camera_ids.append(cam)
        keep = max(1, int(max_viewpoints))
        extra = len(obj.viewpoints) - keep
        if extra > 0:
            del obj.viewpoints[:extra]
            del obj.viewpoint_camera_ids[:extra]

    @staticmethod
    def _rgb_image_from_array(rgb: np.ndarray, header: Header) -> Optional[Image]:
        if rgb is None:
            return None
        try:
            arr = np.asarray(rgb)
        except Exception:
            return None
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3:
            return None
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        if arr.shape[2] != 3:
            return None
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        arr = np.ascontiguousarray(arr)

        msg = Image()
        msg.header = header
        msg.height = int(arr.shape[0])
        msg.width = int(arr.shape[1])
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = int(arr.shape[1] * 3)
        msg.data = arr.tobytes()
        return msg

    @staticmethod
    def _assign_bbox(bbox_msg, xyxy: Sequence[float]) -> None:
        if not (isinstance(xyxy, Sequence) and len(xyxy) >= 4):
            return
        try:
            x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
        except Exception:
            return
        size_x = max(0.0, x2 - x1)
        size_y = max(0.0, y2 - y1)
        cx = x1 + size_x * 0.5
        cy = y1 + size_y * 0.5
        center = bbox_msg.center
        if hasattr(center, "x"):
            center.x = cx
            center.y = cy
            if hasattr(center, "theta"):
                center.theta = 0.0
        elif hasattr(center, "_x"):
            center._x = cx
            center._y = cy
            if hasattr(center, "_theta"):
                center._theta = 0.0
        elif hasattr(center, "position"):
            center.position.x = cx
            center.position.y = cy
            if hasattr(center.position, "z"):
                center.position.z = 0.0
            if hasattr(center, "orientation"):
                center.orientation.w = 1.0
        bbox_msg.size_x = size_x
        bbox_msg.size_y = size_y

    @staticmethod
    def _bbox_touches_image_border(
        xyxy: Sequence[float],
        *,
        width: int,
        height: int,
        border_px: float,
    ) -> bool:
        if width <= 0 or height <= 0:
            return True
        if not (isinstance(xyxy, Sequence) and len(xyxy) >= 4):
            return True
        try:
            x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
        except Exception:
            return True
        if not np.isfinite([x1, y1, x2, y2]).all():
            return True
        if x2 <= x1 or y2 <= y1:
            return True

        m = max(0.0, float(border_px))
        if m <= 0.0:
            return False

        w = float(width)
        h = float(height)
        return (x1 <= m) or (y1 <= m) or (x2 >= (w - m)) or (y2 >= (h - m))

    @staticmethod
    def _label_from_id(names, cls_id: int) -> str:
        if (isinstance(names, dict) and cls_id in names) or (
            isinstance(names, Sequence) and not isinstance(names, (str, bytes)) and 0 <= cls_id < len(names)
        ):
            return str(names[cls_id])
        return str(cls_id)

    def _predict(self, bgr: np.ndarray, *, imgsz: int, conf: float, iou: float):
        self._log_predictor_state()
        use_cuda = bool(torch is not None and str(self._device).startswith("cuda") and torch.cuda.is_available())
        if torch is None:
            return self._model.predict(source=bgr, imgsz=imgsz, conf=conf, iou=iou, verbose=False)

        inference_ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with inference_ctx():
            if use_cuda:
                with torch.cuda.amp.autocast(enabled=False):
                    return self._model.predict(source=bgr, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
            return self._model.predict(source=bgr, imgsz=imgsz, conf=conf, iou=iou, verbose=False)

    @staticmethod
    def _sanitize_component(value: str) -> str:
        raw = str(value or "").strip()
        sanitized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw)
        sanitized = sanitized.strip("._")
        return sanitized or "unknown"

    @staticmethod
    def _crop_bgr(bgr: np.ndarray, xyxy: Sequence[float]) -> Optional[np.ndarray]:
        if bgr is None or bgr.ndim < 2:
            return None
        try:
            x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
        except Exception:
            return None
        if not np.isfinite([x1, y1, x2, y2]).all():
            return None
        h, w = bgr.shape[:2]
        x1_i = max(0, min(w, int(np.floor(x1))))
        y1_i = max(0, min(h, int(np.floor(y1))))
        x2_i = max(0, min(w, int(np.ceil(x2))))
        y2_i = max(0, min(h, int(np.ceil(y2))))
        if x2_i <= x1_i or y2_i <= y1_i:
            return None
        return np.ascontiguousarray(bgr[y1_i:y2_i, x1_i:x2_i])

    @staticmethod
    def _crop_bgr_padded(
        bgr: np.ndarray,
        xyxy: Sequence[float],
        *,
        pad_ratio: float,
        min_side_px: int,
    ) -> Optional[np.ndarray]:
        if bgr is None or bgr.ndim < 2:
            return None
        try:
            x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
        except Exception:
            return None
        if not np.isfinite([x1, y1, x2, y2]).all():
            return None
        if x2 <= x1 or y2 <= y1:
            return None

        img_h, img_w = bgr.shape[:2]
        if img_h <= 0 or img_w <= 0:
            return None

        pad_ratio = max(0.0, float(pad_ratio))
        min_side_px = max(0, int(min_side_px))

        box_w = float(x2 - x1)
        box_h = float(y2 - y1)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        # 0.3 padding on both sides => +0.6 * side length total.
        desired_w = box_w * (1.0 + 2.0 * pad_ratio)
        desired_h = box_h * (1.0 + 2.0 * pad_ratio)
        if min_side_px > 0:
            desired_w = max(desired_w, float(min_side_px))
            desired_h = max(desired_h, float(min_side_px))

        # If the desired crop exceeds image size, just take the full image extent in that axis.
        desired_w = min(desired_w, float(img_w))
        desired_h = min(desired_h, float(img_h))

        x1_f = cx - 0.5 * desired_w
        x2_f = cx + 0.5 * desired_w
        y1_f = cy - 0.5 * desired_h
        y2_f = cy + 0.5 * desired_h

        # Shift to fit inside image bounds while preserving size when possible.
        if x1_f < 0.0:
            x2_f -= x1_f
            x1_f = 0.0
        if x2_f > float(img_w):
            overflow = x2_f - float(img_w)
            x1_f -= overflow
            x2_f = float(img_w)
        if y1_f < 0.0:
            y2_f -= y1_f
            y1_f = 0.0
        if y2_f > float(img_h):
            overflow = y2_f - float(img_h)
            y1_f -= overflow
            y2_f = float(img_h)

        x1_f = max(0.0, min(float(img_w), x1_f))
        x2_f = max(0.0, min(float(img_w), x2_f))
        y1_f = max(0.0, min(float(img_h), y1_f))
        y2_f = max(0.0, min(float(img_h), y2_f))

        x1_i = int(np.floor(x1_f))
        y1_i = int(np.floor(y1_f))
        x2_i = int(np.ceil(x2_f))
        y2_i = int(np.ceil(y2_f))
        x1_i = max(0, min(img_w, x1_i))
        x2_i = max(0, min(img_w, x2_i))
        y1_i = max(0, min(img_h, y1_i))
        y2_i = max(0, min(img_h, y2_i))
        if x2_i <= x1_i or y2_i <= y1_i:
            return None
        return np.ascontiguousarray(bgr[y1_i:y2_i, x1_i:x2_i])

    def _save_adaptation_manifest(self) -> None:
        if not self._adaptation_results_enabled or self._adaptation_manifest_path is None:
            return
        now_mono = time.monotonic()
        interval = float(getattr(self, "_adaptation_manifest_interval_sec", 0.0))
        if interval > 0.0 and (now_mono - float(getattr(self, "_adaptation_last_manifest_write_mono", 0.0))) < interval:
            return
        if interval > 0.0 and not bool(getattr(self, "_adaptation_manifest_dirty", False)):
            return

        payload: Dict[str, Any] = {
            "version": 1,
            "updated_unix_ns": int(time.time_ns()),
            "objects": [],
        }
        with self._adaptation_lock:
            objects = sorted(self._adaptation_objects.values(), key=lambda obj: obj.detection_id)
            for obj in objects:
                payload["objects"].append({
                    "object_id": f"yoloe_{int(obj.detection_id)}",
                    "detection_id": int(obj.detection_id),
                    "class_name": obj.label,
                    "confidence_mean": float(obj.confidence_mean),
                    "confidence_count": int(obj.confidence_count),
                    "position_3d": [
                        float(obj.position_mean[0]),
                        float(obj.position_mean[1]),
                        float(obj.position_mean[2]),
                    ],
                    "viewpoints": [[float(x) for x in vp[:7]] for vp in list(obj.viewpoints)],
                    "viewpoint_camera_ids": [
                        str(cam or "") for cam in list(obj.viewpoint_camera_ids[: len(obj.viewpoints)])
                    ],
                    "full_image": str(obj.full_image_path or ""),
                    "full_image_camera_id": str(obj.full_image_camera_id or ""),
                    "top_cropped_images": [str(p or "") for p in list(obj.top_crop_paths[:2])],
                    "top_confidences": [float(crop.confidence) for crop in obj.top_crops],
                })
        try:
            # Compact JSON is faster to write and parse.
            self._adaptation_manifest_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            self._adaptation_last_manifest_write_mono = now_mono
            self._adaptation_manifest_dirty = False
        except Exception as exc:
            self.get_logger().warning(f"Failed to save adaptation manifest '{self._adaptation_manifest_path}': {exc}")

    def _save_adaptation_object_debug(self, obj: AdaptationObject) -> Tuple[Tuple[float, int, int], List[str]]:
        if not self._adaptation_results_enabled or self._adaptation_debug_dir is None:
            return tuple(), []
        top_sig: Tuple[Tuple[float, int, int], ...] = tuple(
            (float(c.confidence), int(c.stamp_sec), int(c.stamp_nanosec)) for c in obj.top_crops[:2]
        )
        if top_sig == tuple(getattr(obj, "last_saved_top_sig", ())):
            return top_sig, list(obj.top_crop_paths[:2])

        category_dir = self._adaptation_debug_dir / self._sanitize_component(obj.label)
        try:
            category_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.get_logger().warning(f"Failed to create adaptation category dir '{category_dir}': {exc}")
            return tuple(), []

        saved_paths: List[str] = []
        for rank, crop in enumerate(obj.top_crops[:2], start=1):
            # Put category/id/conf in the filename (avoid drawing on the image).
            label_s = self._sanitize_component(obj.label)
            conf_s = f"{float(crop.confidence):.2f}"
            path = category_dir / f"{label_s}_id{int(obj.detection_id)}_conf{conf_s}_top{rank}.jpg"
            try:
                ok = cv2.imwrite(
                    str(path),
                    crop.image_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(self._debug_save_quality)],
                )
                if not ok:
                    self.get_logger().warning(f"cv2.imwrite returned false for adaptation image '{path}'")
                else:
                    with contextlib.suppress(Exception):
                        path = path.resolve()
                    saved_paths.append(str(path))
            except Exception as exc:
                self.get_logger().warning(f"Failed to save adaptation image '{path}': {exc}")
        return top_sig, saved_paths

    def _save_adaptation_full_image(
        self,
        *,
        obj: AdaptationObject,
        bgr: np.ndarray,
        confidence: float,
        header: Header,
    ) -> str:
        if not self._adaptation_save_full_images:
            return str(obj.full_image_path or "")
        if not self._adaptation_results_enabled or self._adaptation_full_dir is None:
            return str(obj.full_image_path or "")
        if bgr is None or bgr.ndim < 2:
            return str(obj.full_image_path or "")

        if float(confidence) < float(obj.best_full_conf):
            return str(obj.full_image_path or "")

        label_s = self._sanitize_component(obj.label)
        stamp = header.stamp
        sec = int(getattr(stamp, "sec", 0))
        nsec = int(getattr(stamp, "nanosec", 0))
        conf_s = f"{float(confidence):.2f}"
        path = self._adaptation_full_dir / f"{label_s}_id{int(obj.detection_id)}_conf{conf_s}_{sec}_{nsec}.jpg"
        try:
            ok = cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self._debug_save_quality)])
            if not ok:
                self.get_logger().warning(f"cv2.imwrite returned false for adaptation full image '{path}'")
                return str(obj.full_image_path or "")
            with contextlib.suppress(Exception):
                path = path.resolve()
            return str(path)
        except Exception as exc:
            self.get_logger().warning(f"Failed to save adaptation full image '{path}': {exc}")
            return str(obj.full_image_path or "")

    def _upsert_adaptation_object(
        self,
        *,
        label: str,
        confidence: float,
        position_world: np.ndarray,
        crop_bgr: np.ndarray,
        full_bgr: np.ndarray,
        viewpoint: Optional[Sequence[float]],
        camera_name: str,
        header: Header,
        obj_size: float,  # Used for adaptive merging of objects
    ) -> int:
        needs_full_save = False
        full_save_obj_id = -1
        full_save_snapshot: Optional[AdaptationObject] = None
        with self._adaptation_lock:
            best_id: Optional[int] = None
            best_dist = float("inf")
            for obj_id, obj in self._adaptation_objects.items():
                if obj.label != label:
                    continue
                dist = float(np.linalg.norm(obj.position_mean - position_world))

                # Merge objects using an adaptive threshold based on the size of objects
                dist_threshold = max(self._adaptation_merge_distance_m, obj_size)
                if dist <= dist_threshold and dist < best_dist:
                    best_id = obj_id
                    best_dist = dist

            if best_id is None:
                obj_id = self._next_adaptation_detection_id
                self._next_adaptation_detection_id += 1
                new_obj = AdaptationObject(
                    detection_id=obj_id,
                    label=label,
                    confidence_mean=float(confidence),
                    confidence_count=1,
                    position_mean=position_world.astype(np.float32, copy=True),
                    top_crops=[],
                )
                self._adaptation_objects[obj_id] = new_obj
                target = new_obj
            else:
                target = self._adaptation_objects[best_id]
                n = int(target.confidence_count)
                target.confidence_mean = float((target.confidence_mean * n + confidence) / (n + 1))
                target.position_mean = (
                    (target.position_mean.astype(np.float64) * n + position_world.astype(np.float64)) / float(n + 1)
                ).astype(np.float32)
                target.confidence_count = n + 1
            self._append_adaptation_viewpoint(
                target,
                viewpoint=viewpoint,
                camera_id=camera_name,
                max_viewpoints=int(self._adaptation_max_viewpoints),
            )

            stamp = header.stamp
            target.top_crops.append(
                AdaptationCrop(
                    confidence=float(confidence),
                    # crop_bgr is already a bounded crop; keep only top-2, so storing a copy is OK.
                    image_bgr=np.ascontiguousarray(crop_bgr),
                    source_camera=camera_name,
                    stamp_sec=int(getattr(stamp, "sec", 0)),
                    stamp_nanosec=int(getattr(stamp, "nanosec", 0)),
                )
            )
            target.top_crops.sort(key=lambda item: item.confidence, reverse=True)
            target.top_crops = target.top_crops[:2]
            out_id = int(target.detection_id)
            if self._adaptation_save_full_images and float(confidence) > float(target.best_full_conf) + 1e-6:
                needs_full_save = True
                full_save_obj_id = out_id
                full_save_snapshot = AdaptationObject(
                    detection_id=target.detection_id,
                    label=target.label,
                    confidence_mean=target.confidence_mean,
                    confidence_count=target.confidence_count,
                    position_mean=target.position_mean.copy(),
                    top_crops=list(target.top_crops),
                    viewpoints=[list(vp) for vp in target.viewpoints],
                    viewpoint_camera_ids=list(target.viewpoint_camera_ids),
                    top_crop_paths=list(target.top_crop_paths),
                    full_image_path=str(target.full_image_path),
                    full_image_camera_id=str(target.full_image_camera_id),
                    best_full_conf=float(target.best_full_conf),
                    last_saved_top_sig=target.last_saved_top_sig,
                )
            snapshot = AdaptationObject(
                detection_id=target.detection_id,
                label=target.label,
                confidence_mean=target.confidence_mean,
                confidence_count=target.confidence_count,
                position_mean=target.position_mean.copy(),
                top_crops=list(target.top_crops),
                viewpoints=[list(vp) for vp in target.viewpoints],
                viewpoint_camera_ids=list(target.viewpoint_camera_ids),
                top_crop_paths=list(target.top_crop_paths),
                full_image_path=str(target.full_image_path),
                full_image_camera_id=str(target.full_image_camera_id),
                best_full_conf=float(target.best_full_conf),
                last_saved_top_sig=target.last_saved_top_sig,
            )
            self._adaptation_manifest_dirty = True

        if self._adaptation_save_full_images and needs_full_save and full_save_snapshot is not None:
            full_path = self._save_adaptation_full_image(
                obj=full_save_snapshot,
                bgr=full_bgr,
                confidence=confidence,
                header=header,
            )
            with self._adaptation_lock:
                target_live = self._adaptation_objects.get(full_save_obj_id)
                if target_live is not None:
                    target_live.full_image_path = str(full_path or target_live.full_image_path)
                    target_live.full_image_camera_id = str(camera_name or target_live.full_image_camera_id)
                    target_live.best_full_conf = max(float(target_live.best_full_conf), float(confidence))
                    self._adaptation_manifest_dirty = True

        top_sig, top_paths = self._save_adaptation_object_debug(snapshot)
        if top_sig:
            with self._adaptation_lock:
                target_live = self._adaptation_objects.get(out_id)
                if target_live is not None:
                    target_live.last_saved_top_sig = top_sig
                    target_live.top_crop_paths = list(top_paths[:2])
                    self._adaptation_manifest_dirty = True
        self._save_adaptation_manifest()
        return out_id

    @staticmethod
    def _position_from_mask(
        mask: np.ndarray,
        depth_u16: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        transform_world_cam: np.ndarray,
        *,
        max_points: int = 0,
    ) -> Optional[np.ndarray]:
        if mask is None or depth_u16 is None:
            return None
        if mask.shape != depth_u16.shape:
            return None
        if fx <= 0.0 or fy <= 0.0:
            return None

        valid = mask & (depth_u16 > 0)
        if not bool(np.any(valid)):
            return None

        ys, xs = np.nonzero(valid)
        if ys.size == 0:
            return None
        if max_points > 0 and ys.size > max_points:
            step = int(np.ceil(float(ys.size) / float(max_points)))
            ys = ys[::step]
            xs = xs[::step]
        z = depth_u16[ys, xs].astype(np.float32) * 0.001
        keep = np.isfinite(z) & (z > 0.0)
        if not bool(np.any(keep)):
            return None
        xs = xs[keep].astype(np.float32)
        ys = ys[keep].astype(np.float32)
        z = z[keep]

        # Compute median in camera frame, then transform once (avoid allocating Nx3 point clouds).
        x = (xs - float(cx)) * z / float(fx)
        y = (ys - float(cy)) * z / float(fy)
        median_cam = np.array([float(np.median(x)), float(np.median(y)), float(np.median(z))], dtype=np.float32)
        if not np.isfinite(median_cam).all():
            return None
        rot = transform_world_cam[:3, :3].astype(np.float32, copy=False)
        trans = transform_world_cam[:3, 3].astype(np.float32, copy=False)
        median_world = median_cam @ rot.T + trans
        if not np.isfinite(median_world).all():
            return None
        return median_world.astype(np.float32, copy=False)

    def _build_result_array(
        self,
        camera_name: str,
        header: Header,
        bgr: np.ndarray,
        result,
        goal_map: dict[int, GoalQuery],
        names_snapshot: Sequence[str],
        num_target_classes: int,
        min_target_confidence: float,
    ) -> Optional[VisualSearchResultArray]:
        height = 0
        width = 0
        try:
            if bgr is not None and getattr(bgr, "ndim", 0) >= 2:
                height = int(bgr.shape[0])
                width = int(bgr.shape[1])
        except Exception:
            height = 0
            width = 0

        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return None
        try:
            xyxy = boxes.xyxy.detach().cpu().numpy()
        except Exception:
            return None
        if xyxy.size == 0:
            return None

        if boxes.conf is not None:
            confs = boxes.conf.detach().cpu().numpy()
        else:
            confs = np.zeros((xyxy.shape[0],), dtype=np.float32)
        if boxes.cls is not None:
            clss = boxes.cls.detach().cpu().numpy().astype(int)
        else:
            clss = np.full((xyxy.shape[0],), -1, dtype=int)

        names = list(names_snapshot or [])
        if clss.size:
            max_cls = int(clss.max())
            if max_cls >= len(names):
                self.get_logger().error(f"YOLOE cls index out of range: max_cls={max_cls}, names={len(names)}")

        msg = VisualSearchResultArray()
        msg.header = header
        msg.robot_pose.orientation.w = 1.0

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image_msg = self._rgb_image_from_array(rgb, header)
        if image_msg is None:
            image_msg = Image()
            image_msg.header = header
        msg.image = image_msg
        msg.camera_name = camera_name

        result_msgs: List[VisualSearchResult] = []
        for idx in range(xyxy.shape[0]):
            box_xyxy = xyxy[idx].tolist()
            if self._bbox_border_filter_px > 0.0 and self._bbox_touches_image_border(
                box_xyxy, width=width, height=height, border_px=self._bbox_border_filter_px
            ):
                continue

            cls_id = int(clss[idx]) if idx < len(clss) else -1
            if cls_id < 0:
                continue
            # Reject detections classified as distractor/background prompts.
            if cls_id >= int(num_target_classes):
                continue
            goal = goal_map.get(cls_id)
            goal_desc = goal.description if goal else ""
            if not goal_desc:
                goal_desc = self._label_from_id(names, cls_id)
            goal_obj_id = goal.object_id if goal else -1

            lc = LocalCaption()
            lc.category = goal_desc
            lc.caption = goal_desc
            self._assign_bbox(lc.bbox, box_xyxy)
            if idx < len(confs):
                lc.confidence = float(confs[idx])
            if float(getattr(lc, "confidence", 0.0)) < float(min_target_confidence):
                continue

            res = VisualSearchResult()
            res.goal_object_id = int(goal_obj_id)
            res.goal_description = str(goal_desc or "")
            res.local_caption = lc
            res.bbox = lc.bbox
            res.position_3d = lc.position_3d
            result_msgs.append(res)
            if self._debug_save_enabled:
                label = res.goal_description or lc.category
                conf = lc.confidence if idx < len(confs) else None
                self._save_debug_detection(camera_name, header, bgr, box_xyxy, label, conf, idx)

        if not result_msgs:
            return None
        msg.results = result_msgs
        return msg

    def _process_adaptation_detections(
        self,
        *,
        camera_name: str,
        header: Header,
        bgr: np.ndarray,
        depth_u16: np.ndarray,
        rot_cw: int,
        transform_world_cam: Transform,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        result,
        names_snapshot: Sequence[str],
        num_target_classes: int,
    ) -> int:
        height = int(bgr.shape[0]) if bgr is not None and bgr.ndim >= 2 else 0
        width = int(bgr.shape[1]) if bgr is not None and bgr.ndim >= 2 else 0

        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return 0
        try:
            xyxy = boxes.xyxy.detach().cpu().numpy()
        except Exception:
            return 0
        if xyxy.size == 0:
            return 0

        confs = (
            boxes.conf.detach().cpu().numpy()
            if boxes.conf is not None
            else np.zeros((xyxy.shape[0],), dtype=np.float32)
        )
        clss = (
            boxes.cls.detach().cpu().numpy().astype(int)
            if boxes.cls is not None
            else np.full((xyxy.shape[0],), -1, dtype=int)
        )
        names = list(names_snapshot or [])
        transform_matrix = self._transform_to_matrix(transform_world_cam)
        viewpoint_payload = self._viewpoint_payload_from_transform(transform_world_cam)
        detection_count = 0
        result_masks = getattr(result, "masks", None)
        masks_np: Optional[np.ndarray] = None
        if result_masks is not None and getattr(result_masks, "data", None) is not None:
            with contextlib.suppress(Exception):
                masks_np = result_masks.data.detach().cpu().numpy()

        for idx in range(xyxy.shape[0]):
            box_xyxy = xyxy[idx].tolist()
            if self._bbox_border_filter_px > 0.0 and self._bbox_touches_image_border(
                box_xyxy, width=width, height=height, border_px=self._bbox_border_filter_px
            ):
                continue

            cls_id = int(clss[idx]) if idx < len(clss) else -1
            if cls_id < 0 or cls_id >= int(num_target_classes):
                continue
            label = self._label_from_id(names, cls_id)
            conf = float(confs[idx]) if idx < len(confs) else 0.0
            if conf < float(self._min_target_confidence):
                continue

            if masks_np is None or masks_np.ndim != 3 or not (0 <= idx < masks_np.shape[0]):
                continue
            mask_arr = masks_np[idx]
            if mask_arr.shape != (height, width):
                mask_arr = cv2.resize(
                    mask_arr.astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = mask_arr > 0.5
            if rot_cw:
                inv_rot = (360 - int(rot_cw)) % 360
                mask_u8 = (mask.astype(np.uint8) * 255).astype(np.uint8, copy=False)
                mask = self._rotate_image_array_cw(mask_u8, inv_rot) > 0
            point_world = self._position_from_mask(
                mask,
                depth_u16,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                transform_world_cam=transform_matrix,
                max_points=int(getattr(self, "_adaptation_max_mask_points", 0)),
            )
            if point_world is None:
                continue

            crop = self._crop_bgr_padded(
                bgr,
                box_xyxy,
                pad_ratio=float(getattr(self, "_adaptation_crop_pad_ratio", self._DEFAULT_ADAPTATION_CROP_PAD_RATIO)),
                min_side_px=int(
                    getattr(self, "_adaptation_crop_min_side_px", self._DEFAULT_ADAPTATION_CROP_MIN_SIDE_PX)
                ),
            )
            if crop is None:
                continue

            # Estimate object size
            x1, y1, x2, y2 = box_xyxy
            w_px = float(abs(x2 - x1))
            h_px = float(abs(y2 - y1))

            zs = depth_u16[mask].astype(np.float32) * 0.001  # m
            zs = zs[np.isfinite(zs) & (zs > 0)]
            if zs.size == 0:
                continue
            z_m = float(np.median(zs))

            w_m = (w_px / fx) * z_m
            h_m = (h_px / fy) * z_m
            obj_size = 0.5 * (w_m + h_m)

            detection_id = self._upsert_adaptation_object(
                label=label,
                confidence=conf,
                position_world=point_world,
                crop_bgr=crop,
                full_bgr=bgr,
                viewpoint=viewpoint_payload,
                camera_name=camera_name,
                header=header,
                obj_size=obj_size,
            )
            if detection_id > 0:
                detection_count += 1
        return int(detection_count)

    def _save_debug_detection(
        self,
        camera_name: str,
        header: Header,
        bgr: np.ndarray,
        xyxy: Sequence[float],
        label: str,
        conf: Optional[float],
        det_idx: int,
    ) -> None:
        if not self._debug_save_enabled:
            return
        try:
            x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
        except Exception:
            return
        if not np.isfinite([x1, y1, x2, y2]).all():
            return
        x1_i = max(0, int(round(x1)))
        y1_i = max(0, int(round(y1)))
        x2_i = max(0, int(round(x2)))
        y2_i = max(0, int(round(y2)))
        if x2_i <= x1_i or y2_i <= y1_i:
            return

        stamp = header.stamp
        sec = int(getattr(stamp, "sec", 0))
        nsec = int(getattr(stamp, "nanosec", 0))
        with self._save_lock:
            seq = self._save_seq
            self._save_seq += 1

        label_text = label
        if conf is not None and np.isfinite(conf):
            label_text = f"{label} {conf:.2f}"

        filename = f"{camera_name}_{sec}_{nsec}_det{det_idx}_{seq}.jpg"
        if self._debug_save_dir is None:
            return
        path = self._debug_save_dir / filename
        self._write_debug_image(path, bgr, (x1_i, y1_i, x2_i, y2_i), label_text, color=(255, 0, 0))

    def _write_debug_image(
        self,
        path: Path,
        bgr: np.ndarray,
        bbox_xyxy: Sequence[int],
        label_text: str,
        *,
        color: tuple[int, int, int],
    ) -> None:
        img = bgr.copy()
        x1_i, y1_i, x2_i, y2_i = bbox_xyxy
        cv2.rectangle(img, (x1_i, y1_i), (x2_i, y2_i), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size, baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        text_w, text_h = text_size
        text_x = x1_i
        text_y = max(0, y1_i - 4)
        box_tl = (text_x, max(0, text_y - text_h - baseline))
        box_br = (text_x + text_w + 2, text_y + 2)
        cv2.rectangle(img, box_tl, box_br, color, -1)
        cv2.putText(
            img,
            label_text,
            (text_x + 1, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

        try:
            ok = cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(self._debug_save_quality)])
            if not ok:
                self.get_logger().warning(f"cv2.imwrite returned false for debug image '{path}'")
        except Exception as exc:
            self.get_logger().warning(f"Failed to save debug image '{path}': {exc}")

    def _on_image(self, camera: str, msg: Image) -> None:
        with self._image_lock:
            self._latest_image_seq += 1
            self._latest_images[camera] = (self._latest_image_seq, msg)
        if not getattr(self, "_use_process_timer", False) and self._get_mode() == self._MODE_VALIDATION:
            self._process_latest_image_validation(camera)

    def _on_rgbd(self, camera: str, msg: RGBDFrame) -> None:
        with self._rgbd_lock:
            self._latest_rgbd_seq += 1
            self._latest_rgbd_frames[camera] = (self._latest_rgbd_seq, msg)
        if not getattr(self, "_use_process_timer", False) and self._get_mode() == self._MODE_ADAPTATION:
            self._process_latest_rgbd_frame_adaptation(camera)

    def _process_latest_inputs(self) -> None:
        mode = self._get_mode()
        if mode == self._MODE_ADAPTATION:
            self._process_latest_rgbd_adaptation()
            return
        self._process_latest_images_validation()

    def _process_latest_images_validation(self) -> None:
        for camera in self._camera_order:
            self._process_latest_image_validation(camera)

    def _process_latest_image_validation(self, camera: str) -> None:
        with self._image_lock:
            item = self._latest_images.get(camera)
            if item is None:
                return
            seq, msg = item
            if self._last_processed_image_seq.get(camera, 0) == seq:
                return
            self._last_processed_image_seq[camera] = seq

        with self._goal_lock:
            goal_state = self._goal_state

        if goal_state is None and not self._run_when_goal_empty:
            return
        # NOTE: Motion/validation gating is intentionally disabled so we always process/publish
        # at a fixed rate (default 5 Hz). Previously we skipped frames when the robot had not
        # moved `min_publish_movement_m`, with a small exception when
        # `/spot/mapping/stopped_for_vlm_validation` was true.

        bgr = self._image_msg_to_bgr(msg)
        if bgr is None:
            self.get_logger().warning(f"Failed to convert image message for camera {camera}")
            return

        rot_cw = int(self._CAMERA_IMAGE_ROTATION_CW.get(camera, 0))
        if rot_cw:
            bgr = self._rotate_image_array_cw(bgr, rot_cw)
        if camera == "rear":
            self._mask_rear_leg_regions(bgr)

        conf = goal_state.conf if goal_state else self._default_conf
        iou = goal_state.iou if goal_state else self._default_iou
        imgsz = goal_state.imgsz if goal_state else self._default_imgsz
        goals = list(goal_state.goals) if goal_state else []

        header = Header()
        if msg.header.stamp.sec or msg.header.stamp.nanosec:
            header.stamp = msg.header.stamp
        else:
            header.stamp = self.get_clock().now().to_msg()
        header.frame_id = msg.header.frame_id or camera

        try:
            with self._model_lock:
                if goals:
                    self._ensure_validation_text_prompt_classes(goals)
                elif self._applied_key is None:
                    return
                goal_map = dict(self._class_goal_map)
                names_snapshot = list(self._applied_key) if self._applied_key is not None else []
                num_target_classes = int(self._num_target_classes)
                results = self._predict(bgr, imgsz=imgsz, conf=conf, iou=iou)
        except Exception as exc:
            self.get_logger().error(f"YOLOE predict failed: {exc}")
            return

        if not results:
            return

        out_msg = self._build_result_array(
            camera,
            header,
            bgr,
            results[0],
            goal_map,
            names_snapshot,
            num_target_classes,
            self._min_target_confidence,
        )
        if out_msg is None:
            return
        try:
            self._results_pub.publish(out_msg)
        except Exception as exc:
            self.get_logger().warning(f"Failed to publish visual search results: {exc}")

    def _process_latest_rgbd_adaptation(self) -> None:
        for camera in self._camera_order:
            self._process_latest_rgbd_frame_adaptation(camera)

    def _process_latest_rgbd_frame_adaptation(self, camera: str) -> None:
        with self._rgbd_lock:
            item = self._latest_rgbd_frames.get(camera)
            if item is None:
                return
            seq, rgbd_msg = item
            if self._last_processed_rgbd_seq.get(camera, 0) == seq:
                return
            self._last_processed_rgbd_seq[camera] = seq

        rgb_msg = self._select_rgb_from_rgbd(rgbd_msg)
        if rgb_msg is None:
            self.get_logger().warning(f"Missing RGB image for adaptation mode camera {camera}")
            return

        bgr_raw = self._decode_bgr(rgb_msg)
        if bgr_raw is None:
            self.get_logger().warning(f"Failed to decode RGB image for adaptation mode camera {camera}")
            return

        depth_u16 = self._decode_depth(rgbd_msg.depth, bgr_raw.shape)
        if depth_u16 is None:
            self.get_logger().warning(f"Failed to decode depth image for adaptation mode camera {camera}")
            return

        bgr = bgr_raw
        rot_cw = int(self._CAMERA_IMAGE_ROTATION_CW.get(camera, 0))
        if rot_cw:
            bgr = self._rotate_image_array_cw(bgr, rot_cw)
        if camera == "rear":
            self._mask_rear_leg_regions(bgr)

        meta = rgbd_msg.meta
        # FrameMetadata fields have evolved; prefer rgb_* but keep backward compatibility.
        try:
            fx = float(getattr(meta, "rgb_fx"))
            fy = float(getattr(meta, "rgb_fy"))
            cx = float(getattr(meta, "rgb_cx"))
            cy = float(getattr(meta, "rgb_cy"))
        except Exception:
            fx = float(getattr(meta, "fx"))
            fy = float(getattr(meta, "fy"))
            cx = float(getattr(meta, "cx"))
            cy = float(getattr(meta, "cy"))
        if fx <= 0.0 or fy <= 0.0:
            self.get_logger().warning(f"Invalid intrinsics for adaptation mode camera {camera}: fx={fx} fy={fy}")
            return

        header = Header()
        if meta.header.stamp.sec or meta.header.stamp.nanosec:
            header.stamp = meta.header.stamp
        elif rgb_msg.header.stamp.sec or rgb_msg.header.stamp.nanosec:
            header.stamp = rgb_msg.header.stamp
        else:
            header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"

        try:
            with self._model_lock:
                self._ensure_adaptation_text_prompt_classes()
                names_snapshot = list(self._applied_key) if self._applied_key is not None else []
                num_target_classes = int(self._num_target_classes)
                results = self._predict(
                    bgr,
                    imgsz=self._default_imgsz,
                    conf=self._default_conf,
                    iou=self._default_iou,
                )
        except Exception as exc:
            self.get_logger().error(f"YOLOE adaptation predict failed: {exc}")
            return

        if not results:
            return

        detections_saved = self._process_adaptation_detections(
            camera_name=camera,
            header=header,
            bgr=bgr,
            depth_u16=depth_u16,
            rot_cw=rot_cw,
            transform_world_cam=meta.transform_world_cam,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            result=results[0],
            names_snapshot=names_snapshot,
            num_target_classes=num_target_classes,
        )
        if detections_saved <= 0:
            self._save_adaptation_manifest()


def main(args: Optional[List[str]] = None) -> int:
    rclpy.init(args=args)
    node = VisualSearchYOLOETextPromptNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
