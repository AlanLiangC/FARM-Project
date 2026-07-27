"""Typed configuration hierarchy for the scene-graph pipeline.

All YAML config files should conform to this schema.  Round-trip example::

    cfg = PipelineConfig.from_yaml("configs/replica.yaml")
    cfg.segmentation.conf_thres = 0.35
    cfg.to_yaml("configs/replica_modified.yaml")
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class CameraParamsConfig:
    image_height: int = 480
    image_width: int = 640
    fx: float = 600.0
    fy: float = 600.0
    cx: float = 319.5
    cy: float = 239.5
    png_depth_scale: float = 1000.0
    crop_edge: int = 0


@dataclass
class DatasetConfig:
    name: str = "Replica"
    base_dir: Optional[str] = None
    sequence: Optional[str] = None
    start: int = 0
    end: int = -1
    stride: int = 1
    # NPZ-specific
    npz_depth_scale: float = 1.0
    relative_pose: bool = False
    # Replica-specific resize
    desired_height: Optional[int] = None
    desired_width: Optional[int] = None
    # Optional camera intrinsics block
    camera_params: Optional[CameraParamsConfig] = None
    # Runtime device (not stored in YAML, can be set via CLI)
    device: str = "cuda:0"

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "DatasetConfig":
        d = dict(d)
        cam = d.pop("camera_params", None)
        obj = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if cam is not None:
            obj.camera_params = CameraParamsConfig(**cam)
        return obj


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


@dataclass
class DinoConfig:
    # When ``weights_path`` is left None, ``DINOFeaturesExtractor`` calls
    # ``resolve_dino_backbone()``: it auto-prefers the gated ViT-S+/16
    # (``dinov3-vits16plus``, paper backbone — more stable merging) when a local
    # copy is present, else falls back to the non-gated ViT-S/16
    # (``dinov3-vits16``, offline-safe). Leave ``model`` at this default so
    # auto-resolution can take effect; set both fields to pin a specific backbone.
    model: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    weights_path: Optional[str] = None
    load_size: int = 512
    stride: Optional[int] = None
    facet: str = "token"
    fp16: bool = True
    channels_last: bool = True


@dataclass
class SegmentationConfig:
    model_id: str = "yoloe-v8l"
    vocab_file: str = "configs/yoloe_vocabulary.txt"
    imgsz: int = 640
    # 0.35 cuts the bottom slice of YOLOE's distribution where labels become
    # noise (random walls labelled "tv", reflections labelled "person",
    # etc.). 0.25/0.30 admitted too many ghosts; 0.35 is still below the
    # high-quality cluster (most real detections sit at 0.45+).
    conf_thres: float = 0.35
    iou_thres: float = 0.5
    device: Optional[str] = None          # None → auto-select at runtime
    mask_erosion_px: int = 3
    # Per-mask outlier rejection: drop points whose Mahalanobis distance
    # from the mask centroid exceeds this many σ. 1.5 σ keeps ~87% of an
    # ideal Gaussian cluster but trims the depth-bleed at mask boundaries
    # more aggressively than the old 2.0 σ default — yields tighter,
    # more-discriminable per-detection Gaussians for downstream Hellinger.
    mahalanobis_thresh: float = 1.5
    # Per-mask **1-D depth-mode** filter applied before computing 3-D
    # stats / Mahalanobis. Cuts background-leak: when YOLOE's mask covers
    # the wall behind an object (or the floor under a table), the leaked
    # pixels are far behind the object surface in depth and would otherwise
    # drag the centroid + cov toward the background, producing an
    # over-extended voxel cloud. We restrict to pixels whose depth is
    # within ``depth_mode_k_mad`` MADs of the median depth in the mask —
    # robust to bi-modal depth distributions in a way the 3-D Mahalanobis
    # step is not (Mahalanobis is fit AFTER the background already
    # contaminated the mean).
    # Set ``depth_mode_filter_enabled = False`` to disable.
    depth_mode_filter_enabled: bool = True
    # k_mad sweep on ScanNet 0011 stride=1 showed tighter is better:
    # k=1.5 → 28/33 @ IoU≥0.10, 17/33 @ IoU≥0.25, 4/33 top-1@0.25.
    # k=3.0 → 25/33, 10/33, 2/33. k=5.0 ≈ filter off.
    depth_mode_k_mad: float = 1.5
    depth_mode_min_mad_m: float = 0.03    # 3 cm floor so flat surfaces still match
    use_dino_features: bool = False
    dino: DinoConfig = field(default_factory=DinoConfig)


# ---------------------------------------------------------------------------
# Detection Filtering
# ---------------------------------------------------------------------------


@dataclass
class FilteringConfig:
    touching_image_border_enabled: bool = True
    touching_image_border_margin_px: int = 5
    # Border filter is *size-aware*: drop only when (touching) AND (mask is
    # small) AND (bbox is small). A door / cabinet / fridge that's clipped
    # near the camera typically has num_pixels ≫ 4000 → kept; a sliver of
    # something at the edge has ≪ 4000 → dropped. See
    # filter_detections_touching_image_border for the predicate.
    touching_image_border_min_kept_num_pixels: int = 4000
    touching_image_border_max_area_fraction: float = 0.05
    by_distance_enabled: bool = True
    # 0.1 m floor: indoor RGBD streams routinely produce close-range valid
    # depth (mounted-bracket cameras, hand-held captures) and a 0.4 m floor
    # was killing legitimate close-up detections. Outdoor large-scale runs
    # are dominated by the upper bound.
    distance_min_m: float = 0.1
    distance_max_m: float = 300.0
    by_num_pixels_enabled: bool = True
    num_pixels_min: int = 100
    uninformative_yoloe_labels_enabled: bool = True
    duplicates_iou_enabled: bool = True
    duplicates_iou_min: float = 0.9


# ---------------------------------------------------------------------------
# Captioning
# ---------------------------------------------------------------------------


@dataclass
class CaptioningModelConfig:
    type: str = "llava"
    model_id: str = "qwen3-vl:4b-instruct"
    device: str = "cuda:0"


@dataclass
class CaptioningConfig:
    enabled: bool = True
    results_queue_size: int = 0
    min_views_for_caption: int = 3
    num_views: int = 5
    top_k: int = 1
    min_fraction_in_fov: float = 0.7
    include_neighbors: bool = False
    neighbor_radius: float = 0.0
    max_retries: int = 1
    retry_delay_sec: float = 20.0
    retry_timeout_sec: float = 60.0
    batch_size: int = 16
    model: CaptioningModelConfig = field(default_factory=CaptioningModelConfig)

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "CaptioningConfig":
        d = dict(d)
        model = d.pop("model", None)
        obj = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if model is not None:
            obj.model = CaptioningModelConfig(**model)
        return obj


@dataclass
class OnlineCaptioningConfig:
    enabled: bool = False
    growth_threshold: float = 1.5
    cooldown_steps: int = 5
    max_contributing_steps: int = 10
    updates_queue_size: int = 0
    drop_oldest_when_full: bool = True
    update_period: int = 20
    captioning: CaptioningConfig = field(default_factory=CaptioningConfig)

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "OnlineCaptioningConfig":
        d = dict(d)
        cap = d.pop("captioning", None)
        obj = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if cap is not None:
            obj.captioning = CaptioningConfig._from_dict(cap)
        return obj


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@dataclass
class CacheWeightsConfig:
    recency: float = 0.5
    proximity: float = 0.3
    size: float = 0.1
    write: float = 0.1


@dataclass
class CacheConfig:
    vram_high_watermark: float = 0.9
    vram_low_watermark: float = 0.75
    ram_high_watermark: float = 0.9
    ram_low_watermark: float = 0.75
    pin_radius: int = 1
    weights: CacheWeightsConfig = field(default_factory=CacheWeightsConfig)

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "CacheConfig":
        d = dict(d)
        w = d.pop("weights", None)
        obj = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if w is not None:
            obj.weights = CacheWeightsConfig(**w)
        return obj


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


@dataclass
class StorageConfig:
    compression: str = "zstd"
    zstd_level: int = 6
    image_dir: str = "log/image_store"


# ---------------------------------------------------------------------------
# Run-time / pipeline-level flags
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    batch_size: int = 5
    log_every: int = 25
    log_time: bool = False
    debug: bool = False
    viser: bool = False
    vis_segmentation_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

_NESTED_KEYS = {
    "dataset", "segmentation", "filtering", "online_captioning",
    "cache", "storage", "run",
}


@dataclass
class PipelineConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    online_captioning: OnlineCaptioningConfig = field(default_factory=OnlineCaptioningConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    run: RunConfig = field(default_factory=RunConfig)

    # ---------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with config_path.open("r", encoding="utf-8") as fh:
            data: Dict[str, Any] = yaml.safe_load(fh) or {}

        obj = cls()

        if "dataset" in data:
            obj.dataset = DatasetConfig._from_dict(data["dataset"])
        if "segmentation" in data:
            seg = dict(data["segmentation"])
            dino = seg.pop("dino", None)
            obj.segmentation = SegmentationConfig(
                **{k: v for k, v in seg.items() if k in SegmentationConfig.__dataclass_fields__}
            )
            if dino is not None:
                obj.segmentation.dino = DinoConfig(**dino)
        if "filtering" in data:
            obj.filtering = FilteringConfig(
                **{k: v for k, v in data["filtering"].items() if k in FilteringConfig.__dataclass_fields__}
            )
        if "online_captioning" in data:
            obj.online_captioning = OnlineCaptioningConfig._from_dict(data["online_captioning"])
        if "cache" in data:
            obj.cache = CacheConfig._from_dict(data["cache"])
        if "storage" in data:
            obj.storage = StorageConfig(
                **{k: v for k, v in data["storage"].items() if k in StorageConfig.__dataclass_fields__}
            )
        if "run" in data:
            obj.run = RunConfig(
                **{k: v for k, v in data["run"].items() if k in RunConfig.__dataclass_fields__}
            )

        return obj

    # ---------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            yaml.dump(self.to_dict(), fh, default_flow_style=False, sort_keys=False)

    def get_worker_config(self, worker_type: str) -> Dict[str, Any]:
        """Return a flat dict suitable for passing to a worker process."""
        base = self.to_dict()
        if worker_type == "captioning":
            return {**base, **base["online_captioning"]["captioning"]}
        return base
