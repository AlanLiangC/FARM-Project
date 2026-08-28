"""Centralized ROS parameter declarations for StreamingMapper."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rclpy.node import Node

from mapping.lib.paths import (
    default_scene_graph_json_save_path,
    default_scene_graph_snapshot_dir,
    default_scene_state_path,
    default_storage_image_dir,
)
from scene_graph.runtime_paths import find_model_dir, find_package_file, resolve_dino_backbone


def declare_mapper_parameters(node: Node) -> None:
    """Declare all StreamingMapper parameters on *node*."""
    import contextlib

    with contextlib.suppress(Exception):
        node.declare_parameter("use_sim_time", True)

    default_vocab = str(find_package_file("configs", "yoloe_vocabulary.txt") or Path("configs/yoloe_vocabulary.txt"))
    # DINO merge backbone: auto-prefer the gated ViT-S+/16 (paper backbone, more
    # stable merging) when a local copy is present, else fall back to the
    # non-gated ViT-S/16 checked in by bootstrap_models.sh (offline-safe).
    # The final value may be supplied by ROS --extra-param, so do not warn
    # about this provisional default before overrides have been applied.
    default_dino_model, default_dino_weights_path = resolve_dino_backbone(warn_if_fallback=False)
    default_dino_weights = str(default_dino_weights_path or "")

    node.declare_parameter("logger_level", "INFO")
    node.declare_parameter(
        "camera_names",
        ["head_left", "head_right", "left", "right", "rear"],
    )
    node.declare_parameter("queue_depth", 10)
    node.declare_parameter("latest_only", False)
    node.declare_parameter("max_pair_skew_sec", 0.05)
    node.declare_parameter("expected_batch", 5)
    node.declare_parameter("batch_timeout", 0.5)
    node.declare_parameter("debug_queue_status", False)
    node.declare_parameter("publish_detected_objects_enabled", True)
    node.declare_parameter("publish_detected_objects_topic", "/mapping/detected_objects")
    node.declare_parameter("publish_local_captions_enabled", False)
    node.declare_parameter("rgb_decode_mode", "bgr2rgb")
    node.declare_parameter("segmenter_model_id", "yoloe-v8l")
    node.declare_parameter("segmenter_vocab_file", default_vocab)
    node.declare_parameter("segmenter_imgsz", 640)
    node.declare_parameter("segmenter_conf", 0.35)
    node.declare_parameter("segmenter_iou", 0.5)
    node.declare_parameter("segmenter_device", "")
    node.declare_parameter("scene_state_device", "cpu")
    node.declare_parameter("segmenter_use_dino", True)
    node.declare_parameter("segmenter_dino_model", default_dino_model)
    node.declare_parameter("segmenter_dino_weights_path", default_dino_weights)
    node.declare_parameter("segmenter_dino_load_size", 512)
    node.declare_parameter("segmenter_dino_facet", "token")
    node.declare_parameter("segmenter_dino_stride", 0)
    node.declare_parameter("vis_segmentation_dir", "")
    node.declare_parameter("segmenter_mask_erosion_px", 3)
    node.declare_parameter("segmenter_mahalanobis_thresh", 2.0)
    # 1-D depth-mode filter: discards mask pixels whose depth is far from
    # the mask's median depth. Cuts background-leak (e.g. wall behind a
    # chair). See SegmentationConfig.depth_mode_filter_enabled.
    node.declare_parameter("segmenter_depth_mode_filter_enabled", True)
    # 1.5 won the ScanNet 0011 stride=1 sweep — see SegmentationConfig.
    node.declare_parameter("segmenter_depth_mode_k_mad", 1.5)
    node.declare_parameter("segmenter_depth_mode_min_mad_m", 0.03)
    node.declare_parameter("segmenter_flush_every_n_messages", 5)
    node.declare_parameter("segmenter_backend", "yoloe")
    node.declare_parameter("segmenter_sam3_manifest", "")

    node.declare_parameter("lock_initial_scene_state", True)

    # Filtering parameters
    node.declare_parameter("filter_detections_touching_image_border_enabled", True)
    node.declare_parameter("filter_touching_image_border_margin_px", 5)
    # Size-aware border filter (see FilteringConfig.touching_image_border_*):
    # drop iff touching AND mask is small AND bbox is small. Both knobs gate
    # the drop together — anything large (num_pixels OR bbox) survives.
    node.declare_parameter("filter_touching_image_border_min_kept_num_pixels", 4000)
    node.declare_parameter("filter_touching_image_border_max_area_fraction", 0.05)
    node.declare_parameter("filter_detections_by_distance_enabled", True)
    # See FilteringConfig.distance_min_m note — 0.1 floors close-range depth
    # streams without giving up the noise rejection we want from this filter.
    node.declare_parameter("filter_by_distance_min_m", 0.1)
    node.declare_parameter("filter_by_distance_max_m", 300.0)
    node.declare_parameter("filter_detections_by_num_pixels_enabled", True)
    node.declare_parameter("filter_by_num_pixels_min", 100)
    node.declare_parameter("filter_uninformative_yoloe_labels_enabled", True)
    node.declare_parameter("filter_detections_duplicates_iou_enabled", True)
    node.declare_parameter("filter_duplicates_iou_min", 0.9)
    node.declare_parameter("correspondence_feature_sim_thresh", 0.5)
    node.declare_parameter("correspondence_hellinger_thresh", 0.8)
    # Deployable merge policy updated on 2026-05-13:
    # conservative best-only detection association plus strict caption-time
    # duplicate cleanup. Hard class-gating remains diagnostic-only because it
    # is slow and did not transfer to HM3D in the Set-A ablation.
    node.declare_parameter("correspondence_hellinger_match_floor_m2", 3.0e-3)
    node.declare_parameter("correspondence_max_merge_distance_m", 1.0)
    node.declare_parameter("correspondence_same_image_one_to_one", True)
    node.declare_parameter("correspondence_use_class_gate", False)
    node.declare_parameter("correspondence_assignment_mode", "best_only")

    node.declare_parameter("storage_image_dir", default_storage_image_dir())
    node.declare_parameter("image_saving_enabled", False)
    node.declare_parameter("storage_image_format", "jpg")
    node.declare_parameter("storage_preview_max_width", 640)
    node.declare_parameter("storage_preview_jpeg_quality", 75)
    node.declare_parameter("image_save_queue_size", 256)
    node.declare_parameter("image_save_max_per_batch", 2)
    node.declare_parameter("object_mask_saving_enabled", False)
    node.declare_parameter("object_mask_storage_dir", "")
    node.declare_parameter("object_mask_observation_max_per_object", 256)
    # JPEG-encode the padded RGB crop already computed for the captioner into
    # the same per-observation .npz sidecar; lets later VL-rerank passes pull
    # a cropped visual without reloading the source frame. Default ON for the
    # final benchmark; older runs that opt out should set this to False.
    node.declare_parameter("object_mask_observation_save_crops", True)
    node.declare_parameter("object_mask_observation_crop_jpeg_quality", 85)
    node.declare_parameter("caption_enabled", True)
    node.declare_parameter("caption_batch_size", 10)
    node.declare_parameter("caption_pad_ratio", 0.25)
    node.declare_parameter("caption_min_bbox_side", 96)
    node.declare_parameter("caption_version", 20)
    node.declare_parameter("caption_visual_prompt_mode", "bbox_crop")
    node.declare_parameter("caption_context_thumbnail_width", 320)
    node.declare_parameter("caption_target_long_side", 512)
    node.declare_parameter("caption_mask_fill_alpha", 0.25)
    node.declare_parameter("caption_mask_outline_px", 4)
    node.declare_parameter("caption_prompt_variant", "default")
    node.declare_parameter("caption_deactivate_unclear_enabled", True)
    node.declare_parameter("caption_device", "cuda:0")
    node.declare_parameter("caption_server", "vllm")
    node.declare_parameter("caption_spatial_context", False)
    node.declare_parameter("caption_spatial_context_include_position", False)
    node.declare_parameter("recaption_time_threshold_sec", 300.0)
    # Shutdown drain timeout for the caption manager. Live ROS uses 5s so the
    # node tears down quickly; offline reconstruction needs longer (captioning
    # is async + bursty) so the .pt actually contains the captions.
    node.declare_parameter("caption_drain_timeout_sec", 5.0)
    node.declare_parameter("caption_step_interval", 5)
    node.declare_parameter("caption_start_step", 5)
    node.declare_parameter("caption_merge_log_path", "")
    node.declare_parameter("caption_merge_enabled", True)
    node.declare_parameter("caption_merge_hellinger_thresh", 0.65)
    node.declare_parameter("caption_merge_caption_thresh", 0.92)
    node.declare_parameter("caption_merge_siglip2_thresh", 0.93)
    node.declare_parameter("caption_merge_require_visual", True)
    node.declare_parameter("caption_merge_require_category_compat", True)
    node.declare_parameter("siglip2_text_embed_service_enabled", True)
    node.declare_parameter("siglip2_text_embed_service_name", "/spot/mapping/siglip2_text_embed")
    node.declare_parameter("siglip2_text_embed_timeout_sec", 0.35)
    node.declare_parameter("siglip2_text_embed_max_texts", 64)
    node.declare_parameter("viser_enabled", False)
    node.declare_parameter("viser_host", "127.0.0.1")
    node.declare_parameter("viser_port", 8080)
    node.declare_parameter("viser_live_rgb_enabled", True)
    node.declare_parameter("viser_live_rgb_max_side", 320)
    node.declare_parameter("viser_live_rgb_max_fps", 5.0)
    node.declare_parameter("viser_voxel_size_m", 0.1)
    node.declare_parameter("viser_point_size_m", 0.01)
    node.declare_parameter("viser_streaming_dashboard_enabled", False)
    node.declare_parameter("viser_stream_total_frames", 0)
    node.declare_parameter("viser_stream_title", "Offline reconstruction")
    node.declare_parameter("config_path", "")
    node.declare_parameter("scene_graph_publish_interval", 60.0)
    node.declare_parameter("scene_graph_json_save_enabled", True)
    node.declare_parameter("scene_graph_json_save_path", default_scene_graph_json_save_path())
    node.declare_parameter("scene_graph_snapshot_save_enabled", True)
    node.declare_parameter("scene_graph_snapshot_dir", default_scene_graph_snapshot_dir())
    node.declare_parameter("scene_graph_snapshot_publish_interval", 0.0)
    node.declare_parameter("scene_graph_snapshot_topic", "/spot/mapping/scene_graph_snapshot")
    node.declare_parameter("scene_graph_persist_distance_m", 40.0)
    node.declare_parameter("prune_enabled", True)
    node.declare_parameter("prune_step_interval", 100)
    node.declare_parameter("prune_caption_keywords", "corn stalk, corn plant, leaf, grass, foliage")
    node.declare_parameter("max_time_sec", 0.0)
    node.declare_parameter("covisibility_enabled", False)
    node.declare_parameter("covisibility_max_objects", 1000)
    node.declare_parameter("covisibility_viz_enabled", False)
    node.declare_parameter("covisibility_viz_path", "log/covisibility_graph.png")
    node.declare_parameter("covisibility_history_batches", 1)
    node.declare_parameter("global_frame", "map")
    node.declare_parameter("timing_enabled", False)
    node.declare_parameter("timing_log_interval_sec", 5.0)
    node.declare_parameter("scene_state_load_path", default_scene_state_path())
    node.declare_parameter("scene_state_save_path", default_scene_state_path())
    node.declare_parameter("scene_state_save_on_shutdown", False)
    node.declare_parameter("scene_state_save_observations", False)
    node.declare_parameter("scene_state_save_service_enabled", True)
    node.declare_parameter("scene_state_shutdown_service_enabled", True)
    node.declare_parameter("scene_state_save_wait_busy_timeout_sec", 2.0)
    node.declare_parameter("scene_state_save_observation_view_limit", 1)
    node.declare_parameter("image_save_close_timeout_sec", 1.0)
    node.declare_parameter("image_save_drain_on_shutdown", False)
    node.declare_parameter("offline_debug", False)
    # Per-frame structured debug trace (JSONL). Empty disables. See
    # scene_graph.debug.tracer / scripts/inspect_pipeline_trace.py.
    node.declare_parameter("debug_trace_path", "")

    # Region clustering
    node.declare_parameter("region_enabled", False)
    node.declare_parameter("region_step_interval", 20)
    node.declare_parameter("region_start_step", 10)
    node.declare_parameter("region_min_objects", 5)
    node.declare_parameter("region_distance_threshold_m", 2.0)
    node.declare_parameter("region_min_cluster_size", 2)
    node.declare_parameter("region_max_diameter_m", 5.0)
