"""Thin offline pipeline orchestrator composing shared step functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

import torch

from scene_graph.captioning.crop_util import compute_caption_observations
from scene_graph.config import FilteringConfig
from scene_graph.map_update.filtering import (
    UNINFORMATIVE_YOLOE_LABELS,
    filter_detections_by_distance,
    filter_detections_by_num_pixels,
    filter_detections_duplicates_iou,
    filter_detections_touching_image_border,
    filter_uninformative_yoloe_labels,
)
from scene_graph.utils.geometry import relative_pose
from scene_graph.storage.image_save_worker import ImageSaveWorker, register_batch_images

from .models import BatchResult, FrameBatch
from .steps import (
    compute_detection_image_ids,
    find_neighbors_for_detections,
    resolve_correspondence,
    save_new_detection_frames,
    segment_and_transform,
    update_state_and_enqueue_captions,
)


class PipelineOrchestrator:
    """Composes the shared pipeline step functions for offline batch processing.

    The streaming mapper calls step functions directly — this class only
    serves the offline ``run_pipeline.py`` entry-point.
    """

    def __init__(
        self,
        segmenter: Any,
        scene_state: dict,
        caption_manager: Any,
        image_save_worker: ImageSaveWorker,
        *,
        image_storage_dir: Path,
        dataset_slug: str = "",
        mask_storage_dir: Path | None = None,
        object_mask_saving_enabled: bool = True,
        max_mask_observations_per_object: int = 256,
        viser_visualizer: Any = None,
        debug: bool = False,
        filtering_config: FilteringConfig | None = None,
        caption_step_interval: int = 5,
        caption_start_step: int = 5,
    ) -> None:
        self._segmenter = segmenter
        self._scene_state = scene_state
        self._caption_manager = caption_manager
        self._image_save_worker = image_save_worker
        self._image_storage_dir = image_storage_dir
        self._mask_storage_dir = mask_storage_dir or (image_storage_dir.parent / f"{image_storage_dir.name}_masks")
        self._object_mask_saving_enabled = bool(object_mask_saving_enabled)
        self._max_mask_observations_per_object = max(1, int(max_mask_observations_per_object))
        self._dataset_slug = dataset_slug
        self._viser_visualizer = viser_visualizer
        self._debug = debug
        self._filtering_config = filtering_config or FilteringConfig()
        self._caption_step_interval = caption_step_interval
        self._caption_start_step = caption_start_step

        self._world_origin_pose: Optional[torch.Tensor] = None
        self._step_index: int = 0
        self._pending_caption_indices: List[int] = []

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def world_origin_pose(self) -> Optional[torch.Tensor]:
        return self._world_origin_pose

    def process_batch(self, batch: FrameBatch) -> BatchResult:
        poses_world = list(batch.poses_world)
        if poses_world:
            if self._world_origin_pose is None:
                self._world_origin_pose = poses_world[0].detach().clone()
            poses_world = [
                relative_pose(self._world_origin_pose, p) for p in poses_world
            ]

        batch_image_ids = register_batch_images(self._scene_state, poses_world)
        batch_image_lookup = {
            img_id: idx for idx, img_id in enumerate(batch_image_ids)
        }

        seg_outputs = segment_and_transform(
            self._segmenter,
            batch.colors,
            batch.depths,
            batch.intrinsics,
            poses_world,
        )
        seg_outputs = self._filter_segmentation_outputs(seg_outputs, batch.colors, poses_world)

        neighbors, _ = find_neighbors_for_detections(seg_outputs, self._scene_state)

        num_detections = len(seg_outputs.get("means", []))
        rgb_observations: list[Any] | None = None
        if self._debug or self._caption_manager.enabled:
            rgb_observations = compute_caption_observations(
                seg_outputs, batch.colors
            )

        if self._viser_visualizer is not None and self._viser_visualizer.enabled:
            from scene_graph.visualization.viser_util import (
                update_viser_visualization,
            )

            rgb_observations = update_viser_visualization(
                viser_visualizer=self._viser_visualizer,
                colors=batch.colors,
                depths=batch.depths,
                intrinsics_batch=batch.intrinsics,
                poses_world=poses_world,
                scene_state=self._scene_state,
                seg_outputs=seg_outputs,
                neighbors=neighbors,
                names=getattr(self._segmenter, "names", None),
                existing_rgb_observations=rgb_observations,
            )

        if rgb_observations is None:
            rgb_observations = [None] * num_detections

        detection_image_ids = compute_detection_image_ids(
            seg_outputs, batch_image_ids, num_detections
        )

        det_idx, obj_idx = resolve_correspondence(
            neighbors,
            self._scene_state["count"].shape[0],
            scene_state=self._scene_state,
            detection_image_ids=detection_image_ids,
            seg_outputs=seg_outputs,
        )

        save_new_detection_frames(
            self._scene_state,
            det_idx,
            detection_image_ids,
            batch_image_lookup,
            batch.colors,
            self._image_storage_dir,
            self._image_save_worker,
            dataset_slug=self._dataset_slug,
        )

        update_info = update_state_and_enqueue_captions(
            self._scene_state,
            seg_outputs,
            det_idx,
            obj_idx,
            rgb_observations,
            detection_image_ids,
            self._caption_manager,
            self._pending_caption_indices,
            mask_storage_dir=self._mask_storage_dir if self._object_mask_saving_enabled else None,
            max_mask_observations_per_object=self._max_mask_observations_per_object,
        )

        should_process = (
            self._caption_manager.enabled
            and self._step_index >= self._caption_start_step
            and (self._step_index - self._caption_start_step)
            % self._caption_step_interval
            == 0
        )
        if should_process and self._pending_caption_indices:
            self._caption_manager.enqueue_objects(self._pending_caption_indices)
            self._pending_caption_indices.clear()
        if should_process:
            self._caption_manager.drain_results()

        new_indices = update_info.get("new_object_indices", []) if update_info else []
        self._step_index += 1

        return BatchResult(
            seg_outputs=seg_outputs,
            neighbors=neighbors,
            det_idx=det_idx,
            obj_idx=obj_idx,
            update_info=update_info,
            new_indices=new_indices,
        )

    def _filter_segmentation_outputs(
        self,
        seg_outputs: dict,
        colors: Sequence[torch.Tensor],
        poses_world: Sequence[torch.Tensor],
    ) -> dict:
        cfg = self._filtering_config
        if cfg.touching_image_border_enabled:
            seg_outputs = filter_detections_touching_image_border(
                seg_outputs,
                colors,
                margin_px=cfg.touching_image_border_margin_px,
                min_kept_num_pixels=cfg.touching_image_border_min_kept_num_pixels,
                max_area_fraction=cfg.touching_image_border_max_area_fraction,
            )
        if cfg.by_num_pixels_enabled:
            seg_outputs = filter_detections_by_num_pixels(
                seg_outputs,
                min_num_pixels=cfg.num_pixels_min,
            )
        if cfg.by_distance_enabled:
            seg_outputs = filter_detections_by_distance(
                seg_outputs,
                poses_world,
                min_distance_m=cfg.distance_min_m,
                max_distance_m=cfg.distance_max_m,
            )
        if cfg.uninformative_yoloe_labels_enabled:
            seg_outputs = filter_uninformative_yoloe_labels(
                seg_outputs,
                names=getattr(self._segmenter, "names", None) or [],
                uninformative_labels=UNINFORMATIVE_YOLOE_LABELS,
            )
        if cfg.duplicates_iou_enabled:
            seg_outputs = filter_detections_duplicates_iou(
                seg_outputs,
                min_iou=cfg.duplicates_iou_min,
            )
        return seg_outputs

    def flush_captions(self, timeout: float = 5.0) -> None:
        if self._caption_manager.enabled and self._pending_caption_indices:
            self._caption_manager.enqueue_objects(self._pending_caption_indices)
            self._pending_caption_indices.clear()
        self._caption_manager.wait_until_idle(timeout=timeout, poll_interval=0.05)
        self._caption_manager.drain_results()

    def shutdown(self) -> None:
        self._caption_manager.shutdown_worker()
        self._image_save_worker.close()
