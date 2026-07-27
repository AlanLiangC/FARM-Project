"""Pure pipeline step functions shared by offline and streaming callers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from scene_graph.captioning.crop_util import compute_caption_observations
from scene_graph.map_update.cannot_link import (
    add_same_frame_cannot_links_from_detection_assignments,
    add_same_frame_cannot_links_from_preliminary_neighbors,
    cannot_link_index_pairs,
)
from scene_graph.map_update.filtering import normalize_seg_outputs
from scene_graph.map_update.get_neighbors import get_neighbors
from scene_graph.map_update.mask_observations import register_detection_mask_observations
from scene_graph.map_update.object_update import update_scene_graph_state
from scene_graph.map_update.union_find import find_object_correspondence
from scene_graph.storage.image_save_worker import ImageSaveWorker, mark_image_saved
from scene_graph.storage.models import ImageRecord
from scene_graph.utils.geometry import transform_segmentation_to_world


def segment_and_transform(
    segmenter: Any,
    colors: Sequence[torch.Tensor],
    depths: Sequence[torch.Tensor],
    intrinsics: Sequence[torch.Tensor],
    poses_world: Sequence[torch.Tensor],
) -> dict:
    seg_outputs = segmenter(colors, depths, intrinsics)
    if isinstance(seg_outputs, list):
        raise ValueError("Segmenter returned list output; expected single dict.")
    seg_outputs = normalize_seg_outputs(seg_outputs, fallback_device=getattr(segmenter, "device", None))
    transform_segmentation_to_world(seg_outputs, poses_world)
    return seg_outputs


def find_neighbors_for_detections(
    seg_outputs: dict,
    scene_state: dict,
    *,
    active_mask: Optional[torch.Tensor] = None,
    feature_sim_thresh: float = 0.5,
    hellinger_thresh: float = 0.8,
    eps_cov: Optional[float] = None,
    return_diagnostics: bool = False,
):
    """Wrapper that ensures detections + state are on the same device.

    When ``return_diagnostics=True``, returns ``(neighbors, k_neighbors, diag)``;
    otherwise the legacy ``(neighbors, k_neighbors)`` 2-tuple.
    """
    det_feats = seg_outputs.get("features")
    state_feats = scene_state.get("features")
    if (
        isinstance(det_feats, torch.Tensor)
        and isinstance(state_feats, torch.Tensor)
        and det_feats.device != state_feats.device
    ):
        state_on_det_device = {
            k: v.to(det_feats.device) if isinstance(v, torch.Tensor) else v
            for k, v in scene_state.items()
        }
        return get_neighbors(
            seg_outputs,
            state_on_det_device,
            active_mask=active_mask,
            feature_sim_thresh=feature_sim_thresh,
            hellinger_thresh=hellinger_thresh,
            **({"eps_cov": float(eps_cov)} if eps_cov is not None else {}),
            return_diagnostics=return_diagnostics,
        )
    return get_neighbors(
        seg_outputs,
        scene_state,
        active_mask=active_mask,
        feature_sim_thresh=feature_sim_thresh,
        hellinger_thresh=hellinger_thresh,
        **({"eps_cov": float(eps_cov)} if eps_cov is not None else {}),
        return_diagnostics=return_diagnostics,
    )


def compute_detection_image_ids(
    seg_outputs: dict,
    batch_image_ids: List[int],
    num_detections: int,
) -> List[Optional[int]]:
    batch_ids_tensor = seg_outputs.get("batch_ids")
    if (
        batch_ids_tensor is not None
        and batch_ids_tensor.numel() > 0
        and batch_image_ids
        and num_detections > 0
    ):
        batch_ids_list = batch_ids_tensor.detach().to(torch.long).tolist()
        detection_image_ids: List[Optional[int]] = []
        for batch_id_value in batch_ids_list:
            batch_idx = int(batch_id_value)
            if 0 <= batch_idx < len(batch_image_ids):
                detection_image_ids.append(batch_image_ids[batch_idx])
            else:
                detection_image_ids.append(None)
        return detection_image_ids
    return [None] * num_detections


def _detection_assignment_priority_tensors(
    seg_outputs: Optional[dict],
    num_detections: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return score/size tensors for resolving same-frame assignment conflicts.

    Higher is better. We primarily trust the segmenter confidence, then mask
    size. The caller applies a deterministic earliest-index tie-break.
    """

    scores = None
    num_pixels = None
    if isinstance(seg_outputs, dict):
        scores = seg_outputs.get("scores")
        num_pixels = seg_outputs.get("num_pixels")

    score_t = torch.zeros((int(num_detections),), dtype=torch.float32)
    if isinstance(scores, torch.Tensor) and scores.numel() >= num_detections:
        with torch.no_grad():
            score_t = scores.detach().to("cpu", dtype=torch.float32).view(-1)[:num_detections].clone()

    pixel_t = torch.zeros((int(num_detections),), dtype=torch.float32)
    if isinstance(num_pixels, torch.Tensor) and num_pixels.numel() >= num_detections:
        with torch.no_grad():
            pixel_t = num_pixels.detach().to("cpu", dtype=torch.float32).view(-1)[:num_detections].clone()

    return score_t, pixel_t


def enforce_same_image_one_to_one_assignments(
    det_idx: torch.Tensor,
    obj_idx: torch.Tensor,
    detection_image_ids: Optional[List[Optional[int]]],
    *,
    seg_outputs: Optional[dict] = None,
) -> Tuple[torch.Tensor, int]:
    """Prevent one object from consuming multiple detections from one image.

    Standard data association is one detection per track per frame. Without
    this guard, two masks in the same RGB-D image can both fuse into the same
    existing object. That corrupts geometry before same-frame cannot-link
    constraints have any object IDs to protect.

    Conflicting lower-priority detections are marked ``-1`` so the update step
    treats them as new objects; the post-update cannot-link pass then records
    them as distinct from other objects seen in that same image.
    """

    if (
        detection_image_ids is None
        or not isinstance(det_idx, torch.Tensor)
        or not isinstance(obj_idx, torch.Tensor)
        or det_idx.numel() == 0
        or obj_idx.numel() == 0
    ):
        return det_idx, 0

    D = int(det_idx.numel())
    det_idx_cpu = det_idx.detach().to("cpu", dtype=torch.long).view(-1)
    obj_idx_cpu = obj_idx.detach().to("cpu", dtype=torch.long).view(-1)
    image_ids = torch.full((D,), -1, dtype=torch.long)
    for det_i, image_id in enumerate(detection_image_ids[:D]):
        if image_id is None:
            continue
        image_ids[det_i] = int(image_id)

    raw_valid = (det_idx_cpu >= 0) & (det_idx_cpu < int(obj_idx_cpu.numel())) & (image_ids >= 0)
    valid_det = torch.nonzero(raw_valid, as_tuple=False).view(-1)
    if valid_det.numel() <= 1:
        return det_idx, 0

    canonical = obj_idx_cpu[det_idx_cpu[valid_det]]
    target_valid = canonical >= 0
    valid_det = valid_det[target_valid]
    canonical = canonical[target_valid]
    if valid_det.numel() <= 1:
        return det_idx, 0

    # One key per (image, canonical-object) pair. The sort/group pass is
    # O(D log D) over detection count, not pairwise over masks, so it stays
    # cheap even for dense YOLOE outputs.
    object_key_stride = max(1, int(obj_idx_cpu.numel()) + 1)
    keys = image_ids[valid_det] * object_key_stride + canonical
    order = torch.argsort(keys)
    keys_sorted = keys[order]
    valid_det_sorted = valid_det[order]
    unique_keys, counts = torch.unique_consecutive(keys_sorted, return_counts=True)
    if unique_keys.numel() == counts.numel() and not bool((counts > 1).any()):
        return det_idx, 0

    score_t, pixel_t = _detection_assignment_priority_tensors(seg_outputs, D)
    forced_new: List[int] = []
    start = 0
    for count_raw in counts.tolist():
        count = int(count_raw)
        end = start + count
        if count > 1:
            det_group = valid_det_sorted[start:end]
            scores = score_t[det_group]
            best_score = torch.max(scores)
            score_ties = det_group[scores == best_score]
            if score_ties.numel() > 1:
                pixels = pixel_t[score_ties]
                best_pixels = torch.max(pixels)
                pixel_ties = score_ties[pixels == best_pixels]
                keep_det = int(torch.min(pixel_ties).item())
            else:
                keep_det = int(score_ties[0].item())
            forced_new.extend(int(idx) for idx in det_group.tolist() if int(idx) != keep_det)
        start = end

    if not forced_new:
        return det_idx, 0

    out = det_idx.clone()
    forced_idx = torch.tensor(forced_new, dtype=torch.long, device=out.device)
    out[forced_idx] = -1
    return out, len(forced_new)


def resolve_correspondence(
    neighbors: list,
    object_count: int,
    *,
    scene_state: Optional[dict] = None,
    detection_image_ids: Optional[List[Optional[int]]] = None,
    seg_outputs: Optional[dict] = None,
    same_image_one_to_one: bool = True,
    assignment_mode: str = "union_all",
) -> Tuple[torch.Tensor, torch.Tensor]:
    blocked_pairs = None
    if scene_state is not None:
        if detection_image_ids is not None:
            add_same_frame_cannot_links_from_preliminary_neighbors(
                scene_state,
                neighbors,
                detection_image_ids,
            )
        blocked_pairs = cannot_link_index_pairs(scene_state, n_objects=object_count)
    det_idx, obj_idx = find_object_correspondence(
        neighbors,
        object_count,
        cannot_link_pairs=blocked_pairs,
        assignment_mode=assignment_mode,
    )
    if same_image_one_to_one:
        det_idx, n_forced = enforce_same_image_one_to_one_assignments(
            det_idx,
            obj_idx,
            detection_image_ids,
            seg_outputs=seg_outputs,
        )
        if scene_state is not None:
            scene_state["_last_same_image_assignment_conflicts"] = int(n_forced)
    return det_idx, obj_idx


def save_new_detection_frames(
    scene_state: dict,
    det_idx: torch.Tensor,
    detection_image_ids: List[Optional[int]],
    batch_image_lookup: Dict[int, int],
    colors: Sequence[torch.Tensor],
    image_storage_dir: Path,
    image_save_worker: ImageSaveWorker,
    *,
    dataset_slug: str = "",
    fmt: str = "h5",
    max_saves: Optional[int] = None,
    on_save: Optional[Callable[[dict, int, Path], None]] = None,
) -> None:
    new_detection_mask = det_idx < 0 if det_idx.numel() else torch.empty(0, dtype=torch.bool)
    frames_to_save: set[int] = set()
    if new_detection_mask.numel() and detection_image_ids:
        new_indices = torch.nonzero(new_detection_mask, as_tuple=False).view(-1).tolist()
        for det_ind in new_indices:
            image_id = detection_image_ids[det_ind] if det_ind < len(detection_image_ids) else None
            if image_id is not None:
                frames_to_save.add(image_id)

    ext = ".h5" if fmt == "h5" else ".jpg"
    prefix = f"{dataset_slug}_" if dataset_slug else ""
    sorted_ids = sorted(frames_to_save)
    if max_saves is not None:
        sorted_ids = sorted_ids[:max_saves]

    for image_id in sorted_ids:
        batch_idx = batch_image_lookup.get(image_id)
        if batch_idx is None or batch_idx >= len(colors):
            continue
        images_meta: List[ImageRecord] = scene_state.get("images", [])
        if image_id < 0 or image_id >= len(images_meta):
            continue
        if images_meta[image_id].storage_path:
            continue
        save_path = image_storage_dir / f"{prefix}frame_{image_id:06d}{ext}"
        if on_save is not None:
            on_save(scene_state, image_id, save_path)
        else:
            mark_image_saved(scene_state, image_id, save_path)
        image_save_worker.submit(colors[batch_idx], save_path, fmt=fmt)


def update_state_and_enqueue_captions(
    scene_state: dict,
    seg_outputs: dict,
    det_idx: torch.Tensor,
    obj_idx: torch.Tensor,
    rgb_observations: list,
    detection_image_ids: List[Optional[int]],
    caption_manager: Any,
    pending_caption_indices: List[int],
    *,
    allow_new_objects: bool = True,
    max_merge_distance_m: Optional[float] = None,
    mask_storage_dir: Optional[Path] = None,
    max_mask_observations_per_object: int = 256,
) -> dict:
    num_detections = len(seg_outputs.get("means", []))
    det_points_flat = seg_outputs.get("det_points_flat")
    det_points_offsets = seg_outputs.get("det_points_offsets")
    update_info = update_scene_graph_state(
        scene_state,
        seg_outputs.get("means", torch.empty(0, 3)),
        seg_outputs.get("cov6", torch.empty(0, 6)),
        seg_outputs.get("features", torch.empty(0, 0)),
        [""] * num_detections,
        det_idx,
        obj_idx,
        rgb_observations=rgb_observations,
        detection_image_ids=detection_image_ids,
        allow_new_objects=allow_new_objects,
        det_points_flat=det_points_flat if isinstance(det_points_flat, torch.Tensor) else None,
        det_points_offsets=det_points_offsets if isinstance(det_points_offsets, torch.Tensor) else None,
        class_ids_d=seg_outputs.get("class_ids"),
        max_merge_distance_m=max_merge_distance_m,
    )
    if update_info is not None and detection_image_ids:
        det_to_obj: List[Optional[int]] = [None] * num_detections
        if det_idx.numel() > 0 and obj_idx.numel() > 0:
            det_idx_cpu = det_idx.detach().to("cpu", dtype=torch.long)
            obj_idx_cpu = obj_idx.detach().to("cpu", dtype=torch.long)
            for det_i, raw_obj_idx in enumerate(det_idx_cpu.tolist()):
                if 0 <= int(raw_obj_idx) < obj_idx_cpu.numel():
                    det_to_obj[det_i] = int(obj_idx_cpu[int(raw_obj_idx)].item())

        new_det_indices = [int(i) for i, value in enumerate(det_idx.detach().to("cpu", dtype=torch.long).tolist()) if value < 0]
        new_object_indices = [int(x) for x in (update_info.get("new_object_indices", []) or [])]
        for det_i, obj_i in zip(new_det_indices, new_object_indices):
            if 0 <= det_i < len(det_to_obj):
                det_to_obj[det_i] = int(obj_i)

        n_added = add_same_frame_cannot_links_from_detection_assignments(
            scene_state,
            detection_image_ids,
            det_to_obj,
        )
        update_info["same_frame_cannot_links_added"] = int(n_added)
        if mask_storage_dir is not None:
            n_masks = register_detection_mask_observations(
                scene_state,
                seg_outputs,
                detection_image_ids,
                det_to_obj,
                Path(mask_storage_dir),
                max_per_object=max_mask_observations_per_object,
            )
            update_info["mask_observations_added"] = int(n_masks)
    new_indices = update_info.get("new_object_indices", []) if update_info else []
    if caption_manager.enabled and new_indices:
        pending_caption_indices.extend(new_indices)
    return update_info or {}
