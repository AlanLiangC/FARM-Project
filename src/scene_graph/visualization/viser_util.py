from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

if TYPE_CHECKING:
    from .viser_visualizer import PipelineViserVisualizer


def _encode_masked_crop(color_chw: torch.Tensor, mask_bool: torch.Tensor, max_size: int = 256) -> np.ndarray | None:
    """Build a small RGB thumbnail for the masked region's bounding box (no alpha mask)."""
    mask_np = mask_bool.detach().cpu().numpy().astype(bool)
    if not mask_np.any():
        return None

    coords = mask_np.nonzero()
    y0, y1 = coords[0].min(), coords[0].max()
    x0, x1 = coords[1].min(), coords[1].max()
    pad = 2  # include a slight border around the crop
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(mask_np.shape[0] - 1, y1 + pad)
    x1 = min(mask_np.shape[1] - 1, x1 + pad)

    color_np = color_chw.detach().cpu().numpy()
    color_np = color_np[:, y0 : y1 + 1, x0 : x1 + 1]

    if color_np.ndim != 3 or color_np.shape[0] not in (1, 3):
        return None

    if color_np.shape[0] == 1:
        color_np = color_np.repeat(3, axis=0)

    color_np = color_np.clip(0.0, 255.0).astype("uint8").transpose(1, 2, 0)  # HWC
    if Image is None:
        return color_np

    rgb = Image.fromarray(color_np, mode="RGB")
    rgb.thumbnail((max_size, max_size))
    return np.array(rgb)


def compute_rgb_observations(seg_outputs: dict[str, Any], colors: list[torch.Tensor]) -> list[Any]:
    """
    Compute per-detection RGB observations (masked thumbnails) for debugging and visualization.

    Returns a list aligned with detections. Entries are image numpy arrays or None.
    """
    masks = seg_outputs.get("masks")
    batch_ids = seg_outputs.get("batch_ids")
    means = seg_outputs.get("means", [])
    if masks is None or batch_ids is None:
        return [None] * len(means)

    observations: list[Any] = []
    for mask, batch_idx in zip(masks, batch_ids):
        try:
            b_idx = int(batch_idx)
        except Exception:
            observations.append(None)
            continue
        if b_idx < 0 or b_idx >= len(colors):
            observations.append(None)
            continue

        color_tensor = colors[b_idx]
        if color_tensor.ndim == 3 and color_tensor.shape[-1] == 3 and color_tensor.shape[0] not in (1, 3):
            color_chw = color_tensor.permute(2, 0, 1)
        elif color_tensor.ndim == 3 and color_tensor.shape[0] in (1, 3):
            color_chw = color_tensor
        else:
            observations.append(None)
            continue

        mask_bool = mask
        if mask_bool.ndim == 3:
            mask_bool = mask_bool.squeeze(0)
        mask_bool = mask_bool.to(dtype=torch.bool, device=color_chw.device)
        if color_chw.shape[1:] != mask_bool.shape:
            observations.append(None)
            continue

        obs_img = _encode_masked_crop(color_chw, mask_bool)
        observations.append(obs_img)

    return observations


def update_viser_visualization(
    viser_visualizer: "PipelineViserVisualizer",
    colors: list[torch.Tensor],
    depths: list[torch.Tensor],
    intrinsics_batch: list[torch.Tensor],
    poses_world: Sequence[torch.Tensor],
    scene_state: dict[str, Any],
    seg_outputs: dict[str, Any],
    neighbors: Sequence[torch.Tensor],
    names: Sequence[str] | None,
    existing_rgb_observations: list[Any] | None = None,
) -> list[Any]:
    """Prepare payloads and update the viser visualizer."""
    num_detections = len(seg_outputs.get("means", []))
    means = seg_outputs.get("means")
    cov6 = seg_outputs.get("cov6")
    if means is None or cov6 is None:
        return existing_rgb_observations if existing_rgb_observations is not None else [None] * num_detections

    detection_rgb_observations = existing_rgb_observations
    if detection_rgb_observations is None:
        detection_rgb_observations = compute_rgb_observations(seg_outputs, colors)

    class_ids = seg_outputs.get("class_ids")
    scores = seg_outputs.get("scores")
    class_ids_list = class_ids.detach().cpu().tolist() if isinstance(class_ids, torch.Tensor) else [-1] * num_detections
    scores_list = scores.detach().cpu().tolist() if isinstance(scores, torch.Tensor) else [None] * num_detections

    detection_captions: list[str] = []
    for det_i in range(num_detections):
        cls_id = class_ids_list[det_i] if det_i < len(class_ids_list) else -1
        score = scores_list[det_i] if det_i < len(scores_list) else None
        if names and isinstance(cls_id, int) and 0 <= cls_id < len(names):
            cls_name = names[cls_id]
        else:
            cls_name = f"id={cls_id}"
        caption = f"{cls_name} ({score:.2f})" if score is not None else cls_name
        detection_captions.append(caption)

    detection_neighbors_vis = []
    for neighbor in neighbors:
        if isinstance(neighbor, torch.Tensor):
            detection_neighbors_vis.append(neighbor.detach().cpu().tolist())
        else:
            detection_neighbors_vis.append(neighbor)

    detection_payload = {
        "means": means,
        "cov6": cov6,
        "captions": detection_captions,
        "images": detection_rgb_observations,
    }
    viser_visualizer.update(
        colors,
        depths,
        intrinsics_batch,
        poses_world,
        scene_state,
        detection_info=detection_payload,
        detection_neighbors=detection_neighbors_vis,
    )

    return detection_rgb_observations
