from __future__ import annotations

import contextlib
from typing import List, Optional, Sequence, Set, Tuple

import torch


UNINFORMATIVE_YOLOE_LABELS: Set[str] = {
    "alley",
    "asphalt",
    "asphalt road",
    "avenue",
    "badlands",
    "beach",
    "bike path",
    "boardwalk",
    "canyon",
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
    "grove",
    "corn field",
    "footprint",
    "spear",
    "seaweed",
    "flare",
    "reed",
    "weed",
    "plantation",
    "hedge",
}


def mask_seg_outputs(seg_outputs: dict, mask: torch.Tensor) -> dict:
    """Return a copy of segmentation outputs with detection-aligned fields masked."""
    if mask is None or not hasattr(mask, "numel") or not mask.numel():
        return seg_outputs

    mask_bool = mask.to(torch.bool)
    masked = dict(seg_outputs)
    mask_len = int(mask_bool.shape[0])

    for key, value in seg_outputs.items():
        if key in ("det_points_flat", "det_points_offsets"):
            continue
        try:
            if isinstance(value, torch.Tensor) and value.shape and value.shape[0] == mask_len:
                mask_for_value = mask_bool
                if mask_for_value.device != value.device:
                    mask_for_value = mask_for_value.to(device=value.device)
                masked[key] = value[mask_for_value]
            elif isinstance(value, (list, tuple)) and len(value) == mask_len:
                keep_list = mask_bool.detach().to("cpu", copy=False).tolist()
                masked[key] = [item for item, keep in zip(value, keep_list) if keep]
        except Exception:
            continue

    det_points_flat = seg_outputs.get("det_points_flat")
    det_points_offsets = seg_outputs.get("det_points_offsets")
    if (
        isinstance(det_points_flat, torch.Tensor)
        and isinstance(det_points_offsets, torch.Tensor)
        and det_points_offsets.numel() == mask_len + 1
    ):
        try:
            mask_dev = mask_bool.to(device=det_points_offsets.device)
            kept_idx = torch.nonzero(mask_dev, as_tuple=False).view(-1)
            if kept_idx.numel() == mask_len:
                masked["det_points_flat"] = det_points_flat
                masked["det_points_offsets"] = det_points_offsets
            else:
                lengths = det_points_offsets[1:] - det_points_offsets[:-1]
                kept_lengths = lengths[kept_idx]
                new_offsets = torch.zeros(kept_idx.numel() + 1, dtype=torch.int64, device=det_points_offsets.device)
                if kept_lengths.numel() > 0:
                    new_offsets[1:] = kept_lengths.to(torch.int64).cumsum(0)

                if kept_idx.numel() > 0 and det_points_flat.numel() > 0:
                    pieces = []
                    starts = det_points_offsets.tolist()
                    for di in kept_idx.tolist():
                        s = int(starts[di])
                        e = int(starts[di + 1])
                        if e > s:
                            pieces.append(det_points_flat[s:e])
                    new_flat = (
                        torch.cat(pieces, dim=0)
                        if pieces
                        else torch.empty((0, 3), dtype=det_points_flat.dtype, device=det_points_flat.device)
                    )
                else:
                    new_flat = torch.empty((0, 3), dtype=det_points_flat.dtype, device=det_points_flat.device)
                masked["det_points_flat"] = new_flat
                masked["det_points_offsets"] = new_offsets
        except Exception:
            masked["det_points_flat"] = torch.empty((0, 3), dtype=det_points_flat.dtype, device=det_points_flat.device)
            masked["det_points_offsets"] = torch.zeros(
                (int(mask_bool.sum().item()) + 1,),
                dtype=torch.int64,
                device=det_points_offsets.device,
            )
    return masked


def filter_duplicate_masks_iou(
    masks: torch.Tensor,
    min_iou: float,
    *,
    max_chunk_bytes: int = 128 * 1024 * 1024,
) -> torch.Tensor:
    """Return a keep mask by suppressing masks with IoU > min_iou.

    This uses chunked comparisons to avoid large temporary allocations.
    """
    if masks is None or not isinstance(masks, torch.Tensor) or masks.numel() == 0:
        return torch.empty((0,), dtype=torch.bool, device="cpu")
    if masks.ndim != 3:
        raise ValueError("Masks must have shape (N, H, W)")

    device = masks.device

    try:
        masks = masks.to(dtype=torch.bool)
        num_masks, height, width = masks.shape
        if num_masks == 0:
            return torch.empty((0,), dtype=torch.bool, device=device)

        areas = masks.sum(dim=(1, 2)).float()
        keep = torch.zeros(num_masks, dtype=torch.bool, device=device)
        kept_indices: list[int] = []

        bytes_per_mask = max(1, height * width)
        chunk_size = max(1, int(max_chunk_bytes // bytes_per_mask))

        for i in range(num_masks):
            if not kept_indices:
                keep[i] = True
                kept_indices.append(i)
                continue

            is_duplicate = False
            m_i = masks[i]
            area_i = areas[i]

            for start in range(0, len(kept_indices), chunk_size):
                idx_chunk = kept_indices[start : start + chunk_size]
                chunk_masks = masks[idx_chunk]
                inter = (chunk_masks & m_i).sum(dim=(1, 2)).float()
                union = area_i + areas[idx_chunk] - inter
                iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
                if torch.any(iou > min_iou):
                    is_duplicate = True
                    break

            if not is_duplicate:
                keep[i] = True
                kept_indices.append(i)

        return keep
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and device.type == "cuda":
            torch.cuda.empty_cache()
            return filter_duplicate_masks_iou(
                masks.detach().to("cpu"),
                min_iou,
                max_chunk_bytes=max_chunk_bytes,
            )
        raise


def filter_detections_by_distance(
    seg_outputs: dict,
    poses_world: Sequence[torch.Tensor],
    min_distance_m: float,
    max_distance_m: float,
) -> dict:
    means = seg_outputs.get("means")
    batch_ids = seg_outputs.get("batch_ids")
    if (
        means is None
        or batch_ids is None
        or not hasattr(means, "shape")
        or means.numel() == 0
        or not hasattr(batch_ids, "shape")
        or batch_ids.numel() == 0
        or not poses_world
    ):
        return seg_outputs

    try:
        batch_ids_long = batch_ids.to(torch.long)
    except Exception:
        return seg_outputs
    if int(batch_ids_long.max().item()) >= len(poses_world) or int(batch_ids_long.min().item()) < 0:
        return seg_outputs

    try:
        camera_positions = torch.stack([pose[:3, 3] for pose in poses_world], dim=0)
        camera_positions = camera_positions.to(device=means.device, dtype=means.dtype)
        det_camera_pos = camera_positions[batch_ids_long]
        distances = torch.linalg.norm(means.to(dtype=det_camera_pos.dtype) - det_camera_pos, dim=1)
        # The upper bound is exclusive: a max of 30.0 drops detections whose
        # reconstructed mean is 30 m or farther from the source camera.
        keep_mask = (distances >= float(min_distance_m)) & (distances < float(max_distance_m))
    except Exception:
        return seg_outputs

    if keep_mask.numel() and not bool(torch.all(keep_mask)):
        return mask_seg_outputs(seg_outputs, keep_mask)
    return seg_outputs


def filter_detections_by_num_pixels(seg_outputs: dict, min_num_pixels: int) -> dict:
    num_pixels = seg_outputs.get("num_pixels")
    if num_pixels is None or not hasattr(num_pixels, "shape") or num_pixels.numel() == 0:
        return seg_outputs
    keep_mask = num_pixels >= float(min_num_pixels)
    if keep_mask.numel() and not bool(torch.all(keep_mask)):
        return mask_seg_outputs(seg_outputs, keep_mask)
    return seg_outputs


def filter_uninformative_yoloe_labels(
    seg_outputs: dict,
    names: Sequence[str],
    uninformative_labels: Set[str] = UNINFORMATIVE_YOLOE_LABELS,
) -> dict:
    class_ids = seg_outputs.get("class_ids")
    if class_ids is None or not names:
        return seg_outputs
    try:
        class_ids_long = class_ids.to(torch.long) if isinstance(class_ids, torch.Tensor) else None
    except Exception:
        return seg_outputs
    if class_ids_long is None or not hasattr(class_ids_long, "numel") or class_ids_long.numel() == 0:
        return seg_outputs

    ignored_ids = [int(idx) for idx, name in enumerate(names) if str(name).strip().lower() in uninformative_labels]
    if not ignored_ids:
        return seg_outputs

    try:
        ignored = torch.tensor(ignored_ids, device=class_ids_long.device, dtype=class_ids_long.dtype)
        if hasattr(torch, "isin"):
            is_ignored = torch.isin(class_ids_long, ignored)
        else:
            is_ignored = (class_ids_long[..., None] == ignored).any(dim=-1)
        keep_mask = ~is_ignored.to(torch.bool)
    except Exception:
        return seg_outputs

    if keep_mask.numel() and not bool(torch.all(keep_mask)):
        return mask_seg_outputs(seg_outputs, keep_mask)
    return seg_outputs


def filter_detections_duplicates_iou(seg_outputs: dict, min_iou: float) -> dict:
    masks = seg_outputs.get("masks")
    batch_ids = seg_outputs.get("batch_ids")
    if masks is None or len(masks) == 0 or batch_ids is None:
        return seg_outputs

    unique_batch_ids = torch.unique(batch_ids)
    all_keep_masks = []
    for unique_batch_id in unique_batch_ids:
        masks_for_batch_id = [mask for mask, batch_id in zip(masks, batch_ids) if batch_id == unique_batch_id]
        masks_for_batch_id = torch.stack(masks_for_batch_id, dim=0)
        keep_mask = filter_duplicate_masks_iou(masks_for_batch_id, min_iou=min_iou)
        all_keep_masks.append(keep_mask)

    all_keep_masks = torch.hstack(all_keep_masks)
    if all_keep_masks.numel() and not bool(torch.all(all_keep_masks)):
        return mask_seg_outputs(seg_outputs, all_keep_masks)
    return seg_outputs


def filter_detections_touching_image_border(
    seg_outputs: dict,
    colors: Sequence[torch.Tensor],
    margin_px: int = 5,
    max_area_fraction: float = 0.05,
    min_kept_num_pixels: int = 4000,
) -> dict:
    """Drop detections whose bbox touches the image border *and* are too small.

    A detection is dropped iff **all** of:
      1. Its bbox sits within ``margin_px`` of any image edge ("touching"), and
      2. Its mask covers fewer than ``min_kept_num_pixels`` pixels (i.e. the
         visible portion is too small to be useful — likely a sliver of a
         partially-observed object), and
      3. Its bbox area is smaller than ``max_area_fraction`` of the frame
         (defence-in-depth: even if num_pixels is unavailable for some reason
         the bbox-area path can still flag slivers).

    Anything fully inside the image always survives. Anything large
    (num_pixels ≥ ``min_kept_num_pixels`` *or* bbox ≥ ``max_area_fraction``
    of the image) survives even if it touches an edge, so doors / cabinets
    / fridges / shelves that are necessarily clipped near the camera don't
    get killed.

    If ``num_pixels`` is missing from ``seg_outputs`` the predicate falls
    back to the bbox-area-only check (legacy behaviour for sources that
    don't emit per-mask pixel counts).
    """
    if not colors:
        return seg_outputs

    boxes_xyxy = seg_outputs.get("boxes_xyxy")
    batch_ids = seg_outputs.get("batch_ids")
    if boxes_xyxy is None or batch_ids is None:
        return seg_outputs

    if isinstance(boxes_xyxy, torch.Tensor) and isinstance(batch_ids, torch.Tensor):
        try:
            boxes = boxes_xyxy
            batch_ids_long = batch_ids.to(torch.long)
        except Exception:
            return seg_outputs

        if boxes.ndim != 2 or boxes.shape[0] == 0 or boxes.shape[1] < 4:
            return seg_outputs
        if batch_ids_long.ndim != 1 or int(batch_ids_long.shape[0]) != int(boxes.shape[0]):
            return seg_outputs

        batch_size = len(colors)
        heights_py: List[int] = []
        widths_py: List[int] = []
        frame_valid_py: List[bool] = []
        for image in colors:
            try:
                shape = tuple(int(dim) for dim in image.shape)
            except Exception:
                heights_py.append(0)
                widths_py.append(0)
                frame_valid_py.append(False)
                continue
            if len(shape) == 2:
                heights_py.append(shape[0])
                widths_py.append(shape[1])
                frame_valid_py.append(True)
            elif len(shape) == 3 and shape[-1] in (1, 3, 4):
                heights_py.append(shape[0])
                widths_py.append(shape[1])
                frame_valid_py.append(True)
            elif len(shape) == 3 and shape[0] in (1, 3, 4):
                heights_py.append(shape[1])
                widths_py.append(shape[2])
                frame_valid_py.append(True)
            else:
                heights_py.append(0)
                widths_py.append(0)
                frame_valid_py.append(False)

        if not any(frame_valid_py):
            return seg_outputs

        try:
            heights = torch.tensor(heights_py, device=boxes.device, dtype=torch.float32)
            widths = torch.tensor(widths_py, device=boxes.device, dtype=torch.float32)
            frame_valid = torch.tensor(frame_valid_py, device=boxes.device, dtype=torch.bool)
        except Exception:
            return seg_outputs

        in_range = (batch_ids_long >= 0) & (batch_ids_long < batch_size)
        idx = batch_ids_long.clamp(min=0, max=batch_size - 1)
        frame_valid_det = in_range & frame_valid[idx]
        if not bool(frame_valid_det.any()):
            return seg_outputs

        try:
            boxes_f = boxes.to(dtype=torch.float32) if not boxes.is_floating_point() else boxes
            x1 = boxes_f[:, 0]
            y1 = boxes_f[:, 1]
            x2 = boxes_f[:, 2]
            y2 = boxes_f[:, 3]
            height_det = heights[idx]
            width_det = widths[idx]
            margin = float(margin_px)
            max_area_fraction_f = float(max_area_fraction)

            right_dist = (width_det - 1.0) - x2
            bottom_dist = (height_det - 1.0) - y2
            touching = frame_valid_det & (
                (x1 <= margin) | (y1 <= margin) | (right_dist <= margin) | (bottom_dist <= margin)
            )
            box_w = (x2 - x1).clamp(min=0.0)
            box_h = (y2 - y1).clamp(min=0.0)
            box_area = box_w * box_h
            img_area = (width_det * height_det).clamp(min=1.0)
            small_bbox = box_area < (max_area_fraction_f * img_area)

            # num_pixels-based "small mask" check, when available. If missing,
            # the bbox-area check carries the predicate alone.
            num_pixels = seg_outputs.get("num_pixels")
            if (
                isinstance(num_pixels, torch.Tensor)
                and num_pixels.numel() == int(boxes.shape[0])
                and int(min_kept_num_pixels) > 0
            ):
                np_dev = num_pixels.to(device=boxes.device, dtype=torch.float32)
                small_mask = np_dev < float(min_kept_num_pixels)
                # Drop only when BOTH small-by-bbox AND small-by-mask.
                small = small_bbox & small_mask
            else:
                small = small_bbox

            keep_mask = ~(touching & small)
        except Exception:
            return seg_outputs

        if keep_mask.numel() and not bool(torch.all(keep_mask)):
            return mask_seg_outputs(seg_outputs, keep_mask)
        return seg_outputs

    sizes: List[Optional[Tuple[int, int]]] = []
    for image in colors:
        try:
            shape = image.shape
        except Exception:
            sizes.append(None)
            continue
        if len(shape) == 2:
            sizes.append((int(shape[0]), int(shape[1])))
        elif len(shape) == 3 and shape[-1] in (1, 3, 4):
            sizes.append((int(shape[0]), int(shape[1])))
        elif len(shape) == 3 and shape[0] in (1, 3, 4):
            sizes.append((int(shape[1]), int(shape[2])))
        else:
            sizes.append(None)

    if not any(size is not None for size in sizes):
        return seg_outputs

    if not isinstance(batch_ids, Sequence) or not isinstance(boxes_xyxy, Sequence):
        return seg_outputs
    batch_ids_list = list(batch_ids)
    boxes_list = list(boxes_xyxy)
    if len(batch_ids_list) != len(boxes_list):
        return seg_outputs

    # Optional num_pixels list for the legacy (Sequence) path.
    np_seq = seg_outputs.get("num_pixels")
    if isinstance(np_seq, torch.Tensor):
        np_seq = np_seq.detach().to("cpu").tolist()
    if isinstance(np_seq, Sequence) and len(np_seq) != len(boxes_list):
        np_seq = None

    margin = float(margin_px)
    max_area_fraction_f = float(max_area_fraction)
    min_kept_pixels_f = float(min_kept_num_pixels)
    keep_mask_list: List[bool] = []
    for d_i, (batch_id, box) in enumerate(zip(batch_ids_list, boxes_list)):
        try:
            batch_idx = int(batch_id)
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        except Exception:
            keep_mask_list.append(True)
            continue
        if not (0 <= batch_idx < len(sizes)) or sizes[batch_idx] is None:
            keep_mask_list.append(True)
            continue
        height, width = sizes[batch_idx]
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        img_area = max(1.0, float(height) * float(width))
        touching = (
            (x1 <= margin)
            or (y1 <= margin)
            or (((width - 1.0) - x2) <= margin)
            or (((height - 1.0) - y2) <= margin)
        )
        small_bbox = box_area < (max_area_fraction_f * img_area)
        if np_seq is not None and min_kept_pixels_f > 0:
            try:
                small_mask = float(np_seq[d_i]) < min_kept_pixels_f
            except Exception:
                small_mask = True
            small = small_bbox and small_mask
        else:
            small = small_bbox
        keep_mask_list.append(not (touching and small))

    if not keep_mask_list:
        return seg_outputs
    keep_mask = torch.tensor(keep_mask_list, dtype=torch.bool)
    return mask_seg_outputs(seg_outputs, keep_mask)


def normalize_seg_outputs(
    seg_outputs: dict,
    *,
    device: torch.device | None = None,
    fallback_device: torch.device | str | None = None,
) -> dict:
    """Coerce common segmentation-output fields to stable tensor types/devices."""
    if not isinstance(seg_outputs, dict):
        return seg_outputs

    target_device = device or fallback_device or torch.device("cpu")
    if not isinstance(target_device, torch.device):
        target_device = torch.device(target_device)

    if device is None:
        for value in seg_outputs.values():
            if isinstance(value, torch.Tensor):
                target_device = value.device
                break

    def _as_tensor(value: object, *, dtype: torch.dtype | None) -> torch.Tensor | object | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            tensor = value
            if tensor.device != target_device or (dtype is not None and tensor.dtype != dtype):
                try:
                    tensor = tensor.to(device=target_device, dtype=dtype or tensor.dtype)
                except Exception:
                    return value
            return tensor
        try:
            return torch.as_tensor(value, dtype=dtype, device=target_device)
        except TypeError:
            try:
                return torch.as_tensor(value, dtype=dtype).to(device=target_device)
            except Exception:
                return None
        except Exception:
            return None

    def _as_1d(tensor: torch.Tensor | object | None) -> torch.Tensor | None:
        if not isinstance(tensor, torch.Tensor):
            return None
        if tensor.ndim == 2 and tensor.shape[1] == 1:
            tensor = tensor[:, 0]
        if tensor.ndim != 1:
            return None
        return tensor

    updated = dict(seg_outputs)

    class_ids_t = _as_1d(_as_tensor(seg_outputs.get("class_ids"), dtype=torch.long))
    if class_ids_t is not None:
        updated["class_ids"] = class_ids_t

    batch_ids_t = _as_1d(_as_tensor(seg_outputs.get("batch_ids"), dtype=torch.long))
    if batch_ids_t is not None:
        updated["batch_ids"] = batch_ids_t

    scores_t = _as_1d(_as_tensor(seg_outputs.get("scores"), dtype=torch.float32))
    if scores_t is not None:
        updated["scores"] = scores_t

    boxes_t = _as_tensor(seg_outputs.get("boxes_xyxy"), dtype=torch.float32)
    if isinstance(boxes_t, torch.Tensor) and boxes_t.ndim == 3 and boxes_t.shape[1] == 1:
        boxes_t = boxes_t[:, 0, :]
    if isinstance(boxes_t, torch.Tensor) and boxes_t.ndim == 2 and boxes_t.shape[1] >= 4:
        updated["boxes_xyxy"] = boxes_t

    means_t = _as_tensor(seg_outputs.get("means"), dtype=torch.float32)
    if isinstance(means_t, torch.Tensor):
        updated["means"] = means_t

    cov6_t = _as_tensor(seg_outputs.get("cov6"), dtype=torch.float32)
    if isinstance(cov6_t, torch.Tensor):
        updated["cov6"] = cov6_t

    masks = seg_outputs.get("masks")
    if masks is not None and not isinstance(masks, torch.Tensor):
        masks_t = _as_tensor(masks, dtype=None)
        if isinstance(masks_t, torch.Tensor):
            updated["masks"] = masks_t
    elif isinstance(masks, torch.Tensor) and masks.device != target_device:
        with contextlib.suppress(Exception):
            updated["masks"] = masks.to(device=target_device)

    return updated
