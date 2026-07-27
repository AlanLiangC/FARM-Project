from __future__ import annotations

import math
from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as F


_VALID_VISUAL_PROMPT_MODES = {"bbox_crop", "mask_crop", "mask_composite"}


def _normalize_visual_prompt_mode(mode: str | None) -> str:
    mode_norm = str(mode or "bbox_crop").strip().lower()
    if mode_norm not in _VALID_VISUAL_PROMPT_MODES:
        return "bbox_crop"
    return mode_norm


def _to_hwc_uint8(value: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = value.detach().to("cpu", copy=False).numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).round().astype(np.uint8)
    return np.ascontiguousarray(arr)


def _to_mask_bool(value: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = value.detach().to("cpu", copy=False).numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    return np.asarray(arr, dtype=bool)


def _resize_hwc(arr: np.ndarray, *, width: int | None = None, height: int | None = None) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
        return arr
    src_h, src_w = int(arr.shape[0]), int(arr.shape[1])
    if width is None and height is None:
        return arr
    if width is None:
        scale = float(height) / float(max(1, src_h))
        width = max(1, int(round(src_w * scale)))
    if height is None:
        scale = float(width) / float(max(1, src_w))
        height = max(1, int(round(src_h * scale)))
    dst_h = max(1, int(height))
    dst_w = max(1, int(width))
    if dst_h == src_h and dst_w == src_w:
        return arr
    tensor = torch.as_tensor(arr).permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=(dst_h, dst_w), mode="bilinear", align_corners=False)
    out = resized.squeeze(0).permute(1, 2, 0).clamp(0, 255).round().to(torch.uint8).cpu().numpy()
    return np.ascontiguousarray(out)


def _resize_mask(mask: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if mask.ndim != 2 or mask.shape[0] <= 0 or mask.shape[1] <= 0:
        return np.zeros((max(1, int(height)), max(1, int(width))), dtype=bool)
    dst_h = max(1, int(height))
    dst_w = max(1, int(width))
    if int(mask.shape[0]) == dst_h and int(mask.shape[1]) == dst_w:
        return np.asarray(mask, dtype=bool)
    tensor = torch.as_tensor(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(dst_h, dst_w), mode="nearest")
    return resized.squeeze(0).squeeze(0).cpu().numpy().astype(bool)


def _resize_long_side_hwc(arr: np.ndarray, long_side: int) -> tuple[np.ndarray, float]:
    if arr.ndim != 3 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
        return arr, 1.0
    target = max(1, int(long_side))
    src_h, src_w = int(arr.shape[0]), int(arr.shape[1])
    longest = max(src_h, src_w)
    if longest <= 0:
        return arr, 1.0
    scale = float(target) / float(longest)
    if abs(scale - 1.0) < 1e-6:
        return arr, 1.0
    return _resize_hwc(arr, width=max(1, int(round(src_w * scale))), height=max(1, int(round(src_h * scale)))), scale


def _mask_outline(mask: np.ndarray, px: int) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2 or not bool(mask_bool.any()):
        return np.zeros_like(mask_bool, dtype=bool)
    radius = max(1, int(px))
    tensor = torch.as_tensor(mask_bool.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    dilated = F.max_pool2d(tensor, kernel_size=2 * radius + 1, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - tensor, kernel_size=2 * radius + 1, stride=1, padding=radius)
    outline = (dilated - eroded).squeeze(0).squeeze(0).cpu().numpy() > 0.0
    return outline


def _highlight_crop(
    image_hwc: np.ndarray,
    mask_hw: np.ndarray,
    *,
    fill_alpha: float,
    outline_px: int,
) -> np.ndarray:
    image = _to_hwc_uint8(image_hwc).astype(np.float32)
    mask = np.asarray(mask_hw, dtype=bool)
    if image.ndim != 3 or image.shape[-1] != 3:
        return _to_hwc_uint8(image_hwc)
    if mask.shape != image.shape[:2]:
        mask = _resize_mask(mask, width=image.shape[1], height=image.shape[0])
    fill_color = np.asarray([255.0, 64.0, 32.0], dtype=np.float32)
    outline_color = np.asarray([255.0, 0.0, 0.0], dtype=np.float32)
    alpha = min(1.0, max(0.0, float(fill_alpha)))
    if alpha > 0.0 and bool(mask.any()):
        image[mask] = (1.0 - alpha) * image[mask] + alpha * fill_color
    outline = _mask_outline(mask, outline_px)
    if bool(outline.any()):
        image[outline] = outline_color
    return np.ascontiguousarray(np.clip(image, 0, 255).round().astype(np.uint8))


def _scale_bbox(bbox: tuple[float, float, float, float], scale: float) -> tuple[float, float, float, float]:
    return tuple(float(x) * float(scale) for x in bbox)  # type: ignore[return-value]


def _compose_mask_composite(
    context_hwc: np.ndarray,
    target_hwc: np.ndarray,
    target_mask: np.ndarray,
    target_bbox: tuple[float, float, float, float],
    *,
    context_thumbnail_width: int,
    target_long_side: int,
    fill_alpha: float,
    outline_px: int,
) -> tuple[np.ndarray, tuple[float, float, float, float], tuple[int, int], np.ndarray]:
    context = _to_hwc_uint8(context_hwc)
    context_w = max(1, int(context_thumbnail_width))
    context = _resize_hwc(context, width=context_w)

    target = _to_hwc_uint8(target_hwc)
    target_resized, target_scale = _resize_long_side_hwc(target, max(1, int(target_long_side)))
    mask_resized = _resize_mask(target_mask, width=target_resized.shape[1], height=target_resized.shape[0])
    highlighted = _highlight_crop(
        target_resized,
        mask_resized,
        fill_alpha=fill_alpha,
        outline_px=outline_px,
    )
    target_bbox_resized = _scale_bbox(target_bbox, target_scale)

    gap = 8
    height = max(int(context.shape[0]), int(highlighted.shape[0]))
    width = int(context.shape[1]) + gap + int(highlighted.shape[1])
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    context_y = (height - int(context.shape[0])) // 2
    target_y = (height - int(highlighted.shape[0])) // 2
    target_x = int(context.shape[1]) + gap
    canvas[context_y:context_y + context.shape[0], 0:context.shape[1]] = context
    canvas[target_y:target_y + highlighted.shape[0], target_x:target_x + highlighted.shape[1]] = highlighted

    x0, y0, x1, y1 = target_bbox_resized
    bbox = (
        float(x0 + target_x),
        float(y0 + target_y),
        float(x1 + target_x),
        float(y1 + target_y),
    )
    return canvas, bbox, (int(width), int(height)), mask_resized


def _build_context_crop(
    color_chw: torch.Tensor,
    mask_bool: torch.Tensor,
    pad_ratio: float = 0.5,
    min_bbox_side: int = 96,
    visual_prompt_mode: str = "bbox_crop",
    context_thumbnail_width: int = 320,
    target_long_side: int = 512,
    mask_fill_alpha: float = 0.25,
    mask_outline_px: int = 4,
) -> dict | None:
    """
    Build a context crop with two stages:
    1) Expand the bbox by pad_ratio and clamp to the image bounds instead of
       padding with empty pixels. For runtime control, if the padded crop's
       scaled side would exceed min_bbox_side * (1 + 2*pad_ratio), clip the
       padded crop to that maximum side length.
    2) If the bbox shortest side exceeds min_bbox_side, downscale the crop so the
       bbox shortest side becomes min_bbox_side, for caption runtime efficiency.

    The returned observation keeps the original-resolution crop in `image`.
    If downscaling is applied, `image_caption` stores the reduced crop used by
    captioning, with matching `bbox_caption` and `size_caption`.
    """
    if mask_bool.dtype is not torch.bool:
        mask_bool = mask_bool.to(dtype=torch.bool, device=color_chw.device)

    if mask_bool.ndim != 2:
        return None
    if color_chw.ndim != 3:
        return None
    if color_chw.shape[1:] != mask_bool.shape:
        return None

    # Bounding box of the mask (inclusive)
    ys, xs = torch.nonzero(mask_bool, as_tuple=True)
    if ys.numel() == 0:
        return None
    y_min = int(ys.min().item())
    y_max = int(ys.max().item())
    x_min = int(xs.min().item())
    x_max = int(xs.max().item())

    box_w = x_max - x_min + 1
    box_h = y_max - y_min + 1
    bbox_area_source = int(max(0, box_w) * max(0, box_h))

    bbox_short = float(min(box_w, box_h))
    effective_pad_ratio = float(pad_ratio)
    if bbox_short < 96.0:
        effective_pad_ratio = float(pad_ratio) * 1.0

    pad_w = int(round(box_w * effective_pad_ratio))
    pad_h = int(round(box_h * effective_pad_ratio))

    desired_w = int(box_w + 2 * pad_w)
    desired_h = int(box_h + 2 * pad_h)

    H, W = mask_bool.shape

    # Runtime control: downscale so bbox shortest side is ~min_bbox_side.
    scale = 1.0
    if bbox_short > float(min_bbox_side):
        scale = float(min_bbox_side) / bbox_short

    # Clip the padded crop to bound runtime/memory. The cap is applied in the
    # scaled space, then mapped back to the source crop size.
    max_side_scaled = float(min_bbox_side) * (1.0 + 2.0 * float(pad_ratio))
    if desired_w > 0 and desired_h > 0 and max_side_scaled > 0.0:
        max_side_source = int(math.floor(max_side_scaled / max(scale, 1e-6)))
        if max_side_source > 0:
            desired_w = min(desired_w, max_side_source)
            desired_h = min(desired_h, max_side_source)

    desired_w = int(max(min(desired_w, W), box_w))
    desired_h = int(max(min(desired_h, H), box_h))

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    x0 = int(math.floor(cx - desired_w / 2.0))
    y0 = int(math.floor(cy - desired_h / 2.0))

    # Keep the crop window centered on the object, but clamp to the image bounds
    # (i.e., do NOT shift the window back into the image).
    x1 = x0 + desired_w
    y1 = y0 + desired_h

    crop_x0 = int(max(0, x0))
    crop_y0 = int(max(0, y0))
    crop_x1 = int(min(W, x1))
    crop_y1 = int(min(H, y1))

    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        return None

    crop_full = color_chw[:, crop_y0:crop_y1, crop_x0:crop_x1]
    mask_full = mask_bool[crop_y0:crop_y1, crop_x0:crop_x1]
    if crop_full.numel() == 0:
        return None

    bbox_crop_full = (
        float(x_min - crop_x0),
        float(y_min - crop_y0),
        float(x_max - crop_x0),
        float(y_max - crop_y0),
    )

    crop_caption = crop_full
    mask_caption = mask_full
    bbox_crop_caption = bbox_crop_full
    if scale < 1.0:
        new_h = max(1, int(round(crop_full.shape[1] * scale)))
        new_w = max(1, int(round(crop_full.shape[2] * scale)))
        crop_caption = F.interpolate(
            crop_full.float().unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        crop_caption = crop_caption.clamp(0, 255).round().to(dtype=torch.uint8)
        mask_caption = F.interpolate(
            mask_full.float().unsqueeze(0).unsqueeze(0),
            size=(new_h, new_w),
            mode="nearest",
        ).squeeze(0).squeeze(0).to(dtype=torch.bool)
        bbox_crop_caption = tuple(coord * scale for coord in bbox_crop_full)

    crop_full_hwc = crop_full.permute(1, 2, 0).contiguous()
    mask_full_hw = mask_full.contiguous()
    full_h, full_w = int(crop_full_hwc.shape[0]), int(crop_full_hwc.shape[1])
    bx0_full = float(np.clip(bbox_crop_full[0], 0, max(full_w - 1, 0)))
    by0_full = float(np.clip(bbox_crop_full[1], 0, max(full_h - 1, 0)))
    bx1_full = float(np.clip(bbox_crop_full[2], bx0_full, max(full_w - 1, bx0_full)))
    by1_full = float(np.clip(bbox_crop_full[3], by0_full, max(full_h - 1, by0_full)))

    if scale < 1.0:
        crop_caption_hwc = crop_caption.permute(1, 2, 0).contiguous()
        mask_caption_hw = mask_caption.contiguous()
        cap_h, cap_w = int(crop_caption_hwc.shape[0]), int(crop_caption_hwc.shape[1])
        bx0_cap = float(np.clip(bbox_crop_caption[0], 0, max(cap_w - 1, 0)))
        by0_cap = float(np.clip(bbox_crop_caption[1], 0, max(cap_h - 1, 0)))
        bx1_cap = float(np.clip(bbox_crop_caption[2], bx0_cap, max(cap_w - 1, bx0_cap)))
        by1_cap = float(np.clip(bbox_crop_caption[3], by0_cap, max(cap_h - 1, by0_cap)))
    else:
        crop_caption_hwc = crop_full_hwc
        mask_caption_hw = mask_full_hw
        cap_w, cap_h = full_w, full_h
        bx0_cap, by0_cap, bx1_cap, by1_cap = bx0_full, by0_full, bx1_full, by1_full

    clean_caption_hwc = _to_hwc_uint8(crop_caption_hwc)
    clean_mask_caption = _to_mask_bool(mask_caption_hw)
    visual_mode = _normalize_visual_prompt_mode(visual_prompt_mode)

    image_caption_out: Any = crop_caption_hwc
    mask_caption_out: Any = mask_caption_hw
    bbox_caption_out = (bx0_cap, by0_cap, bx1_cap, by1_cap)
    size_caption_out = (cap_w, cap_h)

    if visual_mode == "mask_crop":
        highlighted = _highlight_crop(
            clean_caption_hwc,
            clean_mask_caption,
            fill_alpha=float(mask_fill_alpha),
            outline_px=int(mask_outline_px),
        )
        image_caption_out = highlighted
        mask_caption_out = clean_mask_caption
        bbox_caption_out = (bx0_cap, by0_cap, bx1_cap, by1_cap)
        size_caption_out = (int(highlighted.shape[1]), int(highlighted.shape[0]))
    elif visual_mode == "mask_composite":
        context_hwc = _to_hwc_uint8(color_chw)
        composite, comp_bbox, comp_size, comp_mask = _compose_mask_composite(
            context_hwc,
            clean_caption_hwc,
            clean_mask_caption,
            (bx0_cap, by0_cap, bx1_cap, by1_cap),
            context_thumbnail_width=int(context_thumbnail_width),
            target_long_side=int(target_long_side),
            fill_alpha=float(mask_fill_alpha),
            outline_px=int(mask_outline_px),
        )
        image_caption_out = composite
        mask_caption_out = comp_mask
        bbox_caption_out = comp_bbox
        size_caption_out = comp_size

    return {
        "image": crop_full_hwc,
        "mask": mask_full_hw,
        "bbox": (bx0_full, by0_full, bx1_full, by1_full),
        "size": (full_w, full_h),
        "image_caption": image_caption_out,
        "mask_caption": mask_caption_out,
        "bbox_caption": bbox_caption_out,
        "size_caption": size_caption_out,
        "image_embedding": clean_caption_hwc,
        "mask_embedding": clean_mask_caption,
        "bbox_embedding": (bx0_cap, by0_cap, bx1_cap, by1_cap),
        "size_embedding": (cap_w, cap_h),
        "embedding_encoding": "rgb8",
        "caption_visual_prompt_mode": visual_mode,
        # Original (pre-crop/pre-resize) bbox, useful for picking views where the object is largest.
        "bbox_source": (int(x_min), int(y_min), int(x_max), int(y_max)),
        "size_source": (int(W), int(H)),
        "bbox_area_source": bbox_area_source,
    }


def compute_caption_observations(
    seg_outputs: dict[str, Any],
    colors: List[torch.Tensor],
    pad_ratio: float = 0.5,
    min_bbox_side: int = 96,
    visual_prompt_mode: str = "bbox_crop",
    context_thumbnail_width: int = 320,
    target_long_side: int = 512,
    mask_fill_alpha: float = 0.25,
    mask_outline_px: int = 4,
) -> list[Any]:
    """
    Compute per-detection crops with context padding.
    Crops are expanded for context, clamped to the source image (no black padding),
    and include the original-resolution crop in `image`.
    A downscaled caption-specific crop is provided in `image_caption`.

    Returns list aligned with detections; entries are dicts containing image/bbox/size.
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

        obs = _build_context_crop(
            color_chw,
            mask_bool,
            pad_ratio=pad_ratio,
            min_bbox_side=min_bbox_side,
            visual_prompt_mode=visual_prompt_mode,
            context_thumbnail_width=context_thumbnail_width,
            target_long_side=target_long_side,
            mask_fill_alpha=mask_fill_alpha,
            mask_outline_px=mask_outline_px,
        )
        if isinstance(obs, dict):
            # Keep caption crops off the GPU to avoid holding VRAM between batches.
            for key in ("image", "image_caption", "image_embedding", "mask", "mask_caption", "mask_embedding"):
                val = obs.get(key)
                if isinstance(val, torch.Tensor) and val.device.type == "cuda":
                    obs[key] = val.detach().cpu()
        observations.append(obs)

    return observations
