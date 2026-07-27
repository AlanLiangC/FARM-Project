"""Bounding-box and mask utilities for the streaming mapper node."""

from __future__ import annotations

import contextlib
from typing import Optional, Sequence, Tuple

import numpy as np
import torch


def assign_bbox(bbox_msg, xyxy: Sequence[float]) -> None:
    try:
        x1, y1, x2, y2 = (
            float(xyxy[0]),
            float(xyxy[1]),
            float(xyxy[2]),
            float(xyxy[3]),
        )
    except Exception:
        return
    if not np.isfinite([x1, y1, x2, y2]).all():
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


def mask_to_xyxy(mask: object) -> Optional[Tuple[float, float, float, float]]:
    """
    Compute a tight XYXY bbox from a per-detection boolean mask.

    Returns (x1, y1, x2, y2) in pixel coordinates, where x2/y2 are exclusive
    (so width = x2 - x1, height = y2 - y1).
    """
    if not isinstance(mask, torch.Tensor):
        return None
    if mask.numel() == 0:
        return None
    m = mask
    if m.ndim == 3:
        m = m.squeeze(0)
    if m.ndim != 2:
        return None
    m = m.to(dtype=torch.bool)
    with contextlib.suppress(Exception):
        if not bool(m.any().item()):
            return None

    try:
        rows = torch.any(m, dim=1)
        cols = torch.any(m, dim=0)
        if not bool(rows.any().item()) or not bool(cols.any().item()):
            return None
        y_idx = torch.nonzero(rows, as_tuple=False).view(-1)
        x_idx = torch.nonzero(cols, as_tuple=False).view(-1)
        y0 = int(y_idx[0].item())
        y1 = int(y_idx[-1].item())
        x0 = int(x_idx[0].item())
        x1 = int(x_idx[-1].item())
    except Exception:
        return None

    return (float(x0), float(y0), float(x1 + 1), float(y1 + 1))


def mask_rear_leg_regions_tensor(
    color: torch.Tensor,
    depth: torch.Tensor,
    mask_h: int,
    mask_w: int,
) -> None:
    """Zero-out rear-leg corner regions in color and depth tensors in-place."""
    if mask_h <= 0 or mask_w <= 0:
        return

    if depth is not None and isinstance(depth, torch.Tensor) and depth.ndim >= 2:
        if (depth.ndim == 2) or (depth.ndim == 3 and depth.shape[-1] == 1):
            height, width = int(depth.shape[0]), int(depth.shape[1])
        elif depth.ndim == 3 and depth.shape[0] == 1:
            height, width = int(depth.shape[1]), int(depth.shape[2])
        else:
            return
    elif color is not None and isinstance(color, torch.Tensor) and color.ndim == 3:
        if color.shape[-1] in (1, 3, 4):
            height, width = int(color.shape[0]), int(color.shape[1])
        elif color.shape[0] in (1, 3, 4):
            height, width = int(color.shape[1]), int(color.shape[2])
        else:
            return
    else:
        return

    mask_h = min(mask_h, height)
    mask_w = min(mask_w, width)
    if mask_h <= 0 or mask_w <= 0:
        return

    y0 = height - mask_h
    if color is not None and isinstance(color, torch.Tensor) and color.ndim == 3:
        if color.shape[-1] in (1, 3, 4):
            color[y0:height, 0:mask_w] = 0
            color[y0:height, width - mask_w : width] = 0
        elif color.shape[0] in (1, 3, 4):
            color[:, y0:height, 0:mask_w] = 0
            color[:, y0:height, width - mask_w : width] = 0

    if depth is not None and isinstance(depth, torch.Tensor) and depth.ndim >= 2:
        if depth.ndim == 2:
            depth[y0:height, 0:mask_w] = 0.0
            depth[y0:height, width - mask_w : width] = 0.0
        elif depth.ndim == 3 and depth.shape[-1] == 1:
            depth[y0:height, 0:mask_w, :] = 0.0
            depth[y0:height, width - mask_w : width, :] = 0.0
        elif depth.ndim == 3 and depth.shape[0] == 1:
            depth[:, y0:height, 0:mask_w] = 0.0
            depth[:, y0:height, width - mask_w : width] = 0.0
