"""IRef-VLA ground-truth instance + region loaders (HM3D split).

Each scene ships ``<scene>_object_result.csv`` and ``<scene>_region_result.csv``.
Both encode an oriented bbox via ``center + xyz lengths + heading angle``
(rotation is around the world Z axis only). For 3D IoU evaluation we expand
to an axis-aligned bbox by computing the 8 OBB corner points and taking
their min/max.

Object CSV header (HM3D sample, 33 columns):
``object_id, region_id, raw_label, nyu_id, nyu40_id, nyu_label, nyu40_label,
  object_bbox_cx, object_bbox_cy, object_bbox_cz,
  object_bbox_xlength, object_bbox_ylength, object_bbox_zlength,
  object_bbox_heading, object_front_heading,
  object_color_r1, object_color_g1, object_color_b1,
  object_color_scheme1, object_color_scheme_percentage1,
  object_color_scheme_average_dist1,
  object_color_r2, ...   (similar for color 2 + 3)``

Region CSV header (9 columns):
``region_id, region_label,
  region_bbox_cx, region_bbox_cy, region_bbox_cz,
  region_bbox_xlength, region_bbox_ylength, region_bbox_zlength,
  region_bbox_heading``
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .dataset import _object_csv_path, _region_csv_path


@dataclass(frozen=True)
class GTInstance:
    """One IRef-VLA GT object (axis-aligned bbox derived from OBB corners)."""

    instance_id: int                # IRef-VLA object_id (CSV)
    region_id: int                  # IRef-VLA region_id (CSV)
    raw_label: str                  # 'sofa seat', 'lamp stand', ...
    nyu40_label: str                # nyu40-style class
    bbox_min: np.ndarray            # (3,) float32, AABB of OBB corners
    bbox_max: np.ndarray            # (3,) float32
    obb_center: np.ndarray          # (3,) float32 OBB center
    obb_extent: np.ndarray          # (3,) float32 OBB extent (xyz lengths)
    obb_heading: float              # rotation around world Z axis (radians)
    color_labels: Tuple[str, str, str] = field(default_factory=lambda: ("", "", ""))

    @property
    def label(self) -> str:
        """Best-effort class label — prefer the raw label when present."""
        return self.raw_label or self.nyu40_label

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.bbox_min + self.bbox_max)

    @property
    def extent(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min


@dataclass(frozen=True)
class RegionInfo:
    """One IRef-VLA region (room) bbox + label."""

    region_id: int
    label: str                      # 'Living room', 'Bedroom', 'Unknown room', ...
    bbox_min: np.ndarray            # (3,) float32, AABB of OBB corners
    bbox_max: np.ndarray            # (3,) float32
    obb_center: np.ndarray
    obb_extent: np.ndarray
    obb_heading: float


def aabb_from_obb(
    center: np.ndarray,
    extent: np.ndarray,
    heading: float,
    *,
    to_habitat_frame: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box of an oriented box rotated around Z.

    Args:
        center: (3,) box center (xyz, IRef-VLA frame: Z-up).
        extent: (3,) full xyz lengths.
        heading: rotation around world Z axis, in radians.
        to_habitat_frame: When True (default), convert from IRef-VLA's Z-up
            world frame to habitat-sim's Y-up world frame so the resulting
            AABB is directly comparable to scene-graph predictions built from
            habitat renders. Empirically validated on HM3D scene 00009: X
            axis matches identically; Y_iref → -Z_hab; Z_iref → Y_hab.

    Returns:
        ``(min, max)`` of the 8 OBB corner points (in the requested frame).
    """
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    lx, ly, lz = float(extent[0]) / 2.0, float(extent[1]) / 2.0, float(extent[2]) / 2.0
    c = float(np.cos(heading))
    s = float(np.sin(heading))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    # Local-frame corners (8 × 3)
    local = np.array(
        [
            [-lx, +ly, +lz], [+lx, +ly, +lz], [+lx, -ly, +lz], [-lx, -ly, +lz],
            [-lx, +ly, -lz], [+lx, +ly, -lz], [+lx, -ly, -lz], [-lx, -ly, -lz],
        ],
        dtype=np.float64,
    )
    world = (R @ local.T).T + np.array([cx, cy, cz], dtype=np.float64)
    if to_habitat_frame:
        # IRef-VLA (X, Y, Z) → habitat (X, Z, -Y).
        x = world[:, 0]
        y_h = world[:, 2]                # vertical
        z_h = -world[:, 1]
        world = np.stack([x, y_h, z_h], axis=1)
    bbox_min = world.min(axis=0).astype(np.float32)
    bbox_max = world.max(axis=0).astype(np.float32)
    return bbox_min, bbox_max


def _safe_float(value: object) -> float:
    try:
        s = str(value).strip()
        if not s or s == "_":
            return 0.0
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _safe_int(value: object) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


def _color_label_or_empty(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s or s == "_":
        return ""
    return s


def load_scene_objects(
    scene_id: str,
    *,
    dataset_root: Optional[Path] = None,
) -> Dict[int, GTInstance]:
    """Return ``{object_id: GTInstance}`` for ``scene_id``."""
    path = _object_csv_path(scene_id, dataset_root)
    out: Dict[int, GTInstance] = {}
    if not path.exists():
        return out
    with path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            oid = _safe_int(row.get("object_id"))
            if oid is None:
                continue
            rid = _safe_int(row.get("region_id")) or -1
            center = np.array(
                [
                    _safe_float(row.get("object_bbox_cx")),
                    _safe_float(row.get("object_bbox_cy")),
                    _safe_float(row.get("object_bbox_cz")),
                ],
                dtype=np.float64,
            )
            extent = np.array(
                [
                    _safe_float(row.get("object_bbox_xlength")),
                    _safe_float(row.get("object_bbox_ylength")),
                    _safe_float(row.get("object_bbox_zlength")),
                ],
                dtype=np.float64,
            )
            heading = _safe_float(row.get("object_bbox_heading"))
            bbox_min, bbox_max = aabb_from_obb(center, extent, heading)
            colors = (
                _color_label_or_empty(row.get("object_color_scheme1")),
                _color_label_or_empty(row.get("object_color_scheme2")),
                _color_label_or_empty(row.get("object_color_scheme3")),
            )
            out[oid] = GTInstance(
                instance_id=oid,
                region_id=rid,
                raw_label=str(row.get("raw_label", "") or ""),
                nyu40_label=str(row.get("nyu40_label", "") or ""),
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                obb_center=center.astype(np.float32),
                obb_extent=extent.astype(np.float32),
                obb_heading=float(heading),
                color_labels=colors,
            )
    return out


def load_scene_regions(
    scene_id: str,
    *,
    dataset_root: Optional[Path] = None,
) -> Dict[int, RegionInfo]:
    """Return ``{region_id: RegionInfo}`` for ``scene_id``."""
    path = _region_csv_path(scene_id, dataset_root)
    out: Dict[int, RegionInfo] = {}
    if not path.exists():
        return out
    with path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rid = _safe_int(row.get("region_id"))
            if rid is None:
                continue
            center = np.array(
                [
                    _safe_float(row.get("region_bbox_cx")),
                    _safe_float(row.get("region_bbox_cy")),
                    _safe_float(row.get("region_bbox_cz")),
                ],
                dtype=np.float64,
            )
            extent = np.array(
                [
                    _safe_float(row.get("region_bbox_xlength")),
                    _safe_float(row.get("region_bbox_ylength")),
                    _safe_float(row.get("region_bbox_zlength")),
                ],
                dtype=np.float64,
            )
            heading = _safe_float(row.get("region_bbox_heading"))
            bbox_min, bbox_max = aabb_from_obb(center, extent, heading)
            out[rid] = RegionInfo(
                region_id=rid,
                label=str(row.get("region_label", "") or ""),
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                obb_center=center.astype(np.float32),
                obb_extent=extent.astype(np.float32),
                obb_heading=float(heading),
            )
    return out


def list_object_ids(scene_id: str, *, dataset_root: Optional[Path] = None) -> List[int]:
    """Cheap listing of object IDs without instantiating GTInstance objects."""
    return sorted(load_scene_objects(scene_id, dataset_root=dataset_root).keys())
