#!/usr/bin/env python3
"""Render an HM3D scene to NPZ frames with habitat-sim.

Runs OUTSIDE our docker — in the host's ``apg`` micromamba env which already
has ``habitat-sim 0.2.5`` installed. Output NPZ frames are in the format
:class:`scene_graph.offline.frame_sources.npz.NPZFrameSource` (and
:class:`scene_graph.datasets.npz.NPZDataset`) expects.

Trajectory strategy: **region-aware navmesh sampling**. For each region listed
in IRef-VLA's ``<scene>_region_result.csv``, sample K navigable points whose
XY coordinates fall inside the region's bbox, then build a path-planned tour on
Habitat-sim's navmesh. In trajectory mode, each interpolated pose is on the
navmesh and the yaw can be selected by an object-coverage heuristic so views
face nearby annotated objects instead of arbitrary directions. By default each
trajectory pose is rendered with four yaw rotations, preserving the
360-degree view coverage of the original region renderer while keeping
robot-like translation.

Usage::

    # in a habitat-sim (~0.2.5) environment, on the host (not in docker)
    python scripts/render_hm3d_trajectory.py \
        --scene-id      00238-j6fHrce9pHR \
        --hm3d-root     /path/to/hm3d \
        --iref-vla-root /path/to/iref_vla/HM3D \
        --out           /path/to/rendered_trajectory/00238-j6fHrce9pHR \
        --points-per-region 8 \
        --num-yaws 4 \
        --image-size 640 480 \
        --eye-height 1.5

Output::

    <out>/frames_000.npz   (one chunk; bigger scenes may produce multiple)
    <out>/render_meta.json (counts, per-region tally, intrinsics, eye-height)

NPZ schema per chunk::

    images          : (N, H, W, 3) uint8
    depths          : (N, H, W)    float32   metres
    camtoworlds     : (N, 4, 4)    float32   OpenGL convention (camtoworld)
    K               : (3, 3)       float32   shared
    pose_convention : 'opengl'                (scalar string)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("render_hm3d_trajectory")


# ---------------------------------------------------------------------
# IRef-VLA region helpers (kept dep-free so we can run in apg env)
# ---------------------------------------------------------------------


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


def load_iref_vla_regions(
    scene_id: str, iref_vla_root: Path
) -> List[Dict]:
    """Read IRef-VLA region_result.csv → list of region dicts (no scene_graph dep).

    Each entry has: region_id, label, center (xyz), extent (xyz), heading.
    """
    csv_path = iref_vla_root / scene_id / f"{scene_id}_region_result.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"IRef-VLA region CSV not found: {csv_path}")
    out: List[Dict] = []
    with csv_path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rid = _safe_int(row.get("region_id"))
            if rid is None:
                continue
            cx = _safe_float(row.get("region_bbox_cx"))
            cy = _safe_float(row.get("region_bbox_cy"))
            cz = _safe_float(row.get("region_bbox_cz"))
            lx = _safe_float(row.get("region_bbox_xlength"))
            ly = _safe_float(row.get("region_bbox_ylength"))
            lz = _safe_float(row.get("region_bbox_zlength"))
            heading = _safe_float(row.get("region_bbox_heading"))
            out.append(
                {
                    "region_id": rid,
                    "label": str(row.get("region_label", "") or ""),
                    "center": (cx, cy, cz),
                    "extent": (lx, ly, lz),
                    "heading": heading,
                }
            )
    return out


STRUCTURAL_OBJECT_LABELS = {
    "ceiling",
    "floor",
    "wall",
    "walls",
    "room",
    "unknown",
}


def _iref_to_habitat_point(x: float, y: float, z: float) -> np.ndarray:
    """Convert IRef-VLA/HM3D Z-up point to Habitat-sim Y-up point."""
    return np.array([float(x), float(z), float(-y)], dtype=np.float32)


def load_iref_vla_objects(scene_id: str, iref_vla_root: Path) -> List[Dict]:
    """Read IRef-VLA object_result.csv for yaw selection.

    The renderer only uses object centers and sizes to choose camera heading;
    object positions are not used to alter camera positions, which remain
    navmesh/pathfinder controlled.
    """
    csv_path = iref_vla_root / scene_id / f"{scene_id}_object_result.csv"
    if not csv_path.exists():
        return []
    out: List[Dict] = []
    with csv_path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            oid = _safe_int(row.get("object_id"))
            rid = _safe_int(row.get("region_id"))
            if oid is None:
                continue
            label = str(
                row.get("nyu40_label")
                or row.get("nyu_label")
                or row.get("raw_label")
                or ""
            ).strip().lower()
            if label in STRUCTURAL_OBJECT_LABELS:
                continue
            cx = _safe_float(row.get("object_bbox_cx"))
            cy = _safe_float(row.get("object_bbox_cy"))
            cz = _safe_float(row.get("object_bbox_cz"))
            lx = max(0.0, _safe_float(row.get("object_bbox_xlength")))
            ly = max(0.0, _safe_float(row.get("object_bbox_ylength")))
            lz = max(0.0, _safe_float(row.get("object_bbox_zlength")))
            if lx <= 0.0 and ly <= 0.0 and lz <= 0.0:
                continue
            out.append(
                {
                    "object_id": oid,
                    "region_id": rid,
                    "label": label,
                    "center_hab": _iref_to_habitat_point(cx, cy, cz),
                    "extent": (lx, ly, lz),
                    "size_score": max(0.1, math.sqrt(max(lx * ly, lx * lz, ly * lz, 1.0e-4))),
                }
            )
    return out


# ---------------------------------------------------------------------
# Magnet-viewpoint helpers (per-object guaranteed coverage + multi-floor)
# ---------------------------------------------------------------------


# Same hard-blocked classes as ``scene_graph.eval.fair_gt_filter`` plus the
# IRef-VLA structural superset; we copy them inline so the renderer stays
# importable in the bare ``apg`` env without our package on PYTHONPATH.
_FAIR_BLOCKED_CLASSES = {
    "wall", "walls", "ceiling", "ceilings", "floor", "floors",
    "picture", "pictures", "painting", "paintings", "poster", "posters",
    "mirror", "mirrors", "framed picture", "wall calendar", "calendar",
    "book", "books", "magazine", "magazines", "paper", "papers",
    "cardboard", "unknown", "object", "room",
}


def _keep_as_magnet_target(label: str, extent_xyz: Sequence[float]) -> bool:
    """Inline copy of ``fair_gt_filter.keep_fair_gt`` (lenient bound)."""
    lab = (label or "").strip().lower()
    if lab in _FAIR_BLOCKED_CLASSES:
        return False
    if len(extent_xyz) < 3:
        return False
    dims = sorted(float(x) for x in extent_xyz[:3])
    if dims[0] < 0.05:
        return False
    if dims[-1] > 5.0:
        return False
    if dims[-1] >= 1e-6 and (dims[-1] / max(dims[0], 1e-6)) > 15.0:
        return False
    return True


def load_statement_referenced_object_ids(
    scene_id: str, iref_vla_root: Path
) -> set:
    """Return the set of object_ids referenced anywhere in this scene's
    referential statements (as target_index or anchor.index).

    The renderer will magnet these objects even if they fail the lenient fair
    filter — they're what the benchmark actually asks about.
    """
    path = iref_vla_root / scene_id / f"{scene_id}_referential_statements.json"
    if not path.exists():
        return set()
    try:
        blob = json.loads(path.read_text())
    except Exception:
        return set()
    regions = blob.get("regions") or {}
    out: set = set()
    if not isinstance(regions, dict):
        return out
    for region_dict in regions.values():
        if not isinstance(region_dict, dict):
            continue
        for variants in region_dict.values():
            if not isinstance(variants, list):
                continue
            for v in variants:
                if not isinstance(v, dict):
                    continue
                t = v.get("target_index")
                try:
                    out.add(int(str(t)))
                except (ValueError, TypeError):
                    pass
                anchors = v.get("anchors") or {}
                if isinstance(anchors, dict):
                    for ad in anchors.values():
                        if isinstance(ad, dict):
                            ai = ad.get("index")
                            try:
                                out.add(int(str(ai)))
                            except (ValueError, TypeError):
                                pass
    return out


def load_magnet_targets(
    scene_id: str,
    iref_vla_root: Path,
    *,
    include_statement_refs: bool = True,
) -> List[Dict]:
    """Return GTs that deserve a guaranteed viewpoint.

    Union of two sets:
      1. ``keep_fair_gt``-passing objects (drops walls/ceilings/pictures/books
         and degenerate-size boxes).
      2. (optional) Any object referenced as ``target_index`` or anchor in any
         referential statement for this scene.

    Each entry has the same shape as ``load_iref_vla_objects`` plus:
      - ``is_fair``: bool
      - ``is_referenced``: bool
    """
    csv_path = iref_vla_root / scene_id / f"{scene_id}_object_result.csv"
    if not csv_path.exists():
        return []
    referenced = (
        load_statement_referenced_object_ids(scene_id, iref_vla_root)
        if include_statement_refs else set()
    )
    out: List[Dict] = []
    with csv_path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            oid = _safe_int(row.get("object_id"))
            if oid is None:
                continue
            rid = _safe_int(row.get("region_id"))
            raw_label = str(row.get("raw_label", "") or "").strip().lower()
            nyu40_label = str(row.get("nyu40_label", "") or "").strip().lower()
            label = raw_label or nyu40_label
            cx = _safe_float(row.get("object_bbox_cx"))
            cy = _safe_float(row.get("object_bbox_cy"))
            cz = _safe_float(row.get("object_bbox_cz"))
            lx = max(0.0, _safe_float(row.get("object_bbox_xlength")))
            ly = max(0.0, _safe_float(row.get("object_bbox_ylength")))
            lz = max(0.0, _safe_float(row.get("object_bbox_zlength")))
            if lx <= 0.0 and ly <= 0.0 and lz <= 0.0:
                continue
            extent = (lx, ly, lz)
            is_fair = _keep_as_magnet_target(label, extent)
            is_referenced = oid in referenced
            if not is_fair and not is_referenced:
                continue
            out.append(
                {
                    "object_id": oid,
                    "region_id": rid,
                    "label": label,
                    "center_hab": _iref_to_habitat_point(cx, cy, cz),
                    "extent_hab": np.array(
                        # IRef-VLA (lx, ly, lz) → habitat (lx, lz, ly)
                        # (X stays X; vertical Z becomes habitat Y; Y becomes -Z magnitude.)
                        [lx, lz, ly],
                        dtype=np.float32,
                    ),
                    "size_score": max(0.1, math.sqrt(max(lx * ly, lx * lz, ly * lz, 1.0e-4))),
                    "is_fair": is_fair,
                    "is_referenced": is_referenced,
                }
            )
    return out


def _island_of_object(
    pathfinder, gt_center_hab: np.ndarray, *, max_drop: float = 4.0
) -> Tuple[int, Optional[np.ndarray]]:
    """Identify the navmesh island (floor) for a GT object.

    Returns ``(island_id, floor_point)`` where ``floor_point`` is the
    object's projection onto the navmesh on the chosen floor (Habitat Y is
    the navmesh height). Returns ``(-1, None)`` if no navmesh point exists
    within ``max_drop`` metres below the object.
    """
    # Try a direct snap first. snap_point may grab a navmesh point at the
    # wrong floor, but if it succeeds we get a valid island id.
    snapped = pathfinder.snap_point(gt_center_hab)
    if snapped is not None and all(math.isfinite(v) for v in snapped):
        cand = np.asarray(snapped, dtype=np.float32)
        if cand[1] <= gt_center_hab[1] + 0.1 and cand[1] >= gt_center_hab[1] - max_drop:
            try:
                isl = int(pathfinder.get_island(cand))
            except Exception:
                isl = -1
            if isl >= 0:
                return isl, cand
    # Sweep candidate seed Y from gt_y down to gt_y - max_drop and find the
    # closest navmesh sample that is *below* the object center.
    best: Optional[np.ndarray] = None
    best_isl = -1
    for dy in np.linspace(0.05, max_drop, 9):
        seed = np.array(
            [float(gt_center_hab[0]), float(gt_center_hab[1] - dy), float(gt_center_hab[2])],
            dtype=np.float32,
        )
        s = pathfinder.snap_point(seed)
        if s is None or not all(math.isfinite(v) for v in s):
            continue
        arr = np.asarray(s, dtype=np.float32)
        if arr[1] > gt_center_hab[1] + 0.05:
            continue
        if arr[1] < gt_center_hab[1] - max_drop:
            continue
        try:
            isl = int(pathfinder.get_island(arr))
        except Exception:
            isl = -1
        if isl < 0:
            continue
        # Prefer the highest-Y (closest-floor) hit.
        if best is None or arr[1] > best[1]:
            best = arr
            best_isl = isl
    return best_isl, best


def _five_ray_occlusion_check(
    sim,
    camera_pos: np.ndarray,
    obj_center: np.ndarray,
    obj_extent: np.ndarray,
    *,
    min_passes: int,
    distance_eps: float = 0.10,
) -> Tuple[bool, int]:
    """Return ``(accept, n_pass)``: shoot 5 rays from ``camera_pos`` toward
    ``obj_center`` and 4 horizontally-offset points around it; count how
    many reach the object surface within ``distance_eps`` metres of the
    expected distance.
    """
    import habitat_sim

    halfw = 0.4 * float(obj_extent[0])
    halfd = 0.4 * float(obj_extent[2])
    targets = [
        obj_center,
        obj_center + np.array([+halfw, 0.0, 0.0], dtype=np.float32),
        obj_center + np.array([-halfw, 0.0, 0.0], dtype=np.float32),
        obj_center + np.array([0.0, 0.0, +halfd], dtype=np.float32),
        obj_center + np.array([0.0, 0.0, -halfd], dtype=np.float32),
    ]
    n_pass = 0
    for tgt in targets:
        delta = tgt - camera_pos
        dist = float(np.linalg.norm(delta))
        if dist < 0.10:
            continue
        direction = (delta / dist).astype(np.float32)
        ray = habitat_sim.geo.Ray()
        ray.origin = camera_pos.astype(np.float32)
        ray.direction = direction
        try:
            results = sim.cast_ray(ray, max_distance=dist + 0.5)
        except Exception:
            continue
        hits = getattr(results, "hits", []) or []
        # First-hit distance: empty hits => unbounded line of sight, treat as
        # pass (no occluder).
        if not hits:
            n_pass += 1
            continue
        first = hits[0]
        hit_d = float(getattr(first, "ray_distance", float("inf")))
        # Accept when the first hit lands on (or just past) the object.
        if hit_d >= dist - distance_eps:
            n_pass += 1
    return n_pass >= min_passes, n_pass


def _build_magnet_views(
    sim,
    pathfinder,
    magnet_targets: Sequence[Dict],
    *,
    eye_height: float,
    hfov_deg: float,
    r_min: float,
    r_max: float,
    n_candidates: int,
    occl_min_passes: int,
    rng: np.random.Generator,
) -> Tuple[List[Dict], Dict[str, int]]:
    """For each magnet target, pick one navmesh viewpoint and yaw.

    Returns ``(views, stats)`` where each view is a dict with
    ``{pos, yaw, label, region_id, is_waypoint=True, island_id, object_id,
        candidate_distance, n_occlusion_pass}`` plus magnet bookkeeping.
    """
    half_fov_rad = math.radians(0.5 * float(hfov_deg))
    cos_half_fov = math.cos(half_fov_rad)
    views: List[Dict] = []
    stats = {
        "n_targets": len(magnet_targets),
        "n_no_island": 0,
        "n_no_navmesh_candidates": 0,
        "n_no_unoccluded_view": 0,
        "n_views": 0,
    }
    for tgt in magnet_targets:
        center = np.asarray(tgt["center_hab"], dtype=np.float32)
        extent = np.asarray(tgt["extent_hab"], dtype=np.float32)
        island_id, floor_pt = _island_of_object(pathfinder, center)
        if island_id < 0 or floor_pt is None:
            stats["n_no_island"] += 1
            continue
        floor_y = float(floor_pt[1])
        # Sample candidate (x, z) pairs in the annulus, snap to navmesh, and
        # accept only those on the same island as the object's floor.
        # NB: the renderer's CameraSensorSpec already offsets the sensor by
        # +eye_height in the Y axis. So the *agent* position we store as
        # ``pos`` must be the navmesh-floor point (snap_point's Y), not the
        # eye-level Y — habitat-sim will add eye_height itself.
        candidates: List[np.ndarray] = []
        tries = 0
        max_tries = max(8, n_candidates * 6)
        while len(candidates) < n_candidates and tries < max_tries:
            tries += 1
            theta = float(rng.random()) * 2.0 * math.pi
            r = float(rng.random()) * (r_max - r_min) + r_min
            x = float(center[0]) + r * math.cos(theta)
            z = float(center[2]) + r * math.sin(theta)
            seed = np.array([x, floor_y, z], dtype=np.float32)
            snapped = pathfinder.snap_point(seed)
            if snapped is None:
                continue
            arr = np.asarray(snapped, dtype=np.float32)
            if not all(math.isfinite(v) for v in arr):
                continue
            # Stay on the object's floor: same island_id, no more than 0.6m
            # vertical drift from the projected floor_y.
            try:
                isl = int(pathfinder.get_island(arr))
            except Exception:
                isl = -1
            if isl != island_id:
                continue
            if abs(arr[1] - floor_y) > 0.6:
                continue
            # ``arr`` is the navmesh-floor position. The camera (RGB sensor)
            # sits at arr.y + eye_height — only used for the 5-ray occlusion
            # test and yaw-vertical-frustum sanity below.
            cam_eye = np.array(
                [float(arr[0]), float(arr[1]) + float(eye_height), float(arr[2])],
                dtype=np.float32,
            )
            horiz = math.hypot(cam_eye[0] - center[0], cam_eye[2] - center[2])
            if horiz < r_min:
                continue
            candidates.append((arr, cam_eye))
        if not candidates:
            stats["n_no_navmesh_candidates"] += 1
            continue
        # Sort by horizontal distance to the object: closer = better, but the
        # raycast still has to confirm visibility.
        candidates.sort(key=lambda fc: math.hypot(fc[1][0] - center[0], fc[1][2] - center[2]))
        chosen: Optional[Dict] = None
        for floor_pos, cam_eye in candidates:
            # Yaw aim: face the object horizontally.
            dx = float(center[0] - cam_eye[0])
            dz = float(center[2] - cam_eye[2])
            yaw = math.atan2(-dx, -dz)
            # Vertical-frustum sanity: object must sit within ±half-vfov of
            # the horizontal sightline (we don't pitch the agent). VFOV is
            # roughly hfov × 3/4 for 4:3 aspect; use 0.5 × hfov as a
            # conservative bound.
            vert_angle = math.atan2(float(center[1]) - float(cam_eye[1]),
                                    math.hypot(dx, dz))
            if abs(vert_angle) > half_fov_rad:
                continue
            ok, n_pass = _five_ray_occlusion_check(
                sim, cam_eye, center, extent,
                min_passes=occl_min_passes,
            )
            if not ok:
                continue
            chosen = {
                "pos": floor_pos,       # agent (navmesh-floor) position
                "pos_floor": floor_pos, # alias kept for downstream code
                "yaw": yaw,
                "label": tgt.get("label", ""),
                "region_id": tgt.get("region_id"),
                "object_id": tgt.get("object_id"),
                "island_id": island_id,
                "is_waypoint": True,
                "is_magnet": True,
                "is_fair": tgt.get("is_fair", False),
                "is_referenced": tgt.get("is_referenced", False),
                "candidate_distance": math.hypot(dx, dz),
                "n_occlusion_pass": n_pass,
            }
            break
        if chosen is None:
            stats["n_no_unoccluded_view"] += 1
            continue
        views.append(chosen)
        stats["n_views"] += 1
    return views, stats


def _build_per_island_trajectories(
    pathfinder,
    magnet_views: Sequence[Dict],
    *,
    step_m: float,
    yaw_offsets_rad: Sequence[float] = (0.0,),
    objects_for_yaw: Optional[Sequence[Dict]] = None,
    hfov_deg: float = 90.0,
    yaw_candidates: int = 16,
) -> List[Dict]:
    """Group magnet views by island, then build a connected sub-tour for each
    island (greedy nearest-unvisited tour + geodesic interpolation), preserving
    each magnet's yaw at its waypoint and choosing an **object-coverage yaw**
    at each interpolated pose against ``objects_for_yaw`` (so the camera looks
    at nearby fair / statement-referenced objects rather than inheriting the
    preceding magnet's heading).

    Each magnet view supplies a navmesh-snapped ``pos`` (the agent position;
    habitat-sim's sensor spec adds ``eye_height`` to get camera Y). Path-
    planning runs on the same navmesh-Y positions; interpolated poses are
    likewise navmesh-Y.

    Returns a flat list of view dicts in render order; ``is_waypoint=True``
    flags the magnet anchors. Interpolated views inherit ``island_id`` from
    the preceding magnet.
    """
    import habitat_sim

    if not magnet_views:
        return []

    # Group by island.
    by_island: Dict[int, List[int]] = {}
    for i, v in enumerate(magnet_views):
        by_island.setdefault(int(v["island_id"]), []).append(i)

    out: List[Dict] = []
    for isl_id, idxs in sorted(by_island.items(), key=lambda kv: -len(kv[1])):
        # Greedy tour within the island.
        n = len(idxs)
        visited = [False] * n
        # Start near the centroid of magnets on this island (stable).
        positions = np.stack([magnet_views[i]["pos_floor"] for i in idxs])
        centroid = positions.mean(axis=0)
        d0 = np.linalg.norm(positions - centroid, axis=1)
        start = int(np.argmin(d0))
        order = [start]
        visited[start] = True

        def _geodesic(a, b):
            sp = habitat_sim.ShortestPath()
            sp.requested_start = np.asarray(a, dtype=np.float32)
            sp.requested_end = np.asarray(b, dtype=np.float32)
            if not pathfinder.find_path(sp):
                return float("inf"), None
            return float(getattr(sp, "geodesic_distance", float("inf"))), sp

        while len(order) < n:
            last_floor = magnet_views[idxs[order[-1]]]["pos_floor"]
            best = -1
            best_d = float("inf")
            for j in range(n):
                if visited[j]:
                    continue
                d, _ = _geodesic(last_floor, magnet_views[idxs[j]]["pos_floor"])
                if d < best_d:
                    best_d = d
                    best = j
            if best < 0 or not math.isfinite(best_d):
                # Disconnected within this island — rare but possible if
                # snap_point grouped points across a thin pinch. Skip the rest.
                LOGGER.info(
                    "island %d: skipping %d unreachable magnets after %d visited",
                    isl_id, n - len(order), len(order),
                )
                break
            visited[best] = True
            order.append(best)

        # Walk the ordered magnets and geodesic-interpolate between them.
        for k, oi in enumerate(order):
            v = magnet_views[idxs[oi]]
            out.append(v)
            if k == len(order) - 1:
                continue
            nxt = magnet_views[idxs[order[k + 1]]]
            d, sp = _geodesic(v["pos_floor"], nxt["pos_floor"])
            if sp is None or not math.isfinite(d):
                continue
            polyline = [np.asarray(p, dtype=np.float32) for p in sp.points]
            if len(polyline) < 2:
                continue
            yaw_idx = 0
            for interp in _walk_polyline(polyline, step_m):
                snapped = pathfinder.snap_point(interp)
                if snapped is None:
                    continue
                arr = np.asarray(snapped, dtype=np.float32)
                if not all(math.isfinite(v_) for v_ in arr):
                    continue
                base_yaw = float(v["yaw"])
                if objects_for_yaw:
                    chosen_yaw, _yaw_score = _choose_object_coverage_yaw(
                        arr,
                        base_yaw,
                        objects_for_yaw,
                        region_id=v.get("region_id"),
                        hfov_deg=float(hfov_deg),
                        num_candidates=int(yaw_candidates),
                    )
                    base_yaw = float(chosen_yaw)
                yaw = base_yaw + float(yaw_offsets_rad[yaw_idx % len(yaw_offsets_rad)])
                out.append({
                    "pos": arr,            # agent (navmesh-floor) position
                    "pos_floor": arr,
                    "yaw": yaw,
                    "label": v["label"],
                    "region_id": v.get("region_id"),
                    "object_id": None,
                    "island_id": isl_id,
                    "is_waypoint": False,
                    "is_magnet": False,
                })
                yaw_idx += 1
    return out


# ---------------------------------------------------------------------
# Scene mesh resolution
# ---------------------------------------------------------------------


def resolve_scene_mesh(
    scene_id: str,
    hm3d_root: Path,
    *,
    allow_semantic: bool = False,
) -> Path:
    """Find the photorealistic ``.basis.glb`` mesh file for an HM3D scene.

    HM3D ships **two** ``.glb`` files per scene with identical geometry:

    * ``<short_id>.basis.glb``   — photorealistic textured mesh (use this for RGB).
    * ``<short_id>.semantic.glb`` — flat-shaded by HM3D-Sem instance id (use ONLY
      with the dedicated semantic sensor, never as a stand-in for RGB).

    A previous version of this resolver fell through to a permissive
    ``rglob('<short_id>*.glb')``, which silently returned ``.semantic.glb``
    on roughly half the scenes (filesystem-entry-order dependent), and
    produced unrecognisable flat-color RGB renders. This version:

    1. tries explicit candidate paths covering every HM3D layout we've seen
       in the wild, including the ``versioned_data/hm3d-0.2/hm3d/{split}/...``
       layout produced by ``habitat_sim.utils.datasets_download``,
    2. falls back to a two-pass ``rglob``: prefer ``*.basis.glb`` first, only
       consider ``*.semantic.glb`` if no basis exists,
    3. when ``allow_semantic=False`` (the default), refuses to return a
       semantic mesh and raises with an explicit error,
    4. when ``allow_semantic=True``, logs a loud warning before returning it.
    """
    parts = scene_id.split("-", 1)
    short_id = parts[1] if len(parts) == 2 else scene_id

    # All the HM3D layouts we've seen in the wild. ``versioned_data/...`` is
    # what ``habitat_sim.utils.datasets_download`` produces; the others are
    # raw Matterport ToS-gated downloads or symlinked subsets.
    layout_prefixes = [
        Path("versioned_data") / "hm3d-0.2" / "hm3d",
        Path("versioned_data") / "hm3d-1.0" / "hm3d",
        Path("hm3d"),
        Path("."),
    ]
    splits = ("train", "val", "minival", "example")

    basis_candidates: List[Path] = []
    for prefix in layout_prefixes:
        for split in splits:
            basis_candidates.append(
                hm3d_root / prefix / split / scene_id / f"{short_id}.basis.glb"
            )
        basis_candidates.append(
            hm3d_root / prefix / scene_id / f"{short_id}.basis.glb"
        )
    basis_candidates.extend([
        hm3d_root / scene_id / f"{short_id}.basis.glb",
    ])

    for c in basis_candidates:
        if c.exists():
            return c

    # Two-pass rglob, basis first.
    found_basis = sorted(
        p for p in hm3d_root.rglob(f"{short_id}.basis.glb")
        if scene_id in str(p) or short_id in p.name
    )
    if found_basis:
        return found_basis[0]

    # Only now consider semantic — and only if the caller explicitly opted in.
    found_semantic = sorted(
        p for p in hm3d_root.rglob(f"{short_id}.semantic.glb")
        if scene_id in str(p) or short_id in p.name
    )
    if found_semantic:
        if not allow_semantic:
            raise FileNotFoundError(
                f"No '.basis.glb' (photorealistic) mesh found for scene_id "
                f"'{scene_id}' under {hm3d_root}, only '.semantic.glb' "
                f"(flat-shaded HM3D-Sem instance-id mesh): {found_semantic[0]}.\n"
                f"Rendering against the semantic mesh produces unrecognisable "
                f"flat-color RGB. Either install the photorealistic HM3D "
                f"download (HM3D-v0.2 includes basis.glb in every scene dir) or "
                f"pass allow_semantic=True if you really want the semantic mesh."
            )
        LOGGER.warning(
            "Falling back to semantic mesh for scene %s: %s. "
            "RGB renders will be flat-shaded HM3D-Sem instance-id colors, "
            "not photorealistic textures.",
            scene_id, found_semantic[0],
        )
        return found_semantic[0]

    raise FileNotFoundError(
        f"Could not find any .glb mesh for scene_id '{scene_id}' under "
        f"{hm3d_root}.\nTried explicit candidates:\n  "
        + "\n  ".join(str(c) for c in basis_candidates[:8])
        + f"\n... and rglob('{short_id}.basis.glb') / rglob('{short_id}.semantic.glb')."
    )


# ---------------------------------------------------------------------
# habitat-sim config + render
# ---------------------------------------------------------------------


def _intrinsics_from_hfov(hfov_deg: float, width: int, height: int) -> np.ndarray:
    """Convert a horizontal field-of-view to a 3×3 pinhole intrinsic matrix.

    habitat-sim cameras model square pixels, so vertical fov follows from
    aspect: ``vfov = hfov * H / W``. Principal point is the image centre.
    """
    hfov = math.radians(float(hfov_deg))
    fx = (width / 2.0) / math.tan(hfov / 2.0)
    fy = fx  # square pixels in habitat-sim
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return K


def _make_sim(
    mesh_path: Path,
    scene_dataset_config: Optional[Path],
    image_size: Tuple[int, int],
    eye_height: float,
    hfov_deg: float,
    enable_semantic: bool,
):
    """Build a Simulator with rgb + depth (+ optional semantic) sensors.

    Imports happen lazily so the rest of the module remains usable for
    smoke-testing the mesh-resolution logic without habitat-sim.
    """
    import habitat_sim

    width, height = image_size

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(mesh_path)
    if scene_dataset_config is not None:
        sim_cfg.scene_dataset_config_file = str(scene_dataset_config)
    sim_cfg.allow_sliding = True
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = int(os.environ.get("HABITAT_SIM_GPU", "0"))

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb_camera"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [height, width]
    rgb_spec.position = [0.0, eye_height, 0.0]
    rgb_spec.hfov = float(hfov_deg)
    rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth_camera"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [height, width]
    depth_spec.position = [0.0, eye_height, 0.0]
    depth_spec.hfov = float(hfov_deg)
    depth_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    sensor_specs = [rgb_spec, depth_spec]
    if enable_semantic:
        sem_spec = habitat_sim.CameraSensorSpec()
        sem_spec.uuid = "semantic_camera"
        sem_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        sem_spec.resolution = [height, width]
        sem_spec.position = [0.0, eye_height, 0.0]
        sem_spec.hfov = float(hfov_deg)
        sem_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(sem_spec)

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {}  # we teleport directly via set_state

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    return sim


def _yaw_coeffs(yaw_rad: float) -> np.ndarray:
    """Return (x, y, z, w) coefficients of the unit quaternion rotating by
    ``yaw_rad`` around world Y. habitat-sim's ``quat_from_coeffs`` (called
    inside ``AgentState`` setters) expects this 4-tuple form, not a
    ``magnum.Quaternion``.
    """
    half = yaw_rad / 2.0
    return np.array([0.0, math.sin(half), 0.0, math.cos(half)], dtype=np.float32)


def _agent_state(position, yaw_rad: float):
    import habitat_sim

    return habitat_sim.AgentState(
        position=np.asarray(position, dtype=np.float32),
        rotation=_yaw_coeffs(yaw_rad),
    )


def _camera_to_world_opengl(sim) -> np.ndarray:
    """Compute the rgb sensor's camera-to-world transform (OpenGL convention).

    habitat-sim sensors are placed as children of the agent node; the sensor's
    state in world frame is exposed via ``agent.get_state().sensor_states``.
    Avoid the magnum→numpy buffer-order trap by computing the rotation matrix
    directly from the quaternion components.
    """
    sensor = sim.get_agent(0).get_state().sensor_states["rgb_camera"]
    pos = np.asarray(sensor.position, dtype=np.float32)
    q = sensor.rotation  # magnum.Quaternion (or numpy quaternion-like)
    # Pull (w, x, y, z) components from any of the conventions habitat-sim
    # has shipped — magnum uses ``scalar`` + ``vector``; numpy-quaternion has
    # ``real`` + ``imag``; some builds expose tuple indexing.
    try:
        w = float(q.scalar)
        v = q.vector
        x, y, z = float(v.x), float(v.y), float(v.z)
    except AttributeError:
        if hasattr(q, "real") and hasattr(q, "imag"):
            w = float(q.real)
            try:
                x, y, z = float(q.imag.x), float(q.imag.y), float(q.imag.z)
            except AttributeError:
                imag = np.asarray(q.imag, dtype=np.float64).reshape(3)
                x, y, z = float(imag[0]), float(imag[1]), float(imag[2])
        else:  # last-ditch: tuple-like (x, y, z, w)
            arr = np.asarray(q, dtype=np.float64).reshape(4)
            x, y, z, w = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def _walk_polyline(
    polyline: List[np.ndarray], step_m: float
) -> List[np.ndarray]:
    """Walk a polyline at constant ``step_m`` spacing, skipping the start vertex.

    Returns positions at ``step_m, 2*step_m, ...`` along the polyline arc-length,
    interpolated between consecutive vertices. The end vertex is *not* appended
    here — callers handle endpoints to avoid duplication when chaining segments.
    """
    if len(polyline) < 2 or step_m <= 0:
        return []
    cumdist = [0.0]
    for j in range(1, len(polyline)):
        cumdist.append(cumdist[-1] + float(np.linalg.norm(polyline[j] - polyline[j - 1])))
    total = cumdist[-1]
    out: List[np.ndarray] = []
    d = step_m
    seg = 0
    while d < total - 1e-6:
        while seg < len(cumdist) - 2 and cumdist[seg + 1] < d:
            seg += 1
        denom = cumdist[seg + 1] - cumdist[seg]
        if denom <= 1e-9:
            d += step_m
            continue
        t = (d - cumdist[seg]) / denom
        out.append(polyline[seg] + t * (polyline[seg + 1] - polyline[seg]))
        d += step_m
    return out


def _angle_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def _forward_xz_for_yaw(yaw: float) -> Tuple[float, float]:
    # Habitat cameras look down local -Z at yaw=0.
    return -math.sin(yaw), -math.cos(yaw)


def _score_yaw_for_objects(
    pos: np.ndarray,
    yaw: float,
    objects: Sequence[Dict],
    *,
    region_id: Optional[int],
    hfov_deg: float,
) -> float:
    if not objects:
        return 0.0
    fx, fz = _forward_xz_for_yaw(float(yaw))
    half_fov = math.radians(float(hfov_deg)) * 0.5
    cos_half = math.cos(half_fov)
    score = 0.0
    px, pz = float(pos[0]), float(pos[2])
    for obj in objects:
        center = obj["center_hab"]
        dx = float(center[0]) - px
        dz = float(center[2]) - pz
        dist = math.hypot(dx, dz)
        if dist < 0.25 or dist > 12.0:
            continue
        cosang = (dx * fx + dz * fz) / max(dist, 1.0e-6)
        if cosang <= cos_half:
            continue
        centrality = (cosang - cos_half) / max(1.0 - cos_half, 1.0e-6)
        # Close, large, centered objects should dominate, but avoid letting one
        # huge wall-like object erase multi-object coverage. Structural labels
        # are already filtered above.
        size_score = min(2.5, float(obj.get("size_score", 1.0)))
        distance_score = 1.0 / (0.75 + dist)
        region_score = 1.0
        if region_id is not None and obj.get("region_id") is not None:
            region_score = 1.0 if int(obj["region_id"]) == int(region_id) else 0.35
        score += region_score * size_score * distance_score * (0.25 + centrality * centrality)
    return score


def _choose_object_coverage_yaw(
    pos: np.ndarray,
    fallback_yaw: float,
    objects: Sequence[Dict],
    *,
    region_id: Optional[int],
    hfov_deg: float,
    num_candidates: int,
) -> Tuple[float, float]:
    if not objects or num_candidates <= 0:
        return float(fallback_yaw), 0.0

    # Include headings directly toward nearby objects, plus a uniform sweep for
    # coverage when object centers are sparse or behind occluders.
    candidates: List[float] = [
        2.0 * math.pi * j / max(1, num_candidates) for j in range(max(1, num_candidates))
    ]
    px, pz = float(pos[0]), float(pos[2])
    for obj in objects:
        if region_id is not None and obj.get("region_id") is not None and int(obj["region_id"]) != int(region_id):
            continue
        center = obj["center_hab"]
        dx = float(center[0]) - px
        dz = float(center[2]) - pz
        dist = math.hypot(dx, dz)
        if 0.25 <= dist <= 8.0:
            candidates.append(math.atan2(-dx, -dz))

    best_yaw = float(fallback_yaw)
    best_score = -1.0
    for cand in candidates:
        score = _score_yaw_for_objects(
            pos,
            cand,
            objects,
            region_id=region_id,
            hfov_deg=hfov_deg,
        )
        # Small tie-breaker preserves deterministic cycling when scores match.
        score -= 1.0e-4 * abs(_angle_diff(cand, fallback_yaw))
        if score > best_score:
            best_score = score
            best_yaw = cand
    return float(best_yaw), max(0.0, float(best_score))


def build_trajectory(
    pathfinder,
    waypoints: List[Dict],
    *,
    step_m: float,
    yaws: Sequence[float],
    yaw_policy: str = "cycle",
    objects: Optional[Sequence[Dict]] = None,
    hfov_deg: float = 90.0,
    yaw_candidates: int = 16,
) -> List[Dict]:
    """Greedy nearest-unvisited tour through waypoints, geodesic-interpolated.

    Produces a robot-trajectory-style view sequence:

    1. **Tour**: starting from waypoint 0, repeatedly walk to the
       geodesically-nearest unvisited waypoint (habitat-sim's
       ``geodesic_distance`` runs A* on the navmesh). Unreachable
       waypoints are skipped.
    2. **Interpolation**: between consecutive waypoints, fetch the
       geodesic polyline via ``pathfinder.find_path`` and step along it
       at ``step_m`` increments. Each interpolated point is snapped back
       to the navmesh.
    3. **Yaws**: assign one yaw per view. ``cycle`` repeats ``yaws``;
       ``object_coverage`` chooses a yaw that faces more nearby annotated
       objects, falling back to the cycling yaw when no object score is useful.

    Each waypoint dict has ``{"pos": np.ndarray(3,), "label": str}``.
    Returns a list of ``{"pos", "yaw", "label", "is_waypoint"}`` dicts in
    render order.
    """
    import habitat_sim  # lazy

    if not waypoints:
        return []

    n = len(waypoints)
    visited = [False] * n
    visited[0] = True
    order: List[int] = [0]
    # When the navmesh has disconnected components, the greedy tour will
    # exhaust the current component long before all waypoints are visited.
    # We then "teleport" to the next unvisited waypoint and start a new
    # sub-tour; teleport_after marks indices in `order` that follow a
    # disconnected jump, so polyline interpolation can be skipped there.
    teleport_after: set = set()

    def _geodesic(a, b):
        # habitat-sim's PathFinder doesn't expose a standalone geodesic_distance
        # in 0.2.5; route through ShortestPath which fills .geodesic_distance.
        sp = habitat_sim.ShortestPath()
        sp.requested_start = np.asarray(a, dtype=np.float32)
        sp.requested_end = np.asarray(b, dtype=np.float32)
        if not pathfinder.find_path(sp):
            return float("inf")
        d = getattr(sp, "geodesic_distance", float("inf"))
        return float(d) if d is not None and math.isfinite(d) else float("inf")

    n_teleports = 0
    while len(order) < n:
        last_pos = waypoints[order[-1]]["pos"]
        best = -1
        best_d = float("inf")
        for j in range(n):
            if visited[j]:
                continue
            d = _geodesic(last_pos, waypoints[j]["pos"])
            if d < best_d:
                best_d = d
                best = j
        if best < 0 or not math.isfinite(best_d):
            # Current component is exhausted. Teleport to the next unvisited
            # waypoint and continue — this preserves coverage on scenes with
            # multiple disconnected navmesh components (separate floors,
            # balconies, walled-off rooms, etc.).
            next_start = -1
            for j in range(n):
                if not visited[j]:
                    next_start = j
                    break
            if next_start < 0:
                break
            visited[next_start] = True
            order.append(next_start)
            teleport_after.add(len(order) - 1)
            n_teleports += 1
            continue
        visited[best] = True
        order.append(best)
    if n_teleports:
        LOGGER.info(
            "greedy tour: %d teleport(s) across disconnected navmesh components",
            n_teleports,
        )

    views: List[Dict] = []
    yaw_idx = 0
    objects = list(objects or [])

    def _resolve_yaw(pos: np.ndarray, fallback: float, region_id: Optional[int]) -> Tuple[float, float]:
        if yaw_policy == "object_coverage":
            return _choose_object_coverage_yaw(
                pos,
                fallback,
                objects,
                region_id=region_id,
                hfov_deg=hfov_deg,
                num_candidates=yaw_candidates,
            )
        return float(fallback), 0.0

    for k in range(len(order)):
        wp = waypoints[order[k]]
        fallback_yaw = float(yaws[yaw_idx % len(yaws)])
        yaw, yaw_score = _resolve_yaw(
            np.asarray(wp["pos"], dtype=np.float32),
            fallback_yaw,
            wp.get("region_id"),
        )
        views.append({
            "pos": np.asarray(wp["pos"], dtype=np.float32),
            "yaw": yaw,
            "fallback_yaw": fallback_yaw,
            "yaw_score": yaw_score,
            "label": wp["label"],
            "region_id": wp.get("region_id"),
            "is_waypoint": True,
        })
        yaw_idx += 1
        if k == len(order) - 1:
            break
        # Skip interpolation across a teleport — order[k+1] is in a different
        # connected component and there's no continuous polyline.
        if (k + 1) in teleport_after:
            continue
        nxt = waypoints[order[k + 1]]
        sp = habitat_sim.ShortestPath()
        sp.requested_start = np.asarray(wp["pos"], dtype=np.float32)
        sp.requested_end = np.asarray(nxt["pos"], dtype=np.float32)
        if not pathfinder.find_path(sp):
            continue
        polyline = [np.asarray(p, dtype=np.float32) for p in sp.points]
        if len(polyline) < 2:
            continue
        for interp in _walk_polyline(polyline, step_m):
            snapped = pathfinder.snap_point(interp)
            if snapped is None:
                continue
            snapped_arr = np.asarray(snapped, dtype=np.float32)
            if any(math.isnan(v) for v in snapped_arr):
                continue
            fallback_yaw = float(yaws[yaw_idx % len(yaws)])
            yaw, yaw_score = _resolve_yaw(snapped_arr, fallback_yaw, wp.get("region_id"))
            views.append({
                "pos": snapped_arr,
                "yaw": yaw,
                "fallback_yaw": fallback_yaw,
                "yaw_score": yaw_score,
                "label": wp["label"],
                "region_id": wp.get("region_id"),
                "is_waypoint": False,
            })
            yaw_idx += 1
    return views


def _sample_navigable_in_region(
    pathfinder,
    region: Dict,
    *,
    n_points: int,
    max_tries: int = 200,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Rejection-sample navigable points whose floor-plane lies inside the region.

    IRef-VLA stores region bboxes in the source dataset's world frame
    (HM3D = Z-up). Habitat-sim's runtime world frame is Y-up, with the
    transform ``(x, y, z)_iref → (x, z, -y)_habitat`` empirically validated
    on HM3D scene 00009. Floor-plane sampling therefore takes IRef-VLA
    ``(cx, cy)`` and renders at habitat ``(cx, -cy)`` (Y is snapped to the
    navmesh by ``pathfinder.snap_point``).

    Falls back to ``snap_point`` when the random XZ candidate isn't directly
    navigable: we snap to the nearest navmesh point and keep it only if
    the snap moved less than half the region's smaller floor extent.
    """
    cx, cy, cz = region["center"]
    lx, ly, _lz = region["extent"]
    # IRef-VLA → habitat coords.  The floor plane is X/Y in IRef-VLA and X/Z
    # in Habitat; IRef-VLA Z is the vertical Habitat Y.  Using a hard-coded
    # Habitat Y=0 here silently biased multi-floor scenes toward the lower
    # floor, because snap_point() prefers the closest navmesh component.
    hcx = float(cx)
    hcy = float(cz)
    hcz = float(-cy)              # IRef-VLA Y axis points opposite to habitat Z
    half_x, half_z = lx / 2.0, ly / 2.0
    snap_radius = 0.5 * min(half_x, half_z) if min(half_x, half_z) > 0 else 1.0
    vertical_snap_radius = max(0.75, 0.5 * float(_lz) + 0.35)

    out: List[np.ndarray] = []
    tries = 0
    while len(out) < n_points and tries < max_tries:
        tries += 1
        rx = hcx + (rng.random() * 2.0 - 1.0) * half_x
        rz = hcz + (rng.random() * 2.0 - 1.0) * half_z
        candidate = np.array([rx, hcy, rz], dtype=np.float32)  # Y is snapped to navmesh
        snapped = pathfinder.snap_point(candidate)
        if snapped is None or any(math.isnan(v) for v in snapped):
            continue
        snapped_arr = np.asarray(snapped, dtype=np.float32)
        dx = snapped_arr[0] - rx
        dz = snapped_arr[2] - rz
        if math.hypot(dx, dz) > snap_radius:
            continue
        if abs(float(snapped_arr[1]) - hcy) > vertical_snap_radius:
            continue
        out.append(snapped_arr)
    return out


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-id", required=True, help="IRef-VLA scene id, e.g. 00238-j6fHrce9pHR.")
    p.add_argument("--hm3d-root", type=Path, required=True,
                   help="HM3D dataset root (contains hm3d/{train,val,minival}/...).")
    p.add_argument("--iref-vla-root", type=Path, default=None,
                   help="IRef-VLA HM3D extracted root (contains <scene_id>/...). "
                        "If omitted or the scene's region_result.csv is missing, falls back to "
                        "uniform random-navmesh sampling across the whole scene.")
    p.add_argument("--out", type=Path, required=True, help="Output directory for NPZ chunks + meta.")
    p.add_argument("--mesh-path", type=Path, default=None,
                   help="Optional explicit .glb path; bypasses the auto-resolver.")
    p.add_argument("--scene-dataset-config", type=Path, default=None,
                   help="Optional scene_dataset_config.json (HM3D-Sem ships one).")
    p.add_argument("--mode", choices=("region", "random", "trajectory", "magnet", "auto"), default="auto",
                   help="Sampling mode: 'region' = per-IRef-VLA-region (4 yaws per point); "
                        "'random' = uniform random-navmesh (4 yaws per point); "
                        "'trajectory' = greedy nearest-unvisited tour through region "
                        "waypoints, geodesic-interpolated at --step-m spacing, then "
                        "--trajectory-yaws-per-view rotations per pose; base yaw cycles "
                        "through 4 directions offset by --yaw-offset-deg "
                        "from the region-mode set; "
                        "'magnet' = one guaranteed un-occluded close view per fair / "
                        "statement-referenced GT object (per-floor via navmesh island "
                        "grouping); magnets are connected by geodesic interpolation at "
                        "--step-m spacing; "
                        "'auto' = region if regions are available, else random.")
    p.add_argument("--step-m", type=float, default=0.15,
                   help="Inter-view step in metres along the geodesic path. "
                        "Default 0.15m gives ~0.75 m/s at 5 fps and dense per-island "
                        "trajectories that don't skip past small objects. (trajectory + "
                        "magnet modes; original trajectory mode used 0.6.)")
    p.add_argument("--yaw-offset-deg", type=float, default=45.0,
                   help="(trajectory mode) yaw offset from the region-mode 4 directions. "
                        "Default 45° → views look at 45/135/225/315 instead of 0/90/180/270.")
    p.add_argument("--yaw-policy", choices=("cycle", "object_coverage"), default="object_coverage",
                   help="(trajectory mode) choose view heading. 'cycle' repeats the four yaws; "
                        "'object_coverage' scores candidate yaws by nearby IRef-VLA object centers.")
    p.add_argument("--yaw-candidates", type=int, default=16,
                   help="Number of uniform yaw candidates for object_coverage before adding object-facing headings.")
    p.add_argument("--trajectory-yaws-per-view", type=int, default=4,
                   help="(trajectory mode) number of yaw rotations rendered at each "
                        "trajectory pose. Default 4 restores 360-degree coverage; "
                        "set 1 for the older single-heading behavior.")
    p.add_argument("--points-per-region", type=int, default=8,
                   help="(region mode) navigable points sampled per IRef-VLA region.")
    p.add_argument("--total-points", type=int, default=80,
                   help="(random mode) total navigable points sampled across the whole scene.")
    p.add_argument("--num-yaws", type=int, default=4)
    p.add_argument("--image-size", nargs=2, type=int, default=[640, 480], metavar=("W", "H"))
    p.add_argument("--eye-height", type=float, default=1.5,
                   help="Eye height above the navmesh (metres).")
    p.add_argument("--hfov-deg", type=float, default=90.0)
    p.add_argument("--frames-per-chunk", type=int, default=128,
                   help="Frames per NPZ archive (smaller = faster pickup but more files).")
    p.add_argument("--regions", nargs="*", type=int, default=None,
                   help="(region mode) Restrict to these region_ids (default: all regions).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--enable-semantic", action="store_true",
                   help="Also render the semantic sensor (HM3D-Sem only).")
    p.add_argument("--depth-clip", type=float, default=10.0,
                   help="Clip rendered depth to <= this many metres (saves zero on overshoot).")
    p.add_argument("--depth-min", type=float, default=0.05,
                   help="Mask out depths below this (sub-near-plane noise).")
    p.add_argument("--min-valid-depth-fraction", type=float, default=0.20,
                   help="Skip rendered frames whose clipped depth has fewer than this "
                        "fraction of valid pixels. Default 0.20 removes mostly-empty "
                        "views caused by bad poses or views into unmodeled space. "
                        "Set 0 to disable.")
    # Magnet-mode parameters
    p.add_argument("--magnet-r-min", type=float, default=0.8,
                   help="(magnet mode) Minimum horizontal distance camera-to-object (m).")
    p.add_argument("--magnet-r-max", type=float, default=3.0,
                   help="(magnet mode) Maximum horizontal distance camera-to-object (m).")
    p.add_argument("--magnet-candidates", type=int, default=24,
                   help="(magnet mode) Navmesh sample candidates per GT before "
                        "scoring with the occlusion test.")
    p.add_argument("--magnet-occlusion-rays", type=int, default=5,
                   help="(magnet mode) Number of rays in the occlusion check.")
    p.add_argument("--magnet-occlusion-pass", type=int, default=3,
                   help="(magnet mode) Minimum number of rays that must reach "
                        "the object surface for a candidate to be accepted.")
    p.add_argument("--magnet-include-statement-refs", action="store_true", default=True,
                   help="(magnet mode) Magnet any GT referenced as a target or anchor in "
                        "the scene's referential statements, even if it fails the fair "
                        "geometry filter.")
    p.add_argument("--magnet-no-statement-refs", dest="magnet_include_statement_refs",
                   action="store_false",
                   help="(magnet mode) Disable statement-reference union; use only the "
                        "fair-geometry filter.")
    p.add_argument("--magnet-yaws-per-magnet", type=int, default=1,
                   help="(magnet mode) Yaws rendered at each magnet pose. The first yaw "
                        "always faces the object; extras cycle 90° apart.")
    p.add_argument("--save-previews", type=int, default=16,
                   help="Save this many uniformly-sampled frames as PNG previews "
                        "into <out>/_previews/. 0 = disabled.")
    return p.parse_args(argv)


def _save_chunk(
    out_dir: Path,
    chunk_idx: int,
    images: List[np.ndarray],
    depths: List[np.ndarray],
    poses: List[np.ndarray],
    K: np.ndarray,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"frames_{chunk_idx:03d}.npz"
    np.savez_compressed(
        path,
        images=np.stack(images, axis=0).astype(np.uint8, copy=False),
        depths=np.stack(depths, axis=0).astype(np.float32, copy=False),
        camtoworlds=np.stack(poses, axis=0).astype(np.float32, copy=False),
        K=K.astype(np.float32, copy=False),
        pose_convention=np.array("opengl"),
    )
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    args = _parse_args(argv)

    mesh_path = args.mesh_path or resolve_scene_mesh(args.scene_id, args.hm3d_root)
    if mesh_path.name.endswith(".semantic.glb"):
        mesh_kind = "semantic"
        LOGGER.warning(
            "scene mesh is %s (HM3D-Sem instance-id flat colors); "
            "RGB renders will NOT be photorealistic.", mesh_path,
        )
    elif mesh_path.name.endswith(".basis.glb"):
        mesh_kind = "basis"
        LOGGER.info("scene mesh: %s (photorealistic basis)", mesh_path)
    else:
        mesh_kind = "unknown"
        LOGGER.info("scene mesh: %s", mesh_path)

    # Resolve sampling mode + load IRef-VLA regions if available.
    regions: List[Dict] = []
    region_csv_available = False
    if args.iref_vla_root is not None:
        csv_path = args.iref_vla_root / args.scene_id / f"{args.scene_id}_region_result.csv"
        region_csv_available = csv_path.exists()
        if region_csv_available:
            regions = load_iref_vla_regions(args.scene_id, args.iref_vla_root)
            if args.regions is not None:
                wanted = set(int(r) for r in args.regions)
                regions = [r for r in regions if r["region_id"] in wanted]
    objects: List[Dict] = []
    if args.iref_vla_root is not None:
        objects = load_iref_vla_objects(args.scene_id, args.iref_vla_root)

    mode = args.mode
    if mode == "auto":
        mode = "region" if regions else "random"
    elif mode == "region" and not regions:
        LOGGER.error("--mode region requested but no regions available for %s "
                     "(iref_vla_root=%s, csv exists=%s). "
                     "Use --mode random for scenes outside IRef-VLA.",
                     args.scene_id, args.iref_vla_root, region_csv_available)
        return 1
    elif mode == "magnet" and args.iref_vla_root is None:
        LOGGER.error("--mode magnet requires --iref-vla-root for per-object GT lookup.")
        return 1
    # Trajectory mode without regions: derive waypoints from random navmesh
    # sampling instead. The tour itself works identically.

    trajectory_yaws_per_view = max(1, int(args.trajectory_yaws_per_view))
    min_valid_depth_fraction = max(0.0, min(1.0, float(args.min_valid_depth_fraction)))

    if mode == "region":
        LOGGER.info("mode=region: %d regions × %d points/region × %d yaws = %d frames",
                    len(regions), args.points_per_region, args.num_yaws,
                    len(regions) * args.points_per_region * args.num_yaws)
    elif mode == "trajectory":
        LOGGER.info("mode=trajectory: %d regions × %d points/region waypoints, "
                    "step %.2fm, yaw-offset %.1f°, %d yaw(s) per pose",
                    len(regions), args.points_per_region,
                    args.step_m, args.yaw_offset_deg, trajectory_yaws_per_view)
    elif mode == "magnet":
        LOGGER.info("mode=magnet: one viewpoint per fair/referenced GT, "
                    "r∈[%.2f, %.2f]m, %d candidates/GT, %d-ray occlusion (≥%d pass), "
                    "interp step %.2fm",
                    args.magnet_r_min, args.magnet_r_max, args.magnet_candidates,
                    args.magnet_occlusion_rays, args.magnet_occlusion_pass, args.step_m)
    else:
        LOGGER.info("mode=random: %d total points × %d yaws = %d frames",
                    args.total_points, args.num_yaws, args.total_points * args.num_yaws)

    rng = np.random.default_rng(int(args.seed))
    width, height = args.image_size
    K = _intrinsics_from_hfov(args.hfov_deg, width, height)

    sim = _make_sim(
        mesh_path=mesh_path,
        scene_dataset_config=args.scene_dataset_config,
        image_size=(width, height),
        eye_height=args.eye_height,
        hfov_deg=args.hfov_deg,
        enable_semantic=args.enable_semantic,
    )
    try:
        # Make sure a navmesh exists. HM3D ships pre-computed; if not, recompute.
        if not sim.pathfinder.is_loaded:
            import habitat_sim
            settings = habitat_sim.NavMeshSettings()
            settings.set_defaults()
            settings.agent_height = max(0.5, float(args.eye_height) - 0.1)
            settings.agent_radius = 0.18
            # older habitat-sim drops `include_static_objects=True` kwarg
            try:
                sim.recompute_navmesh(sim.pathfinder, settings, include_static_objects=True)
            except TypeError:
                sim.recompute_navmesh(sim.pathfinder, settings)
            LOGGER.info("recomputed navmesh (no precomputed .navmesh found)")

        # Generate sample points per region, then render.
        out_dir = args.out
        out_dir.mkdir(parents=True, exist_ok=True)

        # Log navmesh island layout for multi-floor diagnostics.
        try:
            n_islands = int(sim.pathfinder.num_islands)
            areas = []
            for i in range(n_islands):
                try:
                    areas.append((i, float(sim.pathfinder.island_area(i))))
                except Exception:
                    pass
            areas.sort(key=lambda x: -x[1])
            LOGGER.info(
                "navmesh: %d islands; top areas (m²): %s",
                n_islands,
                ", ".join(f"{i}:{a:.1f}" for i, a in areas[:6]),
            )
        except Exception as exc:
            n_islands = -1
            LOGGER.warning("could not read navmesh islands: %s", exc)

        meta = {
            "scene_id": args.scene_id,
            "mesh_path": str(mesh_path),
            "mesh_kind": mesh_kind,
            "image_size": [width, height],
            "eye_height": args.eye_height,
            "hfov_deg": args.hfov_deg,
            "mode": mode,
            "yaw_policy": args.yaw_policy,
            "yaw_candidates": int(args.yaw_candidates),
            "trajectory_yaws_per_view": trajectory_yaws_per_view,
            "points_per_region": args.points_per_region,
            "total_points": args.total_points,
            "num_yaws": args.num_yaws,
            "min_valid_depth_fraction": min_valid_depth_fraction,
            "n_yaw_objects": len(objects),
            "n_navmesh_islands": n_islands,
            "K": K.tolist(),
            "per_region": [],
            "random_points": 0,
            "frame_count": 0,
            "frame_attempt_count": 0,
            "frame_skip_invalid_depth_count": 0,
            "chunk_paths": [],
        }

        images: List[np.ndarray] = []
        depths: List[np.ndarray] = []
        poses: List[np.ndarray] = []
        chunk_idx = 0
        frames_total = 0
        frames_attempted = 0
        frames_skipped_invalid_depth = 0
        per_chunk = max(8, int(args.frames_per_chunk))

        yaws = [2 * math.pi * j / max(1, args.num_yaws) for j in range(args.num_yaws)]

        # Magnet mode: pick one viewpoint per fair/referenced GT, then
        # interpolate per-island connecting trajectories.
        magnet_views: List[Dict] = []
        magnet_stats: Dict[str, int] = {}
        if mode == "magnet":
            magnet_targets = load_magnet_targets(
                args.scene_id,
                args.iref_vla_root,
                include_statement_refs=bool(args.magnet_include_statement_refs),
            )
            n_fair = sum(1 for t in magnet_targets if t.get("is_fair"))
            n_ref = sum(1 for t in magnet_targets if t.get("is_referenced"))
            LOGGER.info(
                "magnet targets: %d (%d fair, %d statement-referenced; union)",
                len(magnet_targets), n_fair, n_ref,
            )
            magnet_anchors, magnet_stats = _build_magnet_views(
                sim,
                sim.pathfinder,
                magnet_targets,
                eye_height=float(args.eye_height),
                hfov_deg=float(args.hfov_deg),
                r_min=float(args.magnet_r_min),
                r_max=float(args.magnet_r_max),
                n_candidates=int(args.magnet_candidates),
                occl_min_passes=int(args.magnet_occlusion_pass),
                rng=rng,
            )
            LOGGER.info(
                "magnet build: %d views accepted / %d targets; "
                "rejects: no_island=%d, no_navmesh_candidates=%d, "
                "no_unoccluded_view=%d",
                magnet_stats["n_views"], magnet_stats["n_targets"],
                magnet_stats["n_no_island"], magnet_stats["n_no_navmesh_candidates"],
                magnet_stats["n_no_unoccluded_view"],
            )
            magnet_yaws_per_magnet = max(1, int(args.magnet_yaws_per_magnet))
            extra_yaw_offsets = tuple(
                2.0 * math.pi * j / max(1, magnet_yaws_per_magnet)
                for j in range(magnet_yaws_per_magnet)
            )
            # Reuse magnet_targets as the object-coverage pool for interp
            # yaws: at each interpolated pose between magnets the camera
            # picks a heading that maximally faces nearby fair / referenced
            # objects.
            magnet_views = _build_per_island_trajectories(
                sim.pathfinder,
                magnet_anchors,
                step_m=float(args.step_m),
                yaw_offsets_rad=extra_yaw_offsets,
                objects_for_yaw=magnet_targets,
                hfov_deg=float(args.hfov_deg),
                yaw_candidates=int(args.yaw_candidates),
            )
            n_anchors = sum(1 for v in magnet_views if v.get("is_magnet"))
            n_interp_m = len(magnet_views) - n_anchors
            islands_seen = sorted({int(v["island_id"]) for v in magnet_views})
            LOGGER.info(
                "magnet trajectory: %d total poses (%d anchors + %d interp) "
                "across %d navmesh islands: %s",
                len(magnet_views), n_anchors, n_interp_m, len(islands_seen), islands_seen,
            )
            meta["magnet"] = {
                **magnet_stats,
                "n_anchors": n_anchors,
                "n_interp": n_interp_m,
                "n_islands_used": len(islands_seen),
                "islands": islands_seen,
                "r_min": float(args.magnet_r_min),
                "r_max": float(args.magnet_r_max),
                "n_candidates_per_target": int(args.magnet_candidates),
                "occlusion_rays": int(args.magnet_occlusion_rays),
                "occlusion_min_pass": int(args.magnet_occlusion_pass),
                "yaws_per_magnet": magnet_yaws_per_magnet,
                "step_m": float(args.step_m),
                "include_statement_refs": bool(args.magnet_include_statement_refs),
            }

        # Build a single flat list of (label, points) groups depending on mode.
        groups: List[Dict] = []
        if mode in ("region", "trajectory"):
            for region in regions:
                pts = _sample_navigable_in_region(
                    sim.pathfinder,
                    region,
                    n_points=args.points_per_region,
                    rng=rng,
                )
                LOGGER.info(
                    "region %d (%s): sampled %d/%d navigable points",
                    region["region_id"], region["label"], len(pts), args.points_per_region,
                )
                meta["per_region"].append(
                    {
                        "region_id": region["region_id"],
                        "label": region["label"],
                        "n_points": len(pts),
                        "n_frames": len(pts) * (trajectory_yaws_per_view if mode == "trajectory" else args.num_yaws),
                        "n_seed_frames": len(pts) * (trajectory_yaws_per_view if mode == "trajectory" else args.num_yaws),
                    }
                )
                if pts:
                    groups.append({"label": region["label"], "region_id": region["region_id"], "points": pts})
            total_points_sampled = sum(len(g["points"]) for g in groups)
            if total_points_sampled == 0:
                LOGGER.warning(
                    "region-aware sampling produced 0 points (no IRef-VLA regions, or "
                    "Z-up vs Y-up mismatch). Falling back to uniform random sampling — "
                    "trajectory mode will tour random waypoints."
                )
                # NB: keep mode='trajectory' so the tour still runs over the
                # random waypoints; only switch to 'random' if originally asked.
                if mode == "region":
                    mode = "random"
                    meta["mode"] = mode
                fallback_to_random = True
            else:
                fallback_to_random = False
        else:
            fallback_to_random = False
        if mode == "random" or fallback_to_random:
            pts: List[np.ndarray] = []
            for _ in range(args.total_points * 4):  # over-sample then trim
                if len(pts) >= args.total_points:
                    break
                p = sim.pathfinder.get_random_navigable_point()
                if any(math.isnan(v) for v in p):
                    continue
                pts.append(np.asarray(p, dtype=np.float32))
            LOGGER.info("random navmesh sampling: got %d / %d navigable points", len(pts), args.total_points)
            meta["random_points"] = len(pts)
            groups = [{"label": "<all>", "points": pts}]

        # Trajectory mode bundles per-region waypoints into a single tour. Build
        # the tour + interpolated view list now; render with a single (pos, yaw)
        # loop below.
        trajectory_views: List[Dict] = []
        if mode == "trajectory":
            waypoints: List[Dict] = []
            for g in groups:
                for p in g["points"]:
                    waypoints.append({
                        "pos": p,
                        "label": g["label"],
                        "region_id": g.get("region_id"),
                    })
            yaw_offset = math.radians(float(args.yaw_offset_deg))
            traj_yaws = [yaw_offset + 2 * math.pi * j / 4.0 for j in range(4)]
            LOGGER.info("trajectory: %d waypoints; building greedy nearest-unvisited tour ...",
                        len(waypoints))
            trajectory_views = build_trajectory(
                sim.pathfinder, waypoints,
                step_m=float(args.step_m), yaws=traj_yaws,
                yaw_policy=str(args.yaw_policy),
                objects=objects,
                hfov_deg=float(args.hfov_deg),
                yaw_candidates=int(args.yaw_candidates),
            )
            n_waypoints_kept = sum(1 for v in trajectory_views if v["is_waypoint"])
            n_interp = len(trajectory_views) - n_waypoints_kept
            LOGGER.info(
                "trajectory: %d poses (%d waypoints + %d interpolated at %.2fm step), "
                "%d yaw(s)/pose, base yaws %s°",
                len(trajectory_views), n_waypoints_kept, n_interp, args.step_m,
                trajectory_yaws_per_view, [round(math.degrees(y), 1) for y in traj_yaws],
            )
            meta["trajectory"] = {
                "step_m": float(args.step_m),
                "yaw_offset_deg": float(args.yaw_offset_deg),
                "yaw_policy": str(args.yaw_policy),
                "yaw_candidates": int(args.yaw_candidates),
                "yaws_per_view": trajectory_yaws_per_view,
                "n_yaw_objects": len(objects),
                "yaws_deg": [round(math.degrees(y), 2) for y in traj_yaws],
                "n_waypoints": n_waypoints_kept,
                "n_interpolated": n_interp,
                "n_poses": len(trajectory_views),
                "n_views_before_depth_filter": len(trajectory_views) * trajectory_yaws_per_view,
                "yaw_score_mean": (
                    float(np.mean([float(v.get("yaw_score", 0.0)) for v in trajectory_views]))
                    if trajectory_views else 0.0
                ),
            }

        def _render_one(pos: np.ndarray, yaw: float) -> bool:
            nonlocal frames_total, frames_attempted, frames_skipped_invalid_depth, chunk_idx
            frames_attempted += 1
            sim.get_agent(0).set_state(_agent_state(pos, yaw))
            obs = sim.get_sensor_observations()
            rgb = obs["rgb_camera"]
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            depth = np.asarray(obs["depth_camera"], dtype=np.float32)
            if args.depth_min > 0.0:
                depth = np.where(depth < args.depth_min, 0.0, depth)
            if args.depth_clip > 0.0:
                depth = np.where(depth > args.depth_clip, 0.0, depth)
            if min_valid_depth_fraction > 0.0:
                valid_fraction = float(np.count_nonzero(depth > 0.0)) / float(depth.size)
                if valid_fraction < min_valid_depth_fraction:
                    frames_skipped_invalid_depth += 1
                    return False
            pose = _camera_to_world_opengl(sim)
            images.append(np.ascontiguousarray(rgb.astype(np.uint8, copy=False)))
            depths.append(np.ascontiguousarray(depth))
            poses.append(pose)
            frames_total += 1
            if len(images) >= per_chunk:
                path = _save_chunk(out_dir, chunk_idx, images, depths, poses, K)
                meta["chunk_paths"].append(str(path))
                LOGGER.info("wrote chunk %s (%d frames)", path.name, len(images))
                chunk_idx += 1
                images.clear()
                depths.clear()
                poses.clear()
            return True

        if mode == "trajectory":
            yaw_offsets = [
                2.0 * math.pi * j / trajectory_yaws_per_view
                for j in range(trajectory_yaws_per_view)
            ]
            for v in trajectory_views:
                base_yaw = float(v["yaw"])
                for yaw_offset in yaw_offsets:
                    _render_one(v["pos"], base_yaw + yaw_offset)
        elif mode == "magnet":
            # Each magnet anchor renders only at its target-facing yaw; interp
            # poses already have their own yaw assigned by
            # _build_per_island_trajectories.
            for v in magnet_views:
                _render_one(v["pos"], float(v["yaw"]))
        else:
          for group in groups:
            for pos in group["points"]:
                for yaw in yaws:
                    _render_one(pos, yaw)

        if images:
            path = _save_chunk(out_dir, chunk_idx, images, depths, poses, K)
            meta["chunk_paths"].append(str(path))
            LOGGER.info("wrote chunk %s (%d frames)", path.name, len(images))

        meta["frame_count"] = frames_total
        meta["frame_attempt_count"] = frames_attempted
        meta["frame_skip_invalid_depth_count"] = frames_skipped_invalid_depth
        meta["frame_skip_invalid_depth_fraction"] = (
            float(frames_skipped_invalid_depth) / float(frames_attempted)
            if frames_attempted else 0.0
        )

        # Magnet metadata: per-anchor (object_id, label, region_id, island_id,
        # yaw_deg, distance) so the user can audit magnet placement without
        # re-running the renderer. Written even with 0 previews requested.
        if mode == "magnet" and magnet_views:
            anchor_records: List[Dict] = []
            for i, v in enumerate(magnet_views):
                if not v.get("is_magnet"):
                    continue
                obj_id = v.get("object_id")
                isl_id = v.get("island_id")
                anchor_records.append({
                    "trajectory_index": i,
                    "object_id": int(obj_id) if obj_id is not None else -1,
                    "label": str(v.get("label") or ""),
                    "region_id": v.get("region_id"),
                    "island_id": int(isl_id) if isl_id is not None else -1,
                    "pos": [float(x) for x in v["pos"]],
                    "yaw_rad": float(v["yaw"]),
                    "yaw_deg": float(math.degrees(float(v["yaw"]))),
                    "candidate_distance_m": float(v.get("candidate_distance", 0.0)),
                    "n_occlusion_pass": int(v.get("n_occlusion_pass", 0)),
                    "is_fair": bool(v.get("is_fair", False)),
                    "is_referenced": bool(v.get("is_referenced", False)),
                })
            (out_dir / "magnet_summary.json").write_text(
                json.dumps(
                    {
                        "scene_id": args.scene_id,
                        "n_anchors": len(anchor_records),
                        "anchors": anchor_records,
                    },
                    indent=2,
                )
            )

        # Preview PNGs: uniformly-sampled frames across the trajectory, plus
        # one tagged "first frame on island N" per visited island. Writes to
        # <out>/_previews/.
        n_previews = int(args.save_previews)
        if n_previews > 0 and frames_total > 0:
            try:
                import imageio.v2 as imageio
                previews_dir = out_dir / "_previews"
                previews_dir.mkdir(parents=True, exist_ok=True)
                # Walk the chunk files again to pull sampled frames without
                # holding everything in memory.
                chunk_paths = [Path(p) for p in meta["chunk_paths"]]
                # Build a per-frame index of (chunk_path, intra_chunk_idx, yaw).
                idx_table: List[Tuple[Path, int]] = []
                for cp in chunk_paths:
                    with np.load(cp) as d:
                        n_in_chunk = int(d["images"].shape[0])
                    for j in range(n_in_chunk):
                        idx_table.append((cp, j))
                if not idx_table:
                    raise RuntimeError("no frames to preview")
                # Pick uniform sample indices.
                sample_idxs = np.linspace(
                    0, len(idx_table) - 1, num=n_previews
                ).round().astype(int).tolist()
                # Per-island representatives (first magnet anchor on each island).
                island_first: Dict[int, int] = {}
                if mode == "magnet":
                    cumulative = 0
                    for v in magnet_views:
                        # Each rendered pose contributes exactly 1 frame
                        # (magnet_yaws_per_magnet defaults to 1).
                        isl_raw = v.get("island_id")
                        isl = int(isl_raw) if isl_raw is not None else -1
                        if isl >= 0 and v.get("is_magnet") and isl not in island_first:
                            island_first[isl] = cumulative
                        cumulative += 1
                preview_records: List[Dict] = []
                # Combine uniform + per-island sample sets.
                tagged: List[Tuple[str, int]] = [("uniform", i) for i in sample_idxs]
                for isl, k in sorted(island_first.items()):
                    tagged.append((f"island_{isl}", k))
                for tag, frame_idx in tagged:
                    cp, j = idx_table[frame_idx]
                    with np.load(cp) as d:
                        img = d["images"][j]
                    fname = f"preview_{tag}_f{frame_idx:05d}.png"
                    imageio.imwrite(previews_dir / fname, img)
                    preview_records.append({
                        "tag": tag,
                        "frame_index": int(frame_idx),
                        "filename": fname,
                    })
                (previews_dir / "index.json").write_text(
                    json.dumps(preview_records, indent=2)
                )
                LOGGER.info(
                    "wrote %d preview PNGs to %s",
                    len(preview_records), previews_dir,
                )
            except Exception as exc:
                LOGGER.warning("preview PNG write failed: %s", exc)

        (out_dir / "render_meta.json").write_text(json.dumps(meta, indent=2))
        LOGGER.info("done. %d/%d frames kept across %d chunks (%d skipped by depth) → %s",
                    frames_total, frames_attempted, len(meta["chunk_paths"]),
                    frames_skipped_invalid_depth, out_dir)
        return 0
    finally:
        sim.close()


if __name__ == "__main__":
    sys.exit(main())
