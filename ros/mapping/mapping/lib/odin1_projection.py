"""Self-contained Odin1 fisheye + LiDAR projection helpers.

The Odin1 spatial-sensing module is a single fisheye camera rigidly mounted with
a rotating LiDAR. Its ROS2 bag ships:

  * ``/odin1/image[/compressed]``  — BGR8 / JPEG fisheye frames (~10 Hz)
  * ``/odin1/cloud_slam``          — ``PointCloud2`` already in the odom/world
                                     frame (x, y, z[, rgb])
  * ``/odin1/odometry[_highfreq]`` — ``nav_msgs/Odometry`` (odom -> base_link)

There is no depth image. To feed the RGBD mapping pipeline we synthesise a depth
image per RGB frame by projecting the world-frame LiDAR points into the camera at
the RGB timestamp. The camera uses a polynomial fisheye ("FishPoly") model::

    theta   = atan2(sqrt(X^2 + Y^2), Z)
    theta_d = theta + sum_i k[i] * theta**(i+2)     # k = k2..k7
    phi     = atan2(Y, X)
    u, v    = A11*theta_d*cos(phi) + A12*theta_d*sin(phi) + u0,
              A22*theta_d*sin(phi) + v0

``Tcl`` in the calibration maps base(=LiDAR) points into the camera frame.

This module is a trimmed, dependency-light port of the offline largescale
adapter so the public ROS layer has no dependency on the (eval-only) largescale
package. Only numpy is required at import time; OpenCV is imported lazily.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


@dataclass
class OdinCalibration:
    """FishPoly intrinsics + camera<-base extrinsic, loaded from the vendor YAML."""

    image_width: int
    image_height: int
    A11: float
    A12: float
    A22: float
    u0: float
    v0: float
    k: List[float]              # k2..k7
    T_camera_base: np.ndarray   # 4x4, maps base(LiDAR) points into the camera frame (Tcl)

    @staticmethod
    def from_yaml(path: "Path | str") -> "OdinCalibration":
        import yaml

        raw = yaml.safe_load(Path(path).read_text())
        cam = raw["cam_0"]
        tcl = raw.get("Tcl_0")
        if tcl is None:
            raise ValueError(f"{path}: missing 'Tcl_0' transform")
        return OdinCalibration(
            image_width=int(cam["image_width"]),
            image_height=int(cam["image_height"]),
            A11=float(cam["A11"]),
            A12=float(cam.get("A12", 0.0)),
            A22=float(cam["A22"]),
            u0=float(cam["u0"]),
            v0=float(cam["v0"]),
            k=[float(cam[f"k{i}"]) for i in (2, 3, 4, 5, 6, 7)],
            T_camera_base=np.asarray(tcl, dtype=np.float64).reshape(4, 4),
        )

    @property
    def K(self) -> np.ndarray:
        """Affine intrinsic block (used verbatim as the fisheye-mode camera_info)."""
        return np.array(
            [[self.A11, self.A12, self.u0],
             [0.0,      self.A22, self.v0],
             [0.0,      0.0,      1.0]],
            dtype=np.float64,
        )

    def pinhole_K(self, *, focal_scale: float = 1.0) -> np.ndarray:
        """Pinhole intrinsics for the undistorted (rectified) output."""
        s = float(focal_scale)
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError(f"focal_scale must be positive; got {focal_scale!r}")
        return np.array(
            [[self.A11 * s, 0.0, self.u0],
             [0.0, self.A22 * s, self.v0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# projection
# ---------------------------------------------------------------------------


def project_fishpoly(pts_cam: np.ndarray, calib: OdinCalibration) -> np.ndarray:
    """Camera-frame points -> pixel coords via FishPoly. Z<=0 -> NaN."""
    pts_cam = np.asarray(pts_cam, dtype=np.float64)
    X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    r = np.sqrt(X * X + Y * Y)
    theta = np.arctan2(r, Z)
    phi = np.arctan2(Y, X)
    theta_d = theta.copy()
    power = theta.copy()
    for k in calib.k:
        power = power * theta
        theta_d = theta_d + float(k) * power
    xd = theta_d * np.cos(phi)
    yd = theta_d * np.sin(phi)
    u = calib.A11 * xd + calib.A12 * yd + calib.u0
    v = calib.A22 * yd + calib.v0
    behind = Z <= 0
    if np.any(behind):
        u = u.copy(); v = v.copy()
        u[behind] = np.nan; v[behind] = np.nan
    return np.stack([u, v], axis=-1)


def project_world_fishpoly(
    pts_world: np.ndarray, T_world_cam: np.ndarray, calib: OdinCalibration
) -> Tuple[np.ndarray, np.ndarray]:
    if pts_world.size == 0:
        return np.zeros((0, 2)), np.zeros((0,))
    T_cam_world = np.linalg.inv(T_world_cam)
    ones = np.ones((pts_world.shape[0], 1))
    pts_cam = (T_cam_world @ np.concatenate([pts_world, ones], axis=1).T).T[:, :3]
    return project_fishpoly(pts_cam, calib), pts_cam[:, 2]


def project_world_pinhole(
    pts_world: np.ndarray, T_world_cam: np.ndarray, K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if pts_world.size == 0:
        return np.zeros((0, 2)), np.zeros((0,))
    T_cam_world = np.linalg.inv(T_world_cam)
    ones = np.ones((pts_world.shape[0], 1))
    pts_cam = (T_cam_world @ np.concatenate([pts_world, ones], axis=1).T).T[:, :3]
    z = pts_cam[:, 2]
    safe = z > 1e-6
    u = np.full_like(z, np.nan); v = np.full_like(z, np.nan)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    u[safe] = fx * (pts_cam[safe, 0] / z[safe]) + cx
    v[safe] = fy * (pts_cam[safe, 1] / z[safe]) + cy
    return np.stack([u, v], axis=-1), z


def rasterize_zbuffer(
    pixels: np.ndarray, depth_z: np.ndarray, W: int, H: int,
    *, min_depth: float = 0.1, max_depth: float = 80.0,
) -> np.ndarray:
    """Splat (pixels, depth_z) into an (H, W) float32 z-buffer; NaN elsewhere."""
    out = np.full((H, W), np.nan, dtype=np.float32)
    if pixels.shape[0] == 0:
        return out
    u, v, z = pixels[:, 0], pixels[:, 1], depth_z
    keep = (np.isfinite(u) & np.isfinite(v) & np.isfinite(z)
            & (z >= float(min_depth)) & (z <= float(max_depth)))
    u, v, z = u[keep], v[keep], z[keep]
    iu = np.round(u).astype(np.int64); iv = np.round(v).astype(np.int64)
    ok = (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
    iu, iv, z = iu[ok], iv[ok], z[ok]
    if iu.size == 0:
        return out
    flat = iv * W + iu
    order = np.lexsort((-z, flat))  # nearest (smallest z) written last -> wins
    flat = flat[order]; z = z[order]
    out.reshape(-1)[flat] = z
    return out


def make_fishpoly_rectification_maps(
    calib: OdinCalibration, K_pinhole: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """cv2.remap maps: pinhole output pixel -> source fisheye pixel."""
    h, w = int(calib.image_height), int(calib.image_width)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    fx, fy = float(K_pinhole[0, 0]), float(K_pinhole[1, 1])
    cx, cy = float(K_pinhole[0, 2]), float(K_pinhole[1, 2])
    rays = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)], axis=-1).reshape(-1, 3)
    pix = project_fishpoly(rays, calib).reshape(h, w, 2)
    return pix[..., 0].astype(np.float32), pix[..., 1].astype(np.float32)


def dilate_sparse_depth(depth: np.ndarray, *, radius_px: int) -> np.ndarray:
    """Fill invalid pixels from the nearest valid depth within ``radius_px``."""
    radius = int(radius_px)
    if radius <= 0:
        return depth
    src = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(src) & (src > 0.0)
    if not np.any(valid) or np.all(valid):
        return src
    import cv2

    dist_src = np.where(valid, 0, 255).astype(np.uint8)
    dist, labels = cv2.distanceTransformWithLabels(
        dist_src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    max_label = int(labels.max())
    if max_label <= 0:
        return src
    label_depth = np.full(max_label + 1, np.nan, dtype=np.float32)
    label_depth[labels[valid]] = src[valid]
    nearest = label_depth[labels]
    fill = (~valid) & (dist <= float(radius)) & np.isfinite(nearest) & (nearest > 0.0)
    out = src.copy()
    out[fill] = nearest[fill]
    return out


# ---------------------------------------------------------------------------
# PointCloud2 decode
# ---------------------------------------------------------------------------


def pointcloud2_to_xyz(msg) -> np.ndarray:
    """Decode a ``sensor_msgs/PointCloud2`` to (N,3) float32 XYZ (finite only)."""
    fields = {f.name: int(f.offset) for f in msg.fields}
    if not all(c in fields for c in ("x", "y", "z")):
        return np.zeros((0, 3), dtype=np.float32)
    point_step = int(msg.point_step)
    n = int(msg.width) * int(msg.height)
    if n == 0 or point_step <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    buf = bytes(msg.data) if not isinstance(msg.data, bytes) else msg.data
    arr = np.frombuffer(buf, dtype=np.uint8)
    if arr.size < n * point_step:
        n = arr.size // point_step
    arr = arr[: n * point_step].reshape(n, point_step)

    def _f32(name: str) -> np.ndarray:
        off = fields[name]
        return np.frombuffer(arr[:, off:off + 4].tobytes(), dtype=np.float32)

    pts = np.stack([_f32("x"), _f32("y"), _f32("z")], axis=-1)
    return pts[np.isfinite(pts).all(axis=1)].astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# pose helpers (quaternion xyzw)
# ---------------------------------------------------------------------------


def quat_to_rotmat_xyzw(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    if n <= 1e-24:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array(
        [[1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
         [s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)],
         [s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)]],
        dtype=np.float64,
    )


def _slerp_xyzw(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = q0 / max(float(np.linalg.norm(q0)), 1e-12)
    q1 = q1 / max(float(np.linalg.norm(q1)), 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1; dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        q = q0 + float(alpha) * (q1 - q0)
        return q / max(float(np.linalg.norm(q)), 1e-12)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * float(alpha)
    s0 = math.cos(theta) - dot * math.sin(theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


def interpolate_pose_matrix(
    times: List[float], positions: List, quats: List, t_s: float
) -> np.ndarray:
    """SE(3) pose at ``t_s`` from sorted (times, positions, quats_xyzw) samples."""
    import bisect

    if not times:
        raise ValueError("no odometry samples to interpolate")
    if len(times) == 1 or t_s <= float(times[0]):
        pos = np.asarray(positions[0], dtype=np.float64)
        quat = np.asarray(quats[0], dtype=np.float64)
    elif t_s >= float(times[-1]):
        pos = np.asarray(positions[-1], dtype=np.float64)
        quat = np.asarray(quats[-1], dtype=np.float64)
    else:
        i1 = bisect.bisect_left(times, float(t_s))
        i0 = max(0, i1 - 1)
        t0, t1 = float(times[i0]), float(times[i1])
        alpha = 0.0 if t1 <= t0 else float((t_s - t0) / (t1 - t0))
        p0 = np.asarray(positions[i0], dtype=np.float64)
        p1 = np.asarray(positions[i1], dtype=np.float64)
        pos = (1.0 - alpha) * p0 + alpha * p1
        quat = _slerp_xyzw(np.asarray(quats[i0]), np.asarray(quats[i1]), alpha)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_rotmat_xyzw(quat)
    T[:3, 3] = pos
    return T


__all__ = [
    "OdinCalibration",
    "project_fishpoly",
    "project_world_fishpoly",
    "project_world_pinhole",
    "rasterize_zbuffer",
    "make_fishpoly_rectification_maps",
    "dilate_sparse_depth",
    "pointcloud2_to_xyz",
    "quat_to_rotmat_xyzw",
    "interpolate_pose_matrix",
]
