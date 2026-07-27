"""Geometric transform utilities for the streaming mapper node.

Covers SE(3) pose ↔ matrix conversions, quaternion math, image rotation
helpers (tensor and numpy), and camera-id normalisation.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from geometry_msgs.msg import Pose, Transform


def normalize_camera_id(camera: object) -> str:
    """Normalise a raw camera identifier to a canonical lowercase name."""
    value = str(camera or "").strip().lower()
    if not value:
        return ""
    value = value.replace("-", "_")
    mapping = {
        "head_left": "head_left",
        "frontleft": "head_left",
        "front_left": "head_left",
        "head_right": "head_right",
        "frontright": "head_right",
        "front_right": "head_right",
        "left": "left",
        "right": "right",
        "rear": "rear",
        "back": "rear",
    }
    return mapping.get(value, value)


def matrix_from_xyz_quat(xyz: Sequence[float], quat_xyzw: Sequence[float]) -> np.ndarray:
    """Build a 4×4 SE(3) matrix from a translation vector and xyzw quaternion."""
    if len(xyz) != 3 or len(quat_xyzw) != 4:
        raise ValueError("Expected xyz(3) and quat(4).")
    xyz_arr = np.asarray(xyz, dtype=np.float64).reshape(3)
    quat_arr = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(xyz_arr)):
        raise ValueError("xyz contains NaN/Inf.")
    if not np.all(np.isfinite(quat_arr)):
        raise ValueError("quaternion contains NaN/Inf.")
    norm = float(np.linalg.norm(quat_arr))
    if norm <= 1e-12:
        raise ValueError("Quaternion norm is zero.")
    x, y, z, w = (quat_arr / norm).tolist()

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rotation = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = xyz_arr
    return matrix


def transform_to_matrix(transform: Transform) -> np.ndarray:
    """Convert a ROS ``geometry_msgs/Transform`` message to a 4×4 SE(3) matrix."""
    return matrix_from_xyz_quat(
        [
            float(transform.translation.x),
            float(transform.translation.y),
            float(transform.translation.z),
        ],
        [
            float(transform.rotation.x),
            float(transform.rotation.y),
            float(transform.rotation.z),
            float(transform.rotation.w),
        ],
    )


def xyz_quat_from_matrix(
    matrix: np.ndarray,
) -> Optional[Tuple[List[float], List[float]]]:
    """Extract (xyz, xyzw quaternion) from a 4×4 SE(3) matrix.

    Returns ``None`` if the matrix is invalid or non-finite.
    """
    if matrix is None or matrix.shape != (4, 4):
        return None
    if not np.all(np.isfinite(matrix)):
        return None

    rot = matrix[:3, :3]
    trans = matrix[:3, 3]

    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])) * 2.0
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s if s else 0.0
        qz = (rot[0, 2] + rot[2, 0]) / s if s else 0.0
        qw = (rot[2, 1] - rot[1, 2]) / s if s else 0.0
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])) * 2.0
        qx = (rot[0, 1] + rot[1, 0]) / s if s else 0.0
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s if s else 0.0
        qw = (rot[0, 2] - rot[2, 0]) / s if s else 0.0
    else:
        s = math.sqrt(max(0.0, 1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])) * 2.0
        qx = (rot[0, 2] + rot[2, 0]) / s if s else 0.0
        qy = (rot[1, 2] + rot[2, 1]) / s if s else 0.0
        qz = 0.25 * s
        qw = (rot[1, 0] - rot[0, 1]) / s if s else 0.0

    xyz = [float(trans[0]), float(trans[1]), float(trans[2])]
    quat = [float(qx), float(qy), float(qz), float(qw)]
    return xyz, quat


def pose_from_matrix(matrix: np.ndarray) -> Pose:
    """Build a ROS ``geometry_msgs/Pose`` message from a 4×4 SE(3) matrix."""
    pose_msg = Pose()
    payload = xyz_quat_from_matrix(matrix)
    if payload is None:
        return pose_msg
    xyz, quat = payload
    pose_msg.position.x = xyz[0]
    pose_msg.position.y = xyz[1]
    pose_msg.position.z = xyz[2]
    pose_msg.orientation.x = quat[0]
    pose_msg.orientation.y = quat[1]
    pose_msg.orientation.z = quat[2]
    pose_msg.orientation.w = quat[3]
    return pose_msg


def rotz_homogeneous(deg: int) -> torch.Tensor:
    """Return a 4×4 homogeneous rotation matrix around +Z (right-handed).

    *deg* must be a multiple of 90; non-multiples fall back to the general
    ``cos``/``sin`` path.
    """
    deg = int(deg) % 360
    out = torch.eye(4, dtype=torch.float32)
    if deg == 0:
        return out
    if deg == 90:
        out[0, 0] = 0.0
        out[0, 1] = -1.0
        out[1, 0] = 1.0
        out[1, 1] = 0.0
        return out
    if deg == 180:
        out[0, 0] = -1.0
        out[1, 1] = -1.0
        return out
    if deg == 270:
        out[0, 0] = 0.0
        out[0, 1] = 1.0
        out[1, 0] = -1.0
        out[1, 1] = 0.0
        return out

    rad = math.radians(float(deg))
    c = float(math.cos(rad))
    s = float(math.sin(rad))
    out[0, 0] = c
    out[0, 1] = -s
    out[1, 0] = s
    out[1, 1] = c
    return out


def rotate_image_tensor_cw(tensor: torch.Tensor, deg_cw: int) -> torch.Tensor:
    """Rotate a HW / HWC / CHW tensor by multiples of 90 degrees clockwise."""
    deg_cw = int(deg_cw) % 360
    if deg_cw == 0:
        return tensor

    if tensor.ndim == 2:
        spatial_dims = (0, 1)
    elif tensor.ndim == 3:
        # HWC (common for numpy->torch) or CHW.
        if tensor.shape[-1] in (1, 3, 4):
            spatial_dims = (0, 1)
        elif tensor.shape[0] in (1, 3, 4):
            spatial_dims = (1, 2)
        else:
            raise ValueError(f"Unsupported image tensor shape for rotation: {tuple(tensor.shape)}")
    else:
        raise ValueError(f"Unsupported image tensor rank for rotation: {tensor.ndim}")

    if deg_cw == 90:
        return torch.rot90(tensor, k=-1, dims=spatial_dims)
    if deg_cw == 180:
        return torch.flip(tensor, dims=list(spatial_dims))
    if deg_cw == 270:
        return torch.rot90(tensor, k=1, dims=spatial_dims)
    raise ValueError(f"Rotation must be a multiple of 90 degrees, got {deg_cw}.")


def rotate_image_array_cw(image: np.ndarray, deg_cw: int) -> np.ndarray:
    """Rotate a HWC / HW numpy image by multiples of 90 degrees clockwise."""
    deg_cw = int(deg_cw) % 360
    if deg_cw == 0:
        return image
    if deg_cw == 90:
        return np.ascontiguousarray(np.rot90(image, k=-1))
    if deg_cw == 180:
        return np.ascontiguousarray(np.rot90(image, k=2))
    if deg_cw == 270:
        return np.ascontiguousarray(np.rot90(image, k=1))
    raise ValueError(f"Rotation must be a multiple of 90 degrees, got {deg_cw}.")
