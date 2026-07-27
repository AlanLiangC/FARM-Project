"""GradSLAMDataset — legacy BBQ-era base class for file-based RGBD loaders.

All gradslam utility calls (scale_intrinsics, normalize_image, etc.) have been
inlined so that ``gradslam`` is no longer a hard dependency of this module.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
import torch

from scene_graph.datasets.interfaces import BaseDataset, DatasetFrame


# ---------------------------------------------------------------------------
# Inlined gradslam utilities (replaces `from gradslam.datasets import datautils`)
# ---------------------------------------------------------------------------

def _scale_intrinsics(K: torch.Tensor, h_ratio: float, w_ratio: float) -> torch.Tensor:
    """Scale a (*, 3, 3) or (*, 4, 4) intrinsics tensor by the given ratios."""
    K = K.clone()
    K[..., 0, 0] *= w_ratio   # fx
    K[..., 1, 1] *= h_ratio   # fy
    K[..., 0, 2] *= w_ratio   # cx
    K[..., 1, 2] *= h_ratio   # cy
    return K


def _normalize_image(color: np.ndarray) -> np.ndarray:
    return (color / 255.0).astype(np.float32)


def _channels_first(arr: np.ndarray) -> np.ndarray:
    return np.transpose(arr, (2, 0, 1))


def _relative_transformation(
    trans_01: torch.Tensor,
    trans_12: torch.Tensor,
    orthogonal_rotations: bool = False,
) -> torch.Tensor:
    """Return ``trans_01⁻¹ @ trans_12`` (batched 4×4 SE3 matrices)."""
    if orthogonal_rotations:
        inv = trans_01.clone()
        inv[..., :3, :3] = trans_01[..., :3, :3].transpose(-1, -2)
        inv[..., :3, 3:4] = -(inv[..., :3, :3] @ trans_01[..., :3, 3:4])
        return inv @ trans_12
    return torch.linalg.inv(trans_01) @ trans_12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def as_intrinsics_matrix(intrinsics) -> np.ndarray:
    """Build a 3×3 K matrix from (fx, fy, cx, cy)."""
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


# OpenGL camera axes (x right, y up, z backward) → OpenCV (x right, y down, z forward)
OPENGL_TO_OPENCV = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32
)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class GradSLAMDataset(torch.utils.data.Dataset, BaseDataset):
    """Stride-aware RGBD dataset base class originally from the BBQ project.

    Subclasses must implement :meth:`get_filepaths` and :meth:`load_poses`.
    """

    def __init__(
        self,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: int = 480,
        desired_width: int = 640,
        channels_first: bool = False,
        normalize_color: bool = False,
        device: str = "cuda:0",
        dtype=torch.float,
        relative_pose: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.name = kwargs["name"]
        self.device = device
        self.png_depth_scale = kwargs["camera_params"]["png_depth_scale"]

        self.orig_height = kwargs["camera_params"]["image_height"]
        self.orig_width = kwargs["camera_params"]["image_width"]
        self.fx = kwargs["camera_params"]["fx"]
        self.fy = kwargs["camera_params"]["fy"]
        self.cx = kwargs["camera_params"]["cx"]
        self.cy = kwargs["camera_params"]["cy"]

        self.dtype = dtype
        self.desired_height = desired_height
        self.desired_width = desired_width
        self.height_downsample_ratio = float(self.desired_height) / self.orig_height
        self.width_downsample_ratio = float(self.desired_width) / self.orig_width
        self.channels_first = channels_first
        self.normalize_color = normalize_color
        self.relative_pose = relative_pose

        self.start = start
        self.end = end
        if start < 0:
            raise ValueError(f"start must be non-negative. Got {start}.")
        if not (end == -1 or end > start):
            raise ValueError(
                f"end ({end}) must be -1 (use all images) or greater than start ({start})."
            )

        self.distortion = (
            np.array(kwargs["camera_params"]["distortion"])
            if "distortion" in kwargs["camera_params"]
            else None
        )
        self.crop_size = kwargs["camera_params"].get("crop_size")
        self.crop_edge = kwargs["camera_params"].get("crop_edge")

        self.color_paths, self.depth_paths = self.get_filepaths()
        if len(self.color_paths) != len(self.depth_paths):
            raise ValueError("Number of color and depth images must be the same.")

        self.num_imgs = len(self.color_paths)
        self.poses = self.load_poses()

        if self.end == -1:
            self.end = self.num_imgs

        self.color_paths = self.color_paths[self.start : self.end : stride]
        self.depth_paths = self.depth_paths[self.start : self.end : stride]
        self.poses = self.poses[self.start : self.end : stride]
        self.retained_inds = torch.arange(self.num_imgs)[self.start : self.end : stride]
        self.num_imgs = len(self.color_paths)

        self.poses = torch.stack(self.poses)
        if self.relative_pose:
            self.transformed_poses = self._preprocess_poses(self.poses)
        else:
            self.transformed_poses = self.poses

    def __len__(self) -> int:
        return self.num_imgs

    def get_filepaths(self):
        """Return (color_paths, depth_paths). Implement in subclass."""
        raise NotImplementedError

    def load_poses(self) -> List[torch.Tensor]:
        """Return list of (4, 4) pose tensors. Implement in subclass."""
        raise NotImplementedError

    def _preprocess_color(self, color: np.ndarray) -> np.ndarray:
        color = cv2.resize(
            color,
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_LINEAR,
        )
        if self.normalize_color:
            return _normalize_image(color)
        if self.channels_first:
            return _channels_first(color)
        return color

    def _preprocess_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = cv2.resize(
            depth.astype(float),
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.expand_dims(depth, -1)
        if self.channels_first:
            depth = _channels_first(depth)
        return depth / self.png_depth_scale

    def _preprocess_poses(self, poses: torch.Tensor) -> torch.Tensor:
        return _relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal_rotations=False,
        )

    def get_cam_K(self) -> torch.Tensor:
        K = torch.from_numpy(as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy]))
        K = _scale_intrinsics(K, self.height_downsample_ratio, self.width_downsample_ratio)
        return K

    def __getitem__(self, index: int) -> DatasetFrame:
        import imageio

        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color = np.asarray(imageio.imread(color_path), dtype=float)
        color = self._preprocess_color(color)
        color = torch.from_numpy(color)

        if ".png" in depth_path:
            depth = np.asarray(imageio.imread(depth_path), dtype=np.int64)
        elif ".npy" in depth_path:
            depth = np.load(depth_path)
        else:
            raise NotImplementedError(f"Unsupported depth format: {depth_path}")

        K = torch.from_numpy(as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy]))
        if self.distortion is not None:
            color_np = color.numpy() if isinstance(color, torch.Tensor) else color
            color_np = cv2.undistort(color_np, K.numpy(), self.distortion)
            color = torch.from_numpy(color_np)

        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)

        K = _scale_intrinsics(K, self.height_downsample_ratio, self.width_downsample_ratio)
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K

        pose = self.transformed_poses[index]

        return DatasetFrame(
            color=color.to(self.device).type(self.dtype),
            depth=depth.to(self.device).type(self.dtype),
            intrinsics=intrinsics.to(self.device).type(self.dtype),
            pose=pose.to(self.device).type(self.dtype),
        )
