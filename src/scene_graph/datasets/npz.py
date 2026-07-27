"""NPZDataset — loader for sequences stored as consolidated .npz archives."""

from __future__ import annotations

import glob
import os
from typing import List, Optional

import numpy as np
import torch
from natsort import natsorted

from scene_graph.datasets._gradslam_base import (
    OPENGL_TO_OPENCV,
    GradSLAMDataset,
    _scale_intrinsics,
)
from scene_graph.datasets.interfaces import DatasetFrame


class NPZDataset(GradSLAMDataset):
    """Dataset wrapper for sequences stored as consolidated NPZ archives.

    Each archive must contain::

        images      : (N, H, W, 3) uint8
        depths      : (N, H, W) float32, metric depth in metres
        camtoworlds : (N, 4, 4) or (N, 3, 4) float32
        K           : (3, 3) float32 camera intrinsics (shared across archives)

    An optional ``pose_convention`` scalar string (``"opengl"`` or ``"opencv"``)
    controls axis conversion.  Defaults to ``"opengl"`` when absent.
    """

    def __init__(
        self,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = None,
        desired_width: Optional[int] = None,
        npz_pattern: str = "*.npz",
        **kwargs,
    ):
        pose_convention = kwargs.pop("pose_convention", None)
        if pose_convention is None:
            self.pose_convention = "auto"
        else:
            self.pose_convention = pose_convention.lower()
            if self.pose_convention not in ("opengl", "opencv"):
                raise ValueError(
                    f"pose_convention must be 'opengl' or 'opencv', got '{pose_convention}'"
                )

        self.sequence_dir = os.path.join(kwargs["base_dir"], kwargs["sequence"])
        self.npz_paths = natsorted(glob.glob(os.path.join(self.sequence_dir, npz_pattern)))
        if not self.npz_paths:
            raise FileNotFoundError(
                f"No NPZ files matching '{npz_pattern}' found in {self.sequence_dir}"
            )

        (
            self._full_colors,
            self._full_depths,
            self._full_poses,
            self._intrinsics,
        ) = self._load_npz_sequences(self.npz_paths, self.pose_convention)

        self._num_total = self._full_colors.shape[0]
        self._dummy_color_paths = list(range(self._num_total))
        self._dummy_depth_paths = list(range(self._num_total))

        H, W = self._full_colors.shape[1:3]
        camera_params = kwargs.setdefault("camera_params", {})
        camera_params["image_height"] = H
        camera_params["image_width"] = W
        camera_params["fx"] = float(self._intrinsics[0, 0])
        camera_params["fy"] = float(self._intrinsics[1, 1])
        camera_params["cx"] = float(self._intrinsics[0, 2])
        camera_params["cy"] = float(self._intrinsics[1, 2])
        depth_scale = kwargs.pop("npz_depth_scale", None)
        if depth_scale is None:
            depth_scale = camera_params.get("png_depth_scale", 1.0)
        camera_params["png_depth_scale"] = depth_scale

        if desired_height is None:
            desired_height = H
        if desired_width is None:
            desired_width = W

        super().__init__(
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            **kwargs,
        )

        indices = self.retained_inds.cpu().numpy().astype(int)
        self.colors = self._full_colors[indices]
        self.depths = self._full_depths[indices]
        self.poses = self.poses.to(torch.float32)
        if self.relative_pose:
            self.transformed_poses = self._preprocess_poses(self.poses)
        else:
            self.transformed_poses = self.poses

        self.K = torch.from_numpy(self._intrinsics).float()

        self._full_colors = None
        self._full_depths = None
        self._full_poses = None

    @staticmethod
    def _load_npz_sequences(npz_paths: List[str], pose_convention: str):
        colors, depths, poses = [], [], []
        intrinsics = None
        homog_row = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        for path in npz_paths:
            with np.load(path) as data:
                colors.append(np.array(data["images"], copy=True))
                depths.append(np.array(data["depths"], copy=True).astype(np.float32))
                pose_array = np.array(data["camtoworlds"], copy=True).astype(np.float32)

                file_conv = pose_convention
                if pose_convention == "auto":
                    file_conv = "opengl"
                    if "pose_convention" in data:
                        meta = data["pose_convention"]
                        try:
                            file_conv = str(
                                meta.tolist() if isinstance(meta, np.ndarray) else meta
                            ).lower()
                        except Exception:
                            file_conv = "opengl"
                if file_conv not in ("opengl", "opencv"):
                    raise ValueError(
                        f"Unsupported pose convention '{file_conv}' in {path}"
                    )

                if pose_array.ndim == 2:
                    pose_array = pose_array[None, ...]
                if pose_array.shape[-2:] == (3, 4):
                    pad = np.broadcast_to(
                        homog_row, pose_array.shape[:-2] + (1, 4)
                    )
                    pose_array = np.concatenate([pose_array, pad], axis=-2)
                elif pose_array.shape[-2:] != (4, 4):
                    raise ValueError(
                        f"Unexpected pose shape {pose_array.shape} in {path}"
                    )

                if file_conv == "opengl":
                    pose_array[..., :3, :3] = pose_array[..., :3, :3] @ OPENGL_TO_OPENCV
                poses.append(np.ascontiguousarray(pose_array))

                current_K = np.array(data["K"], copy=True).astype(np.float32)
                if intrinsics is None:
                    intrinsics = current_K
                elif not np.allclose(intrinsics, current_K):
                    raise ValueError(
                        f"Inconsistent intrinsics across NPZ files: {path}"
                    )

        return (
            np.concatenate(colors, axis=0),
            np.concatenate(depths, axis=0),
            np.concatenate(poses, axis=0),
            intrinsics,
        )

    def get_filepaths(self):
        return self._dummy_color_paths, self._dummy_depth_paths

    def load_poses(self) -> List[torch.Tensor]:
        return [torch.from_numpy(p).float() for p in self._full_poses]

    def __getitem__(self, index: int) -> DatasetFrame:
        color = self.colors[index].astype(np.float32, copy=False)
        depth = self.depths[index]

        color = self._preprocess_color(color)
        depth = self._preprocess_depth(depth)

        color = torch.from_numpy(color)
        depth = torch.from_numpy(depth)

        K = self.K.clone()
        K = _scale_intrinsics(K, self.height_downsample_ratio, self.width_downsample_ratio)
        intrinsics = torch.eye(4, dtype=K.dtype, device=K.device)
        intrinsics[:3, :3] = K

        pose = self.transformed_poses[index]

        return DatasetFrame(
            color=color.to(self.device).type(self.dtype),
            depth=depth.to(self.device).type(self.dtype),
            intrinsics=intrinsics.to(self.device).type(self.dtype),
            pose=pose.to(self.device).type(self.dtype),
        )
