"""Replica, Habitat, and Isaac dataset loaders."""

from __future__ import annotations

import glob
import os
from typing import List, Optional

import numpy as np
import torch
from natsort import natsorted

from scene_graph.datasets._gradslam_base import GradSLAMDataset


class ReplicaDataset(GradSLAMDataset):
    """Replica scene dataset (iSDF / Replica format: frame*.jpg + depth*.png + traj.txt)."""

    def __init__(
        self,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        **kwargs,
    ):
        self.input_folder = os.path.join(kwargs["base_dir"], kwargs["sequence"])
        self.pose_path = os.path.join(self.input_folder, "traj.txt")
        super().__init__(
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        return color_paths, depth_paths

    def load_poses(self) -> List[torch.Tensor]:
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            c2w = np.array(list(map(float, lines[i].split()))).reshape(4, 4)
            poses.append(torch.from_numpy(c2w).float())
        return poses


class HabitatDataset(ReplicaDataset):
    """Habitat-Sim variant: OpenGL-convention poses (y-up) stored in traj.txt."""

    def load_poses(self) -> List[torch.Tensor]:
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            c2w = np.array(list(map(float, lines[i].split()))).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            poses.append(torch.from_numpy(c2w).float())
        return poses


class IsaacDataset(ReplicaDataset):
    """Isaac Sim variant: same OpenGL-convention axis flip as HabitatDataset."""

    def load_poses(self) -> List[torch.Tensor]:
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            c2w = np.array(list(map(float, lines[i].split()))).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            poses.append(torch.from_numpy(c2w).float())
        return poses
