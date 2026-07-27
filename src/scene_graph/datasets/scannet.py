"""ScanNet dataset loader."""

from __future__ import annotations

import glob
from typing import List, Optional

import numpy as np
import torch
from natsort import natsorted

from scene_graph.datasets._gradslam_base import GradSLAMDataset


class ScannetDataset(GradSLAMDataset):
    """ScanNet v2 dataset (color/*.jpg, depth/*.png, pose/*.txt)."""

    def __init__(
        self,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 968,
        desired_width: Optional[int] = 1296,
        **kwargs,
    ):
        import os
        self.input_folder = os.path.join(kwargs["base_dir"], kwargs["sequence"])
        self.pose_path = None
        super().__init__(
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        return color_paths, depth_paths

    def load_poses(self) -> List[torch.Tensor]:
        poses = []
        for posefile in natsorted(glob.glob(f"{self.input_folder}/pose/*.txt")):
            poses.append(torch.from_numpy(np.loadtxt(posefile)))
        return poses
