"""Abstract base class and shared data model for RGBD datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import torch


class DatasetFrame(NamedTuple):
    """One RGBD frame returned by any dataset loader.

    Implemented as a NamedTuple so existing callers that unpack as
    ``color, depth, intrinsics, pose = dataset[i]`` continue to work.
    """

    color: torch.Tensor       # (H, W, 3) float32, values in [0, 255] unless normalize_color=True
    depth: torch.Tensor       # (H, W, 1) float32, metric depth in metres
    intrinsics: torch.Tensor  # (4, 4) float32, upper-left 3×3 is the K matrix
    pose: torch.Tensor        # (4, 4) float32, camera-to-world SE(3)


class BaseDataset(ABC):
    """Minimal interface that all dataset loaders must satisfy."""

    @abstractmethod
    def __len__(self) -> int:
        """Total number of frames in the (possibly subsampled) sequence."""

    @abstractmethod
    def __getitem__(self, index: int) -> DatasetFrame:
        """Return the frame at position *index* in the subsampled sequence."""

    @abstractmethod
    def get_cam_K(self) -> torch.Tensor:
        """Return the 3×3 camera intrinsics matrix K."""
