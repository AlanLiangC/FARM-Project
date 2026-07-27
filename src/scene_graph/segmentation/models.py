from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class SegmentationOutput:
    """Container for batched per-detection information produced by a segmentation backend."""

    boxes_xyxy: torch.Tensor
    scores: torch.Tensor
    class_ids: torch.Tensor
    class_names: Sequence[str]
    masks: torch.Tensor
    features: torch.Tensor
    means: torch.Tensor
    covariances: torch.Tensor

    @property
    def num_detections(self) -> int:
        return int(self.boxes_xyxy.shape[0])

    def to(self, device: torch.device | str) -> "SegmentationOutput":
        return SegmentationOutput(
            boxes_xyxy=self.boxes_xyxy.to(device),
            scores=self.scores.to(device),
            class_ids=self.class_ids.to(device),
            class_names=list(self.class_names),
            masks=self.masks.to(device),
            features=self.features.to(device),
            means=self.means.to(device),
            covariances=self.covariances.to(device),
        )

    def cpu(self) -> "SegmentationOutput":
        return self.to(torch.device("cpu"))

    @classmethod
    def empty(
        cls,
        image_shape: tuple[int, int],
        feature_dim: int,
        device: torch.device | str = torch.device("cpu"),
    ) -> "SegmentationOutput":
        h, w = image_shape
        device = torch.device(device)
        return cls(
            boxes_xyxy=torch.empty((0, 4), device=device),
            scores=torch.empty((0,), device=device),
            class_ids=torch.empty((0,), dtype=torch.long, device=device),
            class_names=[],
            masks=torch.empty((0, h, w), dtype=torch.bool, device=device),
            features=torch.empty((0, feature_dim), device=device),
            means=torch.empty((0, 3), device=device),
            covariances=torch.empty((0, 6), device=device),
        )
