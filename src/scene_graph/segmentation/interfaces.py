from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch

from .models import SegmentationOutput


class SegmentationBackend(ABC):
    """Shared interface for segmentation implementations (YOLOE, SAM, etc.)."""

    @abstractmethod
    def __call__(
        self,
        color: torch.Tensor | Sequence[torch.Tensor],
        depth: torch.Tensor | Sequence[torch.Tensor],
        intrinsics: torch.Tensor | Sequence[torch.Tensor],
    ) -> SegmentationOutput | Sequence[SegmentationOutput]:
        raise NotImplementedError
