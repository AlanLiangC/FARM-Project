from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class FrameBatch:
    colors: List[torch.Tensor]
    depths: List[torch.Tensor]
    intrinsics: List[torch.Tensor]
    poses_world: List[torch.Tensor]


@dataclass
class BatchResult:
    seg_outputs: dict
    neighbors: list
    det_idx: torch.Tensor
    obj_idx: torch.Tensor
    update_info: Optional[dict] = None
    new_indices: List[int] = field(default_factory=list)
