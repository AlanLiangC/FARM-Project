"""Shared contract for offline frame sources.

A ``FrameSource`` yields dicts that ``StreamingMapper._run_mapping_batch`` accepts.
Two shapes are supported downstream by ``_decode_batch``:

1. ``{"camera": str, "rgbd_msg": RGBDFrame, "received_time": float}`` — the native
   online format. Used by the rosbag source.
2. A pre-decoded dict with keys ``rgb``, ``depth_f32``, ``T_world_cam``,
   ``rgb_instrinsics``, ``depth_instrinsics``, ``camera``, ``stamp_ns``,
   ``frame_id``, ``received_time`` (no ``rgbd_msg``). Used by dataset readers
   that avoid ROS message construction.

Both paths converge at ``_decode_batch`` in the online node.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator

FrameItem = Dict[str, Any]


class FrameSource(ABC):
    """Yields per-frame dicts to be handed to ``StreamingMapper._run_mapping_batch``."""

    @abstractmethod
    def __iter__(self) -> Iterator[FrameItem]:
        raise NotImplementedError

    def close(self) -> None:
        """Release any held resources. Default: no-op."""
        return None
