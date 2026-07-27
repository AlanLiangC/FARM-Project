"""Shared camera configuration for BBQ ROS nodes."""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class CameraConfig:
    rgb_compressed_topic: str
    rgb_topic: str
    depth_topic: str
    rgb_info_topics: Tuple[str, ...]
    depth_info_topics: Tuple[str, ...]
    frame_ids: Tuple[str, ...]


CAMERA_CONFIG: Dict[str, CameraConfig] = {
    "head_left": CameraConfig(
        rgb_topic="/spot/camera/frontleft/image",
        rgb_compressed_topic="/spot/camera/frontleft/image/compressed",
        depth_topic="/spot/depth/frontleft/image",
        rgb_info_topics=("/spot/camera/frontleft/camera_info",),
        depth_info_topics=("/spot/depth/frontleft/camera_info",),
        frame_ids=(
            "head_left_rgbd_optical",
            "head_left_rgb_optical",
            "head_rgb_left_optical_frame",
            "head_rgb_left_optical",
            "spot/camera_front_left_optical_link",
            "spot/camera_front_left_link",
        ),
    ),
    "head_right": CameraConfig(
        rgb_topic="/spot/camera/frontright/image",
        rgb_compressed_topic="/spot/camera/frontright/image/compressed",
        depth_topic="/spot/depth/frontright/image",
        rgb_info_topics=("/spot/camera/frontright/camera_info",),
        depth_info_topics=("/spot/depth/frontright/camera_info",),
        frame_ids=(
            "head_right_rgbd_optical",
            "head_right_rgb_optical",
            "head_rgb_right_optical_frame",
            "head_rgb_right_optical",
            "spot/camera_front_right_optical_link",
            "spot/camera_front_right_link",
        ),
    ),
    "left": CameraConfig(
        rgb_topic="/spot/camera/left/image",
        rgb_compressed_topic="/spot/camera/left/image/compressed",
        depth_topic="/spot/depth/left/image",
        rgb_info_topics=("/spot/camera/left/camera_info",),
        depth_info_topics=("/spot/depth/left/camera_info",),
        frame_ids=(
            "left_rgbd_optical",
            "left_rgb_optical",
            "left_rgb_optical_frame",
            "spot/camera_left_optical_link",
            "spot/camera_left_link",
        ),
    ),
    "right": CameraConfig(
        rgb_topic="/spot/camera/right/image",
        rgb_compressed_topic="/spot/camera/right/image/compressed",
        depth_topic="/spot/depth/right/image",
        rgb_info_topics=("/spot/camera/right/camera_info",),
        depth_info_topics=("/spot/depth/right/camera_info",),
        frame_ids=(
            "right_rgbd_optical",
            "right_rgb_optical",
            "right_rgb_optical_frame",
            "spot/camera_right_optical_link",
            "spot/camera_right_link",
        ),
    ),
    "rear": CameraConfig(
        rgb_topic="/spot/camera/back/image",
        rgb_compressed_topic="/spot/camera/back/image/compressed",
        depth_topic="/spot/depth/back/image",
        rgb_info_topics=("/spot/camera/back/camera_info",),
        depth_info_topics=("/spot/depth/back/camera_info",),
        frame_ids=(
            "rear_rgbd_optical",
            "rear_rgb_optical",
            "rear_rgb_optical_frame",
            "spot/camera_rear_optical_link",
            "spot/camera_rear_link",
        ),
    ),
    # Odin1: a single fisheye + LiDAR unit. It has no native depth image, so the
    # `odin1_depth_pub` node synthesises depth by projecting the LiDAR cloud into
    # the camera and republishes these standard RGBD topics. From `frame_pub`'s
    # perspective it is just another camera (one interface, any sensor).
    "odin1": CameraConfig(
        rgb_topic="/odin1/rect/image",
        rgb_compressed_topic="/odin1/rect/image",
        depth_topic="/odin1/rect/depth",
        rgb_info_topics=("/odin1/rect/camera_info",),
        depth_info_topics=("/odin1/rect/depth/camera_info",),
        frame_ids=("odin1_optical",),
    ),
}

__all__ = ["CameraConfig", "CAMERA_CONFIG"]
