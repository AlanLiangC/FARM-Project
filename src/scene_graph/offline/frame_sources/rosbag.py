"""Frame source that replays ``RGBDFrame`` messages from a ROS 2 rosbag."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Optional, Set

from .base import FrameItem, FrameSource


class RosbagFrameSource(FrameSource):
    """Yields ``{"camera", "rgbd_msg", "received_time"}`` items from a rosbag2.

    Replays bags recorded from the online pipeline: listens for
    messages on ``/mapping/rgbd_frame/<camera>`` topics, deserializes each to an
    ``RGBDFrame`` message, and forwards to the mapper. ``_decode_batch`` in
    ``StreamingMapper`` handles the rest.

    Requires a ROS 2 environment (``rclpy`` + ``rosbag2_py`` + ``mapping_msgs``).
    """

    def __init__(
        self,
        bag_path: str | Path,
        *,
        cameras: Optional[Iterable[str]] = None,
        every_nth: int = 1,
        topic_prefix: str = "/mapping/rgbd_frame/",
    ) -> None:
        self.bag_path = str(Path(bag_path).expanduser())
        self.topic_prefix = topic_prefix
        self.every_nth = max(1, int(every_nth))
        self._camera_filter: Optional[Set[str]] = set(cameras) if cameras else None

    def __iter__(self) -> Iterator[FrameItem]:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message

        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id="")
        converter_options = rosbag2_py.ConverterOptions("", "")
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
        camera_counts: dict[str, int] = {}

        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            if not topic.startswith(self.topic_prefix):
                continue

            camera = topic[len(self.topic_prefix) :]
            if self._camera_filter and camera not in self._camera_filter:
                continue

            n = camera_counts.get(camera, 0) + 1
            camera_counts[camera] = n
            if self.every_nth > 1 and n % self.every_nth != 1:
                continue

            msg_type_str = topic_types.get(topic)
            if not msg_type_str:
                continue
            msg_class = get_message(msg_type_str)
            rgbd_msg = deserialize_message(data, msg_class)

            yield {
                "camera": camera,
                "rgbd_msg": rgbd_msg,
                "received_time": timestamp / 1e9,
            }
