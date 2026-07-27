#!/usr/bin/env python3
import contextlib
import queue
import sys
import time
from queue import Empty, Queue  # Import Empty directly
from threading import Thread
from typing import Dict, Optional, Union

import message_filters
import rclpy
from geometry_msgs.msg import TransformStamped
from mapping_msgs.msg import FrameMetadata, RGBDFrame
from scene_graph.camera_config import CAMERA_CONFIG
from mapping.nodes.tf_listener import OdomStaticTransformListener
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image


def get_first_message(node, topic, msg_type, timeout_sec=None):
    future = rclpy.task.Future()

    def callback(msg):
        if not future.done():
            future.set_result(msg)

    sub = node.create_subscription(msg_type, topic, callback, qos_profile=1)

    try:
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
        return future.result() if future.done() else None
    finally:
        node.destroy_subscription(sub)


class MappingFrameStreamer(Node):
    def __init__(self):
        super().__init__("mapping_frame_pub")
        # Ensure launch files can reliably override this (and that this node uses ROS time when enabled).
        with contextlib.suppress(Exception):
            self.declare_parameter("use_sim_time", True)

        # Params
        self.declare_parameter("camera", "rear")
        self.declare_parameter("global_frame", "spot/odom")
        self.declare_parameter("tf_buffer_cache_time_sec", 300.0)
        self.declare_parameter("tf_lookup_timeout_sec", 0.2)
        self.declare_parameter("sync_queue_size", 20)
        self.declare_parameter("sync_slop_sec", 0.05)
        self.declare_parameter("max_rgb_depth_skew_sec", 0.05)
        self.declare_parameter("max_tf_skew_sec", 0.05)
        self.declare_parameter("use_compressed_image", False)

        # Params
        self.camera = self.get_parameter("camera").get_parameter_value().string_value
        if self.camera not in CAMERA_CONFIG:
            available = ", ".join(sorted(CAMERA_CONFIG))
            raise ValueError(f"Unknown camera '{self.camera}'. Available: {available}")
        cam_cfg = CAMERA_CONFIG[self.camera]

        use_compressed_image = self.get_parameter("use_compressed_image").get_parameter_value().bool_value
        rgb_topic = cam_cfg.rgb_compressed_topic if use_compressed_image else cam_cfg.rgb_topic
        depth_topic = cam_cfg.depth_topic

        rgb_msg_type = CompressedImage if use_compressed_image else Image

        self.publish_topic = f"mapping/frame_stream/{self.camera}"
        self.rgbd_publish_topic = f"mapping/rgbd_frame/{self.camera}"

        # TF
        self.global_frame = self.get_parameter("global_frame").get_parameter_value().string_value
        tf_cache_time_sec = float(self.get_parameter("tf_buffer_cache_time_sec").value)
        tf_lookup_timeout_sec = float(self.get_parameter("tf_lookup_timeout_sec").value)
        odom_buffer_size = max(100, int(max(1.0, tf_cache_time_sec) * 250.0))
        self.tf_listener = OdomStaticTransformListener(
            self,
            odom_buffer_size=odom_buffer_size,
            tf_buffer_cache_time_sec=tf_cache_time_sec,
            tf_lookup_timeout_sec=tf_lookup_timeout_sec,
        )
        max_rgb_depth_skew_sec = float(self.get_parameter("max_rgb_depth_skew_sec").value)
        if max_rgb_depth_skew_sec <= 0.0:
            max_rgb_depth_skew_sec = 0.05
        self._max_rgb_depth_skew_sec = max_rgb_depth_skew_sec
        self._max_rgb_depth_skew_ns = int(max_rgb_depth_skew_sec * 1e9)
        max_tf_skew_sec = float(self.get_parameter("max_tf_skew_sec").value)
        if max_tf_skew_sec <= 0.0:
            max_tf_skew_sec = 0.05
        self._max_tf_skew_sec = max_tf_skew_sec
        self._max_tf_skew_ns = int(max_tf_skew_sec * 1e9)

        # Background worker thread
        self.work_queue = Queue(maxsize=100)
        self.worker_thread = Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        # CameraInfo cache
        self.camera_info_depth_msg = get_first_message(self, cam_cfg.depth_info_topics[0], CameraInfo)
        self.camera_info_rgb_msg = get_first_message(self, cam_cfg.rgb_info_topics[0], CameraInfo)

        # Subscribers + sync
        self._frame_cb_group = MutuallyExclusiveCallbackGroup()
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.counter_in = 0
        self.counter_out = 0
        rgb_sub = message_filters.Subscriber(
            self, rgb_msg_type, rgb_topic, qos_profile=sensor_qos, callback_group=self._frame_cb_group
        )
        depth_sub = message_filters.Subscriber(
            self, Image, depth_topic, qos_profile=sensor_qos, callback_group=self._frame_cb_group
        )

        sync_sub = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=self.get_parameter("sync_queue_size").get_parameter_value().integer_value,
            slop=float(self.get_parameter("sync_slop_sec").value),
        )
        sync_sub.registerCallback(self.frame_cb)

        # Publisher
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.pub = self.create_publisher(FrameMetadata, self.publish_topic, qos)
        self.rgbd_pub = self.create_publisher(RGBDFrame, self.rgbd_publish_topic, qos)

    def _time_from_msg(self, stamp_msg) -> Time:
        clock_type = self.get_clock().clock_type
        try:
            return Time.from_msg(stamp_msg, clock_type=clock_type)
        except TypeError:
            return Time(
                seconds=int(getattr(stamp_msg, "sec", 0)),
                nanoseconds=int(getattr(stamp_msg, "nanosec", 0)),
                clock_type=clock_type,
            )

    def _get_intrinsics(self, camera_info_msg: CameraInfo):

        return {
            "fx": float(camera_info_msg.k[0]),
            "fy": float(camera_info_msg.k[4]),
            "cx": float(camera_info_msg.k[2]),
            "cy": float(camera_info_msg.k[5]),
            "width": int(camera_info_msg.width),
            "height": int(camera_info_msg.height),
        }

    def _scale_intrinsics(
        self,
        intrinsics: Dict[str, float],
        target_width: int,
        target_height: int,
        source_frame: Optional[str] = None,
    ) -> Dict[str, float]:
        src_width = int(intrinsics.get("width", 0) or 0)
        src_height = int(intrinsics.get("height", 0) or 0)
        if src_width <= 0 or src_height <= 0:
            return intrinsics

        scale_x = float(target_width) / float(src_width)
        scale_y = float(target_height) / float(src_height)
        if abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6:
            return intrinsics

        adjusted = dict(intrinsics)
        adjusted["fx"] = float(adjusted["fx"]) * scale_x
        adjusted["fy"] = float(adjusted["fy"]) * scale_y
        adjusted["cx"] = float(adjusted["cx"]) * scale_x
        adjusted["cy"] = float(adjusted["cy"]) * scale_y
        adjusted["width"] = int(target_width)
        adjusted["height"] = int(target_height)
        if source_frame:
            self.get_logger().debug(
                f"Scaled intrinsics from frame '{source_frame}' by factors ({scale_x:.4f}, {scale_y:.4f})"
            )
        return adjusted

    def frame_cb(self, rgb_msg: CompressedImage, depth_msg: Image):
        """Callback for the frame publisher."""
        rgb_stamp_ns = rgb_msg.header.stamp.sec * 10**9 + rgb_msg.header.stamp.nanosec
        depth_stamp_ns = depth_msg.header.stamp.sec * 10**9 + depth_msg.header.stamp.nanosec
        if abs(rgb_stamp_ns - depth_stamp_ns) > self._max_rgb_depth_skew_ns:
            delta_sec = abs(rgb_stamp_ns - depth_stamp_ns) / 1e9
            self.get_logger().warn(
                f"Dropping RGB/depth pair due to timestamp skew Δ={delta_sec:.3f}s "
                f"(limit={self._max_rgb_depth_skew_sec:.3f}s)"
            )
            return

        self.counter_in += 1
        q_size = self.work_queue.qsize()
        if self.counter_in % 10 == 0:
            self.get_logger().debug(f"In: {self.counter_in} | Queue: {q_size}")

        try:
            self.work_queue.put_nowait((rgb_msg, depth_msg, 0))  # 0 is 'attempts'
        except queue.Full:
            self.get_logger().warn("Work queue full, dropping frame")

    def _worker_loop(self):
        while rclpy.ok():
            try:
                # Correct way to catch the Empty exception
                rgb_msg, depth_msg, attempts = self.work_queue.get(timeout=1.0)
            except Empty:
                continue

            # Attempt TF lookup
            tf_world_cam = self.tf_listener.lookup_transform(
                self.global_frame, rgb_msg.header.frame_id, rgb_msg.header.stamp
            )

            if tf_world_cam is None:
                if attempts < 100:
                    # If the queue is almost empty, slow down slightly to save CPU
                    if self.work_queue.qsize() < 5:
                        time.sleep(0.01)
                    self.work_queue.put((rgb_msg, depth_msg, attempts + 1))
                else:
                    self.get_logger().warn(f"TF timeout for {rgb_msg.header.stamp}")
                continue

            self.process_and_publish(rgb_msg, depth_msg, tf_world_cam)

    def process_and_publish(
        self, rgb_msg: Union[CompressedImage, Image], depth_msg: Image, tf_world_cam: TransformStamped
    ):
        if self.camera_info_depth_msg is None or self.camera_info_rgb_msg is None:
            return
        # Get intrinsics
        rgb_intrinsics = self._get_intrinsics(self.camera_info_rgb_msg)
        W_rgb = int(self.camera_info_rgb_msg.width)
        H_rgb = int(self.camera_info_rgb_msg.height)

        depth_intrinsics = self._get_intrinsics(self.camera_info_depth_msg)
        W_depth = int(self.camera_info_depth_msg.width)
        H_depth = int(self.camera_info_depth_msg.height)

        rgb_time = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9
        depth_time = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec * 1e-9
        tf_time = tf_world_cam.header.stamp.sec + tf_world_cam.header.stamp.nanosec * 1e-9
        rgb_tf_delta = abs(rgb_time - tf_time)
        if rgb_tf_delta > self._max_tf_skew_sec:
            self.get_logger().warn(
                f"Dropping frame due to TF timestamp skew Δ={rgb_tf_delta:.3f}s "
                f"(limit={self._max_tf_skew_sec:.3f}s)"
            )
            return
        depth_tf_delta = abs(depth_time - tf_time)
        if depth_tf_delta > 0.01:
            self.get_logger().warn(f"Depth time is {depth_tf_delta:.6f} seconds later than TF time")

        # Publish metadata describing this frame
        metadata_msg = FrameMetadata()
        metadata_msg.header.stamp = rgb_msg.header.stamp
        metadata_msg.header.frame_id = rgb_msg.header.frame_id
        metadata_msg.camera = self.camera
        metadata_msg.frame_id = rgb_msg.header.frame_id
        metadata_msg.depth_stamp = depth_msg.header.stamp
        metadata_msg.tf_stamp = tf_world_cam.header.stamp
        metadata_msg.transform_world_cam = tf_world_cam.transform
        metadata_msg.rgb_fx = float(rgb_intrinsics["fx"])
        metadata_msg.rgb_fy = float(rgb_intrinsics["fy"])
        metadata_msg.rgb_cx = float(rgb_intrinsics["cx"])
        metadata_msg.rgb_cy = float(rgb_intrinsics["cy"])
        metadata_msg.rgb_width = int(W_rgb)
        metadata_msg.rgb_height = int(H_rgb)
        metadata_msg.depth_fx = float(depth_intrinsics["fx"])
        metadata_msg.depth_fy = float(depth_intrinsics["fy"])
        metadata_msg.depth_cx = float(depth_intrinsics["cx"])
        metadata_msg.depth_cy = float(depth_intrinsics["cy"])
        metadata_msg.depth_width = int(W_depth)
        metadata_msg.depth_height = int(H_depth)
        metadata_msg.intrinsics_frame = self.camera_info_rgb_msg.header.frame_id
        metadata_msg.pose_source = "odom"

        self.pub.publish(metadata_msg)
        bundle = RGBDFrame()
        bundle.meta = metadata_msg
        if isinstance(rgb_msg, Image):
            bundle.rgb = rgb_msg
        else:
            bundle.rgb_compressed = rgb_msg
        bundle.depth = depth_msg
        self.rgbd_pub.publish(bundle)

        self.counter_out += 1
        if self.counter_out % 10 == 0:
            self.get_logger().debug(f"Published frame {self.counter_out}")


def main():
    rclpy.init(args=sys.argv)
    node = MappingFrameStreamer()
    try:
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    finally:
        with contextlib.suppress(Exception):
            executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
