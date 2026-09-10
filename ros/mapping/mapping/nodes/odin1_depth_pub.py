#!/usr/bin/env python3
"""Odin1 LiDAR -> depth republisher: makes a fisheye+LiDAR unit look like an RGBD camera.

The Odin1 module has no depth image — only a fisheye RGB stream and a world-frame
LiDAR ``PointCloud2``. This node fuses them into the standard RGBD camera topics
the rest of the online stack already consumes, so ``odin1`` becomes *just another
entry in CAMERA_CONFIG* handled by the identical ``frame_pub`` -> ``streaming_mapper``
path (one interface, any number / kind of cameras).

Per RGB frame it:
  1. interpolates the IMU pose and composes the factory IMU/LiDAR and per-device
     LiDAR/camera extrinsics,
  2. gathers world-frame LiDAR scans within ``±scan_window_s`` (subsampled),
  3. projects them into the camera to synthesise a depth image,
  4. (optional) rectifies RGB+depth to a pinhole model,
  5. publishes ``image`` + ``depth`` + two ``camera_info`` on the ``odin1`` topics,
     all stamped with the RGB timestamp and framed at ``optical_frame``.

It also broadcasts the static TF ``base_frame -> optical_frame = inv(Tcl)`` so
``frame_pub``'s odom+static-TF pose composition resolves the camera pose exactly
as it does for Spot.

Fisheye (default) vs pinhole is selected with ``output_pinhole``. Fisheye mode
reproduces the offline FARM ``sdh_4_and_7`` convention (affine K, raw fisheye
grid); pinhole mode is geometrically exact for the pinhole-unprojection mapper
but crops the FOV.
"""

from __future__ import annotations

import contextlib
import sys
import threading
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2
from nav_msgs.msg import Odometry
from tf2_ros import StaticTransformBroadcaster
from tf2_msgs.msg import TFMessage

from mapping.lib.odin1_projection import (
    OdinCalibration,
    dilate_sparse_depth,
    interpolate_pose_matrix,
    make_fishpoly_rectification_maps,
    pointcloud2_to_xyz,
    project_world_fishpoly,
    project_world_pinhole,
    quat_to_rotmat_xyzw,
    rasterize_zbuffer,
)


def _stamp_s(header) -> float:
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9


def _matrix_to_quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    import math

    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return x / n, y / n, z / n, w / n


def _pose_matrix(position, orientation) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = quat_to_rotmat_xyzw(
        (orientation.x, orientation.y, orientation.z, orientation.w)
    )
    out[:3, 3] = (position.x, position.y, position.z)
    return out


class Odin1DepthPublisher(Node):
    def __init__(self) -> None:
        super().__init__("odin1_depth_pub")
        with contextlib.suppress(Exception):
            self.declare_parameter("use_sim_time", True)

        self.declare_parameter("calibration_path", "/odin1_data/calib.yaml")
        self.declare_parameter("image_topic", "/odin1/image/compressed")
        self.declare_parameter("cloud_topic", "/odin1/cloud_slam")
        self.declare_parameter("odom_topic", "/odin1/odometry_highfreq")
        self.declare_parameter("out_image_topic", "/odin1/rect/image")
        self.declare_parameter("out_depth_topic", "/odin1/rect/depth")
        self.declare_parameter("out_rgb_info_topic", "/odin1/rect/camera_info")
        self.declare_parameter("out_depth_info_topic", "/odin1/rect/depth/camera_info")
        self.declare_parameter("tf_topic", "/tf")
        self.declare_parameter("out_map_odom_topic", "/odin1/map_odometry")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("optical_frame", "odin1_optical")
        self.declare_parameter("base_frame", "odin1_imu")
        self.declare_parameter("odometry_body_frame", "imu")
        self.declare_parameter("output_pinhole", False)
        self.declare_parameter("rectified_focal_scale", 1.0)
        self.declare_parameter("scan_window_s", 0.15)
        self.declare_parameter("max_scans_per_frame", 3)
        self.declare_parameter("depth_min_m", 0.1)
        self.declare_parameter("depth_max_m", 60.0)
        self.declare_parameter("depth_dilate_px", 2)
        self.declare_parameter("cloud_buffer_size", 40)
        self.declare_parameter("odom_buffer_size", 4000)
        self.declare_parameter("cloud_coordinates", "world")
        self.declare_parameter("output_scale", 1.0)
        self.declare_parameter("depth_encoding", "32FC1")
        self.declare_parameter("max_sensor_age_s", 0.0)

        gp = lambda n: self.get_parameter(n).value
        self._calib = OdinCalibration.from_yaml(str(gp("calibration_path")))
        self._image_topic = str(gp("image_topic"))
        self._cloud_topic = str(gp("cloud_topic"))
        self._odom_topic = str(gp("odom_topic"))
        self._optical_frame = str(gp("optical_frame"))
        self._base_frame = str(gp("base_frame"))
        self._map_frame = str(gp("map_frame"))
        self._odom_frame = str(gp("odom_frame"))
        self._output_pinhole = bool(gp("output_pinhole"))
        self._focal_scale = float(gp("rectified_focal_scale"))
        self._scan_window_s = float(gp("scan_window_s"))
        self._max_scans = int(gp("max_scans_per_frame"))
        self._depth_min = float(gp("depth_min_m"))
        self._depth_max = float(gp("depth_max_m"))
        self._depth_dilate = int(gp("depth_dilate_px"))
        self._output_scale = float(gp("output_scale"))
        if not 0.1 <= self._output_scale <= 1.0:
            raise ValueError("output_scale must be between 0.1 and 1.0")
        self._depth_encoding = str(gp("depth_encoding")).upper()
        if self._depth_encoding not in {"16UC1", "32FC1"}:
            raise ValueError("depth_encoding must be 16UC1 or 32FC1")
        self._max_sensor_age_s = max(0.0, float(gp("max_sensor_age_s")))
        self._cloud_coordinates = str(gp("cloud_coordinates")).lower()
        if self._cloud_coordinates not in {"world", "lidar"}:
            raise ValueError("cloud_coordinates must be 'world' or 'lidar'")

        self._source_W = int(self._calib.image_width)
        self._source_H = int(self._calib.image_height)
        self._W = max(1, int(round(self._source_W * self._output_scale)))
        self._H = max(1, int(round(self._source_H * self._output_scale)))
        odometry_body_frame = str(gp("odometry_body_frame")).lower()
        if odometry_body_frame == "imu":
            self._T_base_cam = self._calib.T_imu_camera
        elif odometry_body_frame == "lidar":
            # Compatibility for old bags that encoded a LiDAR body pose.
            self._T_base_cam = np.linalg.inv(self._calib.T_camera_base)
        else:
            raise ValueError("odometry_body_frame must be 'imu' or 'lidar'")

        if self._output_pinhole:
            self._K = self._calib.pinhole_K(focal_scale=self._focal_scale)
            self._K[0, :] *= self._output_scale
            self._K[1, :] *= self._output_scale
            self._rect_x, self._rect_y = make_fishpoly_rectification_maps(
                self._calib,
                self._K,
                output_width=self._W,
                output_height=self._H,
            )
        else:
            self._K = self._calib.K.copy()
            self._K[0, :] *= self._output_scale
            self._K[1, :] *= self._output_scale
            self._rect_x = self._rect_y = None
        self._fx, self._fy = float(self._K[0, 0]), float(self._K[1, 1])
        self._cx, self._cy = float(self._K[0, 2]), float(self._K[1, 2])

        # Buffers.
        self._lock = threading.Lock()
        self._cloud_buf: Deque[Tuple[float, np.ndarray]] = deque(maxlen=int(gp("cloud_buffer_size")))
        self._odom_max = int(gp("odom_buffer_size"))
        self._odom_t: List[float] = []
        self._odom_p: List[Tuple[float, float, float]] = []
        self._odom_q: List[Tuple[float, float, float, float]] = []
        self._T_odom_map: Optional[np.ndarray] = None

        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                                durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)
        reliable_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                                  durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)
        # PointCloud2 is large (~0.5 MB/scan); BEST_EFFORT drops most of them, so
        # subscribe RELIABLE with a deeper queue (matches how `ros2 topic hz` — a
        # reliable sub — sees the full 8 Hz).
        cloud_qos = QoSProfile(depth=30, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)

        img_type = CompressedImage if self._image_topic.rstrip("/").endswith("compressed") else Image
        self._img_type = img_type
        self.create_subscription(img_type, self._image_topic, self._image_cb, sensor_qos)
        self.create_subscription(PointCloud2, self._cloud_topic, self._cloud_cb, cloud_qos)
        self.create_subscription(Odometry, self._odom_topic, self._odom_cb, reliable_qos)
        self.create_subscription(TFMessage, str(gp("tf_topic")), self._tf_cb, reliable_qos)

        self._pub_img = self.create_publisher(Image, str(gp("out_image_topic")), sensor_qos)
        self._pub_depth = self.create_publisher(Image, str(gp("out_depth_topic")), sensor_qos)
        self._pub_rgb_info = self.create_publisher(CameraInfo, str(gp("out_rgb_info_topic")), reliable_qos)
        self._pub_depth_info = self.create_publisher(CameraInfo, str(gp("out_depth_info_topic")), reliable_qos)
        self._pub_map_odom = self.create_publisher(
            Odometry, str(gp("out_map_odom_topic")), reliable_qos
        )

        # Static TF odometry body -> optical, including the current driver's
        # IMU<-LiDAR factory extrinsic.
        self._static_tf = StaticTransformBroadcaster(self)
        self._broadcast_static_tf()

        # Heavy projection runs in a worker thread so the ROS callbacks stay light
        # (just buffer). Otherwise the single executor thread spends all its time
        # in the image callback and starves the cloud/odom buffers -> no scans match.
        # The worker processes only the *latest* image (latest-only), which both
        # avoids unbounded lag and keeps the image well-aligned with fresh clouds.
        # Large clouds arrive with transmission/deserialize lag, so the newest
        # buffered cloud trails the newest image. We therefore hold images in a
        # pending queue and only process one once the cloud stream has advanced
        # past its timestamp (clouds bracket it) — dropping older ready frames so
        # we stay near real time (latest-ready).
        from collections import deque as _deque
        self._pending = _deque(maxlen=200)
        self._latest_cloud_t = -1.0
        self._latest_odom_t = -1.0
        self._stop = False
        self._n_in = 0
        self._n_out = 0
        self._n_stale = 0
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self.get_logger().info(
            f"odin1_depth_pub: {self._W}x{self._H} "
            f"{'pinhole' if self._output_pinhole else 'fisheye'} K=({self._fx:.1f},{self._fy:.1f},"
            f"{self._cx:.1f},{self._cy:.1f}); image={self._image_topic} cloud={self._cloud_topic} "
            f"odom={self._odom_topic}; base={self._base_frame} optical={self._optical_frame}; "
            f"cloud_coordinates={self._cloud_coordinates} depth={self._depth_encoding} "
            f"max_age={self._max_sensor_age_s:.2f}s"
        )

    # -- static TF --------------------------------------------------------
    def _broadcast_static_tf(self) -> None:
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self._base_frame
        t.child_frame_id = self._optical_frame
        T = self._T_base_cam
        t.transform.translation.x = float(T[0, 3])
        t.transform.translation.y = float(T[1, 3])
        t.transform.translation.z = float(T[2, 3])
        qx, qy, qz, qw = _matrix_to_quat_xyzw(T[:3, :3])
        t.transform.rotation.x = qx; t.transform.rotation.y = qy
        t.transform.rotation.z = qz; t.transform.rotation.w = qw
        self._static_tf.sendTransform(t)

    # -- buffers ----------------------------------------------------------
    def _cloud_cb(self, msg: PointCloud2) -> None:
        t = _stamp_s(msg.header)
        pts = pointcloud2_to_xyz(msg)
        with self._lock:
            self._cloud_buf.append((t, pts))
            if t > self._latest_cloud_t:
                self._latest_cloud_t = t

    def _odom_cb(self, msg: Odometry) -> None:
        t = _stamp_s(msg.header)
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        with self._lock:
            self._odom_t.append(t)
            self._odom_p.append((float(p.x), float(p.y), float(p.z)))
            self._odom_q.append((float(o.x), float(o.y), float(o.z), float(o.w)))
            if t > self._latest_odom_t:
                self._latest_odom_t = t
            if len(self._odom_t) > self._odom_max:
                self._odom_t.pop(0); self._odom_p.pop(0); self._odom_q.pop(0)
        self._publish_map_odometry(msg)

    def _tf_cb(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            parent = transform.header.frame_id.lstrip("/")
            child = transform.child_frame_id.lstrip("/")
            if {parent, child} != {self._odom_frame.lstrip("/"), self._map_frame.lstrip("/")}:
                continue
            value = _pose_matrix(
                transform.transform.translation,
                transform.transform.rotation,
            )
            # The pinned Odin driver publishes odom -> map. Accept the standard
            # inverse spelling as well so this adapter is robust to driver updates.
            T_odom_map = value if parent == self._odom_frame.lstrip("/") else np.linalg.inv(value)
            with self._lock:
                self._T_odom_map = T_odom_map

    def _publish_map_odometry(self, msg: Odometry) -> None:
        with self._lock:
            T_odom_map = None if self._T_odom_map is None else self._T_odom_map.copy()
        T_odom_body = _pose_matrix(msg.pose.pose.position, msg.pose.pose.orientation)
        T_map_body = T_odom_body if T_odom_map is None else np.linalg.inv(T_odom_map) @ T_odom_body
        qx, qy, qz, qw = _matrix_to_quat_xyzw(T_map_body[:3, :3])

        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self._map_frame
        out.child_frame_id = msg.child_frame_id or self._base_frame
        out.pose.pose.position.x = float(T_map_body[0, 3])
        out.pose.pose.position.y = float(T_map_body[1, 3])
        out.pose.pose.position.z = float(T_map_body[2, 3])
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw
        out.pose.covariance = msg.pose.covariance
        out.twist = msg.twist
        self._pub_map_odom.publish(out)

    # -- camera_info ------------------------------------------------------
    def _camera_info(self, stamp) -> CameraInfo:
        ci = CameraInfo()
        ci.header.stamp = stamp
        ci.header.frame_id = self._optical_frame
        ci.width = self._W; ci.height = self._H
        ci.distortion_model = "plumb_bob"
        ci.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        ci.k = [self._fx, 0.0, self._cx, 0.0, self._fy, self._cy, 0.0, 0.0, 1.0]
        ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ci.p = [self._fx, 0.0, self._cx, 0.0, 0.0, self._fy, self._cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return ci

    # -- main -------------------------------------------------------------
    def _decode_bgr(self, msg) -> Optional[np.ndarray]:
        import cv2

        if isinstance(msg, CompressedImage):
            buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        enc = (getattr(msg, "encoding", "") or "").lower()
        h, w = int(msg.height), int(msg.width)
        data = bytes(msg.data)
        if enc in ("bgr8", "rgb8"):
            row = int(msg.step) if int(msg.step) > 0 else w * 3
            arr = np.frombuffer(data, dtype=np.uint8, count=h * row).reshape(h, row)[:, : w * 3].reshape(h, w, 3)
            return arr if enc == "bgr8" else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if enc == "mono8":
            row = int(msg.step) if int(msg.step) > 0 else w
            arr = np.frombuffer(data, dtype=np.uint8, count=h * row).reshape(h, row)[:, :w]
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return None

    def _image_cb(self, msg) -> None:
        # Light: queue the frame; the worker processes it once clouds catch up.
        with self._lock:
            self._pending.append(msg)
            self._n_in += 1

    def _worker_loop(self) -> None:
        import time

        while not self._stop and rclpy.ok():
            target = None
            with self._lock:
                # Advance to the newest pending image the cloud stream has reached,
                # dropping older ready frames (stay near real time).
                ready_t = (
                    self._latest_cloud_t
                    if self._cloud_coordinates == "lidar"
                    else min(self._latest_cloud_t, self._latest_odom_t)
                )
                while self._pending and _stamp_s(self._pending[0].header) <= ready_t:
                    target = self._pending.popleft()
            if target is None:
                time.sleep(0.005)
                continue
            try:
                self._process_image(target)
            except Exception as exc:  # keep the worker alive
                self.get_logger().warn(f"odin1_depth_pub: frame failed: {exc}")

    def _process_image(self, msg) -> None:
        import cv2
        import time

        started = time.perf_counter()
        t_s = _stamp_s(msg.header)
        sensor_age = self.get_clock().now().nanoseconds * 1e-9 - t_s
        if self._max_sensor_age_s > 0.0 and sensor_age > self._max_sensor_age_s:
            self._n_stale += 1
            if self._n_stale == 1 or self._n_stale % 20 == 0:
                self.get_logger().warn(
                    f"odin1_depth_pub: dropped stale input age={sensor_age:.3f}s "
                    f"(limit={self._max_sensor_age_s:.3f}s, dropped={self._n_stale})"
                )
            return

        with self._lock:
            if self._cloud_coordinates == "world" and (
                len(self._odom_t) < 2 or self._odom_t[-1] < t_s
            ):
                return  # not enough odom yet to bracket this frame
            odom_t = list(self._odom_t); odom_p = list(self._odom_p); odom_q = list(self._odom_q)
            nearby = [(t, p) for (t, p) in self._cloud_buf if abs(t - t_s) <= self._scan_window_s]
            _cbuf_n = len(self._cloud_buf)
            _cbuf_lo = self._cloud_buf[0][0] if _cbuf_n else float("nan")
            _cbuf_hi = self._cloud_buf[-1][0] if _cbuf_n else float("nan")
        if not nearby and (self._n_out % 20 == 0):
            self.get_logger().warn(
                f"odin1_depth_pub: no cloud match img_t={t_s:.3f} "
                f"cloud_buf[{_cbuf_n}] span=[{_cbuf_lo:.3f},{_cbuf_hi:.3f}] "
                f"odom_t[-1]={(odom_t[-1] if odom_t else float('nan')):.3f}"
            )

        if self._cloud_coordinates == "world":
            try:
                T_world_base = interpolate_pose_matrix(odom_t, odom_p, odom_q, t_s)
            except Exception:
                return
            T_projection_cam = T_world_base @ self._T_base_cam
        else:
            # cloud_raw is in the LiDAR frame. Treat LiDAR as the temporary
            # projection world so no SLAM/odometry wait or motion transform is
            # needed; Tcl_0 maps those points directly into the camera.
            T_projection_cam = np.linalg.inv(self._calib.T_camera_base)

        bgr = self._decode_bgr(msg)
        if bgr is None:
            return
        if bgr.shape[1] != self._source_W or bgr.shape[0] != self._source_H:
            bgr = cv2.resize(
                bgr, (self._source_W, self._source_H), interpolation=cv2.INTER_AREA
            )

        if nearby:
            if len(nearby) > self._max_scans:
                step = max(1, len(nearby) // self._max_scans)
                nearby = nearby[::step][: self._max_scans]
            pts_world = np.concatenate([p for _t, p in nearby], axis=0).astype(np.float64, copy=False)
        else:
            pts_world = np.zeros((0, 3), dtype=np.float64)

        if self._output_pinhole:
            bgr = cv2.remap(bgr, self._rect_x, self._rect_y, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            pix, z = project_world_pinhole(pts_world, T_projection_cam, self._K)
        else:
            pix, z = project_world_fishpoly(pts_world, T_projection_cam, self._calib)
            if self._output_scale != 1.0:
                bgr = cv2.resize(bgr, (self._W, self._H), interpolation=cv2.INTER_AREA)
                pix *= self._output_scale
        depth = rasterize_zbuffer(pix, z, self._W, self._H, min_depth=self._depth_min, max_depth=self._depth_max)
        if self._depth_dilate > 0:
            depth = dilate_sparse_depth(depth, radius_px=self._depth_dilate)
        depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32, copy=False)  # 0 = invalid

        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        img_msg = Image()
        img_msg.header.stamp = msg.header.stamp
        img_msg.header.frame_id = self._optical_frame
        img_msg.height = self._H; img_msg.width = self._W
        img_msg.encoding = "rgb8"; img_msg.is_bigendian = 0
        img_msg.step = self._W * 3
        img_msg.data = rgb.tobytes()

        depth_msg = Image()
        depth_msg.header.stamp = msg.header.stamp
        depth_msg.header.frame_id = self._optical_frame
        depth_msg.height = self._H; depth_msg.width = self._W
        depth_msg.is_bigendian = 0
        if self._depth_encoding == "16UC1":
            depth_mm = np.clip(np.rint(depth * 1000.0), 0, 65535).astype(np.uint16)
            depth_msg.encoding = "16UC1"
            depth_msg.step = self._W * 2
            depth_msg.data = np.ascontiguousarray(depth_mm).tobytes()
        else:
            depth_msg.encoding = "32FC1"
            depth_msg.step = self._W * 4
            depth_msg.data = np.ascontiguousarray(depth).tobytes()

        # Projection is latest-only, but a slow host can still finish work after
        # the observation is no longer safe for control. Never publish such a
        # frame under its old sensor timestamp as if it were current.
        publish_age = self.get_clock().now().nanoseconds * 1e-9 - t_s
        if self._max_sensor_age_s > 0.0 and publish_age > self._max_sensor_age_s:
            self._n_stale += 1
            if self._n_stale == 1 or self._n_stale % 20 == 0:
                self.get_logger().warn(
                    f"odin1_depth_pub: discarded completed stale frame age={publish_age:.3f}s "
                    f"(limit={self._max_sensor_age_s:.3f}s, dropped={self._n_stale})"
                )
            return

        ci = self._camera_info(msg.header.stamp)
        self._pub_rgb_info.publish(ci)
        self._pub_depth_info.publish(ci)
        self._pub_img.publish(img_msg)
        self._pub_depth.publish(depth_msg)

        self._n_out += 1
        if self._n_out % 20 == 0:
            valid = int(np.count_nonzero(depth > 0.0))
            self.get_logger().info(
                f"odin1_depth_pub: published {self._n_out} (in={self._n_in}) "
                f"last valid_depth_px={valid} scans={len(nearby)} "
                f"processing_ms={(time.perf_counter() - started) * 1000.0:.1f} "
                f"sensor_age_ms={publish_age * 1000.0:.1f} stale_dropped={self._n_stale}"
            )


def main() -> None:
    rclpy.init(args=sys.argv)
    node = Odin1DepthPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node._stop = True
        with contextlib.suppress(Exception):
            node.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.shutdown()


if __name__ == "__main__":
    main()
