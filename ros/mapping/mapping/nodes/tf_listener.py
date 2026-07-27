from __future__ import annotations

import bisect
import threading
from typing import Optional

import numpy as np
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener


def _quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = x * x + y * y + z * z + w * w
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    scale = 2.0 / norm
    xx = x * x * scale
    yy = y * y * scale
    zz = z * z * scale
    xy = x * y * scale
    xz = x * z * scale
    yz = y * z * scale
    wx = w * x * scale
    wy = w * y * scale
    wz = w * z * scale
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
        return x, y, z, w

    diag = np.diag(matrix)
    idx = int(np.argmax(diag))
    if idx == 0:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
        w = (matrix[2, 1] - matrix[1, 2]) / scale
    elif idx == 1:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
        w = (matrix[0, 2] - matrix[2, 0]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
        w = (matrix[1, 0] - matrix[0, 1]) / scale
    return x, y, z, w


def _pose_to_matrix(translation, rotation) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quaternion_to_matrix(rotation.x, rotation.y, rotation.z, rotation.w)
    matrix[:3, 3] = [translation.x, translation.y, translation.z]
    return matrix


class OdomStaticTransformListener:
    """
    Compose camera poses from static body->camera TF and dynamic odometry.

    The legacy mapping stack depended on an external helper for this composition.
    Keeping the logic local avoids a cross-repository runtime dependency while
    preserving the original `spot/odom` lookup behavior.
    """

    def __init__(
        self,
        node: Node,
        *,
        odom_topic: str = "/spot/platform/odom",
        odom_buffer_size: int = 400,
        tf_buffer_cache_time_sec: float = 2.0,
        tf_lookup_timeout_sec: float = 0.2,
    ) -> None:
        self._node = node
        self._lock = threading.Lock()
        self._odom_buffer_size = max(1, int(odom_buffer_size))
        self._tf_lookup_timeout = Duration(seconds=max(0.0, float(tf_lookup_timeout_sec)))
        self._odom_msg_buffer: list[Odometry] = []
        self._odom_time_buffer: list[float] = []
        self._last_odom_ts: Optional[float] = None
        self._base_frame: Optional[str] = None
        self._base_frame_candidates: list[str] = []
        self._tolerance_seconds = 1.0 / 200.0

        cache_time = Duration(seconds=max(0.0, float(tf_buffer_cache_time_sec)))
        try:
            self._tf_buffer = Buffer(cache_time=cache_time, node=node)
        except TypeError:
            self._tf_buffer = Buffer(cache_time=cache_time)
        self._tf_listener = TransformListener(self._tf_buffer, node)

        odom_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._odom_subscription = node.create_subscription(Odometry, odom_topic, self._odom_callback, odom_qos)

    def _odom_callback(self, msg: Odometry) -> None:
        stamp_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if self._last_odom_ts is not None and stamp_s < self._last_odom_ts:
            with self._lock:
                self._odom_msg_buffer.clear()
                self._odom_time_buffer.clear()

        self._last_odom_ts = stamp_s
        if self._base_frame is None:
            self._base_frame = msg.child_frame_id
            self._base_frame_candidates = self._build_base_frame_candidates(self._base_frame)

        with self._lock:
            self._odom_msg_buffer.append(msg)
            self._odom_time_buffer.append(stamp_s)
            if len(self._odom_msg_buffer) > self._odom_buffer_size:
                self._odom_msg_buffer.pop(0)
                self._odom_time_buffer.pop(0)

    @staticmethod
    def _build_base_frame_candidates(base_frame: str) -> list[str]:
        candidates = [base_frame]
        if base_frame == "spot/body":
            candidates.append("spot_body")
        elif base_frame == "spot/base_link":
            candidates.extend(["spot_body", "spot/body"])
        elif base_frame == "spot_body":
            candidates.append("spot/body")
        return list(dict.fromkeys(candidate for candidate in candidates if candidate))

    def _get_closest_idx_in_buffer(self, query_time_s: float) -> Optional[int]:
        if not self._odom_time_buffer:
            return None

        idx = bisect.bisect_right(self._odom_time_buffer, query_time_s)
        if idx == 0:
            delta = abs(self._odom_time_buffer[0] - query_time_s)
            return 0 if delta <= self._tolerance_seconds else None
        if idx == len(self._odom_time_buffer):
            delta = abs(self._odom_time_buffer[-1] - query_time_s)
            return len(self._odom_time_buffer) - 1 if delta <= self._tolerance_seconds else None

        before = self._odom_time_buffer[idx - 1]
        after = self._odom_time_buffer[idx]
        if abs(before - query_time_s) <= abs(after - query_time_s):
            return idx - 1 if abs(before - query_time_s) <= self._tolerance_seconds else None
        return idx if abs(after - query_time_s) <= self._tolerance_seconds else None

    def _lookup_target_body_transform(self, target_frame: str, stamp) -> Optional[TransformStamped]:
        candidates = self._base_frame_candidates or ([self._base_frame] if self._base_frame else [])
        for candidate in candidates:
            try:
                return self._tf_buffer.lookup_transform(
                    target_frame,
                    candidate,
                    stamp,
                    timeout=self._tf_lookup_timeout,
                )
            except Exception:
                continue
        return None

    @staticmethod
    def _matrix_to_transform_stamped(
        matrix: np.ndarray,
        *,
        header_frame: str,
        child_frame: str,
        stamp,
    ) -> TransformStamped:
        result = TransformStamped()
        result.header.stamp = stamp
        result.header.frame_id = header_frame
        result.child_frame_id = child_frame
        result.transform.translation.x = float(matrix[0, 3])
        result.transform.translation.y = float(matrix[1, 3])
        result.transform.translation.z = float(matrix[2, 3])
        qx, qy, qz, qw = _matrix_to_quaternion(matrix[:3, :3])
        result.transform.rotation.x = float(qx)
        result.transform.rotation.y = float(qy)
        result.transform.rotation.z = float(qz)
        result.transform.rotation.w = float(qw)
        return result

    @staticmethod
    def _invert_transform(transform: TransformStamped) -> TransformStamped:
        matrix = _pose_to_matrix(transform.transform.translation, transform.transform.rotation)
        inverse = np.linalg.inv(matrix)
        return OdomStaticTransformListener._matrix_to_transform_stamped(
            inverse,
            header_frame=transform.child_frame_id,
            child_frame=transform.header.frame_id,
            stamp=transform.header.stamp,
        )

    @staticmethod
    def _identity_transform(target_frame: str, source_frame: str, stamp) -> TransformStamped:
        identity = TransformStamped()
        identity.header.frame_id = target_frame
        identity.child_frame_id = source_frame
        if isinstance(stamp, Time):
            identity.header.stamp = stamp.to_msg()
        else:
            identity.header.stamp = stamp
        identity.transform.rotation.w = 1.0
        return identity

    def lookup_transform(self, target_frame: str, source_frame: str, stamp) -> Optional[TransformStamped]:
        if target_frame == source_frame:
            return self._identity_transform(target_frame, source_frame, stamp)

        if "odom" not in source_frame:
            if "odom" not in target_frame:
                raise ValueError("At least one frame must be an odom frame.")
            inverse = self.lookup_transform(source_frame, target_frame, stamp)
            return self._invert_transform(inverse) if inverse is not None else None

        if isinstance(stamp, Time):
            query_time = stamp
        elif hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
            query_time = Time.from_msg(stamp)
        else:
            raise TypeError(f"Unsupported stamp type: {type(stamp)!r}")

        query_time_s = float(query_time.nanoseconds) * 1e-9
        with self._lock:
            closest_idx = self._get_closest_idx_in_buffer(query_time_s)
            if closest_idx is None:
                return None
            odom_msg = self._odom_msg_buffer[closest_idx]

        tf_target_body = self._lookup_target_body_transform(target_frame, odom_msg.header.stamp)
        if tf_target_body is None:
            return None

        target_from_body = _pose_to_matrix(tf_target_body.transform.translation, tf_target_body.transform.rotation)
        odom_from_body = _pose_to_matrix(odom_msg.pose.pose.position, odom_msg.pose.pose.orientation)
        target_from_odom = target_from_body @ np.linalg.inv(odom_from_body)
        return self._matrix_to_transform_stamped(
            target_from_odom,
            header_frame=tf_target_body.header.frame_id,
            child_frame=odom_msg.header.frame_id,
            stamp=tf_target_body.header.stamp,
        )
