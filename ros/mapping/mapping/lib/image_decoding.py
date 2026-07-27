"""ROS image message decoding utilities for the streaming mapper node.

All functions are pure (no ROS node state). ``decode_rgb`` accepts an optional
``logger`` callable so the caller can forward errors to the node's logger without
this module importing ``rclpy``.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image


def has_raw_rgb_payload(msg: Optional[Image]) -> bool:
    """Return True if *msg* is a non-empty ``sensor_msgs/Image`` with a valid header."""
    if msg is None:
        return False
    if int(getattr(msg, "height", 0) or 0) <= 0 or int(getattr(msg, "width", 0) or 0) <= 0:
        return False
    if not (getattr(msg, "encoding", "") or "").strip():
        return False
    return len(getattr(msg, "data", b"") or b"") > 0


def has_compressed_rgb_payload(msg: Optional[CompressedImage]) -> bool:
    """Return True if *msg* is a non-empty ``sensor_msgs/CompressedImage``."""
    if msg is None:
        return False
    return len(getattr(msg, "data", b"") or b"") > 0


def decode_compressed(msg: CompressedImage) -> np.ndarray:
    """Decode a ``sensor_msgs/CompressedImage`` to an RGB uint8 HWC array.

    Raises ``ValueError`` if ``cv2.imdecode`` fails.
    """
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError("cv2.imdecode failed")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_image(msg: Image) -> np.ndarray:
    """Decode a ``sensor_msgs/Image`` (rgb8 / bgr8 / mono8) to an RGB uint8 HWC array.

    Raises ``ValueError`` for unsupported encodings.
    """
    if msg.encoding not in ("rgb8", "bgr8", "mono8"):
        raise ValueError(f"Unsupported encoding: {msg.encoding}")

    dtype = np.uint8
    channels = 1 if msg.encoding == "mono8" else 3

    img = np.frombuffer(msg.data, dtype=dtype)
    img = img.reshape((msg.height, msg.step // channels, channels))
    img = img[:, : msg.width, :]  # handle padding

    if msg.encoding == "bgr8":
        # ascontiguousarray: the ``::-1`` view has a negative channel stride,
        # which torch.from_numpy() downstream rejects (real Spot cameras publish
        # bgr8; the rgb8 path below is already contiguous).
        return np.ascontiguousarray(img[:, :, ::-1])  # BGR → RGB
    if msg.encoding == "mono8":
        return np.ascontiguousarray(np.repeat(img, 3, axis=2))

    return np.ascontiguousarray(img)


def decode_rgb(
    msg: Union[CompressedImage, Image],
    logger: Optional[Callable[[str], None]] = None,
) -> Optional[np.ndarray]:
    """Decode a CompressedImage or Image to an RGB uint8 HWC array.

    *logger* is called with an error string on failure; pass
    ``node.get_logger().error`` to route errors through the ROS logger.
    """
    try:
        if isinstance(msg, CompressedImage):
            return decode_compressed(msg)
        if isinstance(msg, Image):
            return decode_image(msg)
    except Exception as e:
        if logger is not None:
            logger(f"Could not decode image: {e}")
    return None


def decode_depth(msg: Image) -> Optional[np.ndarray]:
    """Decode a ``sensor_msgs/Image`` depth frame to a float32 HW array in metres.

    Supports 16UC1 / MONO16 (millimetres → metres) and 32FC1 encodings.
    Returns ``None`` for invalid or unsupported messages.
    """
    if int(msg.height) <= 0 or int(msg.width) <= 0:
        return None

    encoding = (msg.encoding or "").strip()
    encoding_upper = encoding.upper()

    if encoding_upper in {"16UC1", "MONO16"}:
        dtype = np.dtype(np.uint16)
        bytes_per_pixel = 2
        needs_u16_to_f32 = True
    elif encoding_upper == "32FC1":
        dtype = np.dtype(np.float32)
        bytes_per_pixel = 4
        needs_u16_to_f32 = False
    else:
        return None

    if int(msg.step) <= 0:
        step_bytes = int(msg.width) * bytes_per_pixel
    else:
        step_bytes = int(msg.step)
        if step_bytes % bytes_per_pixel != 0:
            return None

    tight_step_bytes = int(msg.width) * bytes_per_pixel
    # Some publishers incorrectly set `step`; fall back to a tightly packed stride if possible.
    if step_bytes < tight_step_bytes and len(msg.data) >= int(msg.height) * tight_step_bytes:
        step_bytes = tight_step_bytes

    expected_min_bytes = int(msg.height) * step_bytes
    if len(msg.data) < expected_min_bytes:
        return None

    row_elems = step_bytes // bytes_per_pixel
    count = int(msg.height) * row_elems

    if int(getattr(msg, "is_bigendian", 0)):
        dtype = dtype.newbyteorder(">")
    else:
        dtype = dtype.newbyteorder("<")

    depth_np = np.frombuffer(msg.data, dtype=dtype, count=count).reshape(int(msg.height), row_elems)[
        :, : int(msg.width)
    ]

    # If depth is inf, set it to 0
    depth_np = np.where(np.isinf(depth_np), 0.0, depth_np)

    if needs_u16_to_f32:
        depth_np = depth_np.astype(np.float32)
        return depth_np / 1000.0  # convert to metres

    return depth_np
