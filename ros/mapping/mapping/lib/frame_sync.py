"""Frame synchronization and batch assembly helpers for StreamingMapper."""

from __future__ import annotations

from typing import Any, Callable, Deque, Dict, List, Optional, Union

import numpy as np

from mapping.lib.batch_processing import log_queue_depths, oldest_frame_age
from mapping.lib.geometric_transforms import transform_to_matrix
from mapping.lib.image_decoding import (
    decode_depth,
    decode_rgb,
    has_compressed_rgb_payload,
    has_raw_rgb_payload,
)


def assemble_frame_payload(
    camera: str,
    bundle: Any,
    *,
    time_from_msg: Callable,
    now_sec: Callable[[], float],
    logger_warn: Callable[..., None],
    logger_debug: Callable[..., None],
    logger_error: Callable[..., None],
) -> Optional[Dict[str, Any]]:
    """Decode an RGBDFrame *bundle* into the dict consumed by the mapping pipeline."""
    rgb_msg = bundle.rgb
    rgb_compressed_msg = bundle.rgb_compressed
    depth_msg = bundle.depth
    meta_msg = bundle.meta

    if has_compressed_rgb_payload(rgb_compressed_msg):
        rgb_msg = rgb_compressed_msg
    elif not has_raw_rgb_payload(bundle.rgb):
        rgb_msg = None

    if rgb_msg is None:
        logger_warn(f"RGB payload missing for camera {camera}")
        return None

    rgb = decode_rgb(rgb_msg, logger=logger_error)
    if rgb is None:
        logger_warn(f"Failed to decode color image for camera {camera}")
        return None

    depth_f32 = decode_depth(depth_msg)
    if depth_f32 is None:
        logger_warn(f"Failed to decode depth image for camera {camera}")
        return None

    meta_stamp = time_from_msg(meta_msg.header.stamp)

    rgb_fx = float(meta_msg.rgb_fx)
    rgb_fy = float(meta_msg.rgb_fy)
    rgb_cx = float(meta_msg.rgb_cx)
    rgb_cy = float(meta_msg.rgb_cy)
    rgb_width = int(meta_msg.rgb_width)
    rgb_height = int(meta_msg.rgb_height)
    if rgb_fx <= 0.0 or rgb_fy <= 0.0:
        logger_warn(f"Invalid intrinsics for camera {camera} (rgb_fx={rgb_fx:.2f}, rgb_fy={rgb_fy:.2f})")
        return None
    rgb_instrinsics = {
        "fx": rgb_fx,
        "fy": rgb_fy,
        "cx": rgb_cx,
        "cy": rgb_cy,
        "width": rgb_width,
        "height": rgb_height,
    }

    depth_fx = float(meta_msg.depth_fx)
    depth_fy = float(meta_msg.depth_fy)
    depth_cx = float(meta_msg.depth_cx)
    depth_cy = float(meta_msg.depth_cy)
    depth_width = int(meta_msg.depth_width)
    depth_height = int(meta_msg.depth_height)
    if depth_fx <= 0.0 or depth_fy <= 0.0:
        logger_warn(
            f"Invalid intrinsics for camera {camera} (depth_fx={depth_fx:.2f}, depth_fy={depth_fy:.2f})"
        )
        return None

    depth_instrinsics = {
        "fx": depth_fx,
        "fy": depth_fy,
        "cx": depth_cx,
        "cy": depth_cy,
        "width": depth_width,
        "height": depth_height,
    }

    T_world_cam = transform_to_matrix(meta_msg.transform_world_cam)

    width, height = int(meta_msg.rgb_width), int(meta_msg.rgb_height)
    if rgb.shape[1] != width or rgb.shape[0] != height:
        logger_debug(
            f"Metadata resolution mismatch for camera {camera}: color={rgb.shape} meta=({width},{height})"
        )

    stamp_ns = np.int64(meta_stamp.nanoseconds)
    frame_id = meta_msg.frame_id or rgb_msg.header.frame_id or camera

    return {
        "camera": meta_msg.camera or camera,
        "rgb": rgb.astype(np.uint8, copy=False),
        "depth_f32": depth_f32,
        "T_world_cam": T_world_cam.astype(np.float32, copy=False),
        "rgb_instrinsics": rgb_instrinsics,
        "depth_instrinsics": depth_instrinsics,
        "stamp_ns": stamp_ns,
        "frame_id": frame_id,
        "received_time": now_sec(),
        "rgb_msg": rgb_msg,
        "depth_msg": depth_msg,
    }


def try_match_pending(
    camera: str,
    rgb_stamp_ns: int,
    *,
    pending_meta: Dict[str, Dict],
    pending_rgb: Dict[str, Dict],
    pending_depth: Dict[str, Dict],
    camera_callback: Callable,
) -> None:
    """Try to match pending meta/rgb/depth entries and fire *camera_callback* on match."""
    meta_dict = pending_meta.get(camera)
    rgb_dict = pending_rgb.get(camera)
    depth_dict = pending_depth.get(camera)
    if meta_dict is None or rgb_dict is None or depth_dict is None:
        return

    meta_entry = meta_dict.get(rgb_stamp_ns)
    rgb_entry = rgb_dict.get(rgb_stamp_ns)
    if meta_entry is None or rgb_entry is None:
        return
    meta_msg, depth_stamp_ns, _ = meta_entry
    depth_entry = depth_dict.get(depth_stamp_ns)
    if depth_entry is None:
        return

    rgb_msg, _ = rgb_entry
    depth_msg, _ = depth_entry

    meta_dict.pop(rgb_stamp_ns, None)
    rgb_dict.pop(rgb_stamp_ns, None)
    depth_dict.pop(depth_stamp_ns, None)

    camera_callback(camera, rgb_msg, depth_msg, meta_msg)


def get_oldest_frame_age(
    frame_queues: Dict[str, Deque],
    now_sec: float,
) -> Optional[float]:
    """Return the age (seconds) of the oldest queued frame, or None."""
    return oldest_frame_age(frame_queues, now_sec)


def do_log_queue_depths(
    frame_queues: Dict[str, Deque],
    reason: str,
    *,
    debug_queue_status: bool,
    logger_debug: Callable[..., None],
) -> None:
    """Log per-camera queue depths (delegates to lib.batch_processing)."""
    log_queue_depths(
        frame_queues,
        reason,
        debug_queue_status=debug_queue_status,
        logger_debug=logger_debug,
    )


def drain_ready_batch(
    *,
    frame_queues: Dict[str, Deque],
    camera_order: List[str],
    expected_batch: int,
    batch_timeout: float,
    partial_min_cameras: int,
    debug_queue_status: bool,
    now_sec: Callable[[], float],
    logger_debug: Callable[..., None],
    process_frame_batch: Callable[[List[Dict[str, Any]]], None],
    allow_partial: bool,
    queue_debug_state: Dict[str, float],
    queue_debug_interval: float,
) -> bool:
    """Pop one ready batch from *frame_queues* and process it. Returns True if a batch fired."""
    ready_cameras = [camera for camera, queue in frame_queues.items() if queue]
    if not ready_cameras:
        return False

    have_full_batch = len(ready_cameras) == expected_batch
    age = oldest_frame_age(frame_queues, now_sec())
    should_process = have_full_batch

    if not should_process:
        timeout_triggered = age is not None and age >= batch_timeout
        should_process = (allow_partial or timeout_triggered) and len(ready_cameras) >= partial_min_cameras

        if not should_process:
            if debug_queue_status:
                now = now_sec()
                if now - queue_debug_state.get("last", 0.0) >= queue_debug_interval:
                    do_log_queue_depths(frame_queues, "Waiting for additional cameras;",
                                        debug_queue_status=debug_queue_status, logger_debug=logger_debug)
                    queue_debug_state["last"] = now
            return False

    batch_frames: List[Dict[str, Any]] = []
    for camera in camera_order:
        queue = frame_queues.get(camera)
        if queue:
            batch_frames.append(queue.popleft())

    if not batch_frames:
        return False

    if len(batch_frames) != expected_batch:
        if debug_queue_status:
            now = now_sec()
            if now - queue_debug_state.get("last_partial", 0.0) >= queue_debug_interval:
                queue_debug_state["last_partial"] = now
                do_log_queue_depths(
                    frame_queues,
                    f"Processing partial batch ({len(batch_frames)}/{expected_batch} cameras);",
                    debug_queue_status=debug_queue_status,
                    logger_debug=logger_debug,
                )
    else:
        if debug_queue_status:
            now = now_sec()
            if now - queue_debug_state.get("last_full", 0.0) >= queue_debug_interval:
                queue_debug_state["last_full"] = now
                do_log_queue_depths(
                    frame_queues,
                    f"Processing full batch ({len(batch_frames)}/{expected_batch} cameras);",
                    debug_queue_status=debug_queue_status,
                    logger_debug=logger_debug,
                )

    process_frame_batch(batch_frames)
    return True
