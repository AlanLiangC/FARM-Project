"""Frame-queue and batch-processing utilities for the streaming mapper node.

All functions are pure (no ROS node state). Callers pass in the queues,
configuration values, and logger callables rather than a ``self`` reference.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Queue inspection helpers
# ---------------------------------------------------------------------------


def oldest_frame_age(
    frame_queues: Dict[str, object],
    now_sec: float,
) -> Optional[float]:
    """Return the age (seconds) of the oldest frame across all queues.

    Args:
        frame_queues: mapping of camera name → deque of frame dicts.
        now_sec: current wall-clock time in seconds.

    Returns:
        Age of the oldest enqueued frame in seconds, or ``None`` if all queues
        are empty.
    """
    oldest_received: Optional[float] = None
    for queue in frame_queues.values():
        if not queue:
            continue
        # Support both deque and list — first element is the oldest.
        try:
            first = next(iter(queue))
        except StopIteration:
            continue
        received_time = first.get("received_time") if isinstance(first, dict) else None
        if received_time is None:
            continue
        age = now_sec - float(received_time)
        if age < 0.0:
            age = 0.0
        if oldest_received is None or age > oldest_received:
            oldest_received = age
    return oldest_received


def log_queue_depths(
    frame_queues: Dict[str, object],
    reason: str,
    *,
    debug_queue_status: bool,
    logger_debug: Optional[Callable[[str], None]] = None,
) -> None:
    """Log per-camera queue depths when *debug_queue_status* is enabled.

    Args:
        frame_queues: mapping of camera name → deque of frame dicts.
        reason: descriptive prefix for the log message.
        debug_queue_status: guard flag; nothing is logged when ``False``.
        logger_debug: callable that emits a DEBUG-level log line.
    """
    if not debug_queue_status or logger_debug is None:
        return
    queue_depths = {camera: len(queue) for camera, queue in frame_queues.items()}
    logger_debug(f"{reason} queue depths {queue_depths}")
