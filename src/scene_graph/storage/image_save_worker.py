from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from .hdf5 import ArrayLike, save_image_to_hdf5, save_image_to_jpeg
from .models import ImageRecord, ImageSaveRequest

LOGGER = logging.getLogger(__name__)


class ImageSaveWorker:
    """Background worker that serializes RGB frames to disk."""

    def __init__(self, *, max_queue_size: int = 0) -> None:
        queue_size = max(0, int(max_queue_size))
        self._queue: "queue.Queue[ImageSaveRequest]" = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._drain_remaining_on_close = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._closed = False
        self._dropped = 0
        self._thread.start()

    def submit(
        self,
        image: ArrayLike,
        path: Path,
        *,
        fmt: str = "h5",
        jpeg_max_width: int = 640,
        jpeg_quality: int = 75,
        on_success: Optional[Callable[[Path], None]] = None,
        drop_if_full: bool = True,
    ) -> bool:
        if self._closed:
            raise RuntimeError("Cannot submit to closed ImageSaveWorker")
        payload: ArrayLike
        if isinstance(image, torch.Tensor):
            payload = image.detach().cpu()
        else:
            payload = image
        request = ImageSaveRequest(
            image=payload,
            path=path,
            fmt=str(fmt or "h5").strip().lower(),
            jpeg_max_width=int(jpeg_max_width),
            jpeg_quality=int(jpeg_quality),
            on_success=on_success,
        )
        if drop_if_full:
            try:
                self._queue.put_nowait(request)
                return True
            except queue.Full:
                self._dropped += 1
                return False
        self._queue.put(request)
        return True

    def _worker(self) -> None:
        while True:
            try:
                request = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue
            try:
                if request.fmt in {"jpg", "jpeg"}:
                    save_image_to_jpeg(
                        request.image,
                        request.path,
                        max_width=request.jpeg_max_width,
                        quality=request.jpeg_quality,
                    )
                else:
                    save_image_to_hdf5(request.image, request.path)
                if request.on_success is not None:
                    with contextlib.suppress(Exception):
                        request.on_success(request.path)
            except Exception:  # pragma: no cover - log + continue
                LOGGER.exception("Failed to save image to %s", request.path)
            finally:
                self._queue.task_done()

        if self._drain_remaining_on_close:
            while True:
                try:
                    request = self._queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    if request.fmt in {"jpg", "jpeg"}:
                        save_image_to_jpeg(
                            request.image,
                            request.path,
                            max_width=request.jpeg_max_width,
                            quality=request.jpeg_quality,
                        )
                    else:
                        save_image_to_hdf5(request.image, request.path)
                    if request.on_success is not None:
                        with contextlib.suppress(Exception):
                            request.on_success(request.path)
                except Exception:
                    LOGGER.exception("Failed to save image to %s", request.path)
                finally:
                    self._queue.task_done()

    def close(self, *, timeout_sec: float | None = None, drain: bool = True) -> bool:
        """Stop the worker; optionally wait for queued writes to finish.

        Returns True if all queued tasks were drained before shutdown.
        """
        if self._closed:
            return True

        self._drain_remaining_on_close = bool(drain)
        drained = True
        if timeout_sec is None:
            self._queue.join()
        else:
            deadline = time.monotonic() + max(0.0, float(timeout_sec))
            # `queue.Queue.join()` has no timeout, so we wait on the internal condition.
            with self._queue.all_tasks_done:
                while self._queue.unfinished_tasks:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        drained = False
                        break
                    self._queue.all_tasks_done.wait(timeout=min(0.1, remaining))

        self._stop_event.set()
        join_timeout = None if timeout_sec is None else max(0.0, float(timeout_sec))
        self._thread.join(timeout=join_timeout)
        self._closed = True
        if getattr(self._queue, "unfinished_tasks", 0):
            drained = False
        return drained

    def dropped_count(self) -> int:
        return int(self._dropped)


def register_batch_images(
    scene_state: Dict[str, Any],
    poses_world: List[torch.Tensor],
    camera_ids: List[str] | None = None,
    source_refs: List[str] | None = None,
) -> List[int]:
    """Register per-frame metadata and return the image IDs for this batch.

    ``source_refs[i]`` is a back-pointer at the original source data for that
    frame (e.g. ``"<sens-path>#frame=N"`` or ``"<chunk>.npz#frame=N"``). It's
    independent of ``image_saving_enabled``: even when no JPEG copy is written,
    ``source_ref`` lets ``load_scene_state_image`` recover the original RGB
    from the still-on-disk source. Live ROS path leaves it empty (no notion
    of a frame-id-in-source for streaming subscriptions).
    """

    if not poses_world:
        return []

    images: List[ImageRecord] = scene_state.setdefault("images", [])
    image_positions: List[torch.Tensor | None] = scene_state.setdefault("image_positions", [])

    batch_image_ids: List[int] = []
    if camera_ids is not None and len(camera_ids) != len(poses_world):
        camera_ids = None
    if source_refs is not None and len(source_refs) != len(poses_world):
        source_refs = None

    for idx, pose in enumerate(poses_world):
        pose_tensor = None
        if isinstance(pose, torch.Tensor):
            pose_tensor = pose.detach().clone().cpu()
        image_id = len(images)
        record = ImageRecord(
            image_id=image_id,
            pose=pose_tensor,
            camera_id=str(camera_ids[idx]) if camera_ids is not None else "",
            source_ref=str(source_refs[idx]) if source_refs is not None else "",
        )
        images.append(record)
        if pose_tensor is None:
            image_positions.append(None)
        else:
            image_positions.append(pose_tensor[:3, 3].clone())
        batch_image_ids.append(image_id)

    return batch_image_ids


def mark_image_saved(scene_state: Dict[str, Any], image_id: int, storage_path: Path) -> None:
    """Update the metadata to reflect that ``image_id`` now lives at ``storage_path``."""

    images: List[ImageRecord] = scene_state.get("images", [])
    if image_id < 0 or image_id >= len(images):
        return
    images[image_id].storage_path = str(storage_path)
