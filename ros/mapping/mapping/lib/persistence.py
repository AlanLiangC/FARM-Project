"""Scene graph and scene state persistence helpers for StreamingMapper."""

from __future__ import annotations

import contextlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from mapping.lib.scene_graph_io import (
    maybe_save_published_scene_graph_json,
    take_scene_graph_snapshot,
    write_scene_graph_snapshot,
)
from scene_graph.scene_state_io import save_scene_state


def write_snapshot_and_json(
    *,
    scene_state: dict,
    snapshot_dir: str,
    snapshot_version: int,
    snapshot_lock: Any,
    snapshot_save_enabled: bool,
    json_save_enabled: bool,
    json_save_path: str,
    covisibility_enabled: bool,
    covisibility_lock: Any,
    hellinger_thresh: float,
    get_stamp_msg: Callable,
    logger_info: Callable[..., None],
    logger_warn: Callable[..., None],
) -> Tuple[int, Optional[Dict[str, object]], bool]:
    """Take a scene graph snapshot, write it to disk, and save JSON.

    Returns (new_snapshot_version, snapshot_ptr_or_None, saved_any).
    """
    objects, snapshot_ctx = take_scene_graph_snapshot(scene_state, logger_info=logger_info)
    if not objects:
        return snapshot_version, None, False

    payload: Dict[str, object] = {"objects": objects}
    snapshot_ptr: Optional[Dict[str, object]] = None
    new_version = snapshot_version

    if snapshot_save_enabled and snapshot_ctx is not None:
        stamp_msg = get_stamp_msg()
        new_version, snapshot_ptr = write_scene_graph_snapshot(
            snapshot_dir=snapshot_dir,
            graph_version=snapshot_version,
            lock=snapshot_lock,
            save_enabled=snapshot_save_enabled,
            include_idx=snapshot_ctx.get("include_idx", []),
            limit=int(snapshot_ctx.get("limit", 0)),
            active_cpu=snapshot_ctx.get("active_cpu"),
            means_cpu=snapshot_ctx.get("means_cpu"),
            cov6_cpu=snapshot_ctx.get("cov6_cpu"),
            obj_ids_cpu=snapshot_ctx.get("obj_ids_cpu"),
            rgb_observations=snapshot_ctx.get("rgb_observations", []),
            object_image_ids=snapshot_ctx.get("object_image_ids", []),
            viewpoint_image_ids=snapshot_ctx.get("viewpoint_image_ids", []),
            images_meta=snapshot_ctx.get("images_meta", []),
            caption_embeddings=snapshot_ctx.get("caption_embeddings", []),
            siglip2_embeddings=snapshot_ctx.get("siglip2_embeddings", []),
            qwen3_vl_embeddings=snapshot_ctx.get("qwen3_vl_embeddings", []),
            caption_history=snapshot_ctx.get("caption_history", []),
            caption_embedding_history=snapshot_ctx.get("caption_embedding_history", []),
            siglip2_embedding_history=snapshot_ctx.get("siglip2_embedding_history", []),
            qwen3_vl_embedding_history=snapshot_ctx.get("qwen3_vl_embedding_history", []),
            object_captions=[str(x or "") for x in (scene_state.get("object_caption", []) or [])],
            scene_state=scene_state,
            covisibility_enabled=covisibility_enabled,
            covisibility_lock=covisibility_lock,
            hellinger_thresh=hellinger_thresh,
            stamp_sec=int(stamp_msg.sec),
            stamp_nanosec=int(stamp_msg.nanosec),
            logger_info=logger_info,
            logger_warn=logger_warn,
        )
        if snapshot_ptr:
            payload["snapshot"] = snapshot_ptr

    try:
        json_payload = json.dumps(payload)
    except Exception:
        json_payload = "{}"

    saved_json = maybe_save_published_scene_graph_json(
        json_payload,
        save_enabled=json_save_enabled,
        save_path=json_save_path,
        logger_info=logger_info,
        logger_warn=logger_warn,
    )
    saved_snapshot = bool(snapshot_ptr)
    return new_version, snapshot_ptr, bool(saved_json or saved_snapshot)


def persist_all_artifacts(
    *,
    reason: str,
    persist_scene_graph: Callable[[], bool],
    save_scene_state_fn: Callable,
    scene_state_save_path: str,
    scene_state_save_lock: Any,
    logger_warn: Callable[..., None],
    timed_persist_state: Dict[str, Any],
) -> bool:
    """Persist scene graph and scene state artifacts. Returns True if any were saved."""
    saved_scene_graph = False
    try:
        saved_scene_graph = bool(persist_scene_graph())
    except Exception as exc:
        logger_warn(f"Failed to save scene graph artifacts ({reason}): {exc}")

    saved_scene_state = False
    if scene_state_save_path:
        try:
            with scene_state_save_lock:
                save_scene_state_fn(reason=reason, wait_for_idle=False)
            saved_scene_state = True
        except Exception as exc:
            logger_warn(f"Failed to save scene state ({reason}): {exc}")
    elif reason.startswith("max_time_checkpoint") and not timed_persist_state.get("missing_warned", False):
        logger_warn("max_time checkpoint reached but scene_state_save_path is empty; skipping scene_state")
        timed_persist_state["missing_warned"] = True

    return bool(saved_scene_graph or saved_scene_state)


def maybe_persist_by_time(
    *,
    checkpoints: List[Dict[str, Any]],
    sim_time_remaining_sec: Optional[float],
    max_time_sec: float,
    persist_all: Callable[[str], bool],
    logger_info: Callable[..., None],
) -> None:
    """Fire one-shot timed checkpoint saves when enough mission time has elapsed."""
    if not checkpoints:
        return
    if sim_time_remaining_sec is None:
        return
    remaining_sec = float(sim_time_remaining_sec)
    if not math.isfinite(remaining_sec):
        return

    fired_labels: List[str] = []
    for checkpoint in checkpoints:
        if bool(checkpoint.get("fired", False)):
            continue
        threshold = float(checkpoint.get("remaining_sec", 0.0) or 0.0)
        if remaining_sec > threshold:
            continue
        checkpoint["fired"] = True
        labels = checkpoint.get("labels", [])
        if isinstance(labels, list):
            fired_labels.extend(str(label) for label in labels if label)

    if not fired_labels:
        return

    reason = f"max_time_checkpoint:{'|'.join(fired_labels)}"
    saved_any = persist_all(reason)
    save_status = "saved" if saved_any else "nothing_saved"
    elapsed_sec = max(0.0, max_time_sec - remaining_sec)
    logger_info(
        f"Timed persist checkpoint reached ({', '.join(fired_labels)}): "
        f"remaining={remaining_sec:.1f}s elapsed={elapsed_sec:.1f}s/{max_time_sec:.1f}s ({save_status})"
    )


def maybe_persist_by_distance(
    *,
    threshold_m: float,
    scene_state: dict,
    json_save_enabled: bool,
    snapshot_save_enabled: bool,
    scene_state_save_path: str,
    persist_all: Callable[[str], bool],
    persist_position_state: Dict[str, Any],
) -> None:
    """Persist on-disk artifacts when the robot has moved far enough."""
    if threshold_m <= 0.0:
        return
    if not json_save_enabled and not snapshot_save_enabled and not scene_state_save_path:
        return

    cur = scene_state.get("current_robot_position")
    if not isinstance(cur, torch.Tensor) or cur.numel() < 2:
        return

    with contextlib.suppress(Exception):
        cur_cpu = cur.detach().to("cpu", dtype=torch.float32, copy=False).view(-1)[:3]
        if cur_cpu.numel() < 2 or not torch.isfinite(cur_cpu[:2]).all():
            return

        last = persist_position_state.get("last_position")
        if not isinstance(last, torch.Tensor) or last.numel() < 2:
            persist_position_state["last_position"] = cur_cpu.clone()
            persist_all("distance_checkpoint:init_pose")
            return

        last_cpu = last.detach().to("cpu", dtype=torch.float32, copy=False).view(-1)[:3]
        if last_cpu.numel() < 2 or not torch.isfinite(last_cpu[:2]).all():
            persist_position_state["last_position"] = cur_cpu.clone()
            persist_all("distance_checkpoint:reset_ref_pose")
            return

        dx = cur_cpu[0] - last_cpu[0]
        dy = cur_cpu[1] - last_cpu[1]
        dist_m = float(torch.sqrt(dx * dx + dy * dy).item())
        if dist_m < threshold_m:
            return

        persist_position_state["last_position"] = cur_cpu.clone()
        persist_all(f"distance_checkpoint:{dist_m:.2f}m")


def wait_for_save_idle(busy_event: Optional[Any], timeout_sec: float) -> None:
    """Best-effort wait for mapping updates to quiesce before saving."""
    try:
        timeout = float(timeout_sec)
    except Exception:
        timeout = 0.0
    if timeout <= 0.0:
        return
    deadline = time.monotonic() + timeout
    while busy_event is not None and busy_event.is_set() and time.monotonic() < deadline:
        time.sleep(0.05)


def do_save_scene_state(
    *,
    reason: str,
    wait_for_idle: bool,
    scene_state_save_path: str,
    scene_state: dict,
    segmenter_feature_dim: int,
    save_observations: bool,
    observation_view_limit_raw: Any,
    busy_event: Any,
    busy_timeout_sec: float,
) -> Path:
    """Save scene state to disk."""
    if not scene_state_save_path:
        raise RuntimeError("scene_state_save_path is empty; cannot save scene state")

    if wait_for_idle:
        wait_for_save_idle(busy_event, busy_timeout_sec)

    observation_view_limit = None
    if save_observations:
        try:
            limit = int(observation_view_limit_raw)
        except Exception:
            limit = 1
        observation_view_limit = None if limit < 0 else limit

    return save_scene_state(
        scene_state_save_path,
        scene_state,
        feature_dim=segmenter_feature_dim,
        include_observations=save_observations,
        observation_view_limit=observation_view_limit,
        extra_metadata={
            "node": "mapping_streaming_mapper",
            "reason": str(reason),
        },
    )
