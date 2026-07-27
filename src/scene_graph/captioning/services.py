import contextlib
import json
import math
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional

try:
    import torch
except ImportError:  # pragma: no cover - caption manager lives alongside torch-heavy pipeline
    torch = None

from scene_graph.map_update.cannot_link import are_cannot_linked_object_ids
from scene_graph.map_update.object_update import update_scene_graph_state

from .models import ObjectCaptionResult, ObjectCaptionTask
from .worker import CaptionWorker


class CaptionManager:
    """
    Mapping-side wrapper that feeds caption tasks to a worker and applies results
    back onto the scene_state in-place.
    """

    def __init__(
        self,
        scene_state: dict,
        *,
        # Bumped from 1000 -> 10000 (2026-05-15) for HM3D-class dense scenes
        # (e.g. 00475 had 1845 objects with bursty enqueues at room transitions
        # that saturated the old bound and combined with a silent worker death
        # to wedge the producer; see Part A fix and worker.py:532).
        queue_maxsize: int = 10000,
        # Bumped from 3 -> 8 (2026-05-15): shrinks the producer-consumer
        # rate gap on dense scenes so the queue is less likely to back up.
        # CaptionWorker's own default is already 10; this only matched
        # the streaming_mapper default downward.
        caption_batch_size: int = 8,
        caption_device: str = "cuda:0",
        caption_server: str = "ollama",
        caption_max_retries: int = 2,
        worker: Optional[CaptionWorker] = None,
        worker_factory: Optional[Callable[[queue.Queue, queue.Queue], CaptionWorker]] = None,
        enabled: bool = True,
        start_immediately: bool = False,
        debug: bool = False,
        merge_log_path: Optional[str] = None,
        merge_enabled: bool = True,
        caption_spatial_context: bool = False,
        caption_spatial_context_include_position: bool = False,
        recaption_time_threshold_sec: float = 0.0,
        caption_merge_hellinger_thresh: float = 0.65,
        caption_merge_caption_thresh: float = 0.92,
        caption_merge_siglip2_thresh: float = 0.93,
        caption_merge_require_visual: bool = True,
        caption_merge_require_category_compat: bool = True,
        caption_visual_prompt_mode: str = "bbox_crop",
        caption_prompt_variant: str = "default",
        caption_version: int = 20,
        deactivate_unclear_objects: bool = True,
    ) -> None:
        self.scene_state = scene_state
        self.tasks_queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self.results_queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self.tasks_enqueued = 0
        self.results_applied = 0
        self.enabled = enabled
        self.debug = debug
        self._merge_enabled = bool(merge_enabled)
        self._merge_log_path = str(merge_log_path) if merge_log_path else None
        self._merge_log_lock = threading.Lock()

        self._worker_factory = worker_factory
        self.worker: Optional[CaptionWorker] = worker
        self._caption_batch_size = caption_batch_size
        self._caption_device = caption_device
        self._caption_server = caption_server
        self._caption_max_retries = max(0, int(caption_max_retries))
        self._caption_spatial_context = bool(caption_spatial_context)
        self._caption_spatial_context_include_position = bool(caption_spatial_context_include_position)
        self._recaption_time_threshold_sec = float(recaption_time_threshold_sec)
        self._caption_merge_hellinger_thresh = float(caption_merge_hellinger_thresh)
        self._caption_merge_caption_thresh = float(caption_merge_caption_thresh)
        self._caption_merge_siglip2_thresh = float(caption_merge_siglip2_thresh)
        self._caption_merge_require_visual = bool(caption_merge_require_visual)
        self._caption_merge_require_category_compat = bool(caption_merge_require_category_compat)
        self._caption_visual_prompt_mode = str(caption_visual_prompt_mode or "bbox_crop").strip().lower()
        self._caption_prompt_variant = str(caption_prompt_variant or "default").strip().lower()
        self._caption_version = int(caption_version)
        self._deactivate_unclear_objects = bool(deactivate_unclear_objects)
        self._pending_no_image: set[int] = set()
        self._inflight_object_ids: set[int] = set()

        if start_immediately:
            self.maybe_start_worker()

    # Public API -----------------------------------------------------------
    def enqueue_objects(self, object_indices: Iterable[int]) -> None:
        if not self.enabled:
            return
        object_ids = self.scene_state.get("object_id")
        active_flags = self.scene_state.get("active")
        rgb_observations = self.scene_state.get("rgb_observations", [])
        captions = self.scene_state.get("object_caption", []) or []
        if object_ids is None:
            return
        # Re-try previously skipped objects that had no images.
        indices = set(int(i) for i in object_indices)
        indices.update(int(i) for i in self._pending_no_image)

        for idx in sorted(indices):
            if idx < 0 or idx >= len(object_ids):
                self._pending_no_image.discard(idx)
                continue

            # Resolve to canonical object id/index so we don't enqueue work for objects that were already merged.
            raw_id = object_ids[idx]
            try:
                raw_id_int = int(raw_id.item()) if hasattr(raw_id, "item") else int(raw_id)
            except Exception:
                self._pending_no_image.discard(idx)
                continue
            canonical_id = self._resolve_canonical_object_id(raw_id_int)
            canonical_idx = self._find_object_index(canonical_id)
            if canonical_idx is None:
                self._pending_no_image.discard(idx)
                continue

            is_active = True
            if active_flags is not None:
                try:
                    is_active = (
                        bool(active_flags[canonical_idx].item())
                        if hasattr(active_flags[canonical_idx], "item")
                        else bool(active_flags[canonical_idx])
                    )
                except Exception:
                    is_active = False
            if not is_active:
                self._pending_no_image.discard(idx)
                self._pending_no_image.discard(canonical_idx)
                continue

            # Skip if the canonical object already has a caption.
            existing_caption = ""
            if canonical_idx < len(captions):
                try:
                    existing_caption = str(captions[canonical_idx] or "").strip()
                except Exception:
                    existing_caption = ""
            if existing_caption:
                self._pending_no_image.discard(idx)
                self._pending_no_image.discard(canonical_idx)
                continue

            def _obs_is_usable(obs: object) -> bool:
                if obs is None:
                    return False
                if isinstance(obs, dict):
                    return obs.get("image") is not None or obs.get("image_caption") is not None
                return True

            obs_list = rgb_observations[canonical_idx] if canonical_idx < len(rgb_observations) else []
            if not obs_list or not any(_obs_is_usable(o) for o in obs_list):
                self._pending_no_image.add(canonical_idx)
                continue

            self._pending_no_image.discard(idx)
            self._pending_no_image.discard(canonical_idx)
            if canonical_id in self._inflight_object_ids:
                continue

            task = ObjectCaptionTask(object_index=int(canonical_idx), object_id=int(canonical_id))
            # Non-blocking put with a timeout: recon must NEVER block on captions.
            # Captions are advisory; dropping a request just means that object's
            # caption is missing this round. A blocking put here was the proximate
            # cause of the 2026-05-15 00475 caption-drain wedge.
            try:
                self.tasks_queue.put(task, timeout=30.0)
            except queue.Full:
                print(
                    "[CaptionManager] tasks_queue full; dropping caption request for "
                    f"object_id={int(canonical_id)} (qsize={self.tasks_queue.qsize()})"
                )
                continue
            self.tasks_enqueued += 1
            self._inflight_object_ids.add(int(canonical_id))

    def drain_results(self) -> int:
        """Apply all available caption results. Returns number applied."""
        if not self.enabled:
            return 0
        applied = 0
        while True:
            try:
                result = self.results_queue.get_nowait()
            except queue.Empty:
                break
            try:
                self._apply_caption_result(result)
                applied += 1
                self.results_applied += 1
            finally:
                with contextlib.suppress(Exception):
                    self._inflight_object_ids.discard(int(getattr(result, "object_id", -1)))
                with contextlib.suppress(ValueError):
                    self.results_queue.task_done()
        return applied

    def wait_until_idle(self, timeout: Optional[float] = None, poll_interval: float = 0.05) -> bool:
        """
        Block until all enqueued tasks have been processed or timeout expires.
        Returns True if idle, False on timeout.
        """
        if not self.enabled:
            return True
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            self.drain_results()
            pending = self.tasks_enqueued - self.results_applied
            # Consider the worker idle if there is nothing pending and queue is empty
            if self.worker is None:
                return pending <= 0
            if pending <= 0 and self.tasks_queue.empty():
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(max(poll_interval, 0.01))

    def shutdown_worker(self, timeout: Optional[float] = 2.0) -> None:
        if not self.enabled:
            return
        if self.worker:
            with contextlib.suppress(Exception):
                if hasattr(self.worker, "log_timing_summary"):
                    self.worker.log_timing_summary()
            self.worker.stop()
            self.worker.join(timeout=timeout)

    def warm_up_model(self) -> bool:
        """Eagerly load caption model/resources before the main loop."""
        if not self.enabled:
            return False
        self.maybe_start_worker()
        if self.worker and hasattr(self.worker, "warm_up_model"):
            try:
                return bool(self.worker.warm_up_model())
            except Exception:
                return False
        return False

    # Internal helpers -----------------------------------------------------
    def _resolve_canonical_object_id(self, object_id: int) -> int:
        """Resolve id_redirect transitively (with cycle guard) and path-compress."""
        try:
            current = int(object_id)
        except Exception:
            return object_id

        id_redirect = self.scene_state.get("id_redirect") or {}
        if not isinstance(id_redirect, dict) or not id_redirect:
            return current

        chain: List[int] = []
        seen: set[int] = set()
        while True:
            if current in seen:
                break
            seen.add(current)
            nxt = id_redirect.get(current)
            if nxt is None:
                break
            try:
                nxt_int = int(nxt)
            except Exception:
                break
            if nxt_int == current:
                break
            chain.append(current)
            current = nxt_int

        # Path compression (best-effort)
        for oid in chain:
            id_redirect[oid] = current
        return current

    def _log_merge_event(
        self,
        *,
        task_object_id: int,
        winner_object_id: int,
        requested_merge_ids: Optional[List[int]],
        merged_objects: List[dict],
        final_caption: str,
    ) -> None:
        if not self._merge_log_path:
            return
        try:
            path = Path(self._merge_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            event = {
                "ts_unix_s": time.time(),
                "task_object_id": int(task_object_id),
                "winner_object_id": int(winner_object_id),
                "requested_merge_ids": [int(x) for x in (requested_merge_ids or [])],
                "merged_objects": merged_objects,
                "final_caption": str(final_caption),
            }
            line = json.dumps(event, ensure_ascii=False)
            with self._merge_log_lock:
                path.open("a", encoding="utf-8").write(line + "\n")
        except Exception:
            # Logging must never break the main mapping loop.
            return

    def _apply_caption_result(self, result: ObjectCaptionResult) -> None:
        self._record_caption_result_stats(result)
        winner_id = self._resolve_canonical_object_id(int(result.object_id))
        if getattr(result, "is_clear_object", None) is False:
            obj_idx = self._find_object_index(winner_id)
            if obj_idx is None:
                return
            if not self._deactivate_unclear_objects:
                self._ensure_caption_len(obj_idx + 1)
                decisions: List[str] = self.scene_state.get("object_caption_decision", [])
                if obj_idx < len(decisions):
                    decisions[obj_idx] = "drop"
                    self.scene_state["object_caption_decision"] = decisions
                if self.debug:
                    reason = getattr(result, "empty_reason", None) or "unclear_object"
                    print(f"[CaptionManager] Kept unclear object {winner_id} active reason={reason}")
                return
            is_locked = self.scene_state.get("is_locked") or []
            if obj_idx < len(is_locked) and bool(is_locked[obj_idx]):
                return
            self._deactivate_unclear_object(obj_idx, reason=getattr(result, "empty_reason", None))
            if self.debug:
                reason = getattr(result, "empty_reason", None) or "unclear_object"
                print(f"[CaptionManager] Deactivated unclear object {winner_id} reason={reason}")
            return

        requested_merge_ids = getattr(result, "merge_object_ids", None)
        if self._merge_enabled:
            winner_id, merged_objects = self._merge_objects_if_needed(
                winner_id, requested_merge_ids, task_object_id=int(result.object_id)
            )
        else:
            merged_objects = []
        obj_idx = self._find_object_index(winner_id)
        if obj_idx is None:
            return

        is_locked = self.scene_state.get("is_locked") or []
        if obj_idx < len(is_locked) and bool(is_locked[obj_idx]):
            return

        self._ensure_caption_len(obj_idx + 1)
        captions: List[str] = self.scene_state.get("object_caption", [])
        existing_caption = str(captions[obj_idx] or "") if obj_idx < len(captions) else ""
        new_caption = str(result.caption or "")
        is_recaption = bool(getattr(result, "is_recaption", False))

        def _dedupe_or_join(new_text: str, old_text: str) -> str:
            new_clean = str(new_text or "").strip()
            old_clean = str(old_text or "").strip()
            if not new_clean:
                return old_clean
            if not old_clean:
                return new_clean
            old_parts = [p.strip() for p in old_clean.split(" OR ") if p.strip()]
            old_parts_l = {p.lower() for p in old_parts}
            if new_clean.lower() in old_parts_l:
                return old_clean
            return f"{new_clean} OR {old_clean}"

        caption_to_store = _dedupe_or_join(new_caption, existing_caption) if is_recaption else new_caption
        caption_applied = True
        if (is_recaption and not new_caption.strip() and existing_caption.strip()) or (
            not caption_to_store.strip() and str(existing_caption or "").strip()
        ):
            caption_applied = False
        else:
            self.scene_state["object_caption"][obj_idx] = caption_to_store
        final_caption = (
            self.scene_state["object_caption"][obj_idx] if obj_idx < len(self.scene_state["object_caption"]) else ""
        )
        if caption_applied:
            embedding = getattr(result, "caption_embedding", None)
            if (
                final_caption
                and str(final_caption).strip()
                and str(final_caption).strip() != str(new_caption).strip()
                and self.worker is not None
                and hasattr(self.worker, "request_caption_text_embeddings")
            ):
                with contextlib.suppress(Exception):
                    ok, _err, vecs = self.worker.request_caption_text_embeddings(
                        [str(final_caption)], timeout_s=30.0, normalize=True
                    )
                    if ok and vecs and isinstance(vecs[0], list):
                        embedding = list(vecs[0])

            categories: List[str] = self.scene_state.get("object_category", [])
            category = getattr(result, "category", None)
            categories[obj_idx] = str(category or "").strip()
            self.scene_state["object_category"] = categories

            supercategories: List[str] = self.scene_state.get("object_supercategory", [])
            supercategory = getattr(result, "supercategory", None)
            supercategories[obj_idx] = str(supercategory or "").strip()
            self.scene_state["object_supercategory"] = supercategories

            category_candidates: List[List[str]] = self.scene_state.get("object_category_candidates", [])
            candidates = getattr(result, "category_candidates", None)
            if isinstance(candidates, list):
                category_candidates[obj_idx] = [str(x or "").strip() for x in candidates if str(x or "").strip()]
            else:
                category_candidates[obj_idx] = []
            self.scene_state["object_category_candidates"] = category_candidates

            key_attributes: List[List[str]] = self.scene_state.get("object_key_attributes", [])
            attrs = getattr(result, "key_attributes", None)
            if isinstance(attrs, list):
                key_attributes[obj_idx] = [str(x or "").strip() for x in attrs if str(x or "").strip()]
            else:
                key_attributes[obj_idx] = []
            self.scene_state["object_key_attributes"] = key_attributes

            decisions: List[str] = self.scene_state.get("object_caption_decision", [])
            decision = str(getattr(result, "decision", None) or "").strip().lower()
            if decision not in {"keep", "drop"}:
                decision = "keep" if getattr(result, "is_clear_object", None) is True else ""
            decisions[obj_idx] = decision
            self.scene_state["object_caption_decision"] = decisions

            embeddings: List[List[float]] = self.scene_state.get("object_caption_embedding", [])
            if embedding is None:
                embeddings[obj_idx] = []
            else:
                embeddings[obj_idx] = list(embedding)
            self.scene_state["object_caption_embedding"] = embeddings

            siglip2_embeddings: List[List[float]] = self.scene_state.get("object_siglip2_embedding", [])
            siglip2_embedding = getattr(result, "siglip2_cls_embedding", None)
            if siglip2_embedding is None:
                siglip2_embeddings[obj_idx] = []
            else:
                siglip2_embeddings[obj_idx] = list(siglip2_embedding)
            self.scene_state["object_siglip2_embedding"] = siglip2_embeddings

            qwen3_vl_embeddings: List[List[float]] = self.scene_state.get("object_qwen3_vl_embedding", [])
            qwen3_vl_embedding = getattr(result, "qwen3_vl_cls_embedding", None)
            if qwen3_vl_embedding is None:
                qwen3_vl_embeddings[obj_idx] = []
            else:
                qwen3_vl_embeddings[obj_idx] = list(qwen3_vl_embedding)
            self.scene_state["object_qwen3_vl_embedding"] = qwen3_vl_embeddings

            # Persist per-object history for retrieval over merged candidates.
            caption_history = self.scene_state.get("object_caption_history", [])
            caption_history[obj_idx] = self._append_history_value(caption_history[obj_idx], final_caption)
            self.scene_state["object_caption_history"] = caption_history

            caption_embedding_history = self.scene_state.get("object_caption_embedding_history", [])
            caption_embedding_history[obj_idx] = self._append_history_value(
                caption_embedding_history[obj_idx], embedding
            )
            self.scene_state["object_caption_embedding_history"] = caption_embedding_history

            siglip2_embedding_history = self.scene_state.get("object_siglip2_embedding_history", [])
            siglip2_embedding_history[obj_idx] = self._append_history_value(
                siglip2_embedding_history[obj_idx], getattr(result, "siglip2_cls_embedding", None)
            )
            self.scene_state["object_siglip2_embedding_history"] = siglip2_embedding_history

            qwen3_vl_embedding_history = self.scene_state.get("object_qwen3_vl_embedding_history", [])
            qwen3_vl_embedding_history[obj_idx] = self._append_history_value(
                qwen3_vl_embedding_history[obj_idx], getattr(result, "qwen3_vl_cls_embedding", None)
            )
            self.scene_state["object_qwen3_vl_embedding_history"] = qwen3_vl_embedding_history

            if is_recaption:
                prev = int(self.scene_state.get("recaption_applied_total", 0) or 0)
                now_total = prev + 1
                self.scene_state["recaption_applied_total"] = int(now_total)
                self.scene_state["recaption_last_object_id"] = int(winner_id)
                self.scene_state["recaption_last_unix_s"] = float(time.time())
                print(f"[CaptionManager] recaption_applied_total={now_total} object_id={int(winner_id)}")
                hq_flags = self.scene_state.get("high_quality_captioning", [])
                if isinstance(hq_flags, list) and obj_idx < len(hq_flags):
                    hq_flags[obj_idx] = False
                    self.scene_state["high_quality_captioning"] = hq_flags

        if merged_objects or getattr(result, "merge_object_ids", None):
            self._log_merge_event(
                task_object_id=int(result.object_id),
                winner_object_id=int(winner_id),
                requested_merge_ids=getattr(result, "merge_object_ids", None),
                merged_objects=merged_objects,
                final_caption=final_caption,
            )

        if self.debug:
            if merged_objects:
                merged_str = ", ".join(f"{entry['object_id']}:{entry.get('caption', '')!r}" for entry in merged_objects)
                print(f"[CaptionManager] Merged {{{merged_str}}} -> {winner_id} caption={final_caption!r}")
            if not new_caption.strip():
                reason = getattr(result, "empty_reason", None) or "unknown"
                if caption_applied:
                    print(f"[CaptionManager] Empty caption for {winner_id} reason={reason}")
                else:
                    print(
                        "[CaptionManager] Skipped empty caption for"
                        f" {winner_id} reason={reason} existing={existing_caption!r}"
                    )
            elif not merged_objects:
                print(f"[CaptionManager] Caption for {winner_id}: {final_caption!r}")

    def _set_active_flag(self, obj_idx: int, value: bool) -> None:
        active = self.scene_state.get("active")
        if active is None or obj_idx < 0:
            return
        if torch is not None and isinstance(active, torch.Tensor):
            if obj_idx < int(active.numel()):
                active[obj_idx] = bool(value)
            return
        if isinstance(active, list):
            if obj_idx < len(active):
                active[obj_idx] = bool(value)
            return
        with contextlib.suppress(Exception):
            if obj_idx < len(active):
                active[obj_idx] = bool(value)

    def _record_caption_result_stats(self, result: ObjectCaptionResult) -> None:
        self.scene_state["caption_results_total"] = int(self.scene_state.get("caption_results_total", 0) or 0) + 1
        decision = str(getattr(result, "decision", "") or "").strip().lower()
        if decision == "drop" or getattr(result, "is_clear_object", None) is False:
            self.scene_state["caption_results_drop_total"] = int(
                self.scene_state.get("caption_results_drop_total", 0) or 0
            ) + 1
        elif str(getattr(result, "caption", "") or "").strip():
            self.scene_state["caption_results_keep_total"] = int(
                self.scene_state.get("caption_results_keep_total", 0) or 0
            ) + 1

        reason = str(getattr(result, "empty_reason", "") or "").strip()
        if reason:
            self.scene_state["caption_results_empty_total"] = int(
                self.scene_state.get("caption_results_empty_total", 0) or 0
            ) + 1
            counts = self.scene_state.get("caption_error_counts")
            if not isinstance(counts, dict):
                counts = {}
            counts[reason] = int(counts.get(reason, 0) or 0) + 1
            self.scene_state["caption_error_counts"] = counts

    def _deactivate_unclear_object(self, obj_idx: int, *, reason: Optional[str] = None) -> None:
        self._ensure_caption_len(obj_idx + 1)
        self._set_active_flag(obj_idx, False)

        scalar_defaults = {
            "object_caption": "",
            "object_caption_decision": "drop",
            "object_category": "",
            "object_supercategory": "",
            "object_caption_embedding": [],
            "object_siglip2_embedding": [],
            "object_qwen3_vl_embedding": [],
        }
        list_defaults = {
            "object_category_candidates": [],
            "object_key_attributes": [],
            "object_caption_history": [],
            "object_caption_embedding_history": [],
            "object_siglip2_embedding_history": [],
            "object_qwen3_vl_embedding_history": [],
        }
        for key, default in scalar_defaults.items():
            rows = self.scene_state.get(key, [])
            if isinstance(rows, list) and obj_idx < len(rows):
                if isinstance(default, dict):
                    rows[obj_idx] = default.copy()
                elif isinstance(default, list):
                    rows[obj_idx] = list(default)
                else:
                    rows[obj_idx] = default
                self.scene_state[key] = rows
        for key, default in list_defaults.items():
            rows = self.scene_state.get(key, [])
            if isinstance(rows, list) and obj_idx < len(rows):
                rows[obj_idx] = list(default)
                self.scene_state[key] = rows

        hq_flags = self.scene_state.get("high_quality_captioning", [])
        if isinstance(hq_flags, list) and obj_idx < len(hq_flags):
            hq_flags[obj_idx] = False
            self.scene_state["high_quality_captioning"] = hq_flags
        if reason:
            reasons = self.scene_state.get("inactive_reason", [])
            if not isinstance(reasons, list):
                reasons = []
            while len(reasons) <= obj_idx:
                reasons.append("")
            reasons[obj_idx] = str(reason)
            self.scene_state["inactive_reason"] = reasons

    def _merge_objects_if_needed(
        self, target_object_id: int, merge_object_ids: Optional[List[int]], *, task_object_id: Optional[int] = None
    ) -> tuple[int, List[dict]]:
        if torch is None:
            return target_object_id, []
        if not merge_object_ids:
            return target_object_id, []

        means = self.scene_state.get("means")
        features = self.scene_state.get("features")
        cov6 = self.scene_state.get("cov6")
        object_ids = self.scene_state.get("object_id")
        active_flags = self.scene_state.get("active")
        id_redirect = self.scene_state.get("id_redirect")
        captions = self.scene_state.get("object_caption", []) or []

        if (
            means is None
            or features is None
            or cov6 is None
            or object_ids is None
            or not hasattr(means, "shape")
            or means.shape[0] == 0
        ):
            return target_object_id, []
        if id_redirect is None:
            id_redirect = self.scene_state["id_redirect"] = {}

        def _canonical_oid(oid: int) -> int:
            # Resolve transitively; keep id_redirect path-compressed.
            try:
                oid_int = int(oid)
            except Exception:
                return oid
            return self._resolve_canonical_object_id(oid_int)

        # Build the set of all object ids participating in this merge.
        # Same-frame cannot-link pairs are distinct physical instances and
        # should not be collapsed by caption/VLM merge decisions.
        target_canonical_oid = _canonical_oid(target_object_id)
        all_ids = {target_canonical_oid}
        for oid in merge_object_ids:
            try:
                candidate_oid = _canonical_oid(int(oid))
            except Exception:
                continue
            if are_cannot_linked_object_ids(self.scene_state, target_canonical_oid, candidate_oid):
                continue
            all_ids.add(candidate_oid)

        # Resolve to indices
        N = means.shape[0]
        all_indices: List[int] = []
        index_to_oid: dict[int, int] = {}
        for oid in sorted(all_ids):
            idx = self._find_object_index(oid)
            if idx is None:
                continue
            if active_flags is not None and idx < len(active_flags):
                try:
                    is_active = (
                        bool(active_flags[idx].item())
                        if hasattr(active_flags[idx], "item")
                        else bool(active_flags[idx])
                    )
                except Exception:
                    is_active = False
                if not is_active:
                    continue
            if 0 <= idx < N:
                all_indices.append(idx)
                index_to_oid[idx] = oid

        if len(all_indices) <= 1:
            return target_object_id, []

        is_locked = self.scene_state.get("is_locked") or []
        for idx in all_indices:
            if idx < len(is_locked) and bool(is_locked[idx]):
                return target_object_id, []

        # Match mapping's union-find semantics: winner is the lowest *object index*.
        winner_idx = min(all_indices)
        merged_objects: List[dict] = []
        for idx in all_indices:
            if idx == winner_idx:
                continue
            oid = index_to_oid.get(idx)
            if oid is None:
                continue
            cap = ""
            try:
                if idx < len(captions):
                    cap = captions[idx] or ""
            except Exception:
                cap = ""
            merged_objects.append({"object_id": oid, "caption": cap})

        obj_winner_idx = torch.arange(N, dtype=torch.long, device=means.device)
        for idx in all_indices:
            if idx == winner_idx:
                continue
            obj_winner_idx[idx] = winner_idx

        zero_long = torch.zeros((0,), dtype=torch.long, device=means.device)
        zero_feat = features.new_zeros((0, features.shape[1]))
        zero_mu = means.new_zeros((0, 3))
        zero_cov6 = cov6.new_zeros((0, 6))

        update_scene_graph_state(
            self.scene_state,
            zero_mu,
            zero_cov6,
            zero_feat,
            [],
            zero_long,
            obj_winner_idx,
        )
        loser_indices = [idx for idx in all_indices if idx != winner_idx]
        self._merge_object_histories(winner_idx=winner_idx, loser_indices=loser_indices)

        winner_oid = index_to_oid[winner_idx]
        # If the task was for a redirected id, log/apply to the final canonical id.
        winner_oid = _canonical_oid(winner_oid)
        return winner_oid, merged_objects

    def _find_object_index(self, object_id: int) -> Optional[int]:
        object_ids = self.scene_state.get("object_id")
        if object_ids is None:
            return None
        matches = (object_ids == object_id).nonzero(as_tuple=False) if hasattr(object_ids, "nonzero") else None
        if matches is not None and matches.numel() > 0:
            return int(matches.view(-1)[0].item())
        # Fallback for plain lists
        try:
            return list(object_ids).index(object_id)
        except ValueError:
            return None

    def _ensure_caption_len(self, min_len: int) -> None:
        captions: List[str] = self.scene_state.get("object_caption", [])
        while len(captions) < min_len:
            captions.append("")
        self.scene_state["object_caption"] = captions

        decisions: List[str] = self.scene_state.get("object_caption_decision", [])
        if decisions is None or not isinstance(decisions, list):
            decisions = []
        while len(decisions) < min_len:
            decisions.append("")
        self.scene_state["object_caption_decision"] = decisions

        categories: List[str] = self.scene_state.get("object_category", [])
        if categories is None or not isinstance(categories, list):
            categories = []
        while len(categories) < min_len:
            categories.append("")
        self.scene_state["object_category"] = categories

        supercategories: List[str] = self.scene_state.get("object_supercategory", [])
        if supercategories is None or not isinstance(supercategories, list):
            supercategories = []
        while len(supercategories) < min_len:
            supercategories.append("")
        self.scene_state["object_supercategory"] = supercategories

        category_candidates: List[List[str]] = self.scene_state.get("object_category_candidates", [])
        if category_candidates is None or not isinstance(category_candidates, list):
            category_candidates = []
        while len(category_candidates) < min_len:
            category_candidates.append([])
        self.scene_state["object_category_candidates"] = category_candidates

        key_attributes: List[List[str]] = self.scene_state.get("object_key_attributes", [])
        if key_attributes is None or not isinstance(key_attributes, list):
            key_attributes = []
        while len(key_attributes) < min_len:
            key_attributes.append([])
        self.scene_state["object_key_attributes"] = key_attributes

        embeddings = self.scene_state.get("object_caption_embedding", [])
        if embeddings is None or not isinstance(embeddings, list):
            embeddings = []
        while len(embeddings) < min_len:
            embeddings.append([])
        self.scene_state["object_caption_embedding"] = embeddings

        siglip2_embeddings = self.scene_state.get("object_siglip2_embedding", [])
        if siglip2_embeddings is None or not isinstance(siglip2_embeddings, list):
            siglip2_embeddings = []
        while len(siglip2_embeddings) < min_len:
            siglip2_embeddings.append([])
        self.scene_state["object_siglip2_embedding"] = siglip2_embeddings

        qwen3_vl_embeddings = self.scene_state.get("object_qwen3_vl_embedding", [])
        if qwen3_vl_embeddings is None or not isinstance(qwen3_vl_embeddings, list):
            qwen3_vl_embeddings = []
        while len(qwen3_vl_embeddings) < min_len:
            qwen3_vl_embeddings.append([])
        self.scene_state["object_qwen3_vl_embedding"] = qwen3_vl_embeddings

        caption_history = self.scene_state.get("object_caption_history", [])
        if caption_history is None or not isinstance(caption_history, list):
            caption_history = []
        while len(caption_history) < min_len:
            caption_history.append([])
        self.scene_state["object_caption_history"] = caption_history

        caption_embedding_history = self.scene_state.get("object_caption_embedding_history", [])
        if caption_embedding_history is None or not isinstance(caption_embedding_history, list):
            caption_embedding_history = []
        while len(caption_embedding_history) < min_len:
            caption_embedding_history.append([])
        self.scene_state["object_caption_embedding_history"] = caption_embedding_history

        siglip2_embedding_history = self.scene_state.get("object_siglip2_embedding_history", [])
        if siglip2_embedding_history is None or not isinstance(siglip2_embedding_history, list):
            siglip2_embedding_history = []
        while len(siglip2_embedding_history) < min_len:
            siglip2_embedding_history.append([])
        self.scene_state["object_siglip2_embedding_history"] = siglip2_embedding_history

        qwen3_vl_embedding_history = self.scene_state.get("object_qwen3_vl_embedding_history", [])
        if qwen3_vl_embedding_history is None or not isinstance(qwen3_vl_embedding_history, list):
            qwen3_vl_embedding_history = []
        while len(qwen3_vl_embedding_history) < min_len:
            qwen3_vl_embedding_history.append([])
        self.scene_state["object_qwen3_vl_embedding_history"] = qwen3_vl_embedding_history

        high_quality_captioning = self.scene_state.get("high_quality_captioning", [])
        if high_quality_captioning is None or not isinstance(high_quality_captioning, list):
            high_quality_captioning = []
        while len(high_quality_captioning) < min_len:
            high_quality_captioning.append(False)
        self.scene_state["high_quality_captioning"] = high_quality_captioning

        high_quality_views = self.scene_state.get("high_quality_views", [])
        if high_quality_views is None or not isinstance(high_quality_views, list):
            high_quality_views = []
        while len(high_quality_views) < min_len:
            high_quality_views.append([])
        self.scene_state["high_quality_views"] = high_quality_views

    @staticmethod
    def _append_history_value(history_row: object, value: object) -> List[object]:
        row = list(history_row) if isinstance(history_row, list) else []
        if value is None:
            return row
        if isinstance(value, str):
            text = str(value).strip()
            if not text:
                return row
            if row and str(row[-1]) == text:
                return row
            row.append(text)
            return row
        if isinstance(value, (list, tuple)):
            vec: List[float] = []
            if len(value) > 0:
                try:
                    vec = [float(x) for x in value]
                except Exception:
                    vec = []
            if not vec:
                return row
            if row and isinstance(row[-1], list) and len(row[-1]) == len(vec) and row[-1] == vec:
                return row
            row.append(vec)
            return row
        return row

    @staticmethod
    def _to_float_vector(value: object) -> List[float]:
        if not isinstance(value, (list, tuple)) or not value:
            return []
        try:
            return [float(x) for x in value]
        except Exception:
            return []

    @classmethod
    def _cosine_similarity(cls, vec_a: object, vec_b: object) -> Optional[float]:
        a = cls._to_float_vector(vec_a)
        b = cls._to_float_vector(vec_b)
        if not a or not b or len(a) != len(b):
            return None
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na <= 0.0 or nb <= 0.0:
            return None
        return float(dot / (math.sqrt(na) * math.sqrt(nb)))

    @staticmethod
    def _observation_area(obs: object) -> float:
        if isinstance(obs, dict):
            for bbox_key, size_key in (
                ("bbox_caption", "size_caption"),
                ("bbox", "size"),
                ("bbox_source", "size_source"),
            ):
                bbox = obs.get(bbox_key)
                size = obs.get(size_key)
                if (
                    isinstance(bbox, (list, tuple))
                    and isinstance(size, (list, tuple))
                    and len(bbox) >= 4
                    and len(size) >= 2
                ):
                    with contextlib.suppress(Exception):
                        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
                        w = max(0.0, min(float(size[0]), x1) - max(0.0, x0))
                        h = max(0.0, min(float(size[1]), y1) - max(0.0, y0))
                        area = w * h
                        if area > 0.0:
                            return area
            with contextlib.suppress(Exception):
                area_src = float(obs.get("bbox_area_source", 0.0) or 0.0)
                if area_src > 0.0:
                    return area_src
            for image_key in ("image_caption", "image"):
                img = obs.get(image_key)
                shape = getattr(img, "shape", None)
                if shape is None:
                    continue
                with contextlib.suppress(Exception):
                    dims = [int(x) for x in tuple(shape)]
                    if len(dims) >= 2:
                        if len(dims) == 3 and dims[0] in (1, 3, 4):
                            h, w = dims[1], dims[2]
                        else:
                            h, w = dims[0], dims[1]
                        area = float(max(h, 0) * max(w, 0))
                        if area > 0.0:
                            return area
        shape = getattr(obs, "shape", None)
        if shape is not None:
            with contextlib.suppress(Exception):
                dims = [int(x) for x in tuple(shape)]
                if len(dims) >= 2:
                    if len(dims) == 3 and dims[0] in (1, 3, 4):
                        h, w = dims[1], dims[2]
                    else:
                        h, w = dims[0], dims[1]
                    area = float(max(h, 0) * max(w, 0))
                    if area > 0.0:
                        return area
        return 0.0

    def _select_merge_representative_indices(self, obj_idx: int, *, max_keep: int = 2) -> List[int]:
        if max_keep <= 0:
            return []
        rgb_rows_all = self.scene_state.get("rgb_observations", []) or []
        sig_hist_all = self.scene_state.get("object_siglip2_embedding_history", []) or []
        rgb_row = (
            rgb_rows_all[obj_idx] if obj_idx < len(rgb_rows_all) and isinstance(rgb_rows_all[obj_idx], list) else []
        )
        sig_row = (
            sig_hist_all[obj_idx] if obj_idx < len(sig_hist_all) and isinstance(sig_hist_all[obj_idx], list) else []
        )

        row_count = max(len(rgb_row), len(sig_row))
        if row_count <= 0:
            return []

        areas = [self._observation_area(rgb_row[i]) if i < len(rgb_row) else 0.0 for i in range(row_count)]
        if any(area > 0.0 for area in areas):
            primary_idx = max(range(row_count), key=lambda i: (areas[i], -i))
        else:
            primary_idx = 0

        keep_indices: List[int] = [int(primary_idx)]
        if max_keep == 1:
            return keep_indices

        primary_sig = sig_row[primary_idx] if primary_idx < len(sig_row) else None
        secondary_idx: Optional[int] = None
        best_cos: Optional[float] = None
        for idx in range(row_count):
            if idx == primary_idx or idx >= len(sig_row):
                continue
            cos = self._cosine_similarity(primary_sig, sig_row[idx])
            if cos is None:
                continue
            if best_cos is None or cos < best_cos:
                best_cos = cos
                secondary_idx = idx

        if secondary_idx is None:
            area_candidates = [i for i in range(row_count) if i != primary_idx and areas[i] > 0.0]
            if area_candidates:
                secondary_idx = max(area_candidates, key=lambda i: (areas[i], -i))

        if secondary_idx is None:
            for idx in range(row_count):
                if idx != primary_idx:
                    secondary_idx = idx
                    break

        if secondary_idx is not None:
            keep_indices.append(int(secondary_idx))
        return keep_indices[:max_keep]

    @staticmethod
    def _slice_list_row(row_value: object, keep_indices: List[int]) -> List[object]:
        row = list(row_value) if isinstance(row_value, list) else []
        if not row or not keep_indices:
            return []
        out: List[object] = []
        for idx in keep_indices:
            if 0 <= idx < len(row):
                out.append(row[idx])
        return out

    def _prune_object_representatives_after_merge(self, obj_idx: int) -> None:
        keep_indices = self._select_merge_representative_indices(obj_idx, max_keep=2)
        if not keep_indices:
            return

        for key in (
            "object_caption_history",
            "object_caption_embedding_history",
            "object_siglip2_embedding_history",
            "object_qwen3_vl_embedding_history",
            "rgb_observations",
            "view_means",
            "view_cov6",
            "high_quality_views",
        ):
            rows = self.scene_state.get(key, [])
            if not isinstance(rows, list) or obj_idx < 0 or obj_idx >= len(rows):
                continue
            rows[obj_idx] = self._slice_list_row(rows[obj_idx], keep_indices)
            self.scene_state[key] = rows

        rgb_rows = self.scene_state.get("rgb_observations", [])
        view_means_rows = self.scene_state.get("view_means", [])
        view_cov6_rows = self.scene_state.get("view_cov6", [])
        high_quality_rows = self.scene_state.get("high_quality_views", [])
        if (
            isinstance(rgb_rows, list)
            and isinstance(view_means_rows, list)
            and isinstance(view_cov6_rows, list)
            and isinstance(high_quality_rows, list)
            and obj_idx < len(rgb_rows)
            and obj_idx < len(view_means_rows)
            and obj_idx < len(view_cov6_rows)
            and obj_idx < len(high_quality_rows)
            and isinstance(rgb_rows[obj_idx], list)
            and isinstance(view_means_rows[obj_idx], list)
            and isinstance(view_cov6_rows[obj_idx], list)
            and isinstance(high_quality_rows[obj_idx], list)
        ):
            common = min(
                len(rgb_rows[obj_idx]),
                len(view_means_rows[obj_idx]),
                len(view_cov6_rows[obj_idx]),
                len(high_quality_rows[obj_idx]),
            )
            rgb_rows[obj_idx] = rgb_rows[obj_idx][:common]
            view_means_rows[obj_idx] = view_means_rows[obj_idx][:common]
            view_cov6_rows[obj_idx] = view_cov6_rows[obj_idx][:common]
            high_quality_rows[obj_idx] = high_quality_rows[obj_idx][:common]
            self.scene_state["rgb_observations"] = rgb_rows
            self.scene_state["view_means"] = view_means_rows
            self.scene_state["view_cov6"] = view_cov6_rows
            self.scene_state["high_quality_views"] = high_quality_rows
            hq_flags = self.scene_state.get("high_quality_captioning", [])
            if isinstance(hq_flags, list) and obj_idx < len(hq_flags) and common == 0:
                hq_flags[obj_idx] = False
                self.scene_state["high_quality_captioning"] = hq_flags

        object_image_ids = self.scene_state.get("object_image_ids", [])
        if isinstance(object_image_ids, list) and obj_idx < len(object_image_ids):
            selected_obs = rgb_rows[obj_idx] if isinstance(rgb_rows, list) and obj_idx < len(rgb_rows) else []
            selected_image_ids: List[int] = []
            for obs in selected_obs if isinstance(selected_obs, list) else []:
                if not isinstance(obs, dict):
                    continue
                with contextlib.suppress(Exception):
                    image_id = int(obs.get("image_id"))
                    if image_id not in selected_image_ids:
                        selected_image_ids.append(image_id)
            if selected_image_ids:
                object_image_ids[obj_idx] = selected_image_ids
            else:
                object_image_ids[obj_idx] = self._slice_list_row(object_image_ids[obj_idx], list(range(2)))
            self.scene_state["object_image_ids"] = object_image_ids

        scalar_history_pairs = (
            ("object_caption", "object_caption_history"),
            ("object_caption_embedding", "object_caption_embedding_history"),
            ("object_siglip2_embedding", "object_siglip2_embedding_history"),
            ("object_qwen3_vl_embedding", "object_qwen3_vl_embedding_history"),
        )
        for scalar_key, history_key in scalar_history_pairs:
            scalars = self.scene_state.get(scalar_key, [])
            histories = self.scene_state.get(history_key, [])
            if (
                not isinstance(scalars, list)
                or not isinstance(histories, list)
                or obj_idx < 0
                or obj_idx >= len(scalars)
                or obj_idx >= len(histories)
            ):
                continue
            row = histories[obj_idx] if isinstance(histories[obj_idx], list) else []
            latest = row[0] if row else None
            if scalar_key == "object_caption":
                scalars[obj_idx] = str(latest or "")
            elif isinstance(latest, (list, tuple)):
                scalars[obj_idx] = self._to_float_vector(latest)
            else:
                scalars[obj_idx] = []
            self.scene_state[scalar_key] = scalars

    def _merge_object_histories(self, *, winner_idx: int, loser_indices: List[int]) -> None:
        if winner_idx < 0:
            return
        self._ensure_caption_len(winner_idx + 1)
        max_idx = winner_idx
        for idx in loser_indices:
            if idx > max_idx:
                max_idx = idx
        self._ensure_caption_len(max_idx + 1)

        scalar_to_history = (
            ("object_caption", "object_caption_history"),
            ("object_caption_embedding", "object_caption_embedding_history"),
            ("object_siglip2_embedding", "object_siglip2_embedding_history"),
            ("object_qwen3_vl_embedding", "object_qwen3_vl_embedding_history"),
        )
        # Ensure scalar latest values are represented in history before moving losers.
        for scalar_key, history_key in scalar_to_history:
            scalars = self.scene_state.get(scalar_key, []) or []
            history = self.scene_state.get(history_key, []) or []
            for idx in [winner_idx, *loser_indices]:
                if idx < 0:
                    continue
                scalar_val = scalars[idx] if idx < len(scalars) else None
                row = history[idx] if idx < len(history) else []
                history[idx] = self._append_history_value(row, scalar_val)
            self.scene_state[history_key] = history

        merge_keys = (
            "object_caption_history",
            "object_caption_embedding_history",
            "object_siglip2_embedding_history",
            "object_qwen3_vl_embedding_history",
        )
        for key in merge_keys:
            rows = self.scene_state.get(key, []) or []
            winner_row = list(rows[winner_idx]) if winner_idx < len(rows) and isinstance(rows[winner_idx], list) else []
            for idx in loser_indices:
                if idx < 0 or idx >= len(rows):
                    continue
                loser_row = rows[idx] if isinstance(rows[idx], list) else []
                winner_row.extend(loser_row)
                rows[idx] = []
            rows[winner_idx] = winner_row
            self.scene_state[key] = rows

        self._prune_object_representatives_after_merge(winner_idx)

    # Worker lifecycle ----------------------------------------------------
    def maybe_start_worker(self) -> None:
        if not self.enabled:
            return
        if self.worker is None:
            if self._worker_factory is not None:
                self.worker = self._worker_factory(self.tasks_queue, self.results_queue)
            else:
                self.worker = CaptionWorker(
                    self.scene_state,
                    self.tasks_queue,
                    self.results_queue,
                    caption_batch_size=self._caption_batch_size,
                    device=self._caption_device,
                    caption_server=self._caption_server,
                    max_retries=self._caption_max_retries,
                    debug=self.debug,
                    caption_spatial_context=self._caption_spatial_context,
                    caption_spatial_context_include_position=self._caption_spatial_context_include_position,
                    recaption_time_threshold_sec=self._recaption_time_threshold_sec,
                    caption_merge_hellinger_thresh=self._caption_merge_hellinger_thresh,
                    caption_merge_caption_thresh=self._caption_merge_caption_thresh,
                    caption_merge_siglip2_thresh=self._caption_merge_siglip2_thresh,
                    caption_merge_require_visual=self._caption_merge_require_visual,
                    caption_merge_require_category_compat=self._caption_merge_require_category_compat,
                    caption_visual_prompt_mode=self._caption_visual_prompt_mode,
                    caption_prompt_variant=self._caption_prompt_variant,
                    caption_version=self._caption_version,
                )
        # Only start if not already running
        if hasattr(self.worker, "start") and callable(getattr(self.worker, "start")):
            # Some fake workers may have no thread; guard against double-start
            # CaptionWorker.start may raise if the thread already started.
            with contextlib.suppress(RuntimeError):
                self.worker.start()
